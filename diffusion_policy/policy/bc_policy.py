from __future__ import annotations

from typing import Dict, Tuple
import math

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d_critic import DoubleCritic
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules, custom_tree_map
from diffusion_policy.policy.diffusion_guided_ddim import DDIMGuidedScheduler


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            n_obs_steps,
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )


        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]

        model = torch.nn.Sequential(
            nn.Linear(obs_feature_dim, 256),
            nn.SELU(),
            nn.Linear(256, 256),
            nn.SELU(),
            nn.Linear(256, action_dim),
        )

        self.obs_encoder = obs_encoder
        self.model = model

        self.normalizer = LinearNormalizer()
        self.horizon = 3
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = 1
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs


        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], neutral_obs_dict=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key

        neutral_obs_dict: same as obs_dict but with the neutral mass - setting it triggers classification free guidance
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, _ = value.shape[:2]
        To = self.n_obs_steps

        # build input
        device = self.device

        # handle different ways of passing observation
        # condition through global feature
        this_nobs = dict_apply(nobs, lambda x: x[:, :To,...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs).to(device)
        nobs_features = nobs_features.reshape(B, -1)

        naction_pred = self.model(nobs_features)
        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        metrics = dict()

        result = {
            'action': action_pred,
            'action_pred': action_pred,
            'metrics': metrics,
            'full_sample': action_pred,
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])

        this_nobs = dict_apply(nobs,
                               lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])) #  get the first n_obs_steps and flatten

        nactions_dataset = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions_dataset.shape[0]

        nobs_features = self.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(batch_size, -1)

        pred = self.model(nobs_features)
        target = nactions_dataset[self.n_obs_steps]  # Only predicting the actions following observations.

        loss_bc = F.mse_loss(pred, target, reduction='none')
        loss_bc = reduce(loss_bc, 'b ... -> b (...)', 'mean')
        loss_bc = loss_bc.mean()

        loss_actor = loss_bc

        metrics = {
            "diffusion_loss": loss_bc,
        }
        other_data = {
            "nobs_features": nobs_features,
        }
        return loss_actor, metrics, other_data

    def calculate_reward(self, obs, action, next_obs):
        start_index = self.n_obs_steps - 1
        end_index = start_index + self.n_action_steps
        rewards, _ = torch.vmap(DDIMGuidedScheduler.scoring_fn, in_dims=(0, None, None, None))(action, self.horizon, self.n_action_steps, self.n_obs_steps)
        return rewards

    def compute_obs_encoding(self, batch, detach=False):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        # handle different ways of passing observation

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs,
            lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)

        if detach:
            nobs_features = nobs_features.detach()
        return nobs_features
