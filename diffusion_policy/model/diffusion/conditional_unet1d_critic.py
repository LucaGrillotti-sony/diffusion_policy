from __future__ import annotations

from typing import Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusers import DDPMScheduler
from einops.layers.torch import Rearrange

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalResidualBlock1D
from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.policy.diffusion_guided_ddim import DDIMGuidedScheduler

logger = logging.getLogger(__name__)


class ConditionalUnet1DCritic(nn.Module):
    def __init__(self,
                 input_dim,
                 local_cond_dim=None,
                 global_cond_dim=None,
                 diffusion_step_embed_dim=256,
                 down_dims=None,
                 kernel_size=3,
                 n_groups=8,
                 cond_predict_scale=False
                 ):
        super().__init__()
        if down_dims is None:
            down_dims = [256, 512, 1024]
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        # dsed = diffusion_step_embed_dim
        # diffusion_step_encoder = nn.Sequential(
        #     SinusoidalPosEmb(dsed),
        #     nn.Linear(dsed, dsed * 4),
        #     nn.Mish(),
        #     nn.Linear(dsed * 4, dsed),
        # )
        cond_dim = 0
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        final_dim = all_dims[-1]
        critic_value_dim = 1
        final_conv = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=final_dim * 2, out_features=256),
            nn.SELU(),
            nn.Linear(in_features=256, out_features=1)
        )

        # self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self,
                sample: torch.Tensor,
                local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        global_feature = global_cond


        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        x = self.final_conv(x)

        # x = einops.rearrange(x, 'b t h -> b h t')
        return x


class DoubleCritic(nn.Module):
    def __init__(self,
                 obs_feature_dim,
                 shape_meta: dict,
                 n_obs_steps,
                 n_action_steps,
                 horizon,
                 obs_as_global_cond=True,
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 cond_predict_scale=True,
                 gamma=0.99,
                 **_,
                 ):
        super().__init__()
        if down_dims is None:
            down_dims = [256, 512, 1024]

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        self.critic_model_1 = ConditionalUnet1DCritic(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.critic_model_2 = ConditionalUnet1DCritic(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.normalizer = None
        self.gamma = gamma
        self.obs_as_global_cond = obs_as_global_cond
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon

    def forward(self, x, local_cond=None, global_cond=None, **kwargs):
        return (self.critic_model_1(x, local_cond, global_cond, **kwargs),
                self.critic_model_2(x, local_cond, global_cond, **kwargs))

    def get_min_critics(self, x, local_cond=None, global_cond=None, consider_action_full_horizon=True, **kwargs, ):
        if consider_action_full_horizon:
            x = self.extract_executed_actions(x)
        critic_values_1 = self.critic_model_1(x, local_cond, global_cond, **kwargs)
        critic_values_2 = self.critic_model_2(x, local_cond, global_cond, **kwargs)
        concat_critic_values = torch.cat([critic_values_1, critic_values_2], dim=-1)
        return torch.min(concat_critic_values, dim=-1)[0]

    def get_one_critic(self, x, local_cond=None, global_cond=None, consider_action_full_horizon=True, **kwargs):
        if consider_action_full_horizon:
            x = self.extract_executed_actions(x)

        return self.critic_model_1(x, local_cond, global_cond, **kwargs)

    def set_normalizer(self, normalizer):
        if self.normalizer is not None:
            raise ValueError("Normalizer already set")
        self.normalizer = normalizer

    def extract_executed_actions(self, action_full_horizon):
        # get action
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        executed_action = action_full_horizon[:, start:end]
        return executed_action

    def get_global_cond(self, batch, nobs_features):
        nobs_features = nobs_features.detach()

        value = next(iter(batch["obs"].values()))
        B, To = value.shape[:2]

        if self.obs_as_global_cond:
            global_cond = nobs_features.reshape(B, -1)
        else:
            raise NotImplementedError
        return global_cond

    def get_normalized_actions(self, batch):
        actions = batch['action'].detach()
        nactions = self.normalizer['action'].normalize(actions)
        return nactions

    def compute_critic_loss(self, batch, nobs_features, critic_target: DoubleCritic, policy, rewards):
        nobs_features = nobs_features.detach()

        # normalize input
        assert 'valid_mask' not in batch
        nactions = self.get_normalized_actions(batch)

        # handle different ways of passing observation
        local_cond = None

        global_cond = self.get_global_cond(batch, nobs_features)

        next_obs = batch['next_obs']
        # rewards = self.calculate_reward(batch['obs'], batch['action'], next_obs)
        gamma = self.gamma

        # generate impainting mask
        # condition_mask = self.mask_generator(trajectory.shape)  # todo What to do with this?

        self.extract_executed_actions(nactions)

        nactions_executed = self.extract_executed_actions(nactions)
        critic_values_1 = self.critic_model_1(nactions_executed, local_cond=local_cond, global_cond=global_cond)
        critic_values_2 = self.critic_model_2(nactions_executed, local_cond=local_cond, global_cond=global_cond)

        actions_target = policy.predict_action(next_obs)
        nactions_target = self.normalizer['action'].normalize(actions_target['action_pred'])
        critic_target_values = critic_target.get_min_critics(nactions_target, local_cond=local_cond, global_cond=global_cond, consider_action_full_horizon=True)

        critic_target_values = rewards + gamma * critic_target_values.ravel()
        critic_target_values = critic_target_values.detach()

        loss_critic = F.mse_loss(critic_values_1.ravel(), critic_target_values.ravel(), reduction='none').mean() + \
                      F.mse_loss(critic_values_2.ravel(), critic_target_values.ravel(), reduction='none').mean()
        # loss_critic = loss_critic.mean()

        metrics = {
            "critic_loss": loss_critic,
        }

        return loss_critic, metrics
