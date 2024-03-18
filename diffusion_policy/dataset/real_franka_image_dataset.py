import random
from typing import Dict, List
import torch
import numpy as np
import quaternion as quat
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)


class RandomFourierFeatures:
    def __init__(self, encoding_size: int, vector_size: int, period_adjustment_rff: float):
        assert encoding_size % 2 == 0
        self.number_features = encoding_size // 2

        rng = np.random.default_rng(12345)

        self.B = rng.normal(size=(vector_size, self.number_features)) * period_adjustment_rff

    def encode_vector(self, v):
        matmul = np.matmul(v, self.B)
        cos_data = np.cos(2 * np.pi * matmul)
        sin_data = np.sin(2 * np.pi * matmul)
        return np.concatenate([cos_data, sin_data], axis=-1)


class RealFrankaImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            dt: float,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_action_steps=None,
            n_latency_steps=0,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            delta_action=False,
            mass_encoding_size=256,
            proba_diffusion_remove_mass_label=0.1,
            period_adjustment_rff=0.15,
        ):
        # dataset_path = "/home/ros/humble/src/diffusion_policy/data/fake_puree_experiments/diffusion_policy_dataset"
        assert os.path.isdir(dataset_path)
        
        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore(),
                            dt=dt,
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore(),
                dt=dt,
            )
        
        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # key_first_k = dict()
        # if n_obs_steps is not None:
        #     # only take first k obs from images
        #     for key in rgb_keys + lowdim_keys:
        #         key_first_k[key] = n_obs_steps
        empty_key_first_k = dict()

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=empty_key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.mass_encoding_size = mass_encoding_size
        self.period_adjustment_rff = period_adjustment_rff

        self.rff_encoder = RandomFourierFeatures(encoding_size=self.mass_encoding_size, vector_size=2, period_adjustment_rff=self.period_adjustment_rff)
        self.proba_diffusion_remove_mass_label = proba_diffusion_remove_mass_label

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def encode_mass(self, mass):
        mass_len = mass.shape[0]
        if random.random() < self.proba_diffusion_remove_mass_label:
            mass_v = np.asarray([[0., 1.] for _ in range(mass_len)])
        else:
            mass_v = np.hstack((mass.reshape(mass_len, 1), np.zeros(shape=(mass_len, 1))))
            # mass_v = np.asarray([mass, 0.] for _ in range(mass_len))

        mass_encoding = self.rff_encoder.encode_vector(mass_v)
        return mass_encoding

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        # normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
        #     self.replay_buffer['action'])
        print('Fitting normalizer for action')
        all_sequences = list()
        for i in range(self.__len__()):
            if i % 10 == 0:
                print(f'Fitting normalizer for action: {i}/{self.__len__()}', end='\r')
            all_sequences.append(self[i])
        # all_sequences = [self[i] for i in range(self.__len__())]
        all_relative_seq = list()
        for _seq in all_sequences:
            relative_seq = _seq['action']
            all_relative_seq.append(relative_seq)
        all_relative_actions = np.concatenate(all_relative_seq, axis=0)
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(all_relative_actions)
        
        # obs
        for key in self.lowdim_keys:
            if key == 'mass':
                normalizer[key] = SingleFieldLinearNormalizer.create_identity()
            else:
                normalizer[key] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer[key])
        
        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)
        next_T_slice = slice(self.n_action_steps, self.n_obs_steps+self.n_action_steps)

        obs_dict = dict()
        next_obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            next_obs_dict[key] = np.moveaxis(data[key][next_T_slice], -1, 1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            next_obs_dict[key] = data[key][next_T_slice].astype(np.float32)
            # save ram
            del data[key]

        action = data['action'].astype(np.float32)
        # labels = data['label'].astype(np.uint8)
        # labels = labels[T_slice]

        assert 'mass' not in obs_dict
        # if 'mass' in obs_dict:
        #     obs_dict['mass'] = self.encode_mass(obs_dict['mass'])
        #     next_obs_dict['mass'] = self.encode_mass(next_obs_dict['mass'])

        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:self.n_latency_steps+self.horizon]
        else:
            action = action[:self.horizon]

        action = self.compute_action_relative_to_initial_eef(action, action[0])

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
            # 'next_obs': dict_apply(next_obs_dict, torch.from_numpy),
            # 'label': torch.from_numpy(labels),
        }
        return torch_data

    @staticmethod
    def compute_action_relative_to_initial_eef(action, initial_eef):
        assert action.shape[-1] == 7

        xyz_action = action[:, :3]
        xyz_initial_eef = initial_eef[:3]
        relative_xyz = xyz_action - xyz_initial_eef

        # quaternion relative rotations
        q_action = quat.from_float_array(action[:, 3:])
        q_initial_eef = quat.from_float_array(initial_eef[3:])

        q_relative = q_action * q_initial_eef.conjugate()

        q_relative = quat.as_float_array(q_relative)

        return np.concatenate([relative_xyz, q_relative], axis=-1)


def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[...,idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr

def _get_replay_buffer(dataset_path, shape_meta, store, dt):
    # parse shape meta
    rgb_keys = list()
    lowdim_keys = list()
    out_resolutions = dict()
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = tuple(attr.get('shape'))
        if type == 'rgb':
            rgb_keys.append(key)
            c,h,w = shape
            out_resolutions[key] = (w,h)
        elif type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            if 'pose' in key:
                assert tuple(shape) in [(2,),(6,)]
    
    action_shape = tuple(shape_meta['action']['shape'])
    label_shape = tuple(shape_meta['label']['shape'])
    # assert action_shape in [(2,),(6,)]

    # load data
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            # lowdim_keys=lowdim_keys + ['action'] + ['label', 'mass'],
            lowdim_keys=lowdim_keys + ['action'],
            image_keys=rgb_keys,
            dt=dt,
        )

    # transform lowdim dimensions
    if action_shape == (2,):
        # 2D action space, only controls X and Y
        zarr_arr = replay_buffer['action']
        zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])

    # # transform lowdim dimensions
    # if action_shape == (2,):
    #     zarr_arr = replay_buffer['label']
    #     zarr_resize_index_last_dim(zarr_arr, idxs=[0, 1])

    for key, shape in lowdim_shapes.items():
        if 'pose' in key and shape == (2,):
            # only take X and Y
            zarr_arr = replay_buffer[key]
            zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])

    return replay_buffer


def test():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize('../diffusion_policy/config'):
        cfg = hydra.compose('train_robomimic_real_image_workspace')
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'][:])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
    _ = plt.hist(dists, bins=100); plt.title('real action velocity')
