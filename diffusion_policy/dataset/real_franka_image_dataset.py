import random
from typing import Dict, List

import toolz
import torch
import numpy as np
import quaternion as quat
from torchvision.transforms import v2
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


class DataAugmentationRandomShifts:
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        *n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = torch.nn.functional.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        # base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift

        grid = grid.unsqueeze(0).repeat(*n, 1, 1, 1)

        return torch.nn.functional.grid_sample(x,
                                               grid,
                                               padding_mode='zeros',
                                               align_corners=False)


class RealFrankaImageDataset(BaseImageDataset):
    # # Choosing initial EEF from dataset.
    # FIXED_INITIAL_EEF = np.asarray([0.4000259, 0.04169807, 0.43917269,  # XYZ
    #                                 0.0368543, 0.97042745, 0.23850703, 0.00517287],  # Quaternion
    #                                )

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
                 augment_data=False,
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
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start + 1:end] = np.diff(actions[start:end], axis=0)
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
            sequence_length=horizon + n_latency_steps,
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

        self.proba_diffusion_remove_mass_label = proba_diffusion_remove_mass_label

        if augment_data:
            self.augment_data_fn = self.get_augment_data_fn()
            print("data augmented - true")
        else:
            self.augment_data_fn = None
            print("data augmented - false")

        self.NUM_CLASSES = 3

    @property
    def augment_data(self):
        return self.augment_data_fn is not None

    def get_augment_data_fn(self):
        jitter = v2.ColorJitter(brightness=.5, hue=.3)
        random_shift_fn = DataAugmentationRandomShifts(pad=0).forward

        augment_data_fn = toolz.compose(random_shift_fn, jitter)
        return augment_data_fn

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
        )
        val_set.val_mask = ~self.val_mask
        val_set.augment_data_fn = None
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer['action'])

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

    @staticmethod
    def concatenate_rgb_depth(rgb, depth):
        return np.concatenate([rgb, np.expand_dims(np.mean(depth, axis=-1), axis=-1)], axis=-1)

    @staticmethod
    def moveaxis_rgbd(data, single_rgb=False):
        if single_rgb:
            return np.moveaxis(data, -1, 0)
        else:
            return np.moveaxis(data, -1, 1)

    @staticmethod
    def rgbd_255_to_1(data):
        return data.astype(np.float32) / 255.

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # import time
        # print(time.time(), self.__len__())
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)
        next_T_slice = slice(self.n_action_steps, self.n_obs_steps + self.n_action_steps)

        obs_dict = dict()
        next_obs_dict = dict()

        # camera_0 is depth
        # camera_1 is color

        # TODO
        assert len(self.rgb_keys) == 1
        assert "camera_1" in self.rgb_keys
        assert "camera_0" not in self.rgb_keys
        obs_dict["camera_1"] = self.rgbd_255_to_1(self.moveaxis_rgbd(data["camera_1"][T_slice]))
        next_obs_dict["camera_1"] = self.rgbd_255_to_1(self.moveaxis_rgbd(data["camera_1"][next_T_slice]))
        del data["camera_1"]  # save ram

        # TODO
        # for key in self.rgb_keys:
        #     # move channel last to channel first
        #     # T,H,W,C
        #     # convert uint8 image to float32
        #     # TODO: temporary fix to add depth image; change camera_1 for camera_0_depth
        #
        #     # camera_0 is depth
        #     # camera_1 is color
        #
        #     if "camera_0" not in self.rgb_keys:
        #         print("camera 0 not in rgb keys!")
        #
        #
        #     obs_dict["camera_0"] = self.rgbd_255_to_1(self.moveaxis_rgbd(data[key][T_slice]))
        #     next_obs_dict["camera_0"] = self.rgbd_255_to_1(self.moveaxis_rgbd(data[key][next_T_slice]))

        # if key == "camera_1":
        #     continue
        # if key == "camera_0":
        #     obs_dict["camera_0"] = self.concatenate_rgb_depth(data["camera_1"][T_slice], data["camera_0"][T_slice])
        #     next_obs_dict["camera_0"] = self.concatenate_rgb_depth(data["camera_1"][next_T_slice], data["camera_0"][next_T_slice])
        #
        #     obs_dict[key] = self.rgbd_255_to_1(self.moveaxis_rgbd(obs_dict["camera_0"]))
        #     next_obs_dict[key] = self.rgbd_255_to_1(self.moveaxis_rgbd(next_obs_dict["camera_0"]))
        # else:
        #     obs_dict[key] = self.rgbd_255_to_1(self.moveaxis_rgbd(data[key][T_slice]))
        #     next_obs_dict[key] = self.rgbd_255_to_1(self.moveaxis_rgbd(data[key][next_T_slice]))

        # obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
        #     ).astype(np.float32) / 255.
        # next_obs_dict[key] = np.moveaxis(data[key][next_T_slice], -1, 1
        #     ).astype(np.float32) / 255.
        # T,C,H,W
        # save ram
        # del data[key]

        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            next_obs_dict[key] = data[key][next_T_slice].astype(np.float32)

            del data[key]

        action = data['action'].astype(np.float32)
        labels_all = data['label'].astype(np.uint8)
        labels_obs = labels_all[T_slice]
        labels_next_obs = labels_all[next_T_slice]

        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:self.n_latency_steps + self.horizon]
        else:
            action = action[:self.horizon]

        # action = self.compute_action_relative_to_initial_eef(action, action[0])
        # action = self.compute_action_relative_to_initial_eef(action, self.FIXED_INITIAL_EEF)

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
            'next_obs': dict_apply(next_obs_dict, torch.from_numpy),
            'label': torch.from_numpy(labels_obs),
        }

        if 'mass' in obs_dict:
            true_mass = obs_dict['mass']
            torch_data.update({'true_mass': torch.from_numpy(true_mass)})

        torch_data['obs'] = self.add_scooping_accomplished_to_batch(torch_data['obs'], labels_obs)
        torch_data['next_obs'] = self.add_scooping_accomplished_to_batch(torch_data['next_obs'], labels_next_obs)

        if self.augment_data:
            torch_data['obs']["camera_1"] = self.augment_data_fn(torch_data['obs']["camera_1"])
            torch_data['next_obs']["camera_1"] = self.augment_data_fn(torch_data['next_obs']["camera_1"])

        return torch_data

    def add_scooping_accomplished_to_batch(self, obs_dict, labels):
        import torch.nn.functional as F

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()

        scooping_accomplished_one_hot = F.one_hot(labels, num_classes=self.NUM_CLASSES)
        obs_dict["scooping_accomplished"] = scooping_accomplished_one_hot

        return obs_dict

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

    @staticmethod
    def compute_absolute_action(relative_action, initial_eef):
        assert relative_action.shape[-1] == 7

        xyz_relative = relative_action[:, :3]
        xyz_initial_eef = initial_eef[:3]
        absolute_xyz = xyz_relative + xyz_initial_eef

        rot_relative = relative_action[:, 3:]
        rot_relative = rot_relative / np.linalg.norm(rot_relative, axis=1).reshape(-1, 1)
        # print("rot_relative.shape", rot_relative.shape, np.linalg.norm(rot_relative, axis=1).reshape(-1, 1).shape)

        # quaternion relative rotations
        q_relative = quat.from_float_array(rot_relative)
        q_initial_eef = quat.from_float_array(initial_eef[3:])

        q_absolute = q_relative * q_initial_eef

        q_absolute = quat.as_float_array(q_absolute)

        print("xyz_relative", xyz_relative)
        # print("q_absolute", q_absolute, np.linalg.norm(q_absolute, axis=1))

        return np.concatenate([absolute_xyz, q_absolute], axis=-1)


def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[..., idxs]
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
            c, h, w = shape
            out_resolutions[key] = (w, h)
        elif type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            if 'pose' in key:
                assert tuple(shape) in [(2,), (6,)]

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
            lowdim_keys=lowdim_keys + ['action', 'label'],
            image_keys=rgb_keys,
            dt=dt,
        )

    # transform lowdim dimensions
    if action_shape == (2,):
        # 2D action space, only controls X and Y
        zarr_arr = replay_buffer['action']
        zarr_resize_index_last_dim(zarr_arr, idxs=[0, 1])

    for key, shape in lowdim_shapes.items():
        if 'pose' in key and shape == (2,):
            # only take X and Y
            zarr_arr = replay_buffer[key]
            zarr_resize_index_last_dim(zarr_arr, idxs=[0, 1])

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
    _ = plt.hist(dists, bins=100);
    plt.title('real action velocity')
