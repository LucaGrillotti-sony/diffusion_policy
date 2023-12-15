from typing import Dict
import torch
import numpy as np
import copy
import pathlib
from tqdm import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env.kitchen.kitchen_util import parse_mjl_logs


class KitchenCustomDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=True,
            robot_noise_ratio=0.0,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        robot_pos_noise_amp = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        rng = np.random.default_rng(seed=seed)

        obs_size = 7

        data_directory = pathlib.Path(dataset_dir)

        self.replay_buffer = ReplayBuffer.create_empty_numpy()

        CARTESIAN_CONTROL_PATH = "cartesian_control_with_time.npy"
        OBS_DATA_PATH = "obs_with_time.npy"

        for i, np_path in enumerate(tqdm(list(data_directory.glob(f'*/{CARTESIAN_CONTROL_PATH}')))):
            try:
                cartesian_control_with_time = np.load(np_path.absolute())
                cartesian_control = cartesian_control_with_time[:, 1:]
                _res_folder = np_path.parents[0]
                obs_path = _res_folder / OBS_DATA_PATH
                times_to_evaluate = cartesian_control_with_time[:, 0].ravel()
                obs = self.load_obs(obs_path, times_to_evaluate)

                obs = obs[:, :obs_size]  # TODO remove this line and do this in preprocessing

                episode = {
                    'obs': obs.astype(np.float32),
                    'action': cartesian_control.astype(np.float32)
                }
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                print(i, e)
        # for i in range(15):
        #     try:
        #         qpos = np.load(path_qpos).astype(np.float32)
        #         times_array = np.load(path_times).astype(np.float32)
        #         times_array = times_array - times_array[0]
        #         qvel = np.load(path_qvel).astype(np.float32)
        #
        #         obs = qpos[:, :obs_size]
        #         if robot_noise_ratio > 0:
        #             # add observation noise to match real robot
        #             noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
        #                 low=-1., high=1., size=(obs.shape[0], obs_size))
        #             obs[:, :obs_size] += noise
        #
        #             TIME_NOISE = 0.1
        #             noise_time = TIME_NOISE * rng.uniform(low=-1, high=1, size=times_array.shape)
        #             times_array += noise_time
        #         episode = {
        #             'obs': np.hstack([qpos, times_array.reshape(-1, 1)]),
        #             'action': obs,
        #         }
        #         self.replay_buffer.add_episode(episode)
        #     except Exception as e:
        #         print(i, e)
        #         # print(e)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def load_obs(self, obs_path: pathlib.Path, times_to_evaluate: np.ndarray):
        obs_with_times = np.load(obs_path)
        times = obs_with_times[:, 0].ravel()
        obs = obs_with_times[:, 1:]
        all_data = []
        for col in range(obs.shape[1]):
            _col_data = np.interp(
                x=times_to_evaluate,
                xp=times,
                fp=obs[:, col].ravel(),
            )
            all_data.append(_col_data)
        array_all_data = np.hstack(
            [_col_data.reshape(-1, 1) for _col_data in all_data]
        )
        return array_all_data



    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'obs': self.replay_buffer['obs'],
            'action': self.replay_buffer['action']
        }
        if 'range_eps' not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs['range_eps'] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class KitchenMjlLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=True,
            robot_noise_ratio=0.0,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        robot_pos_noise_amp = np.array([0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   ,
            0.1   , ], dtype=np.float32)
        rng = np.random.default_rng(seed=seed)

        obs_size = 9

        data_directory = pathlib.Path(dataset_dir)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i, mjl_path in enumerate(tqdm(list(data_directory.glob('*/*.mjl')))):
            try:
                data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                qpos = data['qpos'].astype(np.float32)
                obs = qpos[:,:obs_size]
                # if robot_noise_ratio > 0:
                    # add observation noise to match real robot
                    # noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                    #     low=-1., high=1., size=(obs.shape[0], obs_size))
                    # obs[:, :obs_size] += noise
                episode = {
                    'obs': obs,
                    'action': data['ctrl'].astype(np.float32)
                }
                print("EPISODE SHAPE", obs.shape)
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                print(i, e)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'obs': self.replay_buffer['obs'],
            'action': self.replay_buffer['action']
        }
        if 'range_eps' not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs['range_eps'] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
