import queue
import os
import os.path as osp
import sys

import click
import dill
import gym
import torch
from gym import spaces
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply, custom_tree_map
from diffusion_policy.dataset.real_franka_image_dataset import RealFrankaImageDataset
from diffusion_policy.env_runner.real_robot_runner import RealRobot
from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace import \
    TrainDiffusionTransformerLowdimWorkspace
from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt


def load_obs(obs_path: pathlib.Path, times_to_evaluate: np.ndarray):
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

def get_dataset(dataset_dir):
    obs_size = 7

    data_directory = pathlib.Path(dataset_dir)

    CARTESIAN_CONTROL_PATH = "end_effector_poses_with_time.npy"
    OBS_DATA_PATH = "obs_with_time.npy"

    results_list = list(data_directory.glob(f'*/{CARTESIAN_CONTROL_PATH}'))

    all_episodes = []

    for i, np_path in enumerate(tqdm(results_list)):
        try:
            cartesian_control_with_time = np.load(np_path.absolute())
            cartesian_control = cartesian_control_with_time[:, 1:]
            _res_folder = np_path.parents[0]
            obs_path = _res_folder / OBS_DATA_PATH
            times_to_evaluate = cartesian_control_with_time[:, 0].ravel()
            obs = load_obs(obs_path, times_to_evaluate)

            assert obs.shape[1] == 9
            obs = obs[:, :obs_size]  # TODO remove this line and do this in preprocessing

            episode = {
                'obs': obs.astype(np.float32),
                'action': cartesian_control.astype(np.float32),
                'time': times_to_evaluate.astype(np.float32)
            }
            all_episodes.append(episode)
        except Exception as e:
            print(i, e)

    return all_episodes

def get_one_episode(dataset: RealFrankaImageDataset, n_action_steps: int, n_latency_steps: int, n_obs_steps: int, index_episode: int = 0):
    sampler = dataset.sampler
    replay_buffer = dataset.replay_buffer
    episode_ends = replay_buffer.episode_ends[:]

    index_start = episode_ends[index_episode]
    index_end = episode_ends[index_episode + 1]

    index_sample_start = index_start
    index_sample_end = index_start + n_obs_steps + n_action_steps
    while index_sample_end < index_end:
        obs = replay_buffer['obs']
        new_obs = {}
        for key in obs.keys():
            if key in ("camera_0", "camera_1"):
                new_obs[key] = obs[key][index_sample_start:index_sample_start + n_obs_steps] / 255.0
            else:
                new_obs[key] = obs[key][index_sample_start:index_sample_start + n_obs_steps]

        yield {
            'obs': new_obs,
            'action': replay_buffer['action'],
        }

        index_sample_start += n_action_steps
        index_sample_end += n_action_steps


def main():
    # ckpt_path = "/home/lucagrillotti/ros/humble/src/diffusion_policy/results/13.09.31_train_diffusion_unet_lowdim_kitchen_lowdim/checkpoints/latest.ckpt"
    # ckpt_path = "/home/lucagrillotti/projects/diffusion_policy/data/outputs/2023.12.21/12.21_12.11.57_end_effector_control/checkpoints/latest.ckpt"
    # ckpt_path = "/home/lucagrillotti/projects/diffusion_policy/data/outputs/2023.12.22/17.22.09_train_diffusion_unet_lowdim_kitchen_lowdim/checkpoints/latest.ckpt"
    ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.03.07/17.56.19_train_diffusion_unet_image_franka_kitchen_lowdim_ok/checkpoints/latest.ckpt"  # basic, no reward, ddpm
    # dataset_dir = "/data/kitchen/kitchen_demos_multitask/cartesian_control/"
    # dataset_dir = "/home/lucagrillotti/projects/diffusion_policy/data/kitchen/kitchen_demos_multitask/cartesian_control_all"
    dataset_dir = "/home/ros/humble/src/diffusion_policy/data/fake_puree_experiments/diffusion_policy_dataset/"

    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.task.dataset.dataset_path = "/home/ros/humble/src/diffusion_policy/data/fake_puree_experiments/diffusion_policy_dataset/"
    dataset: RealFrankaImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None) # TODO
    policy = workspace.model

    # list_episodes = get_dataset(dataset_dir)
    one_episode = next(get_one_episode(dataset, 8, 0, 2, 0))

    sequence_observations = one_episode["obs"]
    sequence_actions = one_episode["action"]

    n_action_steps = 8
    n_latency_steps = 0
    n_obs_steps = 2

    # index_start = 60  # TODO - Test with several
    print("length seq", len(sequence_observations))
    # stacked_true_actions = sequence_actions[index_start:index_start+n_action_steps]

    num_dim_actions = sequence_actions.shape[1]
    color = plt.cm.rainbow(np.linspace(0, 1, num_dim_actions))

    for index_start in range(n_obs_steps - 1, len(sequence_observations) - n_obs_steps, n_action_steps):
        print(index_start)
        stacked_obs = sequence_observations[index_start - n_obs_steps + 1:index_start + 1]


        with torch.no_grad():
            np_obs_dict = {
                'obs': stacked_obs.reshape(1, *stacked_obs.shape).astype(np.float32)
            }

            # device transfer
            obs_dict = dict_apply(np_obs_dict,
                                  lambda x: torch.from_numpy(x).cuda())
            action_dict = policy.predict_action(obs_dict)
            np_action_dict = dict_apply(action_dict,
                                        lambda x: x.detach().to('cpu').numpy())

            predicted_actions = np_action_dict['action'].squeeze()

        for index_action, c in enumerate(color):
            plt.plot(np.arange(len(predicted_actions)), predicted_actions[:, index_action], c=c, linestyle="dotted")
    plt.savefig("predictions.png")
    # plt.show()

if __name__ == '__main__':
    main()