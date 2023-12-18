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

from diffusion_policy.common.pytorch_util import dict_apply
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

    CARTESIAN_CONTROL_PATH = "cartesian_control_with_time.npy"
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


def main():
    ckpt_path = "/home/lucagrillotti/ros/humble/src/diffusion_policy/results/17.05.32_train_diffusion_unet_lowdim_kitchen_lowdim/checkpoints/latest.ckpt"
    dataset_dir = "/data/kitchen/kitchen_demos_multitask/cartesian_control/"

    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model

    list_episodes = get_dataset(dataset_dir)
    one_episode = list_episodes[0]

    sequence_observations = one_episode["obs"]
    sequence_actions = one_episode["action"]
    sequence_times = one_episode["time"]

    n_action_steps = 8
    n_latency_steps = 0
    n_obs_steps = 2

    # index_start = 60  # TODO - Test with several
    print("length seq", len(sequence_observations))
    # stacked_true_actions = sequence_actions[index_start:index_start+n_action_steps]

    num_dim_actions = sequence_actions.shape[1]
    color = plt.cm.rainbow(np.linspace(0, 1, num_dim_actions))

    for index_action, c in enumerate(color):
        plt.plot(sequence_times, sequence_actions[:, index_action], c=c)
        # plt.plot(times, stacked_true_actions[:, index_action], c=c)
        # plt.plot(times, predicted_actions[:, index_action], c=c, linestyle="dotted")

    for index_start in range(n_obs_steps - 1, len(sequence_observations) - n_obs_steps, n_action_steps):
        print(index_start)
        # if index_start != 1:
        #     break
        stacked_obs = sequence_observations[index_start - n_obs_steps + 1:index_start + 1]
        times = sequence_times[index_start:index_start + n_action_steps]

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
            # plt.plot(sequence_times, sequence_actions[:, index_action], c=c)
            # plt.plot(times, stacked_true_actions[:, index_action], c=c)
            plt.plot(times, predicted_actions[:, index_action], c=c, linestyle="dotted")

    plt.show()

if __name__ == '__main__':
    main()