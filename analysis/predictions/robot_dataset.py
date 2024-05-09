import os
import sys

import dill
import torch
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.real_franka_image_dataset import RealFrankaImageDataset
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace

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

import matplotlib.pyplot as plt

import pathlib
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

def get_one_episode(dataset: RealFrankaImageDataset, index_episode: int = 0):
    replay_buffer = dataset.replay_buffer
    episode_ends = replay_buffer.episode_ends[:]
    print("episode_ends", episode_ends)

    index_start = episode_ends[index_episode]
    index_end = episode_ends[index_episode + 1]


    # TODO
    camera_0_rgb = replay_buffer["camera_1"][index_start:index_end]
    camera_0_data = camera_0_rgb
    camera_0_data = RealFrankaImageDataset.moveaxis_rgbd(camera_0_data)
    camera_0_data = RealFrankaImageDataset.rgbd_255_to_1(camera_0_data)

    return {
        "obs": {
            "camera_1": camera_0_data * 0.,
            "eef": replay_buffer["eef"][index_start:index_end],
        },
        "action": replay_buffer['action'][index_start:index_end],
    }



def main():
    # ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.04.08/16.24.23_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/epoch=0280-mse_error_val=0.000.ckpt"  # with images
    # ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.04.09/18.02.53_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/epoch=1530-mse_error_val=0.000.ckpt"  # with images + mass
    # ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.04.10/18.53.16_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/latest.ckpt"  # with images + mass + critic
    # ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.04.12/18.08.50_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/epoch=1555-mse_error_val=0.000.ckpt"  # with images + mass + critic + classifier input.
    ckpt_path = "/home/lucagrillotti/ros/humble/src/diffusion_policy/data/outputs/2024.05.09/19.47.27_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/latest.ckpt"  # with images + mass + critic + classifier input + GC
    dataset_dir = "/home/lucagrillotti/ros/humble/src/project_shokunin/shokunin_common/rl/scooping_agent/puree_agent/dataset_parameterized_motion/"
    path_classifier = "/home/lucagrillotti/ros/humble/src/diffusion_policy/data/outputs/classifier/2024.05.09/19.17.18_train_diffusion_unet_image_franka_kitchen_lowdim/classifier.pt"


    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.task.dataset.dataset_path = dataset_dir
    dataset: RealFrankaImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    cls = hydra.utils.get_class(cfg._target_)

    cfg.task.dataset.dataset_path = dataset_dir
    cfg.training.path_classifier_state_dict = path_classifier

    workspace = cls(cfg)
    workspace: TrainDiffusionUnetHybridWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    critic = workspace.critic
    policy = policy.cuda()
    policy = policy.eval()
    critic = critic.cuda()
    critic = critic.eval()

    workspace.load_classifier(path_classifier)

    workspace.classifier = workspace.classifier.cuda()
    workspace.classifier = workspace.classifier.eval()

    index_episode = 12
    one_episode = get_one_episode(dataset, index_episode=index_episode)

    sequence_observations = one_episode["obs"]
    sequence_observations = dict_apply(sequence_observations, lambda x: np.asarray([x[0], x[0], x[0], *x]))

    sequence_actions = one_episode["action"]
    x = sequence_actions
    sequence_actions = np.asarray([x[0], x[0], x[0], *x])

    n_action_steps = 8
    n_obs_steps = 4


    num_dim_actions = sequence_actions.shape[1]

    color = plt.cm.rainbow(np.linspace(0, 1, num_dim_actions))

    images_debug = "images_debug"
    path_debug = pathlib.Path(images_debug)
    path_debug.mkdir(exist_ok=True)
    num_obs = len(sequence_observations["eef"])


    for index_start in range(n_obs_steps - 1, num_obs - n_obs_steps - n_action_steps, n_action_steps):
        print(index_start)
        stacked_obs = dict_apply(sequence_observations, lambda x: x[index_start - n_obs_steps + 1:index_start + 1])


        with torch.no_grad():
            np_obs_dict = dict_apply(stacked_obs, lambda x: x.reshape(1, *x.shape).astype(np.float32))

            # device transfer
            obs_dict = dict_apply(np_obs_dict,
                                  lambda x: torch.from_numpy(x).cuda())

            # add 'scooping_accomplished' field
            obs_dict = workspace.add_scooping_accomplished_to_batch_from_classifier(obs_dict,
                                                                                    normalizer=policy.normalizer,
                                                                                    no_batch=False)

            action_dict = policy.predict_action_from_several_samples(obs_dict, critic,)
            # action_dict = policy.predict_action_from_several_samples(neutral_obs_dict, critic,)  # no classification guided sampling if reward depends of mass.
            # action_dict = policy.predict_action(obs_dict, neutral_obs_dict)
            # action_dict = policy.predict_action(neutral_obs_dict)
            np_action_dict = dict_apply(action_dict,
                                        lambda x: x.detach().to('cpu').numpy())

            absolute_actions = np_action_dict['action'].squeeze()

        legend = [
            "x",
            "y",
            "z",
            "q1",
            "q2",
            "q3",
            "q4"
        ]
        for index_action, c in enumerate(color):
            plt.plot(np.arange(len(absolute_actions)) + index_start, absolute_actions[:, index_action], c=c, linestyle="dotted")
            plt.plot(np.arange(len(absolute_actions)) + index_start, sequence_actions[index_start:index_start+n_action_steps, index_action], c=c)
    


    folder = "predictions_figures/diffusion_x_primitives/"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/predictions_{index_episode=}.png")
    # plt.show()

if __name__ == '__main__':
    main()