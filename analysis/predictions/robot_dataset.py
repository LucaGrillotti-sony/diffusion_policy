import sys

import dill
import torch
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.real_franka_image_dataset import RealFrankaImageDataset

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

def get_one_episode(dataset: RealFrankaImageDataset, mass, index_episode: int = 0):
    sampler = dataset.sampler
    replay_buffer = dataset.replay_buffer
    episode_ends = replay_buffer.episode_ends[:]
    print("episode_ends", episode_ends)

    index_start = episode_ends[index_episode]
    index_end = episode_ends[index_episode + 1]

    if mass is None:
        raise NotImplementedError("Mass is None")
        # mass_obs = dataset.encode_mass(replay_buffer["mass"][index_start:index_end])
    else:
        new_mass = np.full_like(replay_buffer["mass"][index_start:index_end], mass)
        print("New Mass", new_mass)
        mass_v = dataset.get_vector_mass(new_mass, neutral=False)
        neutral_mass_v = dataset.get_vector_mass(new_mass, neutral=True)
        print(f"{neutral_mass_v=}")

        mass_obs = dataset.rff_encoder.encode_vector(mass_v)
        neutral_mass_obs = dataset.rff_encoder.encode_vector(neutral_mass_v)

    # TODO
    camera_0_rgb = replay_buffer["camera_1"][index_start:index_end]
    # camera_0_depth = replay_buffer["camera_0"][index_start:index_end]
    # camera_0_data = RealFrankaImageDataset.concatenate_rgb_depth(camera_0_rgb, camera_0_depth)\
    camera_0_data = camera_0_rgb
    camera_0_data = RealFrankaImageDataset.moveaxis_rgbd(camera_0_data)
    camera_0_data = RealFrankaImageDataset.rgbd_255_to_1(camera_0_data)

    return {
        "obs": {  # TODO: update this too
            "camera_1": camera_0_data - camera_0_data,
            "eef": replay_buffer["eef"][index_start:index_end],
            # "mass": mass_obs,
        },
        "action": replay_buffer['action'][index_start:index_end],
        "neutral_obs": {  # TODO: update this too
            "camera_1": camera_0_data - camera_0_data,
            "eef": replay_buffer["eef"][index_start:index_end],
            # "mass": neutral_mass_obs,
        },
    }


def _get_mass_encoding(mass, rff_encoder):
    if mass is None:
        _mass_encoding = np.array([[0., 1.]])
    else:
        _mass_encoding = np.array([[mass, 0.]])

    return rff_encoder.encode_vector(_mass_encoding)[0]


def main():
    ckpt_path = "/home/ros/humble/src/diffusion_policy/data/outputs/2024.04.08/16.24.23_train_diffusion_unet_image_franka_kitchen_lowdim/checkpoints/epoch=0280-mse_error_val=0.000.ckpt"  # with images
    dataset_dir = "/home/ros/humble/src/diffusion_policy/data/fake_puree_experiments/diffusion_policy_dataset_exp2_v2/"

    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.task.dataset.dataset_path = dataset_dir
    dataset: RealFrankaImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None) # TODO
    policy = workspace.model
    policy = policy.cuda()
    policy = policy.eval()

    # list_episodes = get_dataset(dataset_dir)
    mass = 1000  # TODO: not taking mass atm
    index_episode = 30
    one_episode = get_one_episode(dataset, mass=mass, index_episode=index_episode)

    sequence_observations = one_episode["obs"]
    sequence_neutral_observations = one_episode["neutral_obs"]
    sequence_actions = one_episode["action"]

    n_action_steps = 8
    n_latency_steps = 0
    n_obs_steps = 4  # TODO

    # index_start = 60  # TODO - Test with several
    num_obs = len(sequence_observations["eef"])

    print("length seq", num_obs)
    print("sequence_observations", sequence_observations["eef"][0])
    # stacked_true_actions = sequence_actions[index_start:index_start+n_action_steps]

    num_dim_actions = sequence_actions.shape[1]

    import pathlib
    import matplotlib.pyplot as plt

    color = plt.cm.rainbow(np.linspace(0, 1, num_dim_actions))

    images_debug = "images_debug"
    path_debug = pathlib.Path(images_debug)
    path_debug.mkdir(exist_ok=True)

    # todo
    # images = sequence_observations["camera_0"]
    #
    # for i in range(num_obs):
    #     plt.clf()
    #     plt.cla()
    #     img = images[i]
    #     img = img[:3] # Taking only RGB
    #     # img = np.repeat(img, 3, axis=0)
    #     print("image", i, images[i])
    #     img = np.moveaxis(img, 0, -1)
    #     plt.imshow(img)
    #     plt.savefig(path_debug / f"image_{i}.png")
    #     plt.clf()
    #     plt.cla()

    for index_start in range(n_obs_steps - 1, num_obs - n_obs_steps - n_action_steps, n_action_steps):
        print(index_start)
        stacked_obs = dict_apply(sequence_observations, lambda x: x[index_start - n_obs_steps + 1:index_start + 1])
        stacked_neutral_obs = dict_apply(sequence_neutral_observations, lambda x: x[index_start - n_obs_steps + 1:index_start + 1])
        # print("stacked obs", np.max(stacked_obs["camera_0"]), stacked_obs["camera_0"])
        initial_eef = sequence_actions[index_start]

        with torch.no_grad():
            np_obs_dict = dict_apply(stacked_obs, lambda x: x.reshape(1, *x.shape).astype(np.float32))
            np_neutral_obs_dict = dict_apply(stacked_neutral_obs, lambda x: x.reshape(1, *x.shape).astype(np.float32))

            # device transfer
            obs_dict = dict_apply(np_obs_dict,
                                  lambda x: torch.from_numpy(x).cuda())
            neutral_obs_dict = dict_apply(np_neutral_obs_dict,
                                          lambda x: torch.from_numpy(x).cuda())
            
            # action_dict = policy.predict_action(obs_dict, neutral_obs_dict) TODO
            action_dict = policy.predict_action(obs_dict)
            np_action_dict = dict_apply(action_dict,
                                        lambda x: x.detach().to('cpu').numpy())

            absolute_actions = np_action_dict['action'].squeeze()
            print("SHAPES", absolute_actions.shape, initial_eef.shape)

            # predicted_actions = RealFrankaImageDataset.compute_absolute_action(predicted_actions, initial_eef)
            # print("predicted_actions", predicted_actions.shape)

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
    



    plt.savefig(f"predictions_{index_episode=}_{mass=}.png")
    # plt.show()

if __name__ == '__main__':
    main()