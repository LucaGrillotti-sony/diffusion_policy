import queue
from typing import Dict

import gym
from gym import spaces
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import time
import dill
import math
import logging
import wandb.sdk.data_types.video as wv
import gym
import gym.spaces
import multiprocessing as mp
import os.path as osp

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

module_logger = logging.getLogger(__name__)



class DummyRobot(BaseLowdimRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        return dict()


class DummyImageRobot(BaseImageRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseImagePolicy) -> Dict:
        return dict()


class RealRobot(BaseLowdimRunner):
    def __init__(self,
                 n_obs_steps,
                 n_action_steps,
                 output_dir):
        super().__init__(output_dir)

        env = EnvControlWrapper(None, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)
        self.env = env
        print("N action steps", n_action_steps)
        print("N obs steps", n_obs_steps)

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        env = self.env

        # start rollout
        obs = env.reset()
        # initial_times = np.zeros((*obs.shape[:2], 1))
        # obs = np.concatenate([obs, initial_times], axis=2)
        # print("SHAPES", obs.shape)
        policy.reset()

        done = False
        counter = 0
        while not done:
            counter += 1

            # step env
            if env.queue_actions.empty():
                # create obs dict
                np_obs_dict = {
                    'obs': obs.astype(np.float32)
                }

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    start = time.time()
                    action_dict = policy.predict_action(obs_dict)
                    end = time.time()
                    print("inference time",  end - start, action_dict["action"].shape)
                    # time.sleep(0.7)
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                action_env = action.reshape(*action.shape[1:])

                env.push_actions(action_env)
            obs, reward, done, info = env.step()

            # times = np.asarray([
            #     info_row["time"]
            #     for info_row in info
            # ])

            # times = times.reshape((*obs.shape[:2], 1))

            # obs = np.concatenate([obs, times], axis=2)
            done = np.all(done)

            # print("obs", obs)
            # print("action", action)
        return dict()

