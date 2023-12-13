import queue

import gym
from gym import spaces
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import logging
import wandb.sdk.data_types.video as wv
import gym
import gym.spaces
import multiprocessing as mp
import os.path as osp
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

module_logger = logging.getLogger(__name__)



class EnvControlWrapper():
    def __init__(self, jpc_pub, n_obs_steps, n_action_steps):
        self.observation_space = gym.spaces.Box(
            -8, 8, shape=(7,), dtype=np.float32)  # TODO
        self.action_space = gym.spaces.Box(
            -8, 8, shape=(7,), dtype=np.float32)  # TODO
        self._jpc_pub = jpc_pub
        # self.init_pos = np.load(osp.join(osp.dirname(__file__), 'init_joint_pos.npy'))
        self.init_pos = np.zeros(7)  # TODO
        self._jstate = np.zeros(7)  # TODO

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = 150  # TODO

        self.all_observations = []  # TODO: same for reward?

        self.queue_actions = queue.Queue()

    def reset(self):
        # TODO: reset all observations, queue actions? Also when a done is achieved?
        self.jpc_send_goal(self.init_pos)
        obs = self.get_obs()
        self.all_observations.append(obs)
        stacked_obs = self._compute_stacked_obs(n_steps=self.n_obs_steps)
        return stacked_obs

    def _compute_obs(self, ):
        jnts = np.array(self._jstate[:7])
        return jnts

    def get_obs(self):
        if self._jstate is None:
            return None
        else:
            return self._compute_obs()

    def push_actions(self, list_actions):
        if not self.queue_actions.empty():
            raise ValueError("Queue actions is not empty, cannot push anything new to it")
        assert len(list_actions) == self.n_action_steps
        for action in list_actions:
            self.queue_actions.put(action)

    def _compute_stacked_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.all_observations) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return self.stack_last_n_obs(self.all_observations, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = self.stack_last_n_obs(
                    [obs[key] for obs in self.all_observations],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    @classmethod
    def stack_last_n_obs(cls, all_obs, n_steps):
        all_obs = list(all_obs)
        result = np.zeros((n_steps,) + all_obs[-1].shape,
                          dtype=all_obs[-1].dtype)
        start_idx = -min(n_steps, len(all_obs))
        result[start_idx:] = np.array(all_obs[start_idx:])
        if n_steps > len(all_obs):
            # pad
            result[:start_idx] = result[start_idx]
        return result

    def step(self):
        if self.queue_actions.empty():
            raise ValueError("Queue actions should not be empty when calling step")
        action = self.queue_actions.get()
        self.jpc_send_goal(action)
        obs = self.get_obs()
        reward = -1.
        info = {}
        done = False

        self.all_observations.append(obs)
        stacked_obs = self._compute_stacked_obs(n_steps=self.n_obs_steps)

        if (self.max_steps is not None) and (len(self.all_observations) > self.max_steps):
            done = True

        return stacked_obs, reward, done, info  # TODO

    def get_jstate(self):
        return self._jstate

    def set_jstate(self, msg):
        self._jstate = msg

    def jpc_send_goal(self, jpos):
        ...
        # print("jpos", jpos)


class RealRobot(BaseLowdimRunner):
    def __init__(self,
                 n_obs_steps,
                 n_action_steps,
                 output_dir):
        super().__init__(output_dir)

        env = EnvControlWrapper(None, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)
        self.max_steps = 150
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
                    action_dict = policy.predict_action(obs_dict)

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

