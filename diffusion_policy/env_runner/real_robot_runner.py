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
    def __init__(self, jpc_pub):
        self.observation_space = gym.spaces.Box(
            -8, 8, shape=(7,), dtype=np.float32) # TODO
        self.action_space = gym.spaces.Box(
            -8, 8, shape=(7,), dtype=np.float32) # TODO
        self._jpc_pub = jpc_pub
        # self.init_pos = np.load(osp.join(osp.dirname(__file__), 'init_joint_pos.npy'))
        self.init_pos = np.zeros(7)  # TODO
        self._jstate = np.zeros(7)  # TODO

    def reset(self, ):
        self.jpc_send_goal(self.init_pos)
        return self.get_obs()

    def _compute_obs(self, ):
        jnts = np.array(self._jstate[:7])
        return jnts

    def get_obs(self):
        if self._jstate is None:
            return None
        else:
            return self._compute_obs()

    def step(self, action):
        # print("STEP from Real Robot Env")
        self.jpc_send_goal(action)
        obs = self.get_obs()
        reward = -1.
        info = {}
        done = False
        return obs, reward, done, info  # TODO

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

        env = EnvControlWrapper(None)
        self.max_steps = 150
        self.env = MultiStepWrapper(
                env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=self.max_steps,  # or None, TODO
            )
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

            # step env
            obs, reward, done, info = env.step(action_env)

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

