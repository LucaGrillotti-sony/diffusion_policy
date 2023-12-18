from typing import List, Dict, Optional, Optional
import numpy as np
import gym
from gym.spaces import Box
from diffusion_policy.env.franka.base import FrankaBase

class KitchenLowdimWrapper(gym.Env):
    def __init__(self,
            env: FrankaBase,
            init_qpos: Optional[np.ndarray]=None,
            init_qvel: Optional[np.ndarray]=None,
            render_hw = (240,360)
        ):
        self.env = env
        self.init_qpos = init_qpos
        self.init_qvel = init_qvel
        self.render_hw = render_hw

    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        if self.init_qpos is not None:
            # reset anyway to be safe, not very expensive
            print("qpos qvel 1", self.init_qpos.shape, self.init_qvel.shape)

            _ = self.env.reset()
            # start from known state
            print("qpos qvel 2", self.init_qpos.shape, self.init_qvel.shape)
            self.env.set_state(self.init_qpos, self.init_qvel)
            obs = self.env._get_obs()
            return obs
            # obs, _, _, _ = self.env.step(np.zeros_like(
            #     self.action_space.sample()))
            # return obs
        else:
            print("qpos qvel reset 1", self.init_qpos, self.init_qvel)
            x = self.env.reset()
            print("qpos qvel reset 2", self.init_qpos.shape, self.init_qvel.shape)
            return x

    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, width=w, height=h)
    
    def step(self, a):
        return self.env.step(a)
