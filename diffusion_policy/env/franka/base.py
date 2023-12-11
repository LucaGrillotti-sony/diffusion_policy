import sys
import os
# hack to import adept envs
ADEPT_DIR = os.path.join(os.path.dirname(__file__), 'relay_policy_learning', 'adept_envs')
sys.path.append(ADEPT_DIR)

import logging
import numpy as np
import adept_envs
from adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3
logger = logging.getLogger()


class FrankaBase(KitchenTaskRelaxV1):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.

    def __init__(
        self,
        use_abs_action=False,
        **kwargs
    ):
        super(FrankaBase, self).__init__(use_abs_action=use_abs_action, **kwargs)

    def reset_model(self):
        return super(FrankaBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(FrankaBase, self)._get_reward_n_score(obs_dict)

        reward_dict["bonus"] = 0.
        reward_dict["r_total"] = 0.

        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(FrankaBase, self).step(a, b=b)
        # TODO: no completion check so far.
        return obs, reward, done, env_info

    def get_goal(self):
        """Loads goal state from dataset for goal-conditioned approaches (like RPL)."""
        raise NotImplementedError

