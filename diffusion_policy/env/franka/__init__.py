"""Environments using kitchen and Franka robot."""
from gym.envs.registration import register

register(
    id="franka-v0",
    entry_point="diffusion_policy.env.franka.base:FrankaBase",
    max_episode_steps=280,
    reward_threshold=1.0,
)
