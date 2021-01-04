import gym
from rl_fusion.env import FactorGraphEnv

gym.envs.register(
     id='FactorGraph-v0',
     entry_point='rl_fusion:FactorGraphEnv',
     max_episode_steps=1000,
)