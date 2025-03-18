import torch
import hydra
import gym
import numpy as np
import pandas as pd
import yaml
import os
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_hub import HfApi, upload_file, create_repo, hf_hub_download

class SequenceEnv(gym.Env):
    def __init__(self, data, render_mode=None):
        super(SequenceEnv, self).__init__()
        self.data = data
        self.current_index = 0
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=-1000, high=1000, shape=(len(data[0]['sequence']),), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1000, high=1000, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Reset to a random sequence each episode."""
        self.current_index = np.random.randint(len(self.data))
        return np.array(self.data[self.current_index]['sequence'], dtype=np.float32), {}

    def step(self, action):
        """Take an action and return the new state, reward, done signal."""
        target = self.data[self.current_index]['target']
        # reward = 1.0 if np.isclose(action[0], target) else -1.0
        # reward = -((action[0] - target) ** 2)
        reward = -np.abs(action[0] - target) 
        done = True  

        self.current_index = np.random.randint(len(self.data))
        next_obs = np.array(self.data[self.current_index]['sequence'], dtype=np.float32)
        return next_obs, reward, done, False, {}
    
    def render(self):
        pass
