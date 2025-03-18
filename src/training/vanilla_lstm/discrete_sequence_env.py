import torch
import gym
import numpy as np

class FullyDiscreteSequenceEnv(gym.Env):
    def __init__(self, data, render_mode=None):
        super(FullyDiscreteSequenceEnv, self).__init__()
        self.data = data
        self.current_index = 0
        self.render_mode = render_mode
        
        self.min_val = -1000
        self.max_val = 1000
        self.num_values = self.max_val - self.min_val + 1

        self.observation_space = gym.spaces.MultiDiscrete([self.num_values] * len(data[0]['sequence']))

        self.action_space = gym.spaces.Discrete(self.num_values)

    def reset(self, seed=None, options=None):
        self.current_index = np.random.randint(len(self.data))
        obs = np.array(self.data[self.current_index]['sequence'], dtype=np.int32)
        
        obs = obs - self.min_val  
        return obs, {}

    def step(self, action):
        target = self.data[self.current_index]['target']

        selected_action = action + self.min_val
        
        reward = -np.abs(selected_action - target)
        
        done = True 

        self.current_index = np.random.randint(len(self.data))
        next_obs = np.array(self.data[self.current_index]['sequence'], dtype=np.int32)

        next_obs = next_obs - self.min_val
        
        return next_obs, reward, done, False, {}

    def render(self):
        pass
