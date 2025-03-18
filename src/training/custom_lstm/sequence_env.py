import gym
import numpy as np

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
        self.current_index = np.random.randint(len(self.data))
        return np.array(self.data[self.current_index]['sequence'], dtype=np.float32), {}

    def step(self, action):
        target = self.data[self.current_index]['target']
        reward = -np.abs(action[0] - target)
        # reward = -((action[0] - target) ** 2)
        done = True  
        self.current_index = np.random.randint(len(self.data))
        next_obs = np.array(self.data[self.current_index]['sequence'], dtype=np.float32)
        return next_obs, reward, done, False, {}
    
    def render(self):
        pass
