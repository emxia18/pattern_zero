import torch
import torch.nn as nn
import gym
import numpy as np
import pandas as pd
import yaml
import os
import wandb

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from huggingface_hub import HfApi, upload_file, create_repo, hf_hub_download

class CustomMultiLayerLstmPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        net_arch=None,
        **kwargs,
    ):

        super(CustomMultiLayerLstmPolicy, self).__init__(
            observation_space, action_space, lr_schedule, net_arch=net_arch, **kwargs
        )

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm = nn.LSTM(
            input_size=self.features_dim, 
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )
        self.lstm_state_size = lstm_hidden_size * lstm_num_layers

    def _forward_lstm(self, obs, lstm_states, episode_starts):
        features = self.extract_features(obs)
        lstm_input = features.unsqueeze(1) 

        if lstm_states is None:
            h_0 = torch.zeros(self.lstm_num_layers, lstm_input.size(0), self.lstm_hidden_size, device=lstm_input.device)
            c_0 = torch.zeros(self.lstm_num_layers, lstm_input.size(0), self.lstm_hidden_size, device=lstm_input.device)
            lstm_states = (h_0, c_0)
        else:
            h, c = lstm_states

            mask = episode_starts.view(1, -1, 1).to(lstm_input.device)
            h = h * (1 - mask)
            c = c * (1 - mask)
            lstm_states = (h, c)

        lstm_out, new_lstm_states = self.lstm(lstm_input, lstm_states)

        lstm_out = lstm_out.squeeze(1)
        return lstm_out, new_lstm_states
