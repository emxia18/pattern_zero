from huggingface_hub import hf_hub_download
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sequence_env import SequenceEnv
import numpy as np
import pandas as pd
import gym
import os
import yaml

CONFIG_PATH = "src/config/ppo_trainer.yaml"

HF_USERNAME = "emxia18"
MODEL_NAME = "pattern-zero"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

model_path = hf_hub_download(repo_id=REPO_ID, filename=f"{MODEL_NAME}.zip")

model = RecurrentPPO.load(model_path)
print(f"Model downloaded and loaded from Hugging Face: {model_path}")

config = load_config(CONFIG_PATH)

test_data = config.get("test_data", [])

print(f"Test dataset loaded: {len(test_data)} samples")

env = DummyVecEnv([lambda: SequenceEnv(test_data)])

lstm_states = None  
episode_starts = np.ones((1,), dtype=bool)

predictions = []
targets = []

for data in test_data:
    obs = np.array(data['sequence'], dtype=np.float32).reshape(1, -1)
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    
    predictions.append(action[0][0])
    targets.append(data['target'])

    print(f"Sequence: {data['sequence']}, Prediction: {action[0][0]}, Target: {data['target']}")

mae = mean_absolute_error(targets, predictions)
mse = mean_squared_error(targets, predictions)

print(f"Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")



