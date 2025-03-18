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

from sequence_env import SequenceEnv
from custom_lstm_policy import CustomMultiLayerLstmPolicy


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
    
    def _on_step(self):
        return True

    def _on_rollout_end(self):
        logs = {
            "value_loss": self.model.logger.name_to_value.get("train/value_loss", None),
            "policy_loss": self.model.logger.name_to_value.get("train/policy_loss", None),
            "entropy_loss": self.model.logger.name_to_value.get("train/entropy_loss", None),
            "explained_variance": self.model.logger.name_to_value.get("train/explained_variance", None),
        }
        logs = {k: v for k, v in logs.items() if v is not None}
        wandb.log(logs)
        return True

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def process_df(df, required_length=6):
    print(f"Original dataset size: {len(df)}")
    if 'nums' in df.columns:
        df = df.rename(columns={'nums': 'sequence'})
    df['sequence'] = df['sequence'].apply(eval)
    df = df[df['sequence'].apply(lambda seq: len(seq) == required_length)]
    print(f"Filtered dataset size: {len(df)}")
    return df.to_dict(orient="records")

def upload_model_to_huggingface(local_model_path, repo_id):
    api = HfApi()
    create_repo(repo_id, exist_ok=True)
    upload_file(
        path_or_fileobj=local_model_path,
        path_in_repo=os.path.basename(local_model_path),
        repo_id=repo_id,
        repo_type="model",
    )
    vec_stats_path = local_model_path.replace(".zip", "_vecnormalize.pkl")
    if os.path.exists(vec_stats_path):
        upload_file(
            path_or_fileobj=vec_stats_path,
            path_in_repo=os.path.basename(vec_stats_path),
            repo_id=repo_id,
            repo_type="model",
        )
    with open("README.md", "w") as f:
        f.write(
            f"# PPO-LSTM Model\n"
            f"This model was trained using a custom multi-layer LSTM with PPO.\n"
            f"\n**Training Data**: Custom sequence dataset\n"
            f"**Algorithm**: Proximal Policy Optimization (PPO) with a custom LSTM\n"
            f"**Library**: Stable-Baselines3\n"
        )
    upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Model uploaded to Hugging Face Hub: https://huggingface.co/{repo_id}")

def main():
    HF_USERNAME = "emxia18"
    WANDB_PROJECT_NAME = "pattern-zero"
    MODEL_NAME = "pattern-zero"
    REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"
    CONFIG_PATH = "src/config/ppo_trainer.yaml"

    wandb.init(project=WANDB_PROJECT_NAME, name="ppo_custom_lstm_run", sync_tensorboard=True)
    
    config = load_config(CONFIG_PATH)
    print(f"Training for {config['training_iterations']} iterations with learning rate {config['learning_rate']}")

    training_data_path = config.get("training_data")
    training_df = pd.read_csv(training_data_path)
    training_data = process_df(training_df)

    env = DummyVecEnv([lambda: SequenceEnv(training_data)])

    policy_kwargs = dict(
        lstm_hidden_size=256,
        lstm_num_layers=2,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    model = RecurrentPPO(
        policy=CustomMultiLayerLstmPolicy,
        env=env,
        verbose=1,
        learning_rate=1e-3,
        clip_range=config["clip_range"],
        n_steps=1024,
        n_epochs=20,
        gae_lambda=0.98,
        batch_size=config["batch_size"],
        policy_kwargs=policy_kwargs,
        tensorboard_log="./wandb_logs",
    )

    model.learn(
        total_timesteps=config["training_iterations"],
        callback=WandbCallback()
    )

    local_model_path = f"{MODEL_NAME}.zip"
    model.save(local_model_path)
    print(f"Model saved locally: {local_model_path}")

    upload_model_to_huggingface(local_model_path, REPO_ID)

    test_data = config.get("test_data", [])
    lstm_states = None  
    episode_starts = np.ones((1,), dtype=bool)
    for data in test_data:
        obs = np.array(data['sequence'], dtype=np.float32).reshape(1, -1)
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        print(f"Sequence: {data['sequence']}, Prediction: {action}, Target: {data['target']}")

if __name__ == '__main__':
    main()