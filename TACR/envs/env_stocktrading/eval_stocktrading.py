import os
import sys

# Get parent of parent of parent directory
parent_of_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(parent_of_parent_dir)

from TACR.tacr import TACR
from TACR.config.config import TACRConfig
from env_stocktrading.utils import create_environment
from env_stocktrading.env_stocktrading import SimpleOneStockStockTradingBaseEnv
from preprocessing.custom_technical_indicators import PCT_RETURN, OBV_PCT_CHANGE, RVI_PCT_CHANGE, ADX, RSI_CATEGORICAL, BINARY_SMA_RISING
from preprocessing.process_yh_finance import YHFinanceProcessor

import numpy as np
import pandas as pd
import gymnasium as gym
import torch    

# Custom action picker function that decodes one-hot encoded action
def action_softmax_to_value(action: np.array) -> int:
    index_of_max_probability = np.argmax(action)
    decoded_action = index_of_max_probability - 1
    return decoded_action
    

def evaluate(env: SimpleOneStockStockTradingBaseEnv, path: str, name_actor="actor", name_config="config", num_episodes=100, max_timesteps_per_episode=10000, silence=True):

    # Prepare TACR
    tacr_config = TACRConfig()
    tacr = TACR(config=tacr_config, load_file_config=True, file_config_path=path, file_config_name=name_config)
             
    # Load Actor Model
    tacr.load_actor(path, name_actor)

    tacr.actor.eval()

    # Get state mean and std
    state_mean = tacr.config.state_mean
    state_std = tacr.config.state_std

    state_dim = tacr.config.state_dim
    action_dim = tacr.config.action_dim

    episode_reward_vector = []
    obs, _info = env.reset()
    obs = np.array(obs)

    # Keep all histories on device
    # Latest action and Reward will be "padding"

    states = torch.from_numpy((obs - state_mean) / state_std).reshape(1, state_dim).to(device=tacr.device, dtype=torch.float32)
    actions = torch.zeros((0, action_dim), device=tacr.device, dtype=torch.float32)
    rewards = torch.zeros(0, device=tacr.device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=tacr.device, dtype=torch.long).reshape(1, 1)

    episode_count = 1
    timestep_count = 1
    total_reward = 0
    while True:
        timestep_count += 1

        # Add padding
        actions = torch.cat([actions, torch.zeros((1, action_dim), device=tacr.device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=tacr.device)])

        action = tacr.actor.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )

        action = action.reshape(1, action_dim)

        actions[-1] = action
        action = action.detach().cpu().numpy()

        action = action_softmax_to_value(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.array(obs)


        cur_state = torch.from_numpy(obs).to(device=tacr.device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=tacr.device, dtype=torch.long) * (timestep_count+1)], dim=1)

        total_reward += reward

        if terminated or timestep_count > max_timesteps_per_episode:
            if not silence:
                print(f"Episode {episode_count} reward: {total_reward} account value: {env.account_value}")
            obs, _info = env.reset()
            timestep_count = 1
            episode_reward_vector.append(total_reward)

            total_reward = 0

            # Reset history for the episode
            states = torch.from_numpy(obs).reshape(1, state_dim).to(device=tacr.device, dtype=torch.float32)
            actions = torch.zeros((0, action_dim), device=tacr.device, dtype=torch.float32)
            rewards = torch.zeros(0, device=tacr.device, dtype=torch.float32)
            timesteps = torch.tensor(0, device=tacr.device, dtype=torch.long).reshape(1, 1)

            episode_count += 1
            if episode_count > num_episodes:
                break
        

    env.close()
    avg_reward = np.mean(episode_reward_vector)

    return episode_reward_vector, avg_reward

if __name__ == "__main__":
    INDICATORS = []
    CUSTOM_INDICATORS = [
        PCT_RETURN(length=2), OBV_PCT_CHANGE(length=8), RVI_PCT_CHANGE(length=20, rvi_pct_change_length=2), ADX(length=16), RSI_CATEGORICAL(length=24), BINARY_SMA_RISING(length=24)
    ]

    yfp = YHFinanceProcessor()

    val_dataset = pd.read_csv("val_stock_data.csv")
    val_env = create_environment(yfp, val_dataset, INDICATORS, CUSTOM_INDICATORS, gamma=0.999)
    episode_reward_vector, avg_reward = evaluate(val_env, "saved_models_tacr", "actor_last", "config_last", num_episodes=1, max_timesteps_per_episode=10000, silence=False)

    print(f"Average reward: {avg_reward}")

    import matplotlib.pyplot as plt
    plt.plot(episode_reward_vector)
    print(val_env.account_value)

