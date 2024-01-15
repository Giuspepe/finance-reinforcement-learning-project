import os
import sys

import numpy as np

# Get parent of parent of parent directory
parent_of_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(parent_of_parent_dir)

from TACR.tacr import TACR
from TACR.config.config import TACRConfig

import gymnasium as gym
import torch    


def evaluate(tacr: TACR, env: gym.Env, path: str, name="actor", num_episodes=100, max_timesteps_per_episode=1000, state_mean=0.0, state_std=1.0, state_dim: int = 2, action_dim: int = 1):

    # Load Actor Model
    tacr.load_actor(path, name)

    tacr.actor.eval()

    # Generate state_mean and state_std using the same dimensions as states, so that we can normalize the states, from state_dim
    state_mean = np.array([state_mean for _ in range(state_dim)])
    state_std = np.array([state_std for _ in range(state_dim)])

    state_mean = torch.from_numpy(state_mean).to(device=tacr.device)
    state_std = torch.from_numpy(state_std).to(device=tacr.device)

    episode_reward_vector = []
    obs, _info = env.reset()
    obs = np.array(obs)

    # Keep all histories on device
    # Latest action and Reward will be "padding"
    states = torch.from_numpy(obs).reshape(1, state_dim).to(device=tacr.device, dtype=torch.float32)
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
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )

        action = action.reshape(1, action_dim)

        actions[-1] = action
        action = action.detach().cpu().numpy()

        action = np.argmax(action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        obs = np.array(obs)


        cur_state = torch.from_numpy(obs).to(device=tacr.device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=tacr.device, dtype=torch.long) * (timestep_count+1)], dim=1)

        total_reward += reward

        env.render()
        if terminated or timestep_count > max_timesteps_per_episode:
            obs, _info = env.reset()
            timestep_count = 1
            episode_reward_vector.append(total_reward)

            print(f"Episode {episode_count} reward: {total_reward}")
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
    
    return episode_reward_vector

env = gym.make("MountainCar-v0", render_mode="human")
action_dim = env.action_space.n
tacr_config = TACRConfig(state_dim=env.observation_space.shape[0], action_dim=action_dim, action_softmax=True)
agent = TACR(config=tacr_config)

episode_reward_vector = evaluate(agent, env, "saved_models_tacr", "actor", num_episodes=100, max_timesteps_per_episode=1000, state_mean=0.0, state_std=1.0, state_dim=env.observation_space.shape[0], action_dim=action_dim)

avg_reward = np.mean(episode_reward_vector)

print(f"Average reward: {avg_reward}")

import matplotlib.pyplot as plt
plt.plot(episode_reward_vector)
