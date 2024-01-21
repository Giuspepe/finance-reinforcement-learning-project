import numpy as np
from irdpg import IRDPG
import gymnasium as gym
from log import TensorBoardHandler


# Custom action picker function that decodes one-hot encoded action
def action_softmax_to_value(action: np.array) -> int:
    index_of_max_probability = np.argmax(action)
    decoded_action = index_of_max_probability - 1
    return decoded_action

def evaluate_during_training(irdpg: IRDPG, env: gym.Env, num_episodes=100, max_timesteps_per_episode=1000):
    
    # Load Actor Model using irdpg
    irdpg.actor.eval()
    irdpg.actor_rh.eval()

    episode_reward_vector = []
    obs, _info = env.reset()

    episode_count = 1
    timestep_count = 1
    total_reward = 0
    while True:
        timestep_count += 1
        action = irdpg.get_action(obs, deterministic=True)
        action = action_softmax_to_value(action)
        obs, rewards, terminated, truncated, info = env.step(action)

        total_reward += rewards
        if terminated or timestep_count > max_timesteps_per_episode:
            obs, _info = env.reset()
            timestep_count = 1
            episode_reward_vector.append(total_reward)
            total_reward = 0
            episode_count += 1
            if episode_count > num_episodes:
                break
        

    env.close()
    
    avg_reward = np.mean(episode_reward_vector)

    # Set back to training mode
    irdpg.actor.train()
    irdpg.actor_rh.train()
    return episode_reward_vector, avg_reward

def evaluate_from_file(irdpg: IRDPG, env: gym.Env, path: str, name="actor", num_episodes=100, max_timesteps_per_episode=1000):
    
    # Load Actor Model using irdpg
    irdpg.load_actor(path, name)

    irdpg.actor.eval()
    irdpg.actor_rh.eval()

    episode_reward_vector = []
    obs, _info = env.reset()

    episode_count = 1
    timestep_count = 1
    total_reward = 0
    while True:
        timestep_count += 1
        action = irdpg.get_action(obs, deterministic=True)
        action = action_softmax_to_value(action)
        obs, rewards, terminated, truncated, info = env.step(action)

        total_reward += rewards
        if terminated or timestep_count > max_timesteps_per_episode:
            obs, _info = env.reset()
            timestep_count = 1
            episode_reward_vector.append(total_reward)
            total_reward = 0
            episode_count += 1
            if episode_count > num_episodes:
                break
        

    env.close()
    
    avg_reward = np.mean(episode_reward_vector)
    return episode_reward_vector, avg_reward

