from rdpg import RDPG
import gymnasium as gym
from log import TensorBoardHandler


def evaluate(rdpg: RDPG, env: gym.Env, path: str, name="actor", num_episodes=100, max_timesteps_per_episode=1000):
    # Initialize TensorBoardHandler
    tb_handler = TensorBoardHandler(log_dir="runs/evaluation_01")

    # Load Actor Model using rdpg
    rdpg.load_actor(path, name)

    rdpg.actor.eval()
    rdpg.actor_rh.eval()

    episode_reward_vector = []
    obs, _info = env.reset()

    episode_count = 1
    timestep_count = 1
    total_reward = 0
    while True:
        timestep_count += 1
        action = rdpg.get_action(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)

        total_reward += rewards

        env.render()
        if terminated or timestep_count > max_timesteps_per_episode:
            obs, _info = env.reset()
            timestep_count = 1
            episode_reward_vector.append(total_reward)
            tb_handler.log_scalar(
                "Reward/episode", total_reward, episode_count
            )  # Log reward per episode
            total_reward = 0
            episode_count += 1
            if episode_count > num_episodes:
                break
        

    env.close()

    tb_handler.close()  # Close the TensorBoard handler
    
    return episode_reward_vector

