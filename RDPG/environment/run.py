import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


import gymnasium as gym
from rdpg import RDPG
from buffer import ReplayBuffer
from train import train

env = gym.make('CartPole-v1')

rdpg = RDPG(input_dim=4, action_dim=1)

batch_size=64

replay_buffer = ReplayBuffer(
                    observation_dim=4, 
                    action_dim=1,
                    max_episode_length=100,
                    capacity=100_000,
                    batch_size=batch_size,
                            )

max_timesteps = 1_000_000
train(rdpg, env, max_timesteps, replay_buffer, batch_size=batch_size)


obs, _info = env.reset()
for i in range(1000):
    action, _states = rdpg.get_action(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()