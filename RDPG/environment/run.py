import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


import gymnasium as gym
from rdpg import RDPG
from buffer import ReplayBuffer
from train import train
from evaluate import evaluate
env = gym.make("MountainCarContinuous-v0", render_mode="human")

rdpg = RDPG(input_dim=2, action_dim=1, lr=3e-4, upper_normalization_bounds=env.observation_space.high, lower_normalization_bounds=env.observation_space.low)

# batch_size=32

# replay_buffer = ReplayBuffer(
#     observation_dim=2,
#     action_dim=1,
#     max_episode_length=env.spec.max_episode_steps,
#     capacity=10_000,
#     batch_size=batch_size,
# )

# max_timesteps = 150_000
# train(rdpg, env, max_timesteps, replay_buffer, batch_size=batch_size, update_after=5000)

# Evaluate the best model
reward_vector = evaluate(rdpg, env, "saved_models", "actor", num_episodes=100, max_timesteps_per_episode=1000)


# Take the average of the reward vector
import numpy as np

print(np.mean(reward_vector))

# Plot the reward vector
import matplotlib.pyplot as plt

plt.plot(reward_vector)
plt.show()
