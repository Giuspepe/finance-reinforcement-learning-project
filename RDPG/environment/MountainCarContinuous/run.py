import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(parent_dir)


import gymnasium as gym
from RDPG.rdpg import RDPG
from buffer import ReplayBuffer
from RDPG.trainer.train import train
from RDPG.trainer.evaluate import evaluate
env = gym.make("MountainCarContinuous-v0", render_mode="human")

UPDATE_AFTER = 30000
MAX_TIMESTEPS = 200000

rdpg = RDPG(input_dim=2, action_dim=1, lr=3e-4, upper_normalization_bounds=env.observation_space.high, lower_normalization_bounds=env.observation_space.low, action_noise_decay_steps=MAX_TIMESTEPS-UPDATE_AFTER)

batch_size=64

replay_buffer = ReplayBuffer(
    observation_dim=2,
    action_dim=1,
    max_episode_length=env.spec.max_episode_steps,
    capacity=2_000,
    batch_size=batch_size,
)

train(rdpg, env, MAX_TIMESTEPS, replay_buffer, batch_size=batch_size, update_after=UPDATE_AFTER)

# Evaluate the best model
reward_vector = evaluate(rdpg, env, "saved_models_rdpg", "actor", num_episodes=100, max_timesteps_per_episode=1000)


# Take the average of the reward vector
import numpy as np

print(np.mean(reward_vector))

# Plot the reward vector
import matplotlib.pyplot as plt

plt.plot(reward_vector)
plt.show()

avgreward = np.mean(reward_vector)
print(avgreward)
