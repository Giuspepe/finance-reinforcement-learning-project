import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


import gymnasium as gym
from rdpg import RDPG
from buffer import ReplayBuffer
from train import train

env = gym.make('MountainCarContinuous-v0', render_mode='human')

rdpg = RDPG(input_dim=2, action_dim=1)

batch_size=32

replay_buffer = ReplayBuffer(
                    observation_dim=2, 
                    action_dim=1,
                    max_episode_length=env.spec.max_episode_steps,
                    capacity=10_000,
                    batch_size=batch_size,
                            )

max_timesteps = 40_000
train(rdpg, env, max_timesteps, replay_buffer, batch_size=batch_size, update_after=2000)


reward_vector = []
obs, _info = env.reset()
for i in range(1000):
    action = rdpg.get_action(obs)
    obs, rewards, terminated, truncated, info = env.step(action)

    reward_vector.append(rewards)
    env.render()
    if terminated:
        obs, _info = env.reset()

env.close()

# Take the average of the reward vector
import numpy as np
print(np.mean(reward_vector))

# Plot the reward vector
import matplotlib.pyplot as plt
plt.plot(reward_vector)
plt.show()

