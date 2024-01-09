import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


import gymnasium as gym
from rdpg import RDPG
from buffer import ReplayBuffer
from train import train


def dim(space):
    """Returns the dimension of a space"""  
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Box):
        return space.shape
    else:
        raise NotImplementedError(f"Unknown space type: {space}")


env = gym.make('CartPole-v1')

rdpg = RDPG(input_dim=dim(env.observation_space), action_dim=dim(env.action_space))


replay_buffer = ReplayBuffer(
                    observation_dim=dim(env.observation_space), 
                    action_dim=dim(env.action_space),
                    max_episode_length=10_000,
                    capacity=100_000,
                    batch_size=64,
                            )

max_timesteps = 1_000_000
train(rdpg, env, max_timesteps, replay_buffer)


obs, _info = env.reset()
for i in range(1000):
    action, _states = rdpg.get_action(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()