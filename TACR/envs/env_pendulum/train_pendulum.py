import os
import sys

# Get parent of parent of parent directory
parent_of_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(parent_of_parent_dir)

import gymnasium as gym
import numpy as np
from TACR.config.config import TACRConfig

from TACR.tacr import TACR
from TACR.trainer.trainer import Trainer
from TACR.trajectory.trajectory import TrajectoryGenerator

NUM_STEPS_PER_ITERATION = 1000
MAX_ITERATIONS = 20
GENERATE_NEW_TRAJECTORIES = False
NUMBER_OF_NEW_TRAJECTORIES = 5

env = gym.make("MountainCar-v0", render_mode="human")

if GENERATE_NEW_TRAJECTORIES:
    # Custom action generator function
    def custom_action_generator() -> int:
        return np.random.choice([0, 1, 2])

    # Trajectory generation:
    generator = TrajectoryGenerator(env, number_of_trajectories=NUMBER_OF_NEW_TRAJECTORIES, action_generator_func=custom_action_generator)

    trajectories = generator.generate_all_trajectories(save_file=True, name_file="trajs")
else:
    # Load trajectories from file tacr_experiment_data/trajs.pkl
    import pickle
    with open('tacr_experiment_data/trajs.pkl', 'rb') as f:
        trajectories = pickle.load(f)



states, traj_lens, returns = [], [], []
for trajectory in trajectories:
    states.append(trajectory.observations)
    traj_lens.append(len(trajectory.observations))
    returns.append(np.array(trajectory.rewards).sum())
traj_lens, returns = np.array(traj_lens), np.array(returns)

# used for input normalization
states = np.concatenate(states, axis=0)
state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
num_timesteps = sum(traj_lens)

tacr_config = TACRConfig(state_dim=env.action_space.shape[0], action_dim=env.observation_space.shape[0], traj_lengths=traj_lens, traj_returns=returns, traj_states=states, trajectories=trajectories)
agent = TACR(config=tacr_config)

trainer = Trainer(agent, env, state_mean, state_std)

for iter in range(MAX_ITERATIONS):
    outputs = trainer.train_it(
        num_steps=NUM_STEPS_PER_ITERATION
    )

trainer.agent.save_actor("saved_models_tacr", "actor")
trainer.agent.save_critic("saved_models_tacr", "critic")
