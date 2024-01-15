import os
import pickle
from typing import Any, Callable, List, Optional
import gymnasium
import numpy as np


class UnfinishedTrajectory:
    def __init__(self):
        self.observations = []
        self.rewards = []
        self.dones = []
        self.actions = []

    def append_observation(self, obs, rew, done, actions):
        self.observations.append(obs)
        self.rewards.append(rew)
        self.dones.append(done)
        self.actions.append(actions)

class Trajectory:
    def __init__(self, observations: List[np.ndarray] = [], rewards: List[float] = [], dones: List[int] = [], actions: List[Any] = []):
        self.observations = np.array(observations)
        self.rewards = np.array(rewards)
        self.dones = np.array(dones)
        self.actions = np.array(actions)


class TrajectoryGenerator:
    def __init__(self, env: gymnasium.Env, number_of_trajectories: int = 10, action_generator_func: Optional[Callable] = None, action_picker_func: Optional[Callable] = None):
        self.env = env
        self.number_of_trajectories = number_of_trajectories
        self.action_generator_func = action_generator_func
        self.action_picker_func = action_picker_func

    def generate_trajectory(self) -> Trajectory:
        trajectory = UnfinishedTrajectory()
        observation = self.env.reset()
        observation = observation[0]
        while True:
            if self.action_generator_func:
                action = self.action_generator_func()
            else:
                action = self.env.action_space.sample()

            if self.action_picker_func:
                picked_action = self.action_picker_func(action)
            else:
                picked_action = action

            next_observation, reward, done, truncated, info = self.env.step(picked_action)
            done_or_truncated = done or truncated
            trajectory.append_observation(np.array(observation), reward, done_or_truncated, action)
            observation = next_observation

            if done_or_truncated: 
                break

        finished_trajectory = Trajectory(trajectory.observations, trajectory.rewards, trajectory.dones, trajectory.actions)
            
        return finished_trajectory
    
    def generate_all_trajectories(self, save_file: bool = False, name_file: str = "trajs") -> List[Trajectory]:
        trajectories: List[Trajectory] = []
        for i in range(self.number_of_trajectories):
            trajectories.append(self.generate_trajectory())
            print(f"Generated trajectory {i+1}/{self.number_of_trajectories}")

        if save_file:
            if not os.path.exists("tacr_experiment_data"):
                os.makedirs("tacr_experiment_data")

            name = f'{"tacr_experiment_data/"+name_file}'
            with open(f'{name}.pkl', 'wb') as f:
                pickle.dump(trajectories, f)

        return trajectories
        



