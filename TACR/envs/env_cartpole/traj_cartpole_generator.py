# Random action generator function
import gymnasium as gym
import numpy as np
from TACR.trajectory.trajectory import TrajectoryGenerator


def random_action_generator(**kwargs) -> np.array:
    env = kwargs["env"]
    action_dim = env.action_space.n
    # Generate random probability distribution over 2 actions
    action_probabilities = np.random.dirichlet(np.ones(action_dim), size=1)[0]
    # Sample action from probability distribution
    action_index = np.random.choice([0, 1], p=action_probabilities)
    # One-hot encode action
    action = np.zeros(action_dim)
    action[action_index] = 1

    return action

def biased_random_action_generator(**kwargs) -> np.array:
    env = kwargs["env"]
    action_dim = env.action_space.n
    bias_towards = 0
    bias_strength = 0.7
    # Bias towards one action with a certain probability
    action_probabilities = [bias_strength, 1 - bias_strength] if bias_towards == 0 else [1 - bias_strength, bias_strength]
    action_index = np.random.choice([0, 1], p=action_probabilities)
    action = np.zeros(action_dim)
    action[action_index] = 1

    return action

def heuristic_action_generator(**kwargs) -> np.array:
    env = kwargs["env"]
    action_dim = env.action_space.n
    observation = kwargs["observation"]
    action_index = 1 if observation[2] > 0 else 0  # observation[2] is the angle of the pole
    action = np.zeros(action_dim)
    action[action_index] = 1

    return action

def oscillating_action_generator(**kwargs) -> np.array:
    env = kwargs["env"]
    action_dim = env.action_space.n
    step = kwargs["step"]
    oscillation_period = 10
    action_index = step // oscillation_period % 2  # Oscillates between 0 and 1 every 'oscillation_period' steps
    action = np.zeros(action_dim)
    action[action_index] = 1

    return action

# Custom action picker function that decodes one-hot encoded action
def custom_action_picker(action: np.array) -> np.array:
    return np.argmax(action)


def generate_random_trajectories(env: gym.Env, number_of_new_trajectories: int = 10):
    generator = TrajectoryGenerator(env, number_of_trajectories=number_of_new_trajectories, action_generator_func=random_action_generator, action_picker_func=custom_action_picker)
    trajectories_random = generator.generate_all_trajectories(save_file=True, name_file="trajs_random")

def generate_biased_trajectories(env: gym.Env, number_of_new_trajectories: int = 10):
    generator = TrajectoryGenerator(env, number_of_trajectories=number_of_new_trajectories, action_generator_func=biased_random_action_generator, action_picker_func=custom_action_picker)
    trajectories_biased = generator.generate_all_trajectories(save_file=True, name_file="trajs_biased_random")

def generate_heuristic_trajectories(env: gym.Env, number_of_new_trajectories: int = 10):
    generator = TrajectoryGenerator(env, number_of_trajectories=number_of_new_trajectories, action_generator_func=heuristic_action_generator, action_picker_func=custom_action_picker)
    trajectories_heuristic = generator.generate_all_trajectories(save_file=True, name_file="trajs_heuristic")

def generate_oscillating_trajectories(env: gym.Env, number_of_new_trajectories: int = 10):
    generator = TrajectoryGenerator(env, number_of_trajectories=number_of_new_trajectories, action_generator_func=oscillating_action_generator, action_picker_func=custom_action_picker)
    trajectories_oscillating = generator.generate_all_trajectories(save_file=True, name_file="trajs_oscillating")
