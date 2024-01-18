from typing import List
import gymnasium
import pandas as pd
import numpy as np
from TACR.envs.env_stocktrading.env_stocktrading import SimpleOneStockStockTradingBaseEnv

from TACR.trajectory.trajectory import Trajectory, TrajectoryGenerator

def best_trade_on_window_action_generator(offset: int = 1, window_size: int = 10, min_variation: float = 0.01, price_array: np.ndarray = None):
    """
    Generate actions based on the best trade within a specified window of stock prices, prophetically.

    Parameters:
    - offset (int): Initial offset for the window.
    - window_size (int): Size of the window to consider for each trade decision.
    - min_variation (float): Minimum variation in price to consider a change significant.
    - price_array (np.ndarray): Array of stock prices.

    Returns:
    - np.ndarray: Array of one-hot encoded actions for the length of the price array.

    This function analyzes windows of stock prices and determines the best action to take (buy, sell, hold)
    at the beginning of each window based on the price change within the window.
    """
        
    # Function that generate windows of size 'window_size' and returns the best trade on that window, offset by 'offset' initially,
    # where the action generated is -1 (sell) if the price is lower at the end of the window than at the beginning, 1 (buy) if the price is higher,
    # and 0 (hold) if the price didn't change enough (min_variation) or if the window is out of bounds.
    # If the window is out of bounds, the action is 0 (hold).

    # Initialize an action array that has all zeros for the dataset length using numpy
    actions = np.zeros(len(price_array))

    # Iterate over the dataset, generating windows of size 'window_size' and offset by 'offset'. Note that
    # actions can only take place at the beginning of a window, and at the end of the window, if the action is non-zero
    # then if the same signal is repeated (ex: beginning of window = 1, end of window = 1), then the action = 0. If this is not the case, close the position by doing new_action = -action
    # Iterate over the dataset
    for i in range(offset, len(price_array) - window_size, window_size):  # Iterate in steps of window_size
        j = i+1 # Prophetic Actions
        start_price = price_array[j]
        end_price = price_array[j + window_size - 1]
        price_change = end_price - start_price

        # Determine action based on price change and min_variation
        if abs(price_change) < min_variation:
            action = 0
        elif price_change > 0:
            action = 1
        else:
            action = -1

        # Set action at the start of the window
        actions[i] = action

        # Closing the position at the end of the window regardless of the signal
        actions[i + window_size - 1] = -action

    # For each action, one-hot encode
    action_dim = 3
    actions_one_hot = np.zeros((len(actions), action_dim))
    for i in range(len(actions)):
        action = np.zeros(action_dim)
        action[int(actions[i]) + 1] = 1
        actions_one_hot[i] = action

    return actions_one_hot


def custom_action_picker(action: np.array) -> int:
    """
    Custom function to decode one-hot encoded actions.

    Parameters:
    - action (np.array): One-hot encoded action.

    Returns:
    - int: Decoded action (-1 for sell, 0 for hold, 1 for buy).

    This function decodes a one-hot encoded action into a discrete action used by the environment.
    """
    if action[0] == 1:
        return -1 # Sell
    elif action[1] == 1:
        return 0 # Hold
    else:
        return 1 # Buy
    
def generate_best_trade_on_window_trajectories(env: SimpleOneStockStockTradingBaseEnv):
    """
    Generate trajectories based on the best trade actions within different window sizes and offsets.

    Parameters:
    - env (SimpleOneStockStockTradingBaseEnv): The stock trading environment instance.

    This function generates trajectories for a stock trading environment based on the best trade decision
    for varying window sizes and offsets, and saves the trajectories to a file.
    """
    # Generate trajectories based on the best trade on window action generator
    generator = TrajectoryGenerator(env, number_of_trajectories=1, action_picker_func=custom_action_picker)

    # Define window sizes and offsets
    window_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20]
    offsets = [0, 1, 2, 3, 4, 5, 8, 10, 12, 14, 16]
    trajectories: List[Trajectory] = []

    # Iterate over different window sizes and offsets
    for window_size in window_sizes:
        for offset in offsets:
            if offset*1.5 < window_size:
                trade_list = best_trade_on_window_action_generator(offset=offset, window_size=window_size, price_array=env.price_array, min_variation=0.003)
                trajectory = generator.generate_trajectory_from_predefined_list_of_actions(trade_list)
                trajectories.append(trajectory)

    # Save trajectories to file
    generator.save_file(trajectories, "trajs_best_trade_on_window")

    


        



