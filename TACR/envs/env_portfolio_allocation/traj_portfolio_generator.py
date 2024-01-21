from typing import List
import pandas as pd
import numpy as np
from env_stocktrading.env_portfolio_allocation import SimplePortfolioAllocationBaseEnv

from TACR.trajectory.trajectory import Trajectory, TrajectoryGenerator

def best_portfolio_on_window_action_generator(env: SimplePortfolioAllocationBaseEnv, window_size: int = 10, i: float = 1):
    """
    Generate actions based on the best trade within a specified window of stock prices, prophetically.

    Parameters:
    - window_size (int): Size of the window to consider for each trade decision.
    - i (float): Rate of increase used in weight calculation.

    Returns:
    - np.ndarray: Array weights for the length of the price array.
    """
    actions_prophetically = env.generate_prophetic_actions(i=i, window_size=window_size)

    return actions_prophetically

def generate_best_portfolio_on_window_trajectories(env: SimplePortfolioAllocationBaseEnv):
    """
    Generate trajectories based on the best portfolio actions within different window sizes and rates of increase.

    Parameters:
    - env (SimpleOneStockStockTradingBaseEnv): The stock portfolio weights instance.

    This function generates trajectories for a stock portfolio allocation environment based on the best weight decision
    for varying window sizes and offsets, and saves the trajectories to a file.
    """
    # Generate trajectories based on the best trade on window action generator
    generator = TrajectoryGenerator(env, number_of_trajectories=1)

    # Define window sizes and offsets
    window_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    rates_of_increase = [1, 2, 3, 4, 5, 7, 10, 15, 25, 40, 50]
    trajectories: List[Trajectory] = []

    # Iterate over different window sizes and offsets
    for window_size in window_sizes:
        for i in rates_of_increase:
            trade_list = best_portfolio_on_window_action_generator(window_size=window_size, env=env, i=i)
            trajectory = generator.generate_trajectory_from_predefined_list_of_actions(trade_list)
            trajectories.append(trajectory)

    # Save trajectories to file
    generator.save_file(trajectories, "trajs_best_portfolio_on_window")

    


        



