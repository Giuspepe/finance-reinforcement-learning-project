import math
from typing import Dict, List
import gymnasium as gym
import numpy as np

class SimplePortfolioAllocationBaseEnv(gym.Env):
    def __init__(
        self,
        initial_account_value: float = 1_000_000,
        discount_factor: float = 0.99,
        transaction_fee_rate: float = 0.00025,
        reward_scaling: float = 1,
        stock_data: Dict[str, Dict[str, np.ndarray]] = None,  # Updated structure
        is_training_mode: bool = True,
    ):
        """
        A simple portfolio allocation environment for multiple stocks.

        This environment simulates stock trading where actions represent the allocation weights for each stock in the portfolio.
        The state includes each stock's technical indicators and the agent's current portfolio weights.

        Parameters:
        - initial_account_value (float): The initial value of the account (default: 1,000,000).
        - discount_factor (float): Discount factor for future rewards (default: 0.99).
        - transaction_fee_rate (float): Transaction fee rate for buying and selling (default: 0.00025).
        - reward_scaling (float): Scaling factor for rewards (default: 1).
        - stock_data (Dict[str, Dict[str, np.ndarray]]): Dictionary containing data for each stock.
        - is_training_mode (bool): Flag to indicate training mode (default: True).
        """
        super().__init__()
        self.stock_data = stock_data
        self.initial_account_value = initial_account_value
        self.discount_factor = discount_factor
        self.transaction_fee_rate = transaction_fee_rate
        self.reward_scaling = reward_scaling
        self.is_training_mode = is_training_mode

        self.num_stocks = len(stock_data)
        self.max_episode_steps = min(len(stock_data[stock]["close"]) for stock in stock_data)  # Adjust for the smallest dataset
        self.day = 0

        # Initialize prophetic actions
        self.prophetic_actions = self.generate_prophetic_actions(i=5, window_size=1)

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)

        # Determine the length of the technical indicators for a single stock
        sample_tech_indicators = next(iter(stock_data.values()))["tech_array"]
        tech_indicator_length = sample_tech_indicators.shape[1]

        # State space is the [portfolio weights, technical indicators for each stock]
        self.state_dim = self.num_stocks + self.num_stocks * tech_indicator_length
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        # Initialize portfolio
        self.portfolio_weights = np.array([1.0 / self.num_stocks] * self.num_stocks)
        self.account_value = initial_account_value

        # Memorize portfolio value, portfolio return, actions, and date
        self.asset_memory = [initial_account_value]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.num_stocks] * self.num_stocks]
        self.date_memory = [0]  # Placeholder for the date

    def reset(self):
        self.day = 0
        self.account_value = self.initial_account_value
        self.portfolio_weights = np.array([1.0 / self.num_stocks] * self.num_stocks)
        self.asset_memory = [self.initial_account_value]
        return self.get_state(), dict()

    def get_state(self):
        # Concatenate technical indicators for all stocks
        tech_indicators = np.concatenate([self.stock_data[stock]["tech_array"][self.day] for stock in self.stock_data.keys()], axis=0)

        # Concatenate portfolio weights
        state = np.concatenate([self.portfolio_weights, tech_indicators], axis=0)

        return state

    def step(self, actions):
        # Normalize actions to ensure they sum to 1
        actions = actions / np.sum(actions)

        # Calculate portfolio value before reallocation
        portfolio_value_before = self.account_value

        # Update portfolio weights and account value
        self.portfolio_weights = actions
        portfolio_growth = sum([actions[i] * (self.stock_data[stock]["close"][self.day + 1] / self.stock_data[stock]["close"][self.day] - 1)
                                for i, stock in enumerate(self.stock_data.keys())])
        self.account_value *= (1 + portfolio_growth - self.transaction_fee_rate * np.sum(np.abs(actions - self.portfolio_weights)))

        # Calculate reward
        portfolio_value_after = self.account_value
        reward = self.reward_scaling * (portfolio_value_after - portfolio_value_before) / portfolio_value_before

        self.asset_memory.append(self.account_value)
        self.day += 1

        # Check if the episode is done
        done = self.day >= self.max_episode_steps - 1

        return self.get_state(), reward, done, False, dict()
    
    def generate_prophetic_actions(self, i, window_size):
        """
        Generate the prophetic actions for the portfolio environment.

        Args:
        - i (float): Rate of increase used in weight calculation.
        - window_size (int): The number of days to look ahead for price change.

        Returns:
        - List of np.array: A list of arrays where each array represents the portfolio weights for a day.
        """
        prophetic_actions = []

        day = 0
        while day < self.max_episode_steps:
            window_end = min(day + window_size, self.max_episode_steps - 1)

            # Calculate the weight for each stock based on the price at the end of the window
            bc = []
            for stock in self.stock_data.keys():
                portion = self.stock_data[stock]["close"][window_end] / self.stock_data[stock]["close"][day]
                bc.append(np.exp(portion * (i + 1)))

            # Normalize the weights to sum to 1
            weights = np.array(bc) / np.sum(bc)

            # Apply the same weights for each day in the window
            for _ in range(day, window_end):
                prophetic_actions.append(weights)

            # Move to the next window
            day += window_size

        return prophetic_actions

    def get_prophetic_action(self, current_index: int) -> int:
        """
        Get the prophetic action for the current state.


        Returns:
        - int: The prophetic action for the current state.

        This method returns the prophetic action for the current state.
        """
        return self.prophetic_actions[current_index]

        
