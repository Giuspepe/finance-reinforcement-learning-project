import math
from typing import List
import gymnasium as gym
import numpy as np

class SimpleOneStockStockTradingBaseEnv(gym.Env):
    def __init__(
        self,
        initial_account_value: float = 1_000_000,
        discount_factor: float = 0.99,
        percentage_acc_value_per_trade: float = 0.1,
        buy_transaction_fee_rate: float = 0.00025,
        sell_transaction_fee_rate: float = 0.00025,
        reward_scaling: float = 1,
        price_array: np.ndarray = None,
        tech_array: np.ndarray = None,
        is_training_mode: bool = True,
    ):
        """
        A simple stock trading environment for a single stock.

        This environment simulates stock trading where actions are discrete (buy, hold, sell) and the state includes
        the stock's technical indicators and the agent's current position.

        Parameters:
        - initial_account_value (float): The initial value of the account (default: 1,000,000).
        - discount_factor (float): Discount factor for future rewards (default: 0.99).
        - percentage_acc_value_per_trade (float): Maximum percentage of account value to use per trade (default: 0.1).
        - buy_transaction_fee_rate (float): Transaction fee rate for buying (default: 0.00025).
        - sell_transaction_fee_rate (float): Transaction fee rate for selling (default: 0.00025).
        - reward_scaling (float): Scaling factor for rewards (default: 1).
        - price_array (np.ndarray): Array of stock prices.
        - tech_array (np.ndarray): Array of technical indicators.
        - is_training_mode (bool): Flag to indicate training mode (default: True).
        """
        super().__init__()
        self.env_name = "SimpleStockTradingBase-v0"
        self.price_array = price_array.astype(np.float32)
        self.tech_array = tech_array.astype(np.float32)

        self.initial_account_value = initial_account_value
        self.discount_factor = discount_factor
        self.percentage_acc_value_per_trade = percentage_acc_value_per_trade
        self.buy_transaction_fee_rate = buy_transaction_fee_rate
        self.sell_transaction_fee_rate = sell_transaction_fee_rate
        self.reward_scaling = reward_scaling
        self.is_training_mode = is_training_mode
        self.max_episode_steps = self.price_array.shape[0]
        
        self.shares_held = 0
        self.account_value = initial_account_value
        self.cash_in_hand = initial_account_value
        self.day = 0
        self.discounted_reward = 0

        # Action space: -1 (sell), 0 (hold), 1 (buy), as integers, being discrete
        self.action_space = gym.spaces.Discrete(n=3, start=-1)
        
        self.action_dim = self.action_space.n

        # Observation space: [shares_held, technical indicators].
        self.state_dim = 1 + self.tech_array.shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

    def reset(self, *, seed: int = None, options: dict = None) -> np.ndarray:
        """
        Reset the environment to the initial state.

        Optional Parameters:
        - seed (int): Seed for the random number generator.
        - options (dict): Additional options.

        Returns:
        - np.ndarray: The initial state of the environment.

        This method resets the environment state, including the account value, shares held, and other variables.
        """
        self.shares_held = 0
        self.account_value = self.initial_account_value
        self.day = 0
        self.discounted_reward = 0
        self.cash_in_hand = self.account_value

        return self.get_state(self.price_array), dict()

    def get_state(self, price) -> np.ndarray:
        """
        Get the current state of the environment.

        Parameters:
        - price (np.ndarray): Current stock price.

        Returns:
        - np.ndarray: The current state, including shares held and technical indicators.

        This method constructs the state representation from the current position and the technical indicators.
        """
        indicators = self.tech_array[self.day]

        return np.hstack((self.shares_held, indicators))
    
    def step(self, actions: int) -> tuple:
        """
        Perform an action in the environment.

        Parameters:
        - actions (int): The action to perform (-1 for sell, 0 for hold, 1 for buy).

        Returns:
        - tuple: A tuple containing the next state, the reward, whether the episode is done, and additional info.

        This method updates the environment state based on the action taken. It handles buying, selling, and holding,
        updates the account value, and computes the reward.
        """
        # Increment the day counter
        self.day += 1

        # Actions are either -1 (Sell), 0 (Hold), or 1 (Buy)
        # If buy action, we first check if we are shorting, if so, then close the position to buy
        if actions == 1:
            if self.shares_held < 0:
                shares_to_move = -self.shares_held
                self.cash_in_hand -= (
                    shares_to_move * self.price_array[self.day] * (1 + self.buy_transaction_fee_rate)
                )
                self.shares_held += shares_to_move
            # Check if we are already long in the previous step, so we can't become "more" long
            elif self.shares_held > 0:
                shares_to_move = 0
            else:
                shares_to_move = np.floor(
                    (actions * self.percentage_acc_value_per_trade * self.account_value)
                    / self.price_array[self.day]
                )

                if self.cash_in_hand < shares_to_move * self.price_array[self.day] * (
                    1 + self.buy_transaction_fee_rate
                ):
                    shares_to_move = 0
                else: 
                    self.cash_in_hand -= (
                        shares_to_move * self.price_array[self.day] * (1 + self.buy_transaction_fee_rate)
                    )
                    self.shares_held += shares_to_move
        # If sell action, we first check if we are long, if so, then close the position to sell
        elif actions == -1:
            if self.shares_held > 0:
                shares_to_move = self.shares_held
                total_volume = shares_to_move * self.price_array[self.day] * (1 - self.sell_transaction_fee_rate)
                self.cash_in_hand += abs(total_volume)
                self.shares_held -= abs(shares_to_move)
            # Check if we are already short in the previous step, so we can't become "more" short
            elif self.shares_held < 0:
                shares_to_move = 0
            else:
                shares_to_move = np.floor(
                    (actions * self.percentage_acc_value_per_trade * self.account_value)
                    / self.price_array[self.day]
                )

                total_volume = shares_to_move * self.price_array[self.day] * (1 - self.sell_transaction_fee_rate)
                self.cash_in_hand += abs(total_volume)
                self.shares_held -= abs(shares_to_move)
        

        # Get state for the next day
        next_state = self.get_state(self.price_array)

        # Update the account value
        previous_account_value = self.account_value
        self.account_value = self.cash_in_hand + self.shares_held * self.price_array[self.day]
        #reward = (self.account_value - previous_account_value)*self.reward_scaling
        # Get the float value of the reward
        #reward = reward[0]
        # Compute the reward as the log difference between the current and previous account value
        reward = math.log(self.account_value[0] / previous_account_value) * self.reward_scaling

        # Compute discounted reward.
        self.discounted_reward = self.discounted_reward * self.discount_factor + reward
        
        # Check if the episode is done.
        done = self.day == (self.max_episode_steps-1)
        if done:
            # The reward is the discounted reward.
            reward = self.discounted_reward
            # Compute the episode return.
            self.episode_return = self.account_value / self.initial_account_value

        # Return the state, reward, done, (truncated), info.
        return next_state, reward, done, False, dict()
    
    @staticmethod
    def scaled_sigmoid(
        array: np.ndarray, threshold: float, output_bounds: tuple = (-0.5, 0.5)
    ) -> np.ndarray:
        """
        Applies a scaled sigmoid function to the input array.

        Parameters:
        - array (np.ndarray): The input array.
        - threshold (float): The threshold value for the sigmoid function.
        - output_bounds (tuple, optional): The output bounds for the scaled sigmoid function. Defaults to (-0.5, 0.5).

        Returns:
        - np.ndarray: The array after applying the scaled sigmoid function.
        """
        # Sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Apply sigmoid function to the array
        array = np.asarray(array)
        sigmoid_output = sigmoid(array / threshold)
        # Scale the sigmoid output to the output bounds
        output_range = output_bounds[1] - output_bounds[0]
        return sigmoid_output * output_range + output_bounds[0]
