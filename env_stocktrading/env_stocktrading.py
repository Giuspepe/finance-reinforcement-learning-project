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
        close_array: np.ndarray = None,
        open_array: np.ndarray = None,
        high_array: np.ndarray = None,
        low_array: np.ndarray = None,
        tech_array: np.ndarray = None,
        index_of_expert_tech_indicator: int = -1,
        prophetic_actions_window_length: int = 10,
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
        - close_array (np.ndarray): Array of stock close prices.
        - tech_array (np.ndarray): Array of technical indicators.
        """
        super().__init__()
        self.env_name = "SimpleStockTradingBase-v0"
        self.close_array = close_array.astype(np.float32)
        self.open_array = open_array.astype(np.float32)
        self.high_array = high_array.astype(np.float32)
        self.low_array = low_array.astype(np.float32)
        self.tech_array = tech_array.astype(np.float32)

        # Get an array for the expert technical indicator
        self.expert_tech_indicator = self.tech_array[:, index_of_expert_tech_indicator]

        # Prophetic Actions
        self.prophetic_actions_window_length = prophetic_actions_window_length
        self.prophetic_actions = self.generate_prophetic_actions()

        self.initial_account_value = initial_account_value
        self.discount_factor = discount_factor
        self.percentage_acc_value_per_trade = percentage_acc_value_per_trade
        self.buy_transaction_fee_rate = buy_transaction_fee_rate
        self.sell_transaction_fee_rate = sell_transaction_fee_rate
        self.reward_scaling = reward_scaling
        self.max_episode_steps = self.close_array.shape[0]
        
        self.shares_held = 0
        self.account_value = initial_account_value
        self.cash_in_hand = initial_account_value
        self.day = 0
        self.discounted_reward = 0

        # Action space: -1 (sell), 0 (hold), 1 (buy), as integers, being discrete
        self.action_space = gym.spaces.Discrete(n=3, start=-1)
        
        self.action_dim = self.action_space.n

        # Observation space: [long_short_hold, technical indicators].
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

        return self.get_state(), dict()

    def get_state(self) -> np.ndarray:
        """
        Get the current state of the environment.

        Returns:
        - np.ndarray: The current state, including shares held and technical indicators.

        This method constructs the state representation from the current position and the technical indicators.
        """
        indicators = self.tech_array[self.day]

        long_short_hold = 0
        if self.shares_held > 0:
            long_short_hold = 1
        elif self.shares_held < 0:
            long_short_hold = -1

        return np.hstack((long_short_hold, indicators))
    
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
                    shares_to_move * self.close_array[self.day] * (1 + self.buy_transaction_fee_rate)
                )
                self.shares_held += shares_to_move
            # Check if we are already long in the previous step, so we can't become "more" long
            elif self.shares_held > 0:
                shares_to_move = 0
            else:
                shares_to_move = np.floor(
                    (actions * self.percentage_acc_value_per_trade * self.account_value)
                    / self.close_array[self.day]
                )

                if self.cash_in_hand < shares_to_move * self.close_array[self.day] * (
                    1 + self.buy_transaction_fee_rate
                ):
                    shares_to_move = 0
                else: 
                    self.cash_in_hand -= (
                        shares_to_move * self.close_array[self.day] * (1 + self.buy_transaction_fee_rate)
                    )
                    self.shares_held += shares_to_move
        # If sell action, we first check if we are long, if so, then close the position to sell
        elif actions == -1:
            if self.shares_held > 0:
                shares_to_move = self.shares_held
                total_volume = shares_to_move * self.close_array[self.day] * (1 - self.sell_transaction_fee_rate)
                self.cash_in_hand += abs(total_volume)
                self.shares_held -= abs(shares_to_move)
            # Check if we are already short in the previous step, so we can't become "more" short
            elif self.shares_held < 0:
                shares_to_move = 0
            else:
                shares_to_move = np.floor(
                    (actions * self.percentage_acc_value_per_trade * self.account_value)
                    / self.close_array[self.day]
                )

                total_volume = shares_to_move * self.close_array[self.day] * (1 - self.sell_transaction_fee_rate)
                self.cash_in_hand += abs(total_volume)
                self.shares_held -= abs(shares_to_move)
        

        # Get state for the next day
        next_state = self.get_state()

        # Update the account value
        previous_account_value = self.account_value
        self.account_value = self.cash_in_hand + self.shares_held * self.close_array[self.day]
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
            #reward = self.discounted_reward
            # The reward is zero
            reward = 0
            # Compute the episode return.
            self.episode_return = self.account_value / self.initial_account_value

        # Return the state, reward, done, (truncated), info.
        return next_state, reward, done, False, dict()
    
    def expert_action(self, current_index: int) -> int:
        """
        Get the expert action for the current state.


        Returns:
        - int: The expert action for the current state.

        This method returns the expert action for the current state.
        Currently uses BINARY_SMA_RISING(24) indicator to determine the action.
        """
        if self.expert_tech_indicator[current_index] == 1:
            if self.shares_held < 1:
                return 1
            else:
                return 0
        else:
            if self.shares_held > -1:
                return -1
            else:
                return 0
            
    def generate_prophetic_actions(self) -> np.ndarray:
        """
        Generate the prophetic actions for the environment.

        Returns:
        - np.ndarray: The prophetic actions for the environment.

        This method analyzes windows of stock prices and determines the best action to take (buy, sell, hold)
        by generating windows of self.prophetic_actions_window_length and calculate the price change withing the start of the window
        and for each day in the window until the end, taking the action that maximizes the gain for any of the days and closing the
        position at the day where the gain is maximum. Then continue from the day after the position is ended and NOT at the end of the window.
        """
        # Initialize an action array with all zeros for the dataset length
        actions = np.zeros(len(self.close_array), dtype=int)

        day = 0
        while day < len(self.close_array):
            # Generate a window of length 10 days
            window_length = self.prophetic_actions_window_length
            window_end = min(day + window_length, len(self.close_array))

            # Initialize variables to track the best action and its day
            best_gain = 0
            best_action_day = day
            best_action = 0  # 0 for hold, 1 for buy, -1 for sell

            for window_day in range(day, window_end):
                price_change = self.close_array[window_day] - self.close_array[day]

                # Check if buying on 'day' and selling on 'window_day' is beneficial
                if price_change > best_gain:
                    best_gain = price_change
                    best_action_day = window_day
                    best_action = 1  # Buy

                # We can also check for selling (if we held stocks) based on other indicators or conditions here

            # Apply the best action found, just for the initial day of the window
            actions[day] = best_action
            # Close the position on the best action day
            actions[best_action_day] = -best_action

            # Move to the day after the best action day to start the next window
            day = best_action_day + 1

        return actions

    def get_prophetic_action(self, current_index: int) -> int:
        """
        Get the prophetic action for the current state.


        Returns:
        - int: The prophetic action for the current state.

        This method returns the prophetic action for the current state.
        """
        return self.prophetic_actions[current_index]

        
