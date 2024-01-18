from typing import List
import gymnasium as gym
import numpy as np


class SimpleOneStockStockTradingBaseEnv(gym.Env):
    # Only gets exposed at discrete amounts of 10% of the account value
    def __init__(
        self,
        initial_account_value: float = 1_000_000,
        discount_factor: float = 0.99,
        percentage_acc_value_per_trade: float = 0.1,
        buy_transaction_fee_rate: float = 0.00025,
        sell_transaction_fee_rate: float = 0.00025,
        reward_scaling: float = pow(2, -11),
        price_array: np.ndarray = None,
        tech_array: np.ndarray = None,
        is_training_mode: bool = True,
    ):
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

        # Observation space: [account_value, shares_held, technical indicators].
        self.state_dim = 1 + 1 + self.tech_array.shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

    def reset(self, *, seed: int = None, options: dict = None) -> np.ndarray:
        """
        Resets the environment.
        """
        self.shares_held = 0
        self.account_value = self.initial_account_value
        self.day = 0
        self.discounted_reward = 0
        self.cash_in_hand = self.account_value

        return self.get_state(self.price_array), dict()

    def get_state(self, price) -> np.ndarray:
        """
        Returns the state of the environment.
        """
        indicators = self.tech_array[self.day]

        return np.hstack((self.shares_held, indicators))
    
    def step(self, actions: int) -> tuple:
        """
        Performs a step in the environment.
        """
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
        reward = (self.account_value - previous_account_value)*self.reward_scaling
        # Get the float value of the reward
        reward = reward[0]

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

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        array = np.asarray(array)
        sigmoid_output = sigmoid(array / threshold)
        output_range = output_bounds[1] - output_bounds[0]
        return sigmoid_output * output_range + output_bounds[0]
