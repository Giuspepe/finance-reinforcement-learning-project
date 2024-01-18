import math
import gymnasium as gym
import numpy as np

# Code adapted from FinRL library: https://github.com/AI4Finance-LLC/FinRL


class StockTradingBase(gym.Env):
    def __init__(
        self,
        config: dict = None,
        initial_account_value: float = 1_000_000,
        discount_factor: float = 0.99,
        turbulence_threshold: float = 80,
        minimum_stock_rate: float = 0.1,
        maximum_shares_per_trade: int = 1_00,
        initial_cash_balance: float = 1_000_000,
        buy_transaction_fee_rate: float = 0.001,
        sell_transaction_fee_rate: float = 0.001,
        reward_scaling: float = pow(2, -11),
        initial_stocks: np.ndarray = None,
    ):
        """
        Initializes the StockTradingBase environment.

        Args:
            config (dict, optional): Configuration parameters for the environment.
                - price_array: Historical price data for stocks.
                - tech_array: Technical indicators data for stocks.
                - turbulence_array: Turbulence data for the market.
                - is_training_mode: True for training mode, False for testing mode.
                - target_return: Target return for the environment. Defaults to 10.0.
                - episode_return: Episode return for the environment. Defaults to 0.0.
                - obs_low: Lower bound for the observation space. Defaults to -3000.
                - obs_high: Upper bound for the observation space. Defaults to 3000.
                - act_low: Lower bound for the action space. Defaults to -1.
                - act_high: Upper bound for the action space. Defaults to 1.
            initial_account_value (float, optional): Initial account value. Defaults to 1_000_000.
            discount_factor (float, optional): Discount factor for future rewards. Defaults to 0.99.
            turbulence_threshold (float, optional): Threshold for determining turbulent market. Defaults to 80.
            minimum_stock_rate (float, optional): Minimum stock rate. Defaults to 0.1.
            maximum_shares_per_trade (int, optional): Maximum number of shares per trade. Defaults to 100.
            initial_cash_balance (float, optional): Initial cash balance. Defaults to 1_000_000.
            buy_transaction_fee_rate (float, optional): Transaction fee rate for buying stocks. Defaults to 0.001.
            sell_transaction_fee_rate (float, optional): Transaction fee rate for selling stocks. Defaults to 0.001.
            reward_scaling (float, optional): Scaling factor for rewards. Defaults to pow(2, -11).
            initial_stocks (np.ndarray, optional): Initial stocks. Defaults to None.
        """
        super().__init__()
        price_array, tech_array, turbulence_array, is_training_mode = (
            config["price_array"],
            config["tech_array"],
            config["turbulence_array"],
            config["is_training_mode"],
        )

        self.price_array = price_array.astype(np.float32)
        self.tech_array = tech_array.astype(np.float32) * pow(2, -7)
        self.turbulence_array = turbulence_array
        self.is_turbulent_market = (turbulence_array > turbulence_threshold).astype(
            np.float32
        )
        self.turbulence_array = (
            self.scaled_sigmoid(turbulence_array, turbulence_threshold) * pow(2, -5)
        ).astype(np.float32)

        stock_dim = self.price_array.shape[1]
        self.discount_factor = discount_factor
        self.maximum_shares_per_trade = maximum_shares_per_trade
        self.minimum_stock_rate = minimum_stock_rate
        self.initial_cash_balance = initial_cash_balance
        self.buy_transaction_fee_rate = buy_transaction_fee_rate
        self.sell_transaction_fee_rate = sell_transaction_fee_rate
        self.reward_scaling = reward_scaling
        self.initial_account_value = initial_account_value
        self.initial_stocks = (
            initial_stocks
            if initial_stocks is not None
            else np.zeros(stock_dim, dtype=np.float32)
        )

        self.day = None
        self.amount = None
        self.stocks = None
        self.total_assets = None
        self.discounted_reward = None
        self.initial_total_assets = None

        self.env_name = "StockTradingBase-v0"
        # Calculate the dimension of the state space:
        # 1 for the agent's account balance,
        # 2 for turbulence level and a boolean flag indicating high turbulence,
        # 3 * stock_dim for stock-related features (e.g., price, quantity, additional feature) for each stock,
        # self.tech_ary.shape[1] for the number of technical indicators per stock.
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_array.shape[1]
        # action_dim is a vector with the same dimension as the number of stocks
        self.action_dim = stock_dim
        # max_episode_steps is the number of trading days
        self.max_episode_steps = self.price_array.shape[0] - 1
        self.is_training_mode = is_training_mode
        self.stocks_cool_down = None
        self.is_discrete_action = False
        self.target_return = config.get("target_return", 10.0)
        self.episode_return = config.get("episode_return", 0.0)
        obs_low = config.get("obs_low", -3000)
        obs_high = config.get("obs_high", 3000)
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, shape=(self.state_dim,), dtype=np.float32
        )
        act_low = config.get("act_low", -1)
        act_high = config.get("act_high", 1)
        self.action_space = gym.spaces.Box(
            low=act_low, high=act_high, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self, *, seed: int = None, options: dict = None) -> np.ndarray:
        """
        Resets the environment to its initial state.

        Parameters:
        - seed (int): The seed for the random number generator. Default is None.
        - options (dict): Additional options for resetting the environment. Default is None.

        Returns:
        - np.ndarray: The initial state of the environment.

        """
        self.day = 0
        price = self.price_array[self.day]
        if self.is_training_mode:
            self.stocks = (
                self.initial_stocks
                + np.random.randint(0, 64, self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_cash_balance * np.random.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            ).astype(np.float32)
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_cash_balance

        self.total_assets = self.amount + (self.stocks * price).sum()
        self.initial_total_assets = self.total_assets
        self.discounted_reward = 0.0
        return self.get_state(price), {}

    def get_state(self, price):
        """
        Returns the state representation for the current trading environment.

        Parameters:
        - price: The current price of the stock.

        Returns:
        - state: The state representation as a numpy array.
        """
        amount = np.array(self.amount * pow(2, -12), dtype=np.float32)
        scale = np.array(pow(2, -6), dtype=np.float32)
        return np.hstack(
            (
                amount,
                self.turbulence_array[self.day],
                self.is_turbulent_market[self.day],
                #price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_array[self.day],
            )
        )

    def step(self, actions: np.ndarray) -> tuple:


        # Scale the actions by the maximum number of shares that can be traded at once, and convert to integer.
        # This ensures that the actions represent a valid number of shares for trading.
        actions = (actions * self.maximum_shares_per_trade).astype(int)

        # Update the day counter.
        self.day += 1

        # Get the current price of the stock.
        price = self.price_array[self.day]

        # Update the stocks cool down counter.
        self.stocks_cool_down += 1

        if self.is_turbulent_market[self.day] == 0:
            # The minimum action is the minimum number of shares that can be traded at once.
            minimum_action = int(
                self.maximum_shares_per_trade * self.minimum_stock_rate
            )

            # Sell stocks if the action is less than the negative minimum action.
            for index in np.where(actions < -minimum_action)[0]:
                # Only sell if the price of the stock is greater than 0.
                if price[index] > 0:
                    # The number of shares to sell is the minimum of the absolute action and the number of shares owned.
                    num_shares_to_sell = min(-actions[index], self.stocks[index])
                    # Update the number of shares owned and the amount of cash.
                    self.stocks[index] -= num_shares_to_sell
                    self.amount += (
                        num_shares_to_sell
                        * price[index]
                        * (1 - self.sell_transaction_fee_rate)
                    )
                    # Reset the cool down counter.
                    self.stocks_cool_down[index] = 0
            # Buy stocks if the action is greater than the minimum action.
            for index in np.where(actions > minimum_action)[0]:
                # Only buy if the price of the stock is greater than 0.
                if price[index] > 0:
                    # The number of shares to buy is the minimum of the action and the amount of cash available.
                    num_shares_to_buy = min(self.amount // price[index], actions[index])
                    self.amount -= (
                        num_shares_to_buy
                        * price[index]
                        * (1 + self.buy_transaction_fee_rate)
                    )
                    # Reset the cool down counter.
                    self.stocks_cool_down[index] = 0
        else:
            # Sell all stocks if the market is turbulent.
            self.amount += (self.stocks * price).sum() * (
                1 - self.sell_transaction_fee_rate
            )
            # Reset the stocks owned.
            self.stocks[:] = 0
            # Reset the cool down counter.
            self.stocks_cool_down[:] = 0
        # Get the state for the next time step.
        state = self.get_state(price)

        # Compute the previous total assets.
        previous_total_assets = self.total_assets
        # Compute the total assets.
        self.total_assets = self.amount + (self.stocks * price).sum()
        # Compute the reward.
        #reward = (self.total_assets - previous_total_assets) * self.reward_scaling
        # Compute the reward as the log difference between the current and previous total assets.
        reward = math.log(self.total_assets / previous_total_assets) * self.reward_scaling
        # Compute discounted reward.
        self.discounted_reward = self.discounted_reward * self.discount_factor + reward
        # Check if the episode is done.
        done = self.day == (self.max_episode_steps-1)
        if done:
            # The reward is the discounted reward.
            reward = self.discounted_reward
            # Compute the episode return.
            self.episode_return = self.total_assets / self.initial_total_assets

        # Return the state, reward, done, (truncated), info.
        return state, reward, done, False, dict()

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