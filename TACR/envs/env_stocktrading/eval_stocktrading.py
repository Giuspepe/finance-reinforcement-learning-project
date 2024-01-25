import os
import sys

# Get parent of parent of parent directory
parent_of_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(parent_of_parent_dir)

from TACR.tacr import TACR
from TACR.config.config import TACRConfig
from env_stocktrading.utils import augment_data, create_environment
from env_stocktrading.env_stocktrading import SimpleOneStockStockTradingBaseEnv
from preprocessing.custom_technical_indicators import PCT_RETURN, OBV_PCT_CHANGE, RSI, RVI_PCT_CHANGE, ADX, RSI_CATEGORICAL, BINARY_SMA_RISING
from preprocessing.process_yh_finance import YHFinanceProcessor

import numpy as np
import pandas as pd
import gymnasium as gym
import torch    

# Custom action picker function that decodes one-hot encoded action
def action_softmax_to_value(action: np.array) -> int:
    index_of_max_probability = np.argmax(action)
    decoded_action = index_of_max_probability - 1
    return decoded_action
    

def evaluate(env: SimpleOneStockStockTradingBaseEnv, path: str, name_actor="actor", name_config="config", num_episodes=100, max_timesteps_per_episode=10000, silence=True):

    # Prepare TACR
    tacr_config = TACRConfig()
    tacr = TACR(config=tacr_config, load_file_config=True, file_config_path=path, file_config_name=name_config)
             
    # Load Actor Model
    tacr.load_actor(path, name_actor)

    tacr.actor.eval()

    # Get state mean and std
    state_mean = tacr.config.state_mean
    state_std = tacr.config.state_std

    state_dim = tacr.config.state_dim
    action_dim = tacr.config.action_dim

    episode_reward_vector = []
    obs, _info = env.reset()
    obs = np.array(obs)

    # Keep all histories on device
    # Latest action and Reward will be "padding"

    states = torch.from_numpy((obs - state_mean) / state_std).reshape(1, state_dim).to(device=tacr.device, dtype=torch.float32)
    actions = torch.zeros((0, action_dim), device=tacr.device, dtype=torch.float32)
    rewards = torch.zeros(0, device=tacr.device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=tacr.device, dtype=torch.long).reshape(1, 1)

    episode_count = 1
    timestep_count = 1
    total_reward = 0
    while True:
        timestep_count += 1

        # Add padding
        actions = torch.cat([actions, torch.zeros((1, action_dim), device=tacr.device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=tacr.device)])

        action = tacr.actor.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )

        action = action.reshape(1, action_dim)

        actions[-1] = action
        action = action.detach().cpu().numpy()

        action = action_softmax_to_value(action)
        
        obs, reward, terminated, truncated, info = env.step(action, eval_mode=True)
        obs = np.array(obs)


        cur_state = torch.from_numpy(obs).to(device=tacr.device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=tacr.device, dtype=torch.long) * (timestep_count+1)], dim=1)

        total_reward += reward

        if terminated or timestep_count > max_timesteps_per_episode:
            if not silence:
                print(f"Episode {episode_count} reward: {total_reward} account value: {env.account_value}")
            # obs, _info = env.reset()
            # timestep_count = 1
            episode_reward_vector.append(total_reward)

            total_reward = 0

            # Visualize Actions
            # actions_decoded = []
            # for action in actions:
            #     action = action.reshape(1, action_dim)
            #     action = action.detach().cpu().numpy()
            #     actions_decoded.append(action_softmax_to_value(action))
            # # Matplotlib
            # import matplotlib.pyplot as plt
            # plt.plot(actions_decoded)
            # plt.show()

            # # Reset history for the episode
            # states = torch.from_numpy(obs).reshape(1, state_dim).to(device=tacr.device, dtype=torch.float32)
            # actions = torch.zeros((0, action_dim), device=tacr.device, dtype=torch.float32)
            # rewards = torch.zeros(0, device=tacr.device, dtype=torch.float32)
            # timesteps = torch.tensor(0, device=tacr.device, dtype=torch.long).reshape(1, 1)

            episode_count += 1
            if episode_count > num_episodes:
                break
        

    env.close()
    avg_reward = np.mean(episode_reward_vector)

    return episode_reward_vector, avg_reward

if __name__ == "__main__":
    TICKERS = ["INTC"]
    data_dir = "data/stocktrading"
    os.makedirs(data_dir, exist_ok=True)
    INDICATORS = []
    CUSTOM_INDICATORS = [PCT_RETURN(length=2), OBV_PCT_CHANGE(length=12), RVI_PCT_CHANGE(length=20, rvi_pct_change_length=2), RSI(length=14)]

    yfp = YHFinanceProcessor()

    val_dataset = pd.read_csv(os.path.join(data_dir, f"val_stock_data_{TICKERS[0]}.csv"))
    val_dataset = augment_data(yfp, val_dataset, INDICATORS, CUSTOM_INDICATORS, vix=False)

    val_env = create_environment(yfp, val_dataset, INDICATORS, CUSTOM_INDICATORS, gamma=0.999)
    
    episode_reward_vector, avg_reward = evaluate(val_env, "saved_models_tacr", "actor_best", "config_best", num_episodes=1, max_timesteps_per_episode=10000, silence=False)

    print(f"Average reward: {avg_reward}")

    # Based on val_env.close_array, generate two arrays with the return per day if you buy and hold the stock and if you sell and hold the stock
    # then multiply it by val_env.initial_account_value to get the account value per day
     
    # Assuming val_env is an instance of SimpleOneStockStockTradingBaseEnv
    close_prices = val_env.close_array
    initial_price = close_prices[0]
    initial_account_value = val_env.initial_account_value

    # Buy and Hold Strategy
    # Calculate the proportionate increase in stock price each day
    buy_and_hold_proportions = close_prices / initial_price
    # Calculate account value for each day
    buy_and_hold_account_values = buy_and_hold_proportions * initial_account_value

    # Sell and Hold (Short and Hold) Strategy
    # Calculate the proportionate change in stock price each day in the opposite direction
    sell_and_hold_proportions = 2 - (close_prices / initial_price)
    # Calculate account value for each day
    sell_and_hold_account_values = sell_and_hold_proportions * initial_account_value

    # Calculate total accumulated percentage return for each strategy
    # Buy and Hold Accumulated Daily Returns
    buy_and_hold_accumulated_returns = [(value - initial_account_value) / initial_account_value * 100 for value in buy_and_hold_account_values]

    # Sell and Hold Accumulated Daily Returns
    sell_and_hold_accumulated_returns = [(value - initial_account_value) / initial_account_value * 100 for value in sell_and_hold_account_values]


    # Calculate the daily accumulated returns for the account values
    accumulated_returns = [(value - val_env.account_value_memory[0]) / val_env.account_value_memory[0] * 100 for value in val_env.account_value_memory]    


    # Adjust accumulated_returns to reflect 100% investment
    # This is under the assumption that the returns would be 10 times higher if 100% of the account was invested
    adjusted_accumulated_returns = [accumulated_return * 10 for accumulated_return in accumulated_returns]

    # Plot the Accumulated Returns for each strategy
    import matplotlib.pyplot as plt
    plt.plot(buy_and_hold_accumulated_returns, label="Buy and Hold")
    plt.plot(sell_and_hold_accumulated_returns, label="Sell and Hold")
    plt.plot(adjusted_accumulated_returns, label="TACR")
    plt.title("Accumulated Returns (Test Dataset)")
    plt.xlabel("Trading Days")
    plt.ylabel("Accumulated Return (%)")
    plt.grid(True)
    plt.legend()
    plt.show()


    # Get all the shares owned from the environment
    trades = val_env.shares_held_memory
    # Plot the number of shares owned for each day
    plt.plot(trades)
    plt.title("Shares Held (Test Dataset)")
    plt.xlabel("Trading Days")
    plt.ylabel("Shares Held")
    plt.grid(True)
    plt.show()

    # Make a histogram of trade returns
    trade_returns = val_env.trades_returns_memory
    # Convert it to a numpy array
    trade_returns = np.array(trade_returns)
    # Reshape it to a 1D array
    trade_returns = trade_returns.reshape(-1)

    # Scale it to 100% of the account value since its 10% of the account value per trade
    trade_returns = trade_returns * 10

    # Convert to percentage
    trade_returns = trade_returns * 100

    plt.hist(trade_returns, bins=50)
    plt.title("Trade Returns (Test Dataset)")
    plt.xlabel("Trade Return (%)")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.show()

    print(f"Average trade return: {np.mean(trade_returns)}")


    
