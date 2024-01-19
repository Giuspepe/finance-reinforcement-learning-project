import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(parent_dir)

import pandas as pd
from stock_trading_base import StockTradingBase
from preprocessing.process_yh_finance import YHFinanceProcessor
from RDPG.rdpg import RDPG
from RDPG.buffer import ReplayBuffer
from RDPG.trainer.train import train


def download_and_clean_data(yfp, tickers, start_date, end_date):
    """
    Downloads and cleans the stock data.
    """
    df_raw = yfp.download_data(tickers, start_date, end_date, "1d")
    return yfp.clean_data(df_raw)


def augment_data(yfp, df, indicators):
    """
    Adds technical indicators and turbulence index to the data.
    """
    df = yfp.add_technical_indicators(df, indicators)
    df = yfp.add_turbulence(df, 252)
    df = yfp.add_vix(df)

    # Drop rows with NaN values
    df = df.dropna().reset_index(drop=True)
    
    return df


def save_data(df, filename):
    """
    Saves the DataFrame to a CSV file.
    """
    df.to_csv(filename)


def create_environment(yfp, dataset, indicators):
    """
    Creates and returns the trading environment instance.
    """
    stock_dimension = len(dataset.tic.unique())
    state_space = 1 + 2 * stock_dimension + (len(indicators) + 1) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    close_array, open_array, high_array, low_array, tech_array, turbulence_array = yfp.df_to_array(dataset, indicators, [], True)
    env_config = {
        "price_array": close_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "is_training_mode": True,
    }
    return StockTradingBase(config=env_config)


def train_model(env: StockTradingBase):
    """
    Trains the model.
    """
    rdpg_agent = RDPG(
        input_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        upper_normalization_bounds=3000,
        lower_normalization_bounds=-3000,
    )
    batch_size = 64

    replay_buffer = ReplayBuffer(
        observation_dim=env.state_dim,
        action_dim=env.action_dim,
        max_episode_length=env.max_episode_steps,
        capacity=10_000,
        batch_size=batch_size,
    )
    max_timesteps = 150_000
    train(
        rdpg_agent,
        env,
        max_timesteps,
        replay_buffer,
        batch_size=batch_size,
        update_after=40000,
    )


def main():
    """
    Main function to create dataset and environment.
    """
    TRAIN_START_DATE = "2021-01-01"
    TRAIN_END_DATE = "2024-01-01"
    INDICATORS = [
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
    ]
    TICKERS = ["GOOG"]
    DOWNLOAD_DATA = False
    
    yfp = YHFinanceProcessor()
    if DOWNLOAD_DATA:
        df = download_and_clean_data(yfp, TICKERS, TRAIN_START_DATE, TRAIN_END_DATE)
        df_aug = augment_data(yfp, df, INDICATORS)
        save_data(df_aug, "stock_data.csv")

    dataset = pd.read_csv("stock_data.csv")
    env = create_environment(yfp, dataset, INDICATORS)
    train_model(env)


if __name__ == "__main__":
    main()