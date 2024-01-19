import os
import sys

# Get parent of parent of parent directory
parent_of_parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(parent_of_parent_dir)

parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_dir)

import gymnasium as gym
import numpy as np
import pandas as pd
from iRDPG.irdpg import IRDPG
from iRDPG.buffer import ReplayBCBuffer
from iRDPG.trainer.train import train

from env_stocktrading.utils import (
    augment_data,
    create_environment,
    download_and_clean_data,
    save_data,
)

from preprocessing.custom_technical_indicators import (
    RVI_PCT_CHANGE,
    OBV_PCT_CHANGE,
    PCT_RETURN,
    ADX,
    RSI_CATEGORICAL,
    BINARY_SMA_RISING,
)
from preprocessing.process_yh_finance import YHFinanceProcessor

# Custom action picker function that decodes one-hot encoded action
def action_softmax_to_value(action: np.array) -> int:
    index_of_max_probability = np.argmax(action)
    decoded_action = index_of_max_probability - 1
    return decoded_action
    

if __name__ == "__main__":
    MAX_TIMESTEPS = 1_000_000
    VALIDATION_PERIOD = 5
    TRAIN_START_DATE = "2004-01-01"
    TRAIN_END_DATE = "2018-01-01"
    VAL_START_DATE = "2018-01-01"
    VAL_END_DATE = "2024-01-01"
    TICKERS = ["OXY"]
    INDICATORS = []
    CUSTOM_INDICATORS = [
        # RSI_CATEGORICAL(length=24),
        PCT_RETURN(length=2),
        # OBV_PCT_CHANGE(length=8),
        # RVI_PCT_CHANGE(length=20, rvi_pct_change_length=2),
        # ADX(length=16),
        BINARY_SMA_RISING(length=24), # Can't be removed since its used as an expert action
    ]
    DOWNLOAD_DATA = False
    DISCOUNT_FACTOR = 0.999
    PROPHETIC_ACTIONS_WINDOW_LENGTH = 10
    BATCH_SIZE = 32
    LR = 3e-4
    LAMBDA_POLICY = 0.7
    ACTION_NOISE_DECAY_STEPS = 800_000


    yfp = YHFinanceProcessor()
    if DOWNLOAD_DATA:
        train_df = download_and_clean_data(
            yfp, TICKERS, TRAIN_START_DATE, TRAIN_END_DATE
        )
        train_df_aug = augment_data(
            yfp, train_df, INDICATORS, CUSTOM_INDICATORS, vix=False
        )
        save_data(train_df_aug, "train_stock_data.csv")
        val_df = download_and_clean_data(yfp, TICKERS, VAL_START_DATE, VAL_END_DATE)
        val_df_aug = augment_data(yfp, val_df, INDICATORS, CUSTOM_INDICATORS, vix=False)
        save_data(val_df_aug, "val_stock_data.csv")

    train_dataset = pd.read_csv("train_stock_data.csv")
    val_dataset = pd.read_csv("val_stock_data.csv")
    train_env = create_environment(
        yfp, train_dataset, INDICATORS, CUSTOM_INDICATORS, gamma=DISCOUNT_FACTOR, prophetic_actions_window_length=PROPHETIC_ACTIONS_WINDOW_LENGTH
    )
    val_env = create_environment(
        yfp, val_dataset, INDICATORS, CUSTOM_INDICATORS, gamma=DISCOUNT_FACTOR, prophetic_actions_window_length=PROPHETIC_ACTIONS_WINDOW_LENGTH
    )

    action_dim = train_env.action_space.n

    irdpg_agent = IRDPG(
        input_dim=train_env.observation_space.shape[0],
        action_dim=action_dim,
        hidden_dim=128,
        lr=LR,
        upper_normalization_bounds=3000,
        lower_normalization_bounds=-3000,
        action_noise_decay_steps=ACTION_NOISE_DECAY_STEPS,
        lambda_policy=LAMBDA_POLICY
    )

    replay_buffer = ReplayBCBuffer(
        observation_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        max_episode_length=train_env.max_episode_steps,
        capacity=10_000,
        batch_size=BATCH_SIZE,
    )
    train(
        irdpg_agent,
        train_env,
        val_env,
        MAX_TIMESTEPS,
        replay_buffer,
        batch_size=BATCH_SIZE,
        validation_period=VALIDATION_PERIOD,
        softmax_conversion_func=action_softmax_to_value
    )