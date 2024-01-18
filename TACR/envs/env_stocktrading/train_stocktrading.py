import os
import sys



# Get parent of parent of parent directory
parent_of_parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(parent_of_parent_dir)

import gymnasium as gym
import numpy as np
import pandas as pd
from TACR.config.config import TACRConfig

from TACR.tacr import TACR
from TACR.trainer.trainer import Trainer
from TACR.trajectory.trajectory import TrajectoryGenerator
from TACR.envs.env_stocktrading.eval_stocktrading import evaluate
from TACR.envs.env_stocktrading.utils import augment_data, create_environment, download_and_clean_data, save_data
from TACR.envs.env_stocktrading.traj_stocktrading_generator import generate_best_trade_on_window_trajectories

from preprocessing.custom_technical_indicators import RVI_PCT_CHANGE, OBV_PCT_CHANGE, PCT_RETURN, ADX, RSI_CATEGORICAL, BINARY_SMA_RISING
from preprocessing.process_yh_finance import YHFinanceProcessor

    
if __name__ == "__main__":
    NUM_STEPS_PER_ITERATION = 1000
    MAX_ITERATIONS = 3000
    GENERATE_NEW_TRAJECTORIES = False
    WARMUP_ITERATIONS = 750
    PATIENCE_VALIDATION = 50
    VALIDATION_PERIOD = 5
    TRAIN_START_DATE = "2010-01-01"
    TRAIN_END_DATE = "2020-06-01"
    VAL_START_DATE = "2020-06-01"
    VAL_END_DATE = "2024-01-01"
    TICKERS = ["OXY"]
    INDICATORS = []
    CUSTOM_INDICATORS = [
        PCT_RETURN(length=2), OBV_PCT_CHANGE(length=8), RVI_PCT_CHANGE(length=20, rvi_pct_change_length=2), ADX(length=16), RSI_CATEGORICAL(length=24), BINARY_SMA_RISING(length=24)
    ]
    DOWNLOAD_DATA = False

    yfp = YHFinanceProcessor()
    if DOWNLOAD_DATA:
        train_df = download_and_clean_data(yfp, TICKERS, TRAIN_START_DATE, TRAIN_END_DATE)
        train_df_aug = augment_data(yfp, train_df, INDICATORS, CUSTOM_INDICATORS)
        save_data(train_df_aug, "train_stock_data.csv")
        val_df = download_and_clean_data(yfp, TICKERS, VAL_START_DATE, VAL_END_DATE)
        val_df_aug = augment_data(yfp, val_df, INDICATORS, CUSTOM_INDICATORS)
        save_data(val_df_aug, "val_stock_data.csv")

    train_dataset = pd.read_csv("train_stock_data.csv")
    val_dataset = pd.read_csv("val_stock_data.csv")
    train_env = create_environment(yfp, train_dataset, INDICATORS, CUSTOM_INDICATORS)
    val_env = create_environment(yfp, val_dataset, INDICATORS, CUSTOM_INDICATORS)

    action_dim = train_env.action_space.n

    if GENERATE_NEW_TRAJECTORIES:
        generate_best_trade_on_window_trajectories(env=train_env)
        print("Done generating trajectories")
    else:
        # Load trajectories from all the files
        import pickle

        # Best trade on window trajectories
        with open("tacr_experiment_data/trajs_best_trade_on_window.pkl", "rb") as f:
            trajectories_best_trade_on_window = pickle.load(f)


        # Concatenate all trajectories, considering that they are lists
        trajectories = (
            trajectories_best_trade_on_window
        )

        train_trajectories = trajectories

        train_states, train_traj_lens, train_returns = [], [], []
        for trajectory in train_trajectories:
            train_states.append(trajectory.observations)
            train_traj_lens.append(len(trajectory.observations))
            train_returns.append(np.array(trajectory.rewards).sum())
        train_traj_lens, train_returns = np.array(train_traj_lens), np.array(
            train_returns
        )

        # # Plot distribution of returns
        # import matplotlib.pyplot as plt

        # plt.hist(returns, bins=50)
        # plt.show()

        # used for input normalization
        train_states = np.concatenate(train_states, axis=0)
        train_state_mean, train_state_std = (
            np.mean(train_states, axis=0),
            np.std(train_states, axis=0) + 1e-6,
        )
        num_timesteps = sum(train_traj_lens)

        tacr_config = TACRConfig(
            state_dim=train_env.observation_space.shape[0],
            action_dim=action_dim,
            train_traj_lenghts=train_traj_lens,
            train_traj_returns=train_returns,
            train_traj_states=train_states,
            train_trajectories=train_trajectories,
            action_softmax=True,
            alpha=0.5,
            critic_lr=1e-5,
            actor_lr=1e-4,
            gamma=0.99,
            state_mean=train_state_mean,
            state_std=train_state_std,
            batch_size=32,
        )
        agent = TACR(config=tacr_config)

        trainer = Trainer(agent, train_env)

        avg_train_actor_loss_per_iteration = []
        best_val_metric = float("inf")
        patience_counter = 0
        for iter in range(MAX_ITERATIONS):
            # Training step
            metrics = trainer.train_it(num_steps=NUM_STEPS_PER_ITERATION)
            train_actor_losses = metrics["actor_loss"]
            train_avg_actor_loss = np.mean(train_actor_losses)
            avg_train_actor_loss_per_iteration.append(train_avg_actor_loss)

            # Every VALIDATION_PERIOD iterations, evaluate the agent on the environment directly, and compute the average reward
            if iter % VALIDATION_PERIOD == 0 and iter > 0:
                trainer.agent.save_actor("saved_models_tacr", "actor_last")
                trainer.agent.save_critic("saved_models_tacr", "critic_last")
                trainer.agent.save_config("saved_models_tacr", "config_last")
                episode_reward_vector, avg_reward = evaluate(
                    val_env,
                    "saved_models_tacr",
                    "actor_last",
                    "config_last",
                    num_episodes=1, # We can only do 1 episode because its a fixed dataset
                    max_timesteps_per_episode=100000,
                    silence=True,
                )
                trainer.tb.log_scalar("avg_reward", avg_reward, iter)
                print(
                    f"Iteration {iter+1}/{MAX_ITERATIONS}, Average Actor Train Loss: {train_avg_actor_loss}, Average Reward on Eval: {avg_reward}"
                )

                
                # Check for improvement
                if iter > WARMUP_ITERATIONS:
                    if avg_reward > best_val_metric:
                        best_val_metric = avg_reward
                        patience_counter = 0

                        # Save model
                        trainer.agent.save_actor("saved_models_tacr", "actor_best")
                        trainer.agent.save_critic("saved_models_tacr", "critic_best")
                        trainer.agent.save_config("saved_models_tacr", "config_best")
                    else:
                        patience_counter += 1

                    if patience_counter >= PATIENCE_VALIDATION:
                        print("Early stopping triggered")
                        break

            else:
                print(
                    f"Iteration {iter+1}/{MAX_ITERATIONS}, Average Actor Train Loss: {train_avg_actor_loss}"
                )

        trainer.finish_training()
        trainer.agent.save_actor("saved_models_tacr", "actor_last")
        trainer.agent.save_critic("saved_models_tacr", "critic_last")
        trainer.agent.save_config("saved_models_tacr", "config_last")
