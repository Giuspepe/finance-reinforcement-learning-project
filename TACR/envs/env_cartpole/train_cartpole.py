import os
import sys

# Get parent of parent of parent directory
parent_of_parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(parent_of_parent_dir)

import gymnasium as gym
import numpy as np
from TACR.config.config import TACRConfig

from TACR.tacr import TACR
from TACR.trainer.trainer import Trainer
from TACR.envs.env_cartpole.eval_cartpole import evaluate

import multiprocessing

from traj_cartpole_generator import (
    generate_heuristic_trajectories,
    generate_biased_trajectories,
    generate_random_trajectories,
    generate_oscillating_trajectories,
)

if __name__ == "__main__":
    NUM_STEPS_PER_ITERATION = 500
    MAX_ITERATIONS = 1000
    GENERATE_NEW_TRAJECTORIES = False
    NUMBER_OF_NEW_TRAJECTORIES = 50000
    VALIDATION_STEPS = 500
    WARMUP_ITERATIONS = 200
    PATIENCE_VALIDATION = 10
    VALIDATION_PERIOD = 10

    env = gym.make("CartPole-v1", render_mode="human")

    action_dim = env.action_space.n

    # Generation Mode (won't train)
    if GENERATE_NEW_TRAJECTORIES:
        # Generate new trajectories, one on each processor core
        processes = []
        processes.append(
            multiprocessing.Process(
                target=generate_random_trajectories,
                args=(env, NUMBER_OF_NEW_TRAJECTORIES),
            )
        )
        processes.append(
            multiprocessing.Process(
                target=generate_biased_trajectories,
                args=(env, NUMBER_OF_NEW_TRAJECTORIES),
            )
        )
        processes.append(
            multiprocessing.Process(
                target=generate_heuristic_trajectories,
                args=(env, NUMBER_OF_NEW_TRAJECTORIES),
            )
        )
        processes.append(
            multiprocessing.Process(
                target=generate_oscillating_trajectories,
                args=(env, NUMBER_OF_NEW_TRAJECTORIES),
            )
        )

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    else:
        # Load trajectories from all the files
        import pickle

        # Random trajectories
        with open("tacr_experiment_data/trajs_random.pkl", "rb") as f:
            trajectories_random = pickle.load(f)

        # Biased random trajectories
        with open("tacr_experiment_data/trajs_biased_random.pkl", "rb") as f:
            trajectories_biased = pickle.load(f)

        # Heuristic trajectories
        with open("tacr_experiment_data/trajs_heuristic.pkl", "rb") as f:
            trajectories_heuristic = pickle.load(f)

        # Oscillating trajectories
        with open("tacr_experiment_data/trajs_oscillating.pkl", "rb") as f:
            trajectories_oscillating = pickle.load(f)

        # Concatenate all trajectories, considering that they are lists
        trajectories = (
            trajectories_random
            + trajectories_biased
            + trajectories_heuristic
            + trajectories_oscillating
        )

        # Create a new trajectories where only games where reward > 50 are kept
        train_trajectories = [traj for traj in trajectories if traj.rewards.sum() > 50]

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
            state_dim=env.observation_space.shape[0],
            action_dim=action_dim,
            train_traj_lenghts=train_traj_lens,
            train_traj_returns=train_returns,
            train_traj_states=train_states,
            train_trajectories=train_trajectories,
            action_softmax=True,
            alpha=0.9,
            critic_lr=1e-4,
            actor_lr=1e-4,
            gamma=0.99,
            state_mean=train_state_mean,
            state_std=train_state_std,
        )
        agent = TACR(config=tacr_config)

        trainer = Trainer(agent, env)

        avg_train_actor_loss_per_iteration = []
        best_val_metric = float("inf")
        patience_counter = 0
        for iter in range(MAX_ITERATIONS):
            # Training step
            metrics = trainer.train_it(num_steps=NUM_STEPS_PER_ITERATION)
            train_actor_losses = metrics["actor_loss"]
            train_avg_actor_loss = np.mean(train_actor_losses)
            avg_train_actor_loss_per_iteration.append(train_avg_actor_loss)

            # Every 25 iterations, evaluate the agent on the environment directly, and compute the average reward
            if iter % VALIDATION_PERIOD == 0 and iter > 0:
                trainer.agent.save_actor("saved_models_tacr", "actor_last")
                trainer.agent.save_critic("saved_models_tacr", "critic_last")
                trainer.agent.save_config("saved_models_tacr", "config_last")
                episode_reward_vector, avg_reward = evaluate(
                    "saved_models_tacr",
                    "actor_last",
                    "config_last",
                    num_episodes=50,
                    max_timesteps_per_episode=500,
                    silence=True,
                )
                trainer.tb.log_scalar("avg_reward", avg_reward, iter)
                print(
                    f"Iteration {iter+1}/{MAX_ITERATIONS}, Average Actor Train Loss: {train_avg_actor_loss}, Average Reward on Eval: {avg_reward}"
                )

                
                # Check for improvement
                if avg_reward > best_val_metric:
                    best_val_metric = avg_reward
                    patience_counter = 0

                    # Save model
                    trainer.agent.save_actor("saved_models_tacr", "actor_best")
                    trainer.agent.save_critic("saved_models_tacr", "critic_best")
                    trainer.agent.save_config("saved_models_tacr", "config_best")
                else:
                    if iter > WARMUP_ITERATIONS:
                        patience_counter += 1

                if iter > WARMUP_ITERATIONS:
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
