from copy import deepcopy

import numpy as np
from env_stocktrading.env_stocktrading import SimpleOneStockStockTradingBaseEnv
from iRDPG.buffer import ReplayBCBuffer
from iRDPG.irdpg import IRDPG
from iRDPG.trainer.evaluate import evaluate_during_training
from log import TensorBoardHandler


def convert_action_int_to_trade_one_hot(action: int) -> np.array:
    # -1 = [1, 0, 0], 0 = [0, 1, 0], 1 = [0, 0, 1]
    return np.eye(3)[action + 1]


def train(agent: IRDPG, train_env: SimpleOneStockStockTradingBaseEnv, val_env: SimpleOneStockStockTradingBaseEnv, max_timesteps: int, replay_buffer: ReplayBCBuffer, batch_size: int, validation_period=0, softmax_conversion_func: callable = np.argmax):
    """
    Trains a given agent in a specified train_environment for a maximum number of timesteps,
    using a separate replay buffer for experience storage.

    Args:
        agent: An agent with methods for action selection, updating, and state reset.
        train_env (gym.Train_env): The Gym train_environment to train in.
        val_env (gym.Train_env): The Gym train_environment to validate in.
        max_timesteps (int): The maximum number of timesteps to train for.
        replay_buffer: A replay buffer instance for storing experiences.
        batch_size (int): The size of the batch to sample
        validation_period (int): The number of iterations between validation steps
    """
    # Initialize TensorBoardHandler
    tb_handler = TensorBoardHandler(log_dir="runs/experiment_01")
    timestep = 0
    episode = 0
    total_reward = 0  # Added for logging

    best_reward = -float('inf')  # Initialize best reward as negative infinity
    best_episode = 0

    started_training = False

    max_ep_steps = 0
    if hasattr(train_env, "spec"):
        if hasattr(train_env.spec, "max_episode_steps"):
            max_ep_steps = train_env.spec.max_episode_steps
    if hasattr(train_env, "max_episode_steps"):
        max_ep_steps = train_env.max_episode_steps

    if max_ep_steps == 0:
        raise ValueError(
            "Train_environment must have a 'train_env.spec.max_episode_steps' or 'train_env.max_episode_steps' attribute"
        )

    while timestep < max_timesteps:
        obs = train_env.reset()  # Reset train_environment at the start of each episode
        obs = obs[0]
        agent.reset_hidden()  # Reset agent's internal state
        done = False
        cutoff = False
        episode_reward = 0  # Added for logging
        
        done_or_cutoff = done or cutoff
        while not done_or_cutoff and timestep < max_timesteps:
            if timestep % 100 == 0:
                print(f"Timestep {timestep}/{max_timesteps}")

            # Select action: Fill up with expert demonstrations and not started training, otherwise use agent's action
            action = (
                train_env.expert_action(train_env.day)
                if not started_training
                else softmax_conversion_func(agent.get_action(obs, deterministic=False))
            )
            
            # Prophetic Action
            action_bc = train_env.get_prophetic_action(train_env.day)

            next_obs, reward, done, truncated, info = train_env.step(action)

            episode_reward += reward  # Added for logging

            
            if (
                replay_buffer.episode_length[replay_buffer.episode_pointer]
                == max_ep_steps
            ):
                cutoff = truncated
                done = False if cutoff else True
            else:
                cutoff = False

            # Convert action to one-hot encoding
            action_encoded = convert_action_int_to_trade_one_hot(action)
            action_bc_encoded = convert_action_int_to_trade_one_hot(action_bc)
            # Store experience in replay buffer
            replay_buffer.push(obs, action_encoded, action_bc_encoded, reward, next_obs, done, cutoff)

            # Update timestep and observation
            timestep += 1
            obs = next_obs

            # Perform learning step if enough timesteps have elapsed or batch is full
            if replay_buffer.num_episodes >= batch_size:
                if not started_training:
                    print("Starting training...")
                    started_training = True
                batch = replay_buffer.sample()
                agent.action_noise_decay_steps = max_timesteps - timestep
                metrics = agent.update(batch)
                tb_handler.log_scalar("Critic Loss", metrics["critic_loss"], timestep)
                tb_handler.log_scalar("Actor Loss", metrics["actor_loss"], timestep)
                tb_handler.log_scalar(
                    "Mean Critic Predictions (Predictions)",
                    metrics["mean_critic_predictions"],
                    timestep,
                )
                tb_handler.log_scalar(
                    "Average Critic Estimate (Q-values)",
                    metrics["average_critic_estimate"],
                    timestep,
                )
                tb_handler.log_scalar(
                    "BC Loss", 
                    metrics["bc_loss"], 
                    timestep,
                )
                tb_handler.log_scalar(
                    "Q filter", 
                    metrics["q_filter"], 
                    timestep,
                )

            done_or_cutoff = done or cutoff
                    
        if started_training:
            total_reward += episode_reward
            tb_handler.log_scalar(
                "(Train) Reward/episode", episode_reward, episode
            )  # Log reward per episode

            # Every validation_period episodes, evaluate the agent on the validation environment
            if episode % validation_period == 0 and episode > 0:
                agent.save_actor("saved_models_irdpg", "actor_last")
                agent.save_critic("saved_models_irdpg", "critic_last")
                # Get average reward over 10 episodes
                episode_reward_vector, avg_reward = evaluate_during_training(
                    agent,
                    val_env,
                    num_episodes=1,
                    max_timesteps_per_episode=100000,
                )
                tb_handler.log_scalar(
                    "(Val) Average Reward/episode", avg_reward, episode
                )

                # Check if this episode's reward is the best and save the model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_episode = episode
                    agent.save_actor("saved_models_irdpg", "actor_best")
                    agent.save_critic("saved_models_irdpg", "critic_best")
                    
                    print(f"New best model saved with reward: {best_reward} at episode {best_episode}")

        episode += 1
        print(f"Episode {episode} completed at timestep {timestep}/{max_timesteps}")

    # Last validation step
    agent.save_actor("saved_models_irdpg", "actor_last")
    agent.save_critic("saved_models_irdpg", "critic_last")
    episode_reward_vector, avg_reward = evaluate_during_training(
        agent,
        val_env,
        "saved_models_irdpg",
        name="actor_last",
        num_episodes=1,
        max_timesteps_per_episode=100000,
        silence=True,
    )
    tb_handler.log_scalar(
        "(Val) Average Reward/episode", avg_reward, episode
    )

    # Check if this episode's reward is the best and save the model
    if avg_reward > best_reward:
        best_reward = avg_reward
        best_episode = episode
        agent.save_actor("saved_models_irdpg", "actor_best")
        agent.save_critic("saved_models_irdpg", "critic_best")
        
        print(f"New best model saved with reward: {best_reward} at episode {best_episode}")    
    tb_handler.close()  # Close the TensorBoard handler
    print("Training completed.")