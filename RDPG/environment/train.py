from copy import deepcopy
from log import TensorBoardHandler


def train(agent, env, max_timesteps, replay_buffer, batch_size, update_after=0):
    """
    Trains a given agent in a specified environment for a maximum number of timesteps,
    using a separate replay buffer for experience storage.

    Args:
        agent: An agent with methods for action selection, updating, and state reset.
        env (gym.Env): The Gym environment to train in.
        max_timesteps (int): The maximum number of timesteps to train for.
        replay_buffer: A replay buffer instance for storing experiences.
        batch_size (int): The size of the batch to sample from the replay buffer for training.
    """
    # Initialize TensorBoardHandler
    tb_handler = TensorBoardHandler(log_dir="runs/experiment_01")
    timestep = 0
    episode = 0
    total_reward = 0  # Added for logging

    best_reward = -float('inf')  # Initialize best reward as negative infinity
    best_episode = 0

    while timestep < max_timesteps:
        obs = env.reset()  # Reset environment at the start of each episode
        obs = obs[0]
        agent.reset_hidden()  # Reset agent's internal state
        done = False
        cutoff = False
        episode_reward = 0  # Added for logging

        while not done and timestep < max_timesteps:
            if timestep % 100 == 0:
                print(f"Timestep {timestep}/{max_timesteps}")

            # Select action: Randomly if before update_after, otherwise use agent's action
            action = (
                env.action_space.sample()
                if timestep < update_after
                else agent.get_action(obs, deterministic=False)
            )

            next_obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward  # Added for logging

            if (
                replay_buffer.episode_length[replay_buffer.episode_pointer]
                == env.spec.max_episode_steps
            ):
                cutoff = truncated
                done = False if cutoff else True
            else:
                cutoff = False

            # Store experience in replay buffer
            replay_buffer.push(obs, action, reward, next_obs, done, cutoff)

            # Update timestep and observation
            timestep += 1
            obs = next_obs

            # Perform learning step if enough timesteps have elapsed
            if timestep >= update_after and replay_buffer.num_episodes >= batch_size:
                batch = replay_buffer.sample()
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
                    "Actor RNN Grad Norm", 
                    metrics["actor_rh_grad_norm"], 
                    timestep,
                )
                tb_handler.log_scalar(
                    "Critic RNN Grad Norm",
                    metrics["critic_rh_grad_norm"],
                    timestep,
                )
                    

        total_reward += episode_reward
        tb_handler.log_scalar(
            "Reward/episode", episode_reward, episode
        )  # Log reward per episode

        # Check if this episode's reward is the best and save the model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_episode = episode
            agent.save_actor("saved_models", "actor")
            agent.save_critic("saved_models", "critic")
            print(f"New best model saved with reward: {best_reward} at episode {best_episode}")

        episode += 1
        print(f"Episode {episode} completed at timestep {timestep}/{max_timesteps}")

    tb_handler.log_scalar("Total Reward", total_reward, episode)  # Log total reward
    tb_handler.close()  # Close the TensorBoard handler
    print("Training completed.")