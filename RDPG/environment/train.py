from copy import deepcopy

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
    timestep = 0
    episode = 0

    while timestep < max_timesteps:
        obs = env.reset()  # Reset environment at the start of each episode
        obs = obs[0]
        agent.reset_hidden()  # Reset agent's internal state
        done = False
        cutoff = False

        while not done and timestep < max_timesteps:
            
            if timestep % 100 == 0:
                print(f'Timestep {timestep}/{max_timesteps}')
            
            # Select action: Randomly if before update_after, otherwise use agent's action
            action = env.action_space.sample() if timestep < update_after else agent.get_action(obs, deterministic=False)
            
            
            next_obs, reward, done, truncated, info = env.step(action)

            if replay_buffer.episode_length[replay_buffer.episode_pointer] == env.spec.max_episode_steps:
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
                print(f'Updating agent with batch of size {batch_size}')
                agent.update(batch)

        episode += 1
        print(f'Episode {episode} completed at timestep {timestep}/{max_timesteps}')
    
    print('Training completed.')