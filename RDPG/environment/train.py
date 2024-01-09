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

    The agent is expected to have the following methods:
    - get_action(observation): Returns an action given an observation from the environment.
    - update(batch): Updates the agent's networks using a batch of experience.
    - reset_hidden(): Resets any internal state the agent may have (if applicable).
    """
    timestep = 0
    while timestep < max_timesteps:
        # Reset environment and agent when new episode
        obs = env.reset()
        agent.reset_hidden()
        done = False
        while not done and timestep < max_timesteps:
            
            # Select action randomly until update_after timesteps
            if timestep >= update_after:
                action = agent.get_action(obs)
            else:
                action = env.action_space.sample()
                
            next_obs, reward, done, _ = env.step(action)

            # Store experience in replay buffer
            replay_buffer.push(obs, action, reward, next_obs, done)

            # Perform learning step
            if replay_buffer.episode_length.sum() > batch_size:
                batch = replay_buffer.sample()
                agent.update(batch)

            obs = next_obs
            timestep += 1

