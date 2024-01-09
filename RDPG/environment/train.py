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

    The agent is expected to have the following methods:
    - get_action(observation): Returns an action given an observation from the environment.
    - update(batch): Updates the agent's networks using a batch of experience.
    - reset_hidden(): Resets any internal state the agent may have (if applicable).
    """
    agent_clone = deepcopy(agent)

    timestep = 0
    episode = 0
    while timestep < max_timesteps:
        print(f'Episode {episode}, timestep {timestep}/{max_timesteps}')
        # Reset environment and agent when new episode
        obs = env.reset()[0] # [0] to get rid of info dict
        agent.copy_network(agent_clone)
        agent.reset_hidden()
        done = False
        cutoff = False
        while not done and timestep < max_timesteps:
            print(timestep)            
            # Select action randomly until update_after timesteps
            if timestep >= update_after:
                action = agent.get_action(obs, deterministic=False)
            else:
                action = env.action_space.sample()
                
            next_obs, reward, done, truncated, info = env.step(action)

            if replay_buffer.episode_length[replay_buffer.episode_pointer] == env.spec.max_episode_steps:
                cutoff = truncated
                done = False if cutoff else True
            else:
                cutoff = False

            # Store experience in replay buffer
            replay_buffer.push(obs, action, reward, next_obs, done, cutoff)
            
            # Perform learning step
            if timestep > update_after:
                batch = replay_buffer.sample()
                agent_clone.update(batch)

            obs = next_obs
            timestep += 1
        episode +=1

