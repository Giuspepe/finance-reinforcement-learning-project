import numpy as np
import math
import torch
from utils import get_device
from collections import namedtuple

# Global variables
RecurrentBatch = namedtuple('RecurrentBatch', ['observations', 'actions', 'rewards', 'done', 'mask'])

class ReplayBuffer():
    """
    A Replay Buffer class for use in Reinforcement Learning. 

    This buffer stores and manages the experiences (observations, actions, rewards, next observations and done signals) of an agent interacting with
    an environment. It is designed to handle episodes of varying lengths and can be used for sampling these experiences to train the agent.

    Attributes:
        observation_dim (int): The dimensionality of the observation space.
        action_dim (int): The dimensionality of the action space.
        max_episode_length (int): The maximum length of an episode.
        capacity (int): The total number of episodes the buffer can store.
        batch_size (int): The number of episodes to sample at a time.
        segment_length (int, optional): The length of segments to sample, if segmentation is desired.
        
        observations (numpy.ndarray): Stores observations for each timestep of each episode.
        actions (numpy.ndarray): Stores actions taken by the agent at each timestep.
        rewards (numpy.ndarray): Stores rewards received by the agent at each timestep.
        done (numpy.ndarray): Flags indicating if a timestep is the last one in an episode.
        mask (numpy.ndarray): Masks to indicate valid timesteps in episodes.
        episode_length (numpy.ndarray): Stores the length of each episode.
        ready_for_sample (numpy.ndarray): Flags indicating if an episode is complete and ready for sampling.
        episode_pointer (int): Points to the current episode in the buffer.
        time_pointer (int): Points to the current timestep in the current episode.
        new_episode (bool): Indicates if a new episode has started.
        num_episodes (int): Tracks the total number of episodes stored in the buffer.

    Methods:
        push: Adds a new experience to the buffer.
        sample: Samples a batch of experiences from the buffer for training.
    """
    def __init__(self, observation_dim, action_dim, max_episode_length, capacity, batch_size, segment_length = None):
        """
        Initializes the ReplayBuffer object.

        Args:
            observation_dim (int): The dimensionality of the observation space.
            action_dim (int): The dimensionality of the action space.
            max_episode_length (int): The maximum number of timesteps in an episode.
            capacity (int): The maximum number of episodes the buffer can store.
            batch_size (int): The number of episodes to sample in one batch.
            segment_length (int, optional): The length of segments to be sampled within each episode. 
                Defaults to None, indicating that full episodes are sampled.

        This function initializes storage for observations, actions, rewards, done flags, and masks for each episode. 
        It also sets up necessary counters and flags for managing the buffer. The buffer is initialized with zero values 
        and will be filled with actual experience data as the agent interacts with the environment.

        The `segment_length` parameter can be used in algorithms that require segment-wise sampling from episodes, 
        such as in recurrent neural network-based policies.
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.batch_size = batch_size
        self.max_episode_length = max_episode_length
        self.segment_length = segment_length
        
        self.observations = np.zeros((capacity, max_episode_length + 1, observation_dim))
        self.actions = np.zeros((capacity, max_episode_length, action_dim))
        self.rewards = np.zeros((capacity, max_episode_length, 1))
        self.done = np.zeros((capacity, max_episode_length, 1))
        self.mask = np.zeros((capacity, max_episode_length, 1))
        self.episode_length = np.zeros((capacity, ))
        self.ready_for_sample = np.zeros((capacity, ))
        
        self.episode_pointer = 0
        self.time_pointer = 0
        
        self.new_episode = True
        self.num_episodes = 0
        
    def push(self, observation, action, reward, next_observation, done, cutoff):
        """
        Adds a new experience to the buffer.

        Args:
            observation (array-like): The observation from the environment.
            action (array-like): The action taken by the agent.
            reward (float): The reward received after taking the action.
            next_observation (array-like): The next observation after taking the action.
            done (bool): Flag indicating if the episode has ended.
            cutoff (bool): Flag indicating if the episode was cut off (e.g., due to reaching a maximum episode length).

        This method is responsible for adding the provided experience tuple to the replay buffer. 
        If a new episode has begun, it first resets the data for the current episode pointer. 
        It then stores each component of the experience in the respective arrays at the current episode and time pointers.

        If the episode is marked as 'done' or a 'cutoff' condition is met, the method updates the 'ready_for_sample' 
        flag for the current episode to indicate that it's complete and ready for sampling. It also increments the 
        episode pointer and resets the time pointer for the next episode. In addition, the method ensures that the 
        next observation is stored as the first observation of the new episode if applicable.

        This function is a key part of managing the data within the buffer and ensuring that complete and valid episodes 
        are available for training the agent.
        """
        
        # Reset and clear the data for a new episode in the replay buffer.
        if self.new_episode:
            self.observations[self.episode_pointer] = 0
            self.actions[self.episode_pointer] = 0
            self.rewards[self.episode_pointer] = 0
            self.done[self.episode_pointer] = 0
            self.mask[self.episode_pointer] = 0
            self.episode_length[self.episode_pointer] =+ 1
            
            self.new_episode = False
        
        # Push the observation, action, reward, and done to the replay buffer.
        self.observations[self.episode_pointer, self.time_pointer] = observation
        self.actions[self.episode_pointer, self.time_pointer] = action
        self.rewards[self.episode_pointer, self.time_pointer] = reward
        self.done[self.episode_pointer, self.time_pointer] = done
        self.mask[self.episode_pointer, self.time_pointer] = 1
        self.episode_length[self.episode_pointer] += 1
        
        if done or cutoff:
            # If the episode is done, then we are ready to sample from this episode.
            self.ready_for_sample[self.episode_pointer] = 1
            # Reset the episode pointer to the next episode.
            self.episode_pointer = (self.episode_pointer + 1) % self.capacity
            # Reset the time pointer to the start of the episode.
            self.time_pointer = 0
            # Set the observation for the next time step to the next observation. This ensures that the next observation is part of
            # the replay buffer since after executing an action, there would be a loss of information regarding its effects 
            # on the environment
            self.observations[self.episode_pointer, self.time_pointer + 1] = next_observation
            # Set the new episode flag to true.
            self.new_episode = True
            
            # Increment the number of episodes.
            if self.num_episodes < self.capacity:
                self.num_episodes += 1
        else:
            # Otherwise, increment the time pointer.
            self.time_pointer += 1
            
    
    
    def sample(self):
        
        # Get all episodes that are ready for sampling.
        episodes = np.argwhere(self.ready_for_sample == 1).flatten()
        
        # Get the lengths of the episodes.
        episode_lengths = self.episode_length[episodes]
        
        # Get the sampling probabilities for the episodes.
        probabilities = episode_lengths / np.sum(episode_lengths)
        
        # Sample episodes from the replay buffer.
        sampled_episodes = np.random.choice(episodes, size = self.batch_size, p = probabilities)
        
        # Get the length of the sampled episodes.
        sampled_episode_lengths = self.episode_length[sampled_episodes]
        
        if self.segment_length is None:
            # Get the maximum length of the sampled episodes.
            max_sampled_episode_length = int(np.max(sampled_episode_lengths))
            
            # Add 1 to the maximum length of the sampled episodes to account for the final observation.
            observations = self.observations[sampled_episodes][:, :max_sampled_episode_length+1, :]
            actions = self.actions[sampled_episodes][:, :max_sampled_episode_length, :]
            rewards = self.rewards[sampled_episodes][:, :max_sampled_episode_length, :]
            done = self.done[sampled_episodes][:, :max_sampled_episode_length, :]
            mask = self.mask[sampled_episodes][:, :max_sampled_episode_length, :]
            
            # convert the observations, actions, rewards, and done to tensors.
            observations = torch.tensor(observations, dtype = torch.float32).to(get_device())
            actions = torch.tensor(actions, dtype = torch.float32).to(get_device())
            rewards = torch.tensor(rewards, dtype = torch.float32).to(get_device())
            done = torch.tensor(done, dtype = torch.float32).to(get_device())
            mask = torch.tensor(mask, dtype = torch.float32).to(get_device())
            
            # return the observations, actions, rewards, done, and mask.
            return RecurrentBatch(observations, actions, rewards, done, mask)
        
        else:
            
            # Get the number of segments for each episode.
            num_segments = int(math.ceil(sampled_episode_lengths / self.segment_length))
            
            observations = self.observations[sampled_episodes]
            actions = self.actions[sampled_episodes]
            rewards = self.rewards[sampled_episodes]
            done = self.done[sampled_episodes]
            mask = self.mask[sampled_episodes]
            
            # initialize buffers for the segmented episodes.
            segmented_observations = np.zeros((self.batch_size, self.segment_length + 1, self.observation_dim))
            segmented_actions = np.zeros((self.batch_size, self.segment_length, self.action_dim))
            segmented_rewards = np.zeros((self.batch_size, self.segment_length, 1))
            segmented_done = np.zeros((self.batch_size, self.segment_length, 1))
            segmented_mask = np.zeros((self.batch_size, self.segment_length, 1))
            
            # Segment the episodes.
            for i in range(self.batch_size):
                # Get a random start index for the episode to be segmented.
                start_index = np.random.randint(num_segments[i]) * self.segment_length
                segmented_observations[i] =  observations[i][start_index:start_index+self.segment_length+1]
                segmented_actions[i] = actions[i][start_index:start_index+self.segment_length]
                segmented_rewards[i] = rewards[i][start_index:start_index+self.segment_length]
                segmented_done[i] = done[i][start_index:start_index+self.segment_length]
                segmented_mask[i] = mask[i][start_index:start_index+self.segment_length]
                
                # Convert the segmented observations, actions, rewards, and done to tensors.
                segmented_observations = torch.tensor(segmented_observations, dtype = torch.float32).to(get_device())
                segmented_actions = torch.tensor(segmented_actions, dtype = torch.float32).to(get_device())
                segmented_rewards = torch.tensor(segmented_rewards, dtype = torch.float32).to(get_device())
                segmented_done = torch.tensor(segmented_done, dtype = torch.float32).to(get_device())
                segmented_mask = torch.tensor(segmented_mask, dtype = torch.float32).to(get_device())
                
                # Return the segmented observations, actions, rewards, done, and mask.
                return RecurrentBatch(segmented_observations, segmented_actions, segmented_rewards, segmented_done, segmented_mask)