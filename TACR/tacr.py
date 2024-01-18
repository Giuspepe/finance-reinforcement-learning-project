import os
import sys

# Get parent directory
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

from TACR.utils.utils import Batch, get_device, get_target_network, load_model, polyak_update, save_model
from models.actor import Actor
from models.critic import Critic
from TACR.config.config import TACRConfig
import torch.optim as optim
import torch
import torch.nn.functional as F


# Transformer Actor-Critic Reinforcement Learning (TACR)
class TACR:
    def __init__(self, config: TACRConfig, load_file_config=False, file_config_path=None, file_config_name=None):
        """
        Initialize the Transformer Actor-Critic with Regularization (TACR) class.

        Parameters:
        - config (TACRConfig): Configuration object containing parameters for TACR.
        - load_file_config (bool, optional): Flag to indicate whether to load config from a file. Defaults to False.
        - file_config_path (str, optional): Path to the configuration file. Required if load_file_config is True.
        - file_config_name (str, optional): Name of the configuration file. Required if load_file_config is True.

        The method initializes the actor and critic networks, along with their target counterparts. It also sets up the
        respective optimizers for these networks. The configuration can be directly passed as an object or loaded from a file.
        """
        # Initialize the TACR class with configuration settings
        # Load configuration from a file if specified, otherwise use the provided config
        if load_file_config:
            self.load_config(file_config_path, file_config_name)
        else:
            self.config = config

        # Determine and set the computation device (CPU or GPU)
        self.device = get_device()

        # Initialize the actor network with specified architecture and parameters
        self.actor = Actor(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_size=self.config.embed_size,
            max_length=self.config.lookback,
            max_episode_length=self.config.max_episode_length,
            action_softmax=self.config.action_softmax,
            action_softmax_dim=self.config.action_softmax_dim
        ).to(self.device)
        # Create a target actor network for stable learning
        self.actor_target = get_target_network(self.actor)

        # Initialize the critic network
        self.critic = Critic(
            state_dim=self.config.state_dim, action_dim=self.config.action_dim
        ).to(self.device)
        # Create a target critic network for stable learning
        self.critic_target = get_target_network(self.critic)

        # Define the optimizer for the actor network
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )

        # Define the optimizer for the critic network
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )

    # Function to update the actor and critic networks based on a given batch of data
    def update(self, batch: Batch):
        """
        Update the actor and critic networks based on a given batch of data.

        Parameters:
        - batch (Batch): A batch of data containing states, actions, rewards, etc.

        Returns:
        - dict: A dictionary containing various metrics like critic loss, actor loss, and average critic estimates.

        This method processes a batch of data to update both the actor and critic networks. It involves forward passes through
        both networks, calculation of loss, and backpropagation. The method also updates the target networks using Polyak averaging.
        The return value is a dictionary of metrics useful for monitoring the training process.
        """
        # Prepare the data from the batch and move it to the computation device
        states = batch.states.clone().detach().to(self.device)
        next_state = batch.next_state.clone().detach().to(self.device)
        actions = batch.actions.clone().detach().to(self.device)
        timesteps = batch.timesteps.clone().detach().to(self.device)
        next_timesteps = batch.next_timesteps.clone().detach().to(self.device)
        rewards = batch.rewards.clone().detach().to(self.device)
        next_rewards = batch.next_rewards.clone().detach().to(self.device)
        dones = batch.dones.clone().detach().to(self.device)
        next_actions = batch.next_actions.clone().detach().to(self.device)
        attention_mask = batch.attention_mask.clone().detach().to(self.device)

        # Predict actions for the current and next states using the actor and its target network
        action_preds = self.actor.forward(
            states, actions, rewards, timesteps, attention_mask=attention_mask,
        )

        next_action_preds = self.actor_target.forward(
            next_state, next_actions, next_rewards, next_timesteps, attention_mask=attention_mask,
        )

        # Reshape and filter the data based on the attention mask
        states = states.reshape(-1, self.config.state_dim)[attention_mask.reshape(-1) > 0]
        next_state = next_state.reshape(-1, self.config.state_dim)[attention_mask.reshape(-1) > 0]
        rewards = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        action_sample = actions.reshape(-1, self.config.action_dim)[attention_mask.reshape(-1) > 0]
        Q_action_preds = action_preds.reshape(-1, self.config.action_dim)[attention_mask.reshape(-1) > 0]
        next_Q_action_preds = next_action_preds.reshape(-1, self.config.action_dim)[attention_mask.reshape(-1) > 0]
        dones = dones.reshape(-1, 1)[attention_mask.reshape(-1) > 0]

        # Compute the target Q value using the critic's target network
        target_Q = self.critic_target(next_state, next_Q_action_preds)
        target_Q = rewards + ((1-dones) * self.config.gamma * target_Q).detach()

        # Compute current Q estimates using the critic network
        current_Q = self.critic(states, action_sample)
        # Compute the loss for the critic using MSE
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Perform backpropagation and update the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute the actor loss with regularization (lambda)
        pi = Q_action_preds
        Q = self.critic(states, pi)
        lmbda = self.config.alpha / Q.abs().mean().detach()
        actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action_sample)

        # Perform backpropagation and update the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Clip the gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        # Update the target networks using Polyak averaging
        # This ensures that the target networks are slowly updated
        polyak_update(self.actor, self.actor_target, self.config.tau)

        polyak_update(self.critic, self.critic_target, self.config.tau)

        # Collect and return metrics for monitoring
        metrics = {
            "critic_loss": critic_loss.detach().cpu().item(),
            "actor_loss": actor_loss.detach().cpu().item(),
            "mean_critic_predictions": Q.mean().detach().cpu().item(),
            "average_critic_estimate": current_Q.mean().detach().cpu().item(),
        }
        return metrics

    def save_actor(self, path, name="actor"):
        """
        Saves the actor model to the specified path.

        Args:
            path (str): The directory path where the model should be saved.
            name (str, optional): The base name for the saved model files. Defaults to "actor".
        """
        save_model(self.actor, path, name)

    def load_actor(self, path, name="actor"):
        """
        Loads the actor model from the specified path.

        Args:
            path (str): The directory path from where the model should be loaded.
            name (str, optional): The base name of the model files to be loaded. Defaults to "actor".
        """
        self.actor = load_model(self.actor, path, name)
        self.actor_target = get_target_network(self.actor)

    def save_critic(self, path, name="critic"):
        """
        Saves the critic model to the specified path.

        Args:
            path (str): The directory path where the model should be saved.
            name (str, optional): The base name for the saved model files. Defaults to "critic".
        """
        save_model(self.critic, path, name)

    def load_critic(self, path, name="critic"):
        """
        Loads the critic model from the specified path.

        Args:
            path (str): The directory path from where the model should be loaded.
            name (str, optional): The base name of the model files to be loaded. Defaults to "critic".
        """
        self.critic = load_model(self.critic, path, name)
        self.critic_target = get_target_network(self.critic)

    def save_config(self, path, name="config"):
        """
        Saves the TACRConfig to the specified path.

        Args:
            path (str): The directory path where the config should be saved.
            name (str, optional): The base name for the saved config file. Defaults to "config".
        """
        import pickle

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, f"{name}.pkl"), "wb") as f:
            pickle.dump(self.config, f)

    def load_config(self, path, name="config"):
        """
        Loads the TACRConfig from the specified path.

        Args:
            path (str): The directory path from where the config should be loaded.
            name (str, optional): The base name of the config file to be loaded. Defaults to "config".
        """
        import pickle

        with open(os.path.join(path, f"{name}.pkl"), "rb") as f:
            self.config = pickle.load(f)

    