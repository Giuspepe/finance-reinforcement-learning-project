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


class TACR:
    def __init__(self, config: TACRConfig):
        # Configuration for TACR
        self.config = config

        self.device = get_device()

        # Actor Networks
        self.actor = Actor(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_size=self.config.embed_size,
            max_length=self.config.lookback,
            max_episode_length=self.config.max_episode_length,
            action_softmax=self.config.action_softmax,
        ).to(self.device)
        self.actor_target = get_target_network(self.actor)

        # Critic Networks
        self.critic = Critic(
            state_dim=self.config.state_dim, action_dim=self.config.action_dim
        ).to(self.device)
        self.critic_target = get_target_network(self.critic)

        # Actor optimizer (local and target)
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config.actor_lr
        )

        # Critic optimizer (local and target)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config.critic_lr
        )

    # From a random minibatch, update the models
    def update(self, batch: Batch):
        states = torch.tensor(batch.states, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(batch.next_state, dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch.actions, dtype=torch.float32).to(self.device)
        timesteps = torch.tensor(batch.timesteps, dtype=torch.long).to(self.device)
        next_timesteps = torch.tensor(batch.next_timesteps, dtype=torch.long).to(self.device)
        rewards = torch.tensor(batch.rewards, dtype=torch.float32).to(self.device)
        next_rewards = torch.tensor(batch.next_rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(batch.dones, dtype=torch.bool).to(self.device)
        next_actions = torch.tensor(batch.next_actions, dtype=torch.float32).to(self.device)
        attention_mask = torch.tensor(batch.attention_mask, dtype=torch.long).to(self.device)

        # Predict the action from the 
        action_preds = self.actor.forward(
            states, actions, rewards, timesteps, attention_mask=attention_mask,
        )

        next_action_preds = self.actor_target.forward(
            next_state, next_actions, next_rewards, next_timesteps, attention_mask=attention_mask,
        )

        states = states.reshape(-1, self.config.state_dim)[attention_mask.reshape(-1) > 0]
        next_state = next_state.reshape(-1, self.config.state_dim)[attention_mask.reshape(-1) > 0]
        rewards = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        action_sample = actions.reshape(-1, self.config.action_dim)[attention_mask.reshape(-1) > 0]
        Q_action_preds = action_preds.reshape(-1, self.config.action_dim)[attention_mask.reshape(-1) > 0]
        next_Q_action_preds = next_action_preds.reshape(-1, self.config.action_dim)[attention_mask.reshape(-1) > 0]
        dones = dones.reshape(-1, 1)[attention_mask.reshape(-1) > 0]

        # Compute the target Q value
        target_Q = self.critic_target(next_state, next_Q_action_preds)
        target_Q = rewards + (dones * self.config.gamma * target_Q).detach()
        # Get current Q estimates
        current_Q = self.critic(states, action_sample)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Set lambda and Compute actor loss
        pi = Q_action_preds
        Q = self.critic(states, pi)
        lmbda = self.config.alpha / Q.abs().mean().detach()
        actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action_sample)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        # Update the target networks using Polyak averaging
        # This ensures that the target networks are slowly updated
        polyak_update(self.actor, self.actor_target, self.config.tau)

        polyak_update(self.critic, self.critic_target, self.config.tau)

        return actor_loss.detach().cpu().item()
        


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
        save_model(self.critic_rh, path, name + "_rh")

    def load_critic(self, path, name="critic"):
        """
        Loads the critic model from the specified path.

        Args:
            path (str): The directory path from where the model should be loaded.
            name (str, optional): The base name of the model files to be loaded. Defaults to "critic".
        """
        self.critic = load_model(self.critic, path, name)
        self.critic_target = get_target_network(self.critic)