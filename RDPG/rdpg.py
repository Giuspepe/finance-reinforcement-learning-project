import numpy as np
import torch
import torch.optim as optim

from networks.history import RecordedHistory
from networks.actor import ActorMLP
from networks.critic import CriticMLP
from utils import get_device, save_model, load_model, get_target_network, polyak_update


class RDPG:
    def __init__(
        self, input_dim, action_dim, hidden_dim=256, gamma=0.99, lr=3e-4, tau=0.995
    ):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.gamma = gamma
        self.lr = lr
        self.tau = tau

        self.device = get_device()

        self.hidden = None

        # rh = recorded history

        # Actor networks (local and target)
        self.actor_rh = RecordedHistory(input_dim, hidden_dim).to(self.device)
        self.actor_rh_target = get_target_network(self.actor_rh)
        self.actor = ActorMLP(hidden_dim, action_dim).to(self.device)
        self.actor_target = get_target_network(self.actor)

        # Critic networks (local and target)
        self.critic_rh = RecordedHistory(input_dim, hidden_dim).to(self.device)
        self.critic_rh_target = get_target_network(self.critic_rh)
        self.critic = CriticMLP(hidden_dim, action_dim).to(self.device)
        self.critic_target = get_target_network(self.critic)

        # Actor optimizer (local and target)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.actor_rh_optimizer = optim.Adam(self.actor_rh.parameters(), lr=self.lr)

        # Critic optimizer (local and target)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.critic_rh_optimizer = optim.Adam(self.critic_rh.parameters(), lr=self.lr)

    def reset_hidden(self):
        self.hidden = None

    def get_action(self, observation):
        with torch.no_grad():
            # Dimension of observation is (1, 1, hidden_dim)
            observation = (
                torch.tensor(observation)
                .unsqueeze(0)
                .unsqueeze(0)
                .float()
                .to(get_device())
            )
            rh, self.hidden = self.actor_rh(observation, self.hidden)
            # For deterministic policy, we don't need to sample from the distribution
            action = self.actor(rh).view(-1).detach().cpu().numpy()
            return action

    def update(self, batch):
        # Get recorded history from actor and critic
        actor_rh, _ = self.actor_rh(batch.observations)
        critic_rh, _ = self.critic_rh(batch.observations)

        # Get recorded history from actor and critic target
        actor_rh_target, _ = self.actor_rh_target(batch.observations)
        critic_rh_target, _ = self.critic_rh_target(batch.observations)

        # Original sequence from the actor network
        # This gives us all actions/states from the start up to the second-to-last time step
        actor_rh_1_T = actor_rh[:, :-1, :]

        # Sequence from the target actor network
        # This gives us all actions/states from the second time step to the end
        actor_rh_2_Tplus1 = actor_rh_target[:, 1:, :]

        # Same as above, but for the critic network
        critic_rh_1_T = critic_rh[:, :-1, :]
        critic_rh_2_Tplus1 = critic_rh_target[:, 1:, :]

        # Get the predicted values from the critic network
        preds = self.critic(critic_rh_1_T, batch.actions)

        # Get the target values from the critic network
        # Note that we use the target actor network to get the actions
        # This is because we want to use the target actor network to get the next state
        # This is the same as the original DDPG paper
        next_actions = self.actor(actor_rh_2_Tplus1)
        targets = batch.rewards + self.gamma * (1 - batch.dones) * self.critic_target(
            critic_rh_2_Tplus1, next_actions
        )

        # Compute the loss for the critic network
        # Similar to TD error
        critic_loss = (preds - targets).pow(2)
        # Mask out the loss for the padded time steps (i.e. where the mask is 0)
        critic_loss = (
            torch.mean(critic_loss * batch.mask)
            / batch.mask.sum()
            * np.prod(batch.mask.shape)
        )

        # Minimize the critic loss
        self.critic_rh_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_rh_optimizer.step()
        self.critic_optimizer.step()

        # Get the predicted actions from the actor network
        # Note that we use the actor recorded history
        actions = self.actor(actor_rh_1_T)
        Q_values = self.critic(critic_rh_1_T.detach(), actions)
        pi_loss = -Q_values
        # Mask out the loss for the padded time steps (i.e. where the mask is 0)
        pi_loss = (
            torch.mean(pi_loss * batch.mask)
            / batch.mask.sum()
            * np.prod(batch.mask.shape)
        )

        # Minimize the actor loss
        self.actor_rh_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        pi_loss.backward()
        self.actor_rh_optimizer.step()
        self.actor_optimizer.step()

        # Update the target networks using Polyak averaging
        # This ensures that the target networks are slowly updated
        polyak_update(self.actor_rh, self.actor_rh_target, self.tau)
        polyak_update(self.actor, self.actor_target, self.tau)

        polyak_update(self.critic_rh, self.critic_rh_target, self.tau)
        polyak_update(self.critic, self.critic_target, self.tau)

        return {
            "Q predictions": float(
                torch.mean(preds)
                * batch.mask
                / batch.mask.sum()
                * np.prod(batch.mask.shape)
            ),
            "Q Loss": float(critic_loss),
            "Actor Q values": float(
                torch.mean(Q_values)
                * batch.mask
                / batch.mask.sum()
                * np.prod(batch.mask.shape)
            ),
        }

    def save_actor(self, path, name="actor"):
        save_model(self.actor, path, name)
        save_model(self.actor_rh, path, name + "_rh")

    def load_actor(self, path, name="actor"):
        self.actor = load_model(self.actor, path, name)
        self.actor_rh = load_model(self.actor_rh, path, name + "_rh")