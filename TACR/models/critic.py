import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Initialize the Critic class.

        Parameters:
        - state_dim (int): Dimension of the state space.
        - action_dim (int): Dimension of the action space.

        The critic network estimates the value of state-action pairs. It's a part of the actor-critic framework where the critic
        helps in evaluating the performance of the actor.
        """
        super(Critic, self).__init__()
        # First fully connected layer taking state and action dimensions as input
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        # Second fully connected layer
        self.fc2 = nn.Linear(512, 256)
        # Third fully connected layer outputting a single value (Value of the state-action pair)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, actions):
        """
        Forward pass for the Critic model.

        Parameters:
        - states (Tensor): Tensor of states.
        - actions (Tensor): Tensor of actions.

        Returns:
        - Tensor: Value of the state-action pairs.

        This method processes the inputs through fully connected layers with ReLU activations (except for the last layer)
        to estimate the value of state-action pairs.
        """
        # Concatenate states and actions as input to the network and process the concatenated input through the fully connected layers 
        # with ReLU activation
        x = F.relu(self.fc1(torch.cat([states, actions], dim=1)))
        x = F.relu(self.fc2(x))
        # Output layer without activation (linear layer)
        x = self.fc3(x)
        return x