import torch
import os
from copy import deepcopy

def get_device():
    """
    Determine and return the available computation device.

    Returns:
    - torch.device: The computation device (CUDA if available, otherwise CPU).

    This function checks if CUDA is available and returns a CUDA device; otherwise, it returns the CPU device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(network, path, name):
    """
    Save a model's state dictionary to a file.

    Parameters:
    - network (torch.nn.Module): The neural network model to save.
    - path (str): Directory path to save the model.
    - name (str): Model name to be used in the filename.

    This function saves the state dictionary of the given network model to the specified path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    print("Saving model to {}".format(path))
    torch.save(network.state_dict(), os.path.join(path, "model_{}.pth".format(name)))


def load_model(network, path, name):
    """
    Load a model's state dictionary from a file.

    Parameters:
    - network (torch.nn.Module): The neural network model to load into.
    - path (str): Directory path from where to load the model.
    - name (str): Model name used in the filename.

    Returns:
    - torch.nn.Module: The network with loaded state dictionary.

    This function loads the state dictionary into the given network model from the specified path.
    """
    print("Loading model from {}".format(path))
    network.load_state_dict(torch.load(os.path.join(path, "model_{}.pth".format(name))))
    return network

def polyak_update(local_model, target_model, tau):
    """
    Update the target model parameters using Polyak averaging.

    Parameters:
    - local_model (torch.nn.Module): The local (source) model.
    - target_model (torch.nn.Module): The target model to be updated.
    - tau (float): The interpolation parameter.

    This function softly updates the target model parameters based on the local model parameters using Polyak averaging.
    """
    for target_param, local_param in zip(
        target_model.parameters(), local_model.parameters()
    ):
        target_param.data.copy_(
            tau * local_param.data + (1.0 - tau) * target_param.data
        )

def get_target_network(network):
    """
    Create and return a target network as a deep copy of the given network.

    Parameters:
    - network (torch.nn.Module): The source neural network model.

    Returns:
    - torch.nn.Module: A deep copy of the source network with gradient computation disabled.

    This function creates a deep copy of the given network and sets 'requires_grad' to False for all its parameters.
    """
    target_net = deepcopy(network)
    for param in target_net.parameters():
        param.requires_grad = False
    return target_net

class Batch:
    def __init__(self, states, actions, rewards, dones, next_state, next_actions, next_rewards, timesteps, next_timesteps, attention_mask):
        """
        Initialize a Batch object to hold data for a batch of experiences.

        Parameters:
        - states (Tensor): States in the batch.
        - actions (Tensor): Actions taken in the batch.
        - rewards (Tensor): Rewards received in the batch.
        - dones (Tensor): Done flags indicating the end of an episode.
        - next_state (Tensor): Next states in the batch.
        - next_actions (Tensor): Next actions in the batch.
        - next_rewards (Tensor): Next rewards in the batch.
        - timesteps (Tensor): Timesteps in the batch.
        - next_timesteps (Tensor): Next timesteps in the batch.
        - attention_mask (Tensor): Attention masks for the batch.

        This class serves as a container for a batch of data used in training reinforcement learning algorithms.
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.next_state = next_state
        self.next_actions = next_actions
        self.next_rewards = next_rewards
        self.timesteps = timesteps
        self.next_timesteps = next_timesteps
        self.attention_mask = attention_mask