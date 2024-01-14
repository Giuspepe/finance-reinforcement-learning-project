import torch
import os
from copy import deepcopy

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(network, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    print("Saving model to {}".format(path))
    torch.save(network.state_dict(), os.path.join(path, "model_{}.pth".format(name)))


def load_model(network, path, name):
    print("Loading model from {}".format(path))
    network.load_state_dict(torch.load(os.path.join(path, "model_{}.pth".format(name))))
    return network

def polyak_update(local_model, target_model, tau):
    for target_param, local_param in zip(
        target_model.parameters(), local_model.parameters()
    ):
        target_param.data.copy_(
            tau * local_param.data + (1.0 - tau) * target_param.data
        )

def get_target_network(network):
    target_net = deepcopy(network)
    for param in target_net.parameters():
        param.requires_grad = False
    return target_net

class Batch:
    def __init__(self, states, actions, rewards, dones, next_state, next_actions, next_rewards, timesteps, next_timesteps, attention_mask):
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