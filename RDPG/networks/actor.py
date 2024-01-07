import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorMLP(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x
