import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticMLP(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(CriticMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, states, actions):
        x = F.relu(self.fc1(torch.cat([states, actions], dim=-1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x