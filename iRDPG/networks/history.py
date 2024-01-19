import torch.nn as nn
import numpy as np
from typing import Union

class RecordedHistory(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, upper_normalization_bounds: Union[np.array, None] = None, lower_normalization_bounds: Union[np.array, None] = None):
        super(RecordedHistory, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.upper_normalization_bounds = upper_normalization_bounds
        self.lower_normalization_bounds = lower_normalization_bounds
        self.init_weights()

    def init_weights(self):
        # Initialize weights for lstm layers
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Xavier initialization for input weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Orthogonal initialization for recurrent weights
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Zero initialization for biases
                nn.init.zeros_(param)
                # Setting the forget gate bias (if using LSTM)
                # This is optional for lstm, but can be beneficial in some cases
                # bias_size = param.size(0)
                # start, end = bias_size // 4, bias_size // 2
                # param.data[start:end].fill_(1.)

    def forward(self, observation, hidden=None, reset=False):
        # Flatten parameters
        self.lstm.flatten_parameters()
        if reset:
            hidden = None
        output, hidden = self.lstm(self.normalize_state(observation), hidden)
        return output, hidden
    
    def normalize_state(self, state):
        if self.upper_normalization_bounds is None or self.lower_normalization_bounds is None:
            return state
        
        min_bounds = self.lower_normalization_bounds
        max_bounds = self.upper_normalization_bounds

        normalized_state = 2 * (state - min_bounds) / (max_bounds - min_bounds) - 1
        return normalized_state
