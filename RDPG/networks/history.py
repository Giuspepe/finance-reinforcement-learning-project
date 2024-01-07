import torch.nn as nn


class RecordedHistory(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(RecordedHistory, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, observation, hidden=None, reset=False):
        if reset:
            hidden = None
        output, hidden = self.gru(observation, hidden)
        return output, hidden
