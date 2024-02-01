import torch
import torch.nn as nn
from torch import device


class LSTMEncoder(nn.Module):
    def __init__(self, input_size=625, hidden_size=128, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        batch, seq_len, channel, width, height = x.size()
        input_size = channel * width * height
        x = x.view(batch, seq_len, input_size)
        output, (hn, cn) = self.lstm(x)
        return output, (hn, cn)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, input_size=625, hidden_size=128, num_layers=1):
        super(LSTMAutoEncoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMPredictor, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded