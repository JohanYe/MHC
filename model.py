# model script
import torch
import torch.nn as nn
from Utils import Flatten

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_shape, hidden_shape=256, Dropout=True, n_layers = 2):
        super(BidirectionalLSTM, self).__init__()

        self.dropout = Dropout
        self.input_shape = input_shape
        self.hidden = hidden_shape
        self.layers = nn.LSTM(input_shape, hidden_size=hidden_shape, num_layers=n_layers,
                              dropout=self.dropout, bidirectional=True, batch_first=True)

    def forward(self, x):
        out = self.layers(x)
        out = nn.Softmax(out, -1)
        return out


class FFN(nn.Module):
    def __init__(self, input_shape, nh1=56, nh2=66):
        super(FFN, self).__init__()

        self.nh1 = nh1
        self.nh2 = nh2
        self.input_shape = input_shape

        self.layers(
            Flatten(),
            nn.Linear(self.input_shape, nh1),
            nn.BatchNorm1d(nh2),
            nn.ReLU(),
            nn.Linear(nh1,nh2),
            nn.BatchNorm1d(nh2),
            nn.ReLU,
        )

    def forward(self, x):
        out = self.layers(x)
        return out



