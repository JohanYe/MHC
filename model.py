# model script
import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_shape, hidden_shape, Dropout=True, n_layers = 2):
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


