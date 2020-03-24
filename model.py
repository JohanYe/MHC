# model script
import torch
import torch.nn as nn
from Utils import Flatten


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_shape, hidden_shape=256, Dropout=0.2, n_layers=2):
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


class ResidualBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride
        self.kernel_size = kernel_size

        self.layers = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size, stride, padding=1),  # Note bias is false in paper code
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.Conv1d(c_out, c_out, kernel_size, stride, padding=1),  # Note bias is false in paper code
            nn.BatchNorm1d(c_out),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.layers(x)


class DeepLigand(nn.Module):
    def __init__(self, filters=256, n_layers=5, seq_len=40):
        super(DeepLigand, self).__init__()

        # Convolutional network
        stride = [1] + [2] * n_layers
        self.stride = stride
        self.filters = filters
        self.n_layers = n_layers
        self.init_convolution = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=3, stride=1, padding=1),  # Note bias is false in paper code
            nn.BatchNorm1d(filters),
            nn.ReLU()
        )

        layers = [ResidualBlock(filters, filters, stride=stride[i]) for i in range(n_layers)]

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.layers = nn.Sequential(*layers)

        # LSTM
        nn.ELMo = BidirectionalLSTM(seq_len)

    def forward(self, x):
        out1 = self.init_convolution(x)
        out1 = self.layers(out1)

        out2 = self.ELMo(x)

        print(out1.shape, out2.shape)

        return



class MortenFFN(nn.Module):
    def __init__(self, input_shape, nh1=56, nh2=66):
        super(MortenFFN, self).__init__()

        self.nh1 = nh1
        self.nh2 = nh2
        self.input_shape = input_shape

        self.layers(
            Flatten(),
            nn.Linear(self.input_shape, nh1),
            nn.BatchNorm1d(nh2),
            nn.ReLU(),
            nn.Linear(nh1, nh2),
            nn.BatchNorm1d(nh2),
            nn.ReLU,
        )

    def forward(self, x):
        out = self.layers(x)
        return out
