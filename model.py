# model script
import torch
import torch.nn as nn
from Utils import Flatten


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_shape, hidden_shape=256, Dropout=0, n_layers=2):
        super(BidirectionalLSTM, self).__init__()

        self.dropout = Dropout
        self.input_shape = input_shape
        self.hidden = hidden_shape
        self.layers = nn.LSTM(input_shape, hidden_size=hidden_shape, num_layers=n_layers,
                              dropout=self.dropout, bidirectional=True, batch_first=True)

    def forward(self, x):
        out = self.layers(x)
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
    def __init__(self, filters=256, n_layers=5, seq_len=49, lstm_hidden=256):
        super(DeepLigand, self).__init__()

        # Convolutional network
        stride = 1
        self.lstm_hidden = lstm_hidden
        self.lstm_linear = 32
        self.seq_len = seq_len
        self.stride = stride
        self.filters = filters
        self.n_layers = n_layers
        self.init_convolution = nn.Sequential(
            nn.Conv1d(40, filters, kernel_size=3, stride=1, padding=1),  # Note bias is false in paper code
            nn.BatchNorm1d(filters),
            nn.ReLU()
        )

        layers = [ResidualBlock(filters, filters, stride=stride) for _ in range(n_layers)]
        # layers = [ResidualBlock(filters, filters, stride=stride[i]) for i in range(n_layers)]

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.layers = nn.Sequential(*layers)

        # LSTM
        self.ELMo = BidirectionalLSTM(seq_len, hidden_shape=lstm_hidden, n_layers=2)
        self.ELMo_Linear = nn.Sequential(
            nn.Linear(2*lstm_hidden, 32),
            nn.BatchNorm1d(32)
        )

    def forward(self, x):
        out1 = self.init_convolution(x)
        out1 = self.layers(out1)

        x_lstm = x.view(x.shape[0], -1, self.seq_len)
        out2 = self.ELMo(x)[0]
        print(out2.shape)

        #print(out1.shape, len(out2), out2[0].shape, out2[1].shape)

        return out1, out2



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
