# model script
import torch
import torch.nn as nn
from Utils import Flatten, PrintLayerShape


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
    # Consider addring gated resnet block instead
    # block_type is a string specifying the structure of the block, where:
    #         a = activation
    #         b = batch norm
    #         c = conv layer
    #         d = dropout.
    # For example, bacd (batchnorm, activation, conv, dropout).
    # TODO: ADDTT uses different number of filters in inner, should we consider that? I've only allowed same currently.

    def __init__(self, c_in, c_out, nonlin=nn.ReLU(), kernel_size=3, block_type=None, dropout=None, stride=2):
        super(ResidualBlock, self).__init__()

        assert all(c in 'abcd' for c in block_type)
        self.c_in, self.c_out = c_in, c_out
        self.nonlin = nonlin
        self.kernel_size = kernel_size
        self.block_type = block_type
        self.dropout = dropout
        self.stride = stride

        self.pre_conv = nn.Conv1d(
            c_in, c_out, kernel_size=kernel_size, padding=self.kernel_size // 2, stride=stride)
        res = []  # Am considering throwing these if statements into separate function
        for character in block_type:
            if character == 'a':
                res.append(nonlin)
            elif character == 'b':
                res.append(nn.BatchNorm1d(c_out))
            elif character == 'c':
                res.append(
                    nn.Conv1d(c_out, c_out, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
                )
            elif character == 'd':
                res.append(nn.Dropout(dropout))
        self.res = nn.Sequential(*res)
        self.post_conv = None  # TODO: Consider implementation of this

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.res(x) + x
        if self.post_conv is not None:
            x = self.post_conv(x)

        return x


class DeepLigand(nn.Module):
    def __init__(self, filters=256, n_layers=5, seq_len=49, lstm_hidden=128, lstm_linear=256, block_type=None):
        super(DeepLigand, self).__init__()

        # Convolutional network
        stride = 1
        self.lstm_hidden = lstm_hidden
        self.lstm_linear = lstm_linear
        self.seq_len = seq_len
        self.stride = stride
        self.filters = filters
        self.n_layers = n_layers
        self.block_type = block_type
        self.ResidualOutDim = round((49 / (2 ** n_layers)))  # No idea why this is round and not int / floor as usuaual
        self.final_linear_dim = int(self.ResidualOutDim*filters + lstm_linear)
        self.init_convolution = nn.Sequential(
            nn.Conv1d(40, filters, kernel_size=3, stride=1, padding=1),  # Note bias is false in paper code
            nn.BatchNorm1d(filters),
            nn.ReLU()
        )

        layers = [ResidualBlock(filters,
                                filters,
                                block_type=block_type,
                                dropout=0.1,
                                nonlin=nn.LeakyReLU()) for _ in range(n_layers)]

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.layers = nn.Sequential(*layers)

        # LSTM
        self.ELMo = BidirectionalLSTM(seq_len, hidden_shape=lstm_hidden, n_layers=3)
        self.ELMo_Linear = nn.Sequential(
            Flatten(),
            nn.Linear(2*lstm_hidden*40, lstm_linear),
            nn.BatchNorm1d(lstm_linear),
            nn.ReLU(),
        )

        self.final_linear = nn.Sequential(
            nn.Linear(self.final_linear_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )


    def forward(self, x):
        out1 = self.init_convolution(x)
        out1 = self.layers(out1).view(x.shape[0],-1)

        x_lstm = x.view(x.shape[0], -1, self.seq_len)
        out2 = self.ELMo(x)[0]
        out2 = self.ELMo_Linear(out2)

        # Network together
        out = torch.cat((out1, out2), dim=1)
        out = self.final_linear(out)
        # out = torch.sigmoid(out)

        return out















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
