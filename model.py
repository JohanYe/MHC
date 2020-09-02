# model script
import torch
import torch.nn as nn
from utils import Flatten, PrintLayerShape
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import utils


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

    def __init__(self, c_in, c_out, nonlin=nn.ReLU(), kernel_size=7, block_type=None, dropout=None, stride=2,
                 rezero=False):
        super(ResidualBlock, self).__init__()

        assert all(c in 'abcd' for c in block_type)
        self.c_in, self.c_out = c_in, c_out
        self.nonlin = nonlin
        self.kernel_size = kernel_size
        self.block_type = block_type
        self.dropout = dropout
        self.stride = stride
        self.alpha = nn.Parameter(torch.Tensor([0]))  # rezero stuff
        self.rezero = rezero  # rezero stuff

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
        x = self.alpha * self.res(x) + x if self.rezero else self.res(x) + x
        if self.post_conv is not None:
            x = self.post_conv(x)

        return x


class ResidualNetwork(nn.Module):
    def __init__(self, filters=256, n_layers=5, seq_len=49, n_Linear=256, block_type=None, rezero=False):
        super(ResidualNetwork, self).__init__()
        self.n_layers = n_layers
        self.block_type = block_type
        self.seq_len = seq_len
        self.n_Linear = n_Linear
        self.rezero = rezero
        self.strides = [2] * n_layers if n_layers <= 5 else [2] * 2 + [1] * (n_layers - 5) + [2] * 3
        # idk why this is round and not int/floor as usual
        self.ResidualOutDim = max(round((49 / (2 ** n_layers))), 2) if n_layers <= 5 else \
            max(round((49 / (2 ** n_layers // 2))), 2)

        self.Residual_initial_MHC = nn.Sequential(
            nn.Conv1d(40, filters, kernel_size=7, stride=1, padding=3), nn.BatchNorm1d(filters), nn.ReLU()
        )  # Note bias is false in paper code
        self.Residual_initial_Peptide = nn.Sequential(
            nn.Conv1d(40, filters, kernel_size=7, stride=1, padding=3), nn.BatchNorm1d(filters), nn.ReLU())
        layers = [ResidualBlock(filters,
                                filters,
                                block_type=block_type,
                                dropout=0.1,
                                nonlin=nn.LeakyReLU(),
                                kernel_size=7,
                                stride=self.strides[i],
                                rezero=rezero) for i in range(n_layers - 1)]
        layers.append(ResidualBlock(filters, filters // 2, block_type=block_type, dropout=0.1, nonlin=nn.LeakyReLU()))

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')  # , nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(int(filters // 2 * self.ResidualOutDim), n_Linear)
        self.mu = nn.Linear(n_Linear, 1)
        self.std = nn.Linear(n_Linear, 1)

    def Residual_init(self, x):
        x_Peptide, x_MHC = torch.split(x, [15, 34], dim=2)
        out1 = self.Residual_initial_MHC(x_MHC)
        out2 = self.Residual_initial_Peptide(x_Peptide)
        out = torch.cat((out1, out2), dim=2)
        return out

    def forward(self, x):
        # Resnet
        out = self.Residual_init(x)
        out = self.layers(out).view(x.shape[0], -1)
        out = self.fc(out)
        # TODO try sigmoid
        mu = nn.Softplus()(self.mu(out))
        std = nn.Softplus()(self.std(out))  # Double parenthesis since it's a class

        return mu, std


class Frozen_resnet(nn.Module):
    def __init__(self, lstm_hidden=64, init_hidden=50, lstm_linear=256, MHC_len=34, Pep_len=15, lstm_layers=2,
                 full_lstm=False):
        super(Frozen_resnet, self).__init__()
        self.full_lstm = full_lstm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lstm_hidden = lstm_hidden
        self.init_hidden = init_hidden
        self.MHC_len = MHC_len
        self.Pep_len = Pep_len
        self.final_linear_dim = 1024
        self.lstm_layers = lstm_layers

        # Linear Init
        if full_lstm:
            self.MHC_init = BidirectionalLSTM(MHC_len, hidden_shape=init_hidden, n_layers=2)
            self.pep_init = BidirectionalLSTM(Pep_len, hidden_shape=init_hidden, n_layers=2)
        else:
            self.MHC_init = nn.Sequential(nn.Linear(MHC_len, init_hidden), nn.ReLU())
            self.pep_init = nn.Sequential(nn.Linear(Pep_len, init_hidden), nn.ReLU())

        # LSTM
        if full_lstm:
            self.LSTM = BidirectionalLSTM(init_hidden * 4, hidden_shape=lstm_hidden, n_layers=lstm_layers)
        else:
            self.LSTM = BidirectionalLSTM(init_hidden * 2, hidden_shape=lstm_hidden, n_layers=lstm_layers)
        self.LSTM_linear = nn.Sequential(
            Flatten(),
            nn.Linear(2 * lstm_hidden * 40, lstm_linear),
            nn.BatchNorm1d(lstm_linear),
            nn.ReLU(), )

        self.final_linear = nn.Linear(lstm_linear + 2, 1)

    def Input_To_LSTM(self, x):
        x_peptide, x_MHC = torch.split(x, [15, 34], dim=2)

        # Peptide
        if self.full_lstm:
            x_peptide = self.pep_init(x_peptide)[0]
            x_MHC = self.MHC_init(x_MHC)[0]
        else:
            x_peptide = self.pep_init(x_peptide)
            x_MHC = self.MHC_init(x_MHC)

        x = torch.cat((x_peptide, x_MHC), dim=2)
        x = self.LSTM(x)[0]
        x = self.LSTM_linear(x)

        return x

    def forward(self, x, Resnet_input):
        x = self.Input_To_LSTM(x)

        Res_mu, Res_std = Resnet_input
        Res_mu = Res_mu.detach().to(self.device)
        Res_std = Res_std.detach().to(self.device)

        # shape stuff
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, Res_mu, Res_std), dim=1)
        x = torch.sigmoid(self.final_linear(x))

        return x


class Resnet_Blosum_direct(nn.Module):
    def __init__(self, filters=256, n_Linear=512, block_type=None, stride=[2, 1, 1, 1, 2]):
        super(Resnet_Blosum_direct, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.final_linear_dim = n_Linear
        self.ResidualOutDim = 6402

        self.Residual_initial_MHC = nn.Sequential(
            nn.Conv1d(40, filters, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(filters), nn.ReLU()
        )  # Note bias is false in paper code
        self.Residual_initial_Peptide = nn.Sequential(
            nn.Conv1d(40, filters, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(filters), nn.ReLU())

        layers = [ResidualBlock(filters,
                                filters,
                                block_type=block_type,
                                dropout=0.1,
                                nonlin=nn.LeakyReLU(),
                                stride=stride[i]) for i in range(5)]

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')  # , nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.layers = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(6400, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        self.final_linear = nn.Linear(514, 1)

    def Residual_init(self, x):
        x_Peptide, x_MHC = torch.split(x, [15, 34], dim=2)
        out1 = self.Residual_initial_MHC(x_MHC)
        out2 = self.Residual_initial_Peptide(x_Peptide)
        out = torch.cat((out1, out2), dim=2)
        return out

    def forward(self, x, Resnet_input):
        out = self.Residual_init(x)

        Res_mu, Res_std = Resnet_input
        Res_mu = Res_mu.detach().to(self.device)
        Res_std = Res_std.detach().to(self.device)

        # shape stuff
        out = out.view(x.shape[0], -1)
        out = self.fc(out)
        out = torch.cat((out, Res_mu, Res_std), dim=1)
        out = torch.sigmoid(self.final_linear(out))

        return out


class DeepLigand(nn.Module):
    def __init__(self, filters=256, n_layers=5, seq_len=49, block_type=None, lstm_hidden=128, lstm_linear=256):
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

        # LSTM
        self.ELMo = BidirectionalLSTM(seq_len, hidden_shape=lstm_hidden, n_layers=n_layers)
        self.ELMo_Linear = nn.Sequential(
            Flatten(),
            nn.Linear(2 * lstm_hidden * 40, lstm_linear),
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
        # LSTM
        x_lstm = x.view(x.shape[0], -1, self.seq_len)
        out2 = self.ELMo(x)[0]
        out2 = self.ELMo_Linear(out2)

        # Network together
        # out = torch.cat((out1, out2), dim=1)
        # out = self.final_linear(out)
        # out = torch.sigmoid(out)

        return out2


class VariationalAutoencoder(nn.Module):
    """ Apparently we assume bernoulli output distribution """

    def __init__(self, c_in=40, n_filters=256, n_latent=256):
        super(VariationalAutoencoder, self).__init__()

        self.n_latent = n_latent
        self.c_in = 33
        self.n_filters = n_filters
        self.flattened_size = (49 // 2 ** 5) * n_filters
        self.Residual_initial_MHC = nn.Sequential(
            nn.Conv1d(c_in, n_filters, kernel_size=5, stride=1, padding=2), nn.BatchNorm1d(n_filters), nn.ReLU()
        )  # Note bias is false in paper code
        self.Residual_initial_Peptide = nn.Sequential(
            nn.Conv1d(c_in, n_filters, kernel_size=5, stride=1, padding=2), nn.BatchNorm1d(n_filters), nn.ReLU())

        self.Deterministic_Encoder = nn.Sequential(
            ResidualBlock(n_filters, n_filters, kernel_size=7, dropout=0.1, block_type='cabd'),
            ResidualBlock(n_filters, n_filters, kernel_size=7, dropout=0.1, block_type='cabd'),
            ResidualBlock(n_filters, n_filters // 2, kernel_size=7, dropout=0.1, block_type='cabd'),
            utils.Flatten(),
            nn.Linear(896, 512),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(512, n_latent)
        self.lv = nn.Linear(512, n_latent)

        # Not the prettiest, will do for nwo
        self.fc2 = nn.Sequential(
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=256, hidden_size=512, num_layers=3, batch_first=True)
        self.fc3 = nn.Linear(512, 49)

    def encode(self, x):
        x_peptide, x_MHC = torch.split(x, [15, 34], dim=2)
        out1 = self.Residual_initial_MHC(x_MHC)
        out2 = self.Residual_initial_Peptide(x_peptide)
        out = torch.cat((out1, out2), dim=2)

        out = self.Deterministic_Encoder(out)

        mu = self.mu(out)
        lv = self.lv(out)
        return mu, lv

    def reparametrize(self, mu, lv):
        std = 0.5 * lv.exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # in place calculations are faster in pytorch

    def decode(self, z):
        z = F.selu(self.fc2(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 60, 1)
        out, h = self.gru(z)
        out_layershape = out.contiguous().view(-1, out.size(-1))
        y0 = self.fc3(out_layershape)
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):

        mu, lv = self.encode(x)
        z = self.reparametrize(mu, lv)

        return self.decode(z)

    def loss_function(self, x):
        mu, lv = self.encode(x)
        q = torch.distributions.Normal(loc=mu, scale=lv.mul_(0.5).exp())
        p = torch.distributions.Normal(loc=0, scale=1)
        KLD = torch.distributions.kl_divergence(q, p).sum() / x.shape[0]
        x_recon = self.decode(q.rsample())

        # changing values to help training
        x_lv = F.softplus(x_recon[:, 40:, :])
        x[:, 20:, :] = x[:, 20:, :] / 15  # largest value in BLOSUM50

        # Binary loss
        bernoulli_error = torch.distributions.Bernoulli(logits=x_recon[:, :20, :]).log_prob(x[:, :20, :]).sum() / x.shape[0]
        gauss_error = torch.distributions.Normal(loc=x_recon[:, 20:40, :], scale=x_lv).log_prob(
            x[:, 20:, :]).sum() / x.shape[0]

        return -(bernoulli_error + gauss_error - KLD)

    def train_model(self, model, train_loader, validation_loader, optimizer, save_dir, crossvalsplit, n_epoch=100):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_epoch_loss, val_epoch_loss, test_epoch_loss = [], [], []
        best_validation_MSE = np.inf

        for epoch in range(1, n_epoch + 1):
            train_batch_loss = []

            for X, y in tqdm(train_loader):
                model.train()
                X = X.permute(0, 2, 1).float().to(device)
                loss = model.loss_function(X)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_batch_loss.append(loss.item())

            val_batch_loss = []
            for X, y in tqdm(validation_loader):
                X = X.permute(0, 2, 1).float().to(device)
                with torch.no_grad():
                    model.eval()
                    loss = model.loss_function(X)
                val_batch_loss.append(loss.item())

                train_epoch_loss.append(np.mean(train_batch_loss))
                val_epoch_loss.append(np.mean(val_batch_loss))

            print('Validation Split: [{}/20], Epoch: {}, Training Loss: {}, Validation Loss {}'.format(
                crossvalsplit, epoch, train_epoch_loss[-1], val_epoch_loss[-1]))

        return model, optimizer
