# Script to evaluate MHC encoding for EL/BA evaluation
# Johan Ziruo Ye
# Spring 2020

import os
from Utils import *
from model import *
import time
import torch.optim as optim
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data', type=str, default="BA")
parser.add_argument('--resnet_output_file', type=str, default='model_output_resnet.txt')
parser.add_argument('--total_output_file', type=str, default='model_output_total.txt')
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--lstm', type=bool, default=True)
parser.add_argument('--fucking_raw', type=bool, default=False)
parser.add_argument('--gauss', action='store_true')
parser.add_argument('--block_type', type=str, default='bacd')
parser.add_argument('--n_reslayers', type=int, default=5)
parser.add_argument('--lstm_nhidden', type=int, default=64)
parser.add_argument('--lstm_nlayers', type=int, default=2)
parser.add_argument('--rezero', type=bool, default=False)
args = parser.parse_args()



BA_EL = args.data  # Expects BA or EL

# Data stuff
data_path = os.getcwd() + '/MHC_I/' + BA_EL.lower() + "_data/"
MHC = pd.read_csv(os.getcwd() + '/MHC_I/' + 'MHC_pseudo.dat', header=None, sep='\s+')
MHC_len = MHC[1].map(len).max()
MHC_dict = MHC.set_index(0).to_dict()[1]
All_data = {0, 1, 2, 3, 4}
torch.manual_seed(args.seed)

# Hyperparams:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Peptide_len = 15
n_epoch = args.n_epochs
batch_size = args.batch_size
lr = args.lr
patience = args.patience
train_epoch_loss, val_epoch_loss, test_epoch_loss = [], [], []
k = 0
exp_path, save_dir, figure_dir = generate_experiment_folders('./experiments', args)

file1 = save_dir + args.resnet_output_file
file2 = save_dir + args.total_output_file

outfile_resnet = open(file1, 'w')
outfile_resnet.write('split\tMHC\tPeptide\ty\ty_pred\n')
outfile_total = open(file2, 'w')
outfile_total.write('split\tMHC\tPeptide\ty\ty_pred\n')


def criterion(y, mu, std=None, normal_dist=True):
    if normal_dist:
        loss = -torch.distributions.Normal(loc=mu, scale=std).log_prob(y)
    else:
        loss = nn.functional.mse_loss(mu, y)
    return loss.mean()


for test_set in range(5):
    test_loader = torch.utils.data.DataLoader(
        MHC_dataset(data_path, test_set, BA_EL, MHC_dict, MHC_len), batch_size=batch_size, shuffle=True)

    for validation_set in range(5):

        t = time.process_time()
        if test_set == validation_set:
            continue
        k += 1
        best_val_MSE = np.inf

        # data loading: [N, Concat_length, Amino acids]
        train_set = list(All_data - set([test_set, validation_set]))
        train_loader = torch.utils.data.DataLoader(
            MHC_dataset(data_path, train_set, BA_EL, MHC_dict, MHC_len), batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            MHC_dataset(data_path, validation_set, BA_EL, MHC_dict, MHC_len), batch_size=batch_size, shuffle=True)

        net = ResidualNetwork(block_type=args.block_type, n_layers=args.n_reslayers, rezero=args.rezero).to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr)

        for epoch in range(1, n_epoch):
            train_batch_loss = []
            for X, y in tqdm(train_loader):
                net.train()
                X = X.permute(0, 2, 1).float()
                mu, std = net(X.to(device))
                loss = criterion(y.to(device).float(), mu, std, normal_dist=args.gauss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_batch_loss.append(loss.item())

            val_batch_loss, test_batch_loss = [], []
            for X, y in tqdm(val_loader):
                X = X.permute(0, 2, 1).float()
                with torch.no_grad():
                    net.eval()
                    mu, std = net(X.to(device))
                    loss = criterion(y.to(device).float(), mu, std, normal_dist=False)
                    val_batch_loss.append(loss.item())

            train_epoch_loss.append(np.mean(train_batch_loss))
            val_epoch_loss.append(np.mean(val_batch_loss))

            print('Validation Split: [{}/20], Epoch: {}, Training Loss: {}, Validation Loss {}'.format(
                k, epoch, train_epoch_loss[-1], val_epoch_loss[-1]))

            if np.mean(val_batch_loss) < best_val_MSE:
                best_epoch = epoch
                best_val_MSE = np.mean(val_batch_loss)
                save_checkpoint({'epoch': best_epoch, 'state_dict': net.state_dict()},
                                save_dir,
                                ckpt_name='best' + str(best_epoch) + '_resnet.pth.tar')

            if epoch - best_epoch > patience:  # Early stopping
                break

        load_checkpoint(save_dir + 'best' + str(best_epoch) + '_resnet.pth.tar', net)

        performance_testing_print(
            data_path, test_set, BA_EL, MHC_dict, batch_size, MHC_len, Peptide_len, net, k, outfile_resnet)

        if args.lstm:
            # LSTM WITH FROZEN RESNET
            best_val_MSE = np.inf
            if args.fucking_raw:
                net2 = Resnet_Blosum_direct(block_type=args.block_type).to(device)
            else:
                net2 = Frozen_resnet(lstm_hidden=args.lstm_nhidden).to(device)
            optimizer = optim.Adam(net2.parameters(), lr=lr)

            for epoch in range(1, n_epoch + 1):
                train_batch_loss = []
                for X, y in tqdm(train_loader):
                    net.eval()
                    net2.train()
                    X = X.permute(0, 2, 1).float().to(device)
                    with torch.no_grad():
                        res_out = net(X)
                    y_pred = net2(X, res_out)  # detach because i'm paranoid about gradients
                    loss = nn.functional.mse_loss(y_pred, y.to(device).float())  # , normal_dist=False)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_batch_loss.append(loss.item())

                val_batch_loss, test_batch_loss = [], []

                for X, y in tqdm(val_loader):
                    X = X.permute(0, 2, 1).float().to(device)
                    with torch.no_grad():
                        net.eval()
                        net2.eval()
                        res_out = net(X)
                        y_pred = net2(X, res_out)
                        loss = nn.functional.mse_loss(y_pred, y.to(device).float())
                        val_batch_loss.append(loss.item())

                train_epoch_loss.append(np.mean(train_batch_loss))
                val_epoch_loss.append(np.mean(val_batch_loss))

                print('Validation Split: [{}/20], Epoch: {}, Training Loss: {}, Validation Loss {}'.format(
                    k, epoch, train_epoch_loss[-1], val_epoch_loss[-1]))

                if np.mean(val_batch_loss) < best_val_MSE:
                    best_epoch = epoch
                    best_val_MSE = np.mean(val_batch_loss)
                    save_checkpoint({'epoch': best_epoch, 'state_dict': net2.state_dict()},
                                    save_dir,
                                    ckpt_name='best' + str(best_epoch) + '_total.pth.tar')

                if epoch - best_epoch > patience:  # Early stopping
                    break

            load_checkpoint(save_dir + 'best' + str(best_epoch) + '_total.pth.tar', net2)
            performance_testing_print(
                data_path, test_set, BA_EL, MHC_dict,
                batch_size, MHC_len, Peptide_len, net, k, outfile_total,
                net2=net2, resnet=True)

outfile_resnet.close()
outfile_total.close()
