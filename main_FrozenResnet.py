# Script to evaluate MHC encoding for EL/BA evaluation
# Johan Ziruo Ye
# Spring 2020

import os
from scipy.stats import pearsonr
import torch
from Utils import *
from model import *
import time
import torch.optim as optim
from tqdm import tqdm

BA_EL = "BA"  # Expects BA or EL

# Data stuff
data_path = os.getcwd() + '/MHC_I/' + BA_EL.lower() + "_data/"
MHC = pd.read_csv(os.getcwd() + '/MHC_I/' + 'MHC_pseudo.dat', header=None, sep='\s+')
MHC_len = MHC[1].map(len).max()
MHC_dict = MHC.set_index(0).to_dict()[1]
All_data = {0, 1, 2, 3, 4}

# Hyperparams:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Peptide_len = 15
n_epoch = 100
batch_size = 128
lr = 1e-4
train_epoch_loss, val_epoch_loss, test_epoch_loss = [], [], []
k = 0
save_dir = './checkpoints/'

outfile = open('model_output.txt', 'w')
outfile.write('split\tMHC\tPeptide\ty\ty_pred\n')

def criterion(y, mu, std=None, normal_dist=True):
    if normal_dist:
        loss = -torch.distributions.Normal(loc=mu, scale=std).log_prob(y)
    else:
        loss = nn.MSELoss()(mu, y)
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

        net = ResidualNetwork(block_type='cabd').to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)

        for epoch in range(1, n_epoch + 1):
            train_batch_loss = []
            for X, y in tqdm(train_loader):
                net.train()
                X = X.permute(0, 2, 1).float()
                mu, std = net(X.to(device))
                loss = criterion(y.to(device).float(), mu, std)#, normal_dist=False)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_batch_loss.append(loss.item())

            val_batch_loss, test_batch_loss = [], []
            net.eval()
            for X, y in tqdm(val_loader):
                X = X.permute(0, 2, 1).float()
                with torch.no_grad():
                    mu, std = net(X.to(device))
                    loss = criterion(y.to(device).float(), mu, std, normal_dist=False)
                    # loss = nn.MSELoss()(mu, y.to(device))
                    val_batch_loss.append(loss.item())

            scheduler.step(np.mean(val_batch_loss))
            train_epoch_loss.append(np.mean(train_batch_loss))
            val_epoch_loss.append(np.mean(val_batch_loss))

            print('Validation Split: [{}/20], Epoch: {}, Training Loss: {}, Validation Loss {}'.format(
                k, epoch, train_epoch_loss[-1], val_epoch_loss[-1]))

            if np.mean(val_batch_loss) < best_val_MSE:
                best_epoch = epoch
                best_val_MSE = np.mean(val_batch_loss)
                save_checkpoint({'epoch': epoch, 'state_dict': net.state_dict()}, save_dir)

            if epoch - best_epoch > 10:  # Early stopping
                break

        net2 = net.to("cpu")



        performance_testing_print(
            data_path, test_set, BA_EL, MHC_dict, batch_size, MHC_len, Peptide_len, net, criterion, k, outfile)

outfile.close()


