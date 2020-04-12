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
n_epoch = 10
batch_size = 128
lr = 1e-4
net = DeepLigand().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.MSELoss()
train_epoch_loss, val_epoch_loss, test_epoch_loss = [], [], []
k=0

for test_set in range(5):
    test_loader = torch.utils.data.DataLoader(
        MHC_dataset(data_path, test_set, BA_EL, MHC_dict, MHC_len), batch_size=batch_size, shuffle=True)

    for validation_set in range(5):

        t = time.process_time()
        if test_set == validation_set:
            continue
        k += 1
        best_test_MSE = np.inf


        # data loading: [N, Concat_length, Amino acids]
        train_set = list(All_data - set([test_set, validation_set]))
        train_loader = torch.utils.data.DataLoader(
            MHC_dataset(data_path, train_set, BA_EL, MHC_dict, MHC_len), batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            MHC_dataset(data_path, validation_set, BA_EL, MHC_dict, MHC_len), batch_size=batch_size, shuffle=True)

        elapsed_time = time.process_time() - t
        print(elapsed_time, test_set, validation_set)

        for epoch in range(n_epoch):
            train_batch_loss = []
            for X, y in tqdm(train_loader):
                net.train()
                X = X.permute(0, 2, 1).float()
                pred_BA = net(X.to(device))
                loss = criterion(pred_BA, y.to(device).float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_batch_loss.append(loss.item())

            val_batch_loss, test_batch_loss = [], []
            net.eval()
            for X, y in tqdm(val_loader):
                X = X.permute(0, 2, 1).float()
                with torch.no_grad():
                    pred_BA = net(X.to(device))
                    loss = criterion(pred_BA, y.to(device).float())
                    val_batch_loss.append(loss.item())

            for X, y in tqdm(test_loader):
                X = X.permute(0, 2, 1).float()
                with torch.no_grad():
                    pred_BA = net(X.to(device))
                    loss = criterion(pred_BA, y.to(device).float())
                    test_batch_loss.append(loss.item())

            train_epoch_loss.append(np.mean(train_batch_loss))
            val_epoch_loss.append(np.mean(val_batch_loss))
            test_epoch_loss.append(np.mean(val_batch_loss))

            print('Validation Split: [{}/20], Epoch: {}, Training Loss: {}, Validation Loss {}, Test Loss: {}'.format(
                k, epoch, train_epoch_loss[-1], val_epoch_loss[-1], test_epoch_loss[-1]
            ))
        break
    break

    # test_df = MHC_df(data_path, test_set, BA_EL, MHC_dict)
    #     batches_per_epoch = int(np.ceil(test_df.shape[0] / batch_size))
    #
    # # LOOP IN ORDER TO MEASURE PERFORMANCE IN THE END.
    # # Test loop is funny due to having to save MHC Allele
    # test_df = test_df.sample(frac=1).reset_index(drop=True)  # Shuffling data set
    # for i in tqdm(range(batches_per_epoch)):
    #     if i == batches_per_epoch:  # Batching
    #         batch_df = test_df.iloc[batch_size * i:]
    #     else:
    #         batch_df = test_df.iloc[batch_size * i:batch_size * (i + 1)]
    #     X, y = df_ToTensor(test_df, MHC_len, Peptide_len)
    #     X = X.permute(0, 2, 1).float()
    #     with torch.no_grad():
    #         pred_BA = net(X.to(device))
    #         loss = criterion(pred_BA, y.to(device).float())
    #         val_batch_loss.append(loss.item())
