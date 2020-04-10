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
n_epoch = 1
train_log = []
batch_size = 128
lr = 1e-4
net = DeepLigand().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.MSELoss()
k=1

for test_set in range(5):
    test_df = MHC_df(data_path, test_set, BA_EL, MHC_dict)
    batches_per_epoch = np.ceil(test_df.shape[0] / batch_size)

    for validation_set in range(5):
        t = time.process_time()
        if test_set == validation_set:
            continue

        best_test_MSE = np.inf


        # data loading: [N, Concat_length, Amino acids]
        train_set = list(All_data - set([test_set, validation_set]))
        train_loader = torch.utils.data.DataLoader(
            MHC_dataset(data_path, train_set, BA_EL, MHC_dict), batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            MHC_dataset(data_path, validation_set, BA_EL, MHC_dict), batch_size=batch_size, shuffle=True)

        elapsed_time = time.process_time() - t
        print(elapsed_time, test_set, validation_set)

        for epoch in range(n_epoch):
            for X, y in train_loader:
                net.train()
                X = X.permute(0, 2, 1).to(device).float()
                pred_BA = net(X)
                loss = criterion(pred_BA, y.to(device).float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_log.append(loss.item())

            print('Epoch: {}, Train loss: {}'.format(
                epoch, np.mean(train_log)))

            # Test loop is funny due to having to save MHC Allele
            test_df = test_df.sample(frac=1).reset_index(drop=True)  # Shuffling data set
            for i in range(batches_per_epoch):
                batch_df = test_df.iloc[batch_size*i:batch_size*(i+1)]  # Batching

                #Fun stuff converting to tensor
                y = torch.from_numpy(np.expand_dims(batch_df['BindingAffinity'].values, 1))
                X = batch_df.drop('BindingAffinity', axis=1)
                Peptide_mat = np.stack(X.Peptide.apply(
                    one_hot_encoding, encoding_dict=onehot_Blosum50, max_len=Peptide_len).values)
                MHC_mat = np.stack(
                    X.MHC.apply(one_hot_encoding, encoding_dict=onehot_Blosum50, max_len=MHC_len).values)
                X = torch.from_numpy(np.concatenate((Peptide_mat, MHC_mat), axis=1).astype(int))









                net.eval()
        break
    break
