# Script to evaluate MHC encoding for EL/BA evaluation
# Johan Ziruo Ye
# Spring 2020

import os
from scipy.stats import pearsonr
import torch
from Utils import *
from model import *
import time

BA_EL = "BA"  # Expects BA or EL

# Data stuff
data_path = os.getcwd() + '/MHC_I/' + BA_EL.lower() + "_data/"
MHC = pd.read_csv(os.getcwd() + '/MHC_I/' + 'MHC_pseudo.dat', header=None, sep='\s+')
MHC_len = MHC[1].map(len).max()
MHC_dict = MHC.set_index(0).to_dict()[1]
All_data = {0, 1, 2, 3, 4}

# Hyperparams:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epoch = 3
batch_size = 128
lr = 1e-4
net = DeepLigand().to(device)

for test_set in range(5):
    for validation_set in range(5):
        t = time.process_time()
        if test_set == validation_set:
            continue

        # data loading: [N, Concat_length, Amino acids]
        train_set = list(All_data - set([test_set, validation_set]))
        train_loader = FileToTensor(data_path, train_set, BA_EL, MHC_dict)
        val_loader = FileToTensor(data_path, validation_set, BA_EL, MHC_dict)
        test_loader = FileToTensor(data_path, test_set, BA_EL, MHC_dict)

        elapsed_time = time.process_time() - t
        print(elapsed_time, test_set, validation_set)

        for batch in train_loader:
            net.train()
            X, y = batch
            X = X.permute(0, 2, 1).to(device).float()
            output = net(X)



        break
    break
