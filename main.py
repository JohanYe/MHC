# Script to evaluate MHC encoding for EL/BA evaluation
# Johan Ziruo Ye
# Spring 2020

import os
from scipy.stats import pearsonr
import torch
from Utils import *
from model import *
import time

BA_EL = "EL"  # Expects BA or EL

# Data stuff
data_path = os.getcwd() + '/MHC_I/' + BA_EL.lower() + "_data/"
MHC = pd.read_csv(os.getcwd() + '/MHC_I/' + 'MHC_pseudo.dat', header=None, sep='\s+')
MHC_len = MHC[1].map(len).max()
MHC_dict = MHC.set_index(0).to_dict()[1]
All_data = {0, 1, 2, 3, 4}

#Hyperparams:
n_epoch = 3
batch_size = 500
lr = 1e-4



for test_set in range(5):
    for validation_set in range(5):
        t = time.process_time()
        if test_set == validation_set:
            continue

        # data loading: [N, Concat_length, Amino acids]
        train_set = list(All_data - set([test_set, validation_set]))
        X_train, y_train = FileToTensor(data_path, train_set, BA_EL, MHC_dict)
        X_val, y_val = FileToTensor(data_path, validation_set, BA_EL, MHC_dict)
        X_test, y_test = FileToTensor(data_path, test_set, BA_EL, MHC_dict)

        elapsed_time = time.process_time() - t
        print(elapsed_time, test_set, validation_set)

        break
    break

