# Script to evaluate MHC encoding for EL/BA evaluation
# Johan Ziruo Ye
# Spring 2020

import os
import pandas as pd
from Utils import *

BA_EL = "BA"  # Expects BA or EL

# Data stuff
data_path = os.getcwd() + '/MHC_I/' + BA_EL.lower() + "_data/"
MHC = pd.read_csv(os.getcwd() + '/MHC_I/' + 'MHC_pseudo.dat', header=None, sep='\s+')
MHC_len = MHC[1].map(len).max()
MHC_dict = MHC.set_index(0).to_dict()[1]
All_data = {0, 1, 2, 3, 4}

for test_set in range(5):
    for validation_set in range(5):

        if test_set == validation_set:
            continue

        # data loading:
        train_set = list(All_data - set([test_set, validation_set]))
        X_train, y_train = FileToTensor(data_path, train_set, BA_EL, MHC_dict)
        X_val, y_val = FileToTensor(data_path, validation_set, BA_EL, MHC_dict)
        X_test, y_test = FileToTensor(data_path, test_set, BA_EL, MHC_dict)


        print(test_set, validation_set)

import pandas as pd

filepath = data_path
Partition = test_set
mapping_dict = MHC_dict
import numpy as np
AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','empty']
seq = 'YSEMYRERAGNFFVSNLYLWSMFYSMAEQNYRWY'
one_hot_encode(seq).shape

def one_hot_encode(seq):
    o = list(set(AA) - set(seq))
    s = pd.DataFrame(list(seq))
    x = pd.DataFrame(np.zeros((len(seq),len(o)),dtype=int),columns=o)
    a = s[0].str.get_dummies(sep=',')
    a = a.join(x)
    a = a.sort_index(axis=1)
    e = a.values
    return e
