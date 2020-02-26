# Script to evaluate MHC encoding for EL/BA evaluation
# Johan Ziruo Ye
# Spring 2020

import os
import pandas as pd
from Utils import *

BA_EL = "BA"  # Expects BA or EL

# Data stuff
data_path = os.getcwd() + '/MHC_I/' + BA_EL.lower() + "_data/"
MHC = pd.read_csv(os.getcwd() + '/MHC_I/' + 'MHC_pseudo.dat',header=None, sep='\s+')
MHC_len = MHC[1].map(len).max()
MHC_dict = MHC.set_index(0).to_dict()[1]

for test_set in range(5):
    for validation_set in range(5):

        if test_set == validation_set:
            continue

        X_val, y_val = FileToTensor(data_path, validation_set, BA_EL, MHC_dict)
        X_test, y_test = FileToTensor(data_path, test_set, BA_EL, MHC_dict)

        #data loading:
        for i in range(5):
            if i != test_set and i != validation_set:
                print('lol')


        print(test_set, validation_set)

import pandas as pd
filepath = data_path
Partition = test_set
mapping_dict = MHC_dict
