import torch
import pandas as pd

# class MHCData(torch.utils.data.Dataset):
#     def __init__(self, BA_EL, test, validation, data_path):
#
#         assert BA_EL.lower() in {"ba","el"}
#         self.test_set = test
#         self.validation_set = validation
#         self.data_path = data_path
#
#     def __getitem__(self, idx):
#         mhc_path =


def FileToTensor(filepath, Partition, BA_EL, mapping_dict):
    """
    Load file to tensor
    :param filepath: path to files
    :param Partition: Current partition
    :return:
    """
    complete_path = filepath + 'c00' + str(Partition) + "_" + BA_EL.lower()
    colnames = ['Peptide', 'BindingAffinity', 'MHC']
    X = pd.read_csv(complete_path, header=None, sep='\s+', names=colnames)
    X['Peptide'] = X['Peptide'].astype(str)
    X['MHC'] = X['MHC'].astype(str)
    X['MHC'] = X['MHC'].map(mapping_dict)
    return X[['Peptide', 'MHC']], X.BindingAffinity


