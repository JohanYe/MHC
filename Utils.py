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
    :return: X_tensor, y_tensor
    """
    if type(Partition) is not list:
        Partition = [Partition]

    colnames = ['Peptide', 'BindingAffinity', 'MHC']
    X = pd.DataFrame(columns=colnames)

    for i in Partition:
        complete_path = filepath + 'c00' + str(i) + "_" + BA_EL.lower()
        tmp = pd.read_csv(complete_path, header=None, sep='\s+', names=colnames)
        tmp['Peptide'] = tmp['Peptide'].astype(str)
        tmp['MHC'] = tmp['MHC'].astype(str)
        tmp['MHC'] = tmp['MHC'].map(mapping_dict)
        X = X.append(tmp)

    return X[['Peptide', 'MHC']], X.BindingAffinity
