import torch
import pandas as pd
import numpy as np
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.shape[0], -1)


class PrintLayerShape(nn.Module):
    def __init__(self):
        super(PrintLayerShape, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

one_hot = {
    'A': np.array((1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    'R': np.array((0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    'N': np.array((0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    'D': np.array((0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    'C': np.array((0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    'Q': np.array((0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    'E': np.array((0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    'G': np.array((0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    'H': np.array((0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    'I': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    'L': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    'K': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
    'M': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0)),
    'F': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)),
    'P': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)),
    'S': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)),
    'T': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)),
    'W': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)),
    'Y': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)),
    'V': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0)),
    'X': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)) #  Placeholder for no AA
}


#  https://github.com/gifford-lab/DeepLigand/blob/master/data/onehot_first20BLOSUM50
onehot_Blosum50 = {'I': np.array((1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -4, -3, -4, -2, -3, -4, -4, -4, 5,
                      2, -3, 2, 0, -3, -3, -1, -3, -1, 4)),
       'L': np.array((0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -3, -4, -4, -2, -2, -3, -4, -3, 2,
                      5, -3, 3, 1, -4, -3, -1, -2, -1, 1)),
       'V': np.array((0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -4, -1, -3, -3, -4, -4, 4,
                      1, -3, 1, -1, -3, -2, 0, -3, -1, 5)),
       'F': np.array((0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, -3, -4, -5, -2, -4, -3, -4, -1, 0,
                      1, -4, 0, 8, -4, -3, -2, 1, 4, -1)),
       'M': np.array((0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -2, -4, -2, 0, -2, -3, -1, 2,
                      3, -2, 7, 0, -3, -2, -1, -1, 0, 1)),
       'C': np.array((
                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -4, -2, -4, 13, -3, -3, -3, -3, -2,
                     -2, -3, -2, -2, -4, -1, -1, -5, -3, -1)),
       'A': np.array((0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, -2, -1, -2, -1, -1, -1, 0, -2, -1,
                      -2, -1, -1, -3, -1, 1, 0, -3, -2, 0)),
       'G': np.array((0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3, 0, -1, -3, -2, -3, 8, -2, -4,
                      -4, -2, -3, -4, -2, 0, -2, -3, -3, -4)),
       'P': np.array((
                     0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -3, -2, -1, -4, -1, -1, -2, -2, -3,
                     -4, -1, -3, -4, 10, -1, -1, -4, -3, -3)),
       'T': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1, -1, -1, -2, -2, -1,
                      -1, -1, -1, -2, -1, 2, 5, -3, -2, 0)),
       'S': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, 0, -1, 0, -1, 0, -1, -3, -3,
                      0, -2, -3, -1, 5, 2, -4, -2, -2)),
       'Y': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, -2, -3, -3, -1, -2, -3, 2, -1,
                      -1, -2, 0, 4, -3, -2, -2, 2, 8, -1)),
       'W': np.array((
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -3, -3, -4, -5, -5, -1, -3, -3, -3, -3,
                     -2, -3, -1, 1, -4, -4, -3, 15, 2, -3)),
       'Q': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, -3, 7, 2, -2, 1, -3, -2,
                      2, 0, -4, -1, 0, -1, -1, -1, -3)),
       'N': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, -1, 7, 2, -2, 0, 0, 0, 1, -3, -4,
                      0, -2, -4, -2, 1, 0, -4, -2, -3)),
       'H': np.array((
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -2, 0, 1, -1, -3, 1, 0, -2, 10, -4, -3,
                     0, -1, -1, -2, -1, -2, -3, 2, -4)),
       'E': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 2, -3, 2, 6, -3, 0, -4, -3,
                      1, -2, -3, -1, -1, -1, -3, -2, -3)),
       'D': np.array((
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -2, -2, 2, 8, -4, 0, 2, -1, -1, -4, -4,
                     -1, -4, -5, -1, 0, -1, -5, -3, -4)),
       'K': np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 3, 0, -1, -3, 2, 1, -2, 0, -3, -3,
                      6, -2, -4, -1, 0, -1, -3, -2, -3)),
       'R': np.array((
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -2, 7, -1, -2, -4, 1, 0, -3, 0, -4, -3,
                     3, -2, -3, -3, -1, -1, -3, -1, -3)),
       'X': np.array((0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                      0.05, 0.05, 0.05, 0.05, -1, -1, -1, -1, -2, -1, -1, -2, -1, -1, -1, -1, -1, -2, -2, -1, 0, -3, -1,
                      -1)),
       'J': np.array((
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5,
                     -5, -5, -5, -5, -5, -5, -5, -5, -5, -5))
       }


def one_hot_encoding(list, encoding_dict, max_len):
    """ One-hot encoding applied on pandas array """
    matrix = []
    idx = 0
    for i in list:
        # OBS: upper is only due to data already being preprocessed + there are leftover 'm's
        try:
            matrix.append(encoding_dict[i.upper()])
        except:
            print(list,i)
        idx += 1
    for i in range(int(max_len) - idx):
        matrix.append(encoding_dict['X'])

    return np.array(matrix)

class MHC_dataset(torch.utils.data.Dataset):
    """
    test set input currently does nothing. May need to change depending on memory usage.
    """

    def __init__(self, filepath, Partition, BA_EL, mapping_dict, Peptide_len=15):

        self.filepath = filepath
        self.partition = Partition
        self.BA_EL = BA_EL
        self.mapping_dict = mapping_dict

        if type(Partition) is not list:
            Partition = [Partition]
        MHC_len = len(max(list(mapping_dict.keys()), key=len))
        colnames = ['Peptide', 'BindingAffinity', 'MHC']
        X = pd.DataFrame(columns=colnames)

        # reading files
        for i in Partition:
            complete_path = filepath + 'c00' + str(i) + "_" + BA_EL.lower()
            tmp = pd.read_csv(complete_path, header=None, sep='\s+', names=colnames)
            tmp['Peptide'] = tmp['Peptide'].astype(str)
            tmp['MHC'] = tmp['MHC'].astype(str)
            tmp['MHC'] = tmp['MHC'].map(mapping_dict)
            X = X.append(tmp, ignore_index=True)

        # modification of shape and conversion to np array
        self.y = torch.from_numpy(np.expand_dims(X['BindingAffinity'].values, 1))
        # self.X_original = X
        X = X.drop('BindingAffinity', axis=1)
        Peptide_mat = np.stack(
            X.Peptide.apply(one_hot_encoding, encoding_dict=onehot_Blosum50, max_len=Peptide_len).values)
        MHC_mat = np.stack(
            X.MHC.apply(one_hot_encoding, encoding_dict=onehot_Blosum50, max_len=MHC_len).values)
        self.X = torch.from_numpy(np.concatenate((Peptide_mat, MHC_mat), axis=1).astype(int))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        batch = self.X[idx]
        label = self.y[idx]
        return batch, label

def MHC_df(filepath, Partition, BA_EL, mapping_dict,Peptide_len=15):

    if type(Partition) is not list:
        Partition = [Partition]
    MHC_len = len(max(list(mapping_dict.keys()), key=len))
    colnames = ['Peptide', 'BindingAffinity', 'MHC']
    X = pd.DataFrame(columns=colnames)

    # reading files
    for i in Partition:
        complete_path = filepath + 'c00' + str(i) + "_" + BA_EL.lower()
        tmp = pd.read_csv(complete_path, header=None, sep='\s+', names=colnames)
        tmp['Peptide'] = tmp['Peptide'].astype(str)
        tmp['MHC'] = tmp['MHC'].astype(str)
        tmp['MHC'] = tmp['MHC'].map(mapping_dict)
        X = X.append(tmp, ignore_index=True)

    return X




def TxtToArray(data_path, Partition, BA_EL, mapping_dict):

    if type(Partition) is not list:
        Partition = [Partition]
    Peptide_len = 15  # Is there a better way to do this?
    MHC_len = len(max(list(mapping_dict.keys()), key=len))
    Peptides = []
    MHC = []

    for file in Partition:
        filepath = data_path + 'c00' + str(file) + "_" + BA_EL.lower()

        infile = open(filepath, 'r')
        for line in infile:
            line = line.strip().split()

            # Peptide
            sequence, idx = [], 0
            for AA in line[0]:
                sequence.append(onehot_Blosum50[AA.upper()])
                idx += 1
            for i in range(int(Peptide_len) - idx):
                sequence.append(onehot_Blosum50['X'])
            Peptides.append(np.array(sequence))

            # MHC
            sequence, idx = [], 0
            for AA in mapping_dict[line[2]]:
                sequence.append(onehot_Blosum50[AA.upper()])
                idx += 1
            for i in range(int(MHC_len) - idx):
                sequence.append(onehot_Blosum50['X'])
            MHC.append(np.array(sequence))

    Peptides = torch.from_numpy(np.array(Peptides))
    MHC = torch.from_numpy(np.array(MHC))
    print('{} are loaded'.format(Partition))
    return Peptides, MHC

def TxtToTensor(data_path, Partition, BA_EL, mapping_dict):

    if type(Partition) is not list:
        Partition = [Partition]
    Peptide_len = 15  # Is there a better way to do this?
    MHC_len = len(max(list(mapping_dict.keys()), key=len))
    Peptides = []
    MHC = []

    for file in Partition:
        filepath = data_path + 'c00' + str(file) + "_" + BA_EL.lower()

        infile = open(filepath, 'r')
        for line in infile:
            line = line.strip().split()

            # Peptide
            sequence, idx = [], 0
            for AA in line[0]:
                sequence.append(torch.from_numpy(onehot_Blosum50[AA.upper()]))
                idx += 1
            for i in range(int(Peptide_len) - idx):
                sequence.append(torch.from_numpy(onehot_Blosum50['X']))
            print(sequence)
            Peptides.append(torch.Tensor(sequence))

            # MHC
            sequence, idx = [], 0
            for AA in mapping_dict[line[2]]:
                sequence.append(torch.from_numpy(onehot_Blosum50[AA.upper()]))
                idx += 1
            for i in range(int(MHC_len) - idx):
                sequence.append(torch.from_numpy(onehot_Blosum50['X']))
            MHC.append(torch.Tensor((sequence)))

    X_train = torch.Tensor(Peptides.shape[0], 49, 40)
    Peptides = torch.Tensor(Peptides)
    MHC = torch.Tensor(MHC)
    X_train = torch.cat((Peptides, MHC),dim=1,out=X_train)
    print('{} are loaded'.format(Partition))
    return X_train, _

