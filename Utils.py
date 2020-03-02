import torch
import pandas as pd
import numpy as np

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
        matrix.append(encoding_dict[i])
        idx += 1
    for i in range(int(max_len) - idx):
        matrix.append(encoding_dict['X'])

    return np.array(matrix)


def FileToTensor(filepath, Partition, BA_EL, mapping_dict):
    """
    Load file to tensor
    :param MHC_len: max len of MHC
    :param filepath: path to files
    :param Partition: Current partition
    :return: X_tensor, y_tensor
    """

    if type(Partition) is not list:
        Partition = [Partition]
    Peptide_len = 15 #  Is there a better way to do this?
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
        X = X.append(tmp,ignore_index=True)

    # modification of shape and conversion to np array
    y = X['BindingAffinity'].values
    X = X.drop('BindingAffinity', axis=1)
    Peptide_mat = np.stack(X.Peptide.apply(one_hot_encoding, encoding_dict=onehot_Blosum50, max_len=Peptide_len).values)
    MHC_mat = np.stack(X.MHC.apply(one_hot_encoding, encoding_dict=onehot_Blosum50, max_len=MHC_len).values)
    X = np.concatenate((Peptide_mat, MHC_mat), axis=1)

    return torch.from_numpy(X), torch.from_numpy(y)

nlf = pd.read_csv('https://raw.githubusercontent.com/dmnfarrell/epitopepredict/master/epitopepredict/mhcdata/NLF.csv',index_col=0)


