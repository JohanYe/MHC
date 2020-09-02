# Script to evaluate MHC encoding for EL/BA evaluation
# Johan Ziruo Ye
# Spring 2020

import os
from utils import *
from model import *
import time
import torch.optim as optim
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data', type=str, default="BA")
parser.add_argument('--resnet_output_file', type=str, default='model_output_resnet.txt')
parser.add_argument('--total_output_file', type=str, default='model_output_total.txt')
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--lstm', type=int, default=1)
parser.add_argument('--fucking_raw', type=bool, default=False)
parser.add_argument('--gauss', action='store_true')
parser.add_argument('--block_type', type=str, default='cabd')
parser.add_argument('--n_reslayers', type=int, default=5)
parser.add_argument('--lstm_nhidden', type=int, default=64)
parser.add_argument('--lstm_nlayers', type=int, default=2)
parser.add_argument('--rezero', type=bool, default=False)
parser.add_argument('--full_lstm', type=bool, default=False)
parser.add_argument('--vae', type=bool, default=False)
args = parser.parse_args()

# lazy workaround
args.lstm = True if args.lstm == 1 else False

BA_EL = args.data  # Expects BA or EL

# Data stuff
data_path = os.getcwd() + '/MHC_I/' + BA_EL.lower() + "_data/"
MHC = pd.read_csv(os.getcwd() + '/MHC_I/' + 'MHC_pseudo.dat', header=None, sep='\s+')
MHC_len = MHC[1].map(len).max()
MHC_dict = MHC.set_index(0).to_dict()[1]
All_data = {0, 1, 2, 3, 4}
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# Hyperparams:
torch.backends.cudnn.benchmark= True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Peptide_len = 15
n_epoch = args.n_epochs
batch_size = args.batch_size
lr = args.lr
patience = args.patience
train_epoch_loss, val_epoch_loss, test_epoch_loss = [], [], []
k = 0
exp_path, save_dir, figure_dir = generate_experiment_folders('./experiments', args)

file1 = save_dir + args.resnet_output_file
file2 = save_dir + args.total_output_file

outfile_resnet = open(file1, 'w')
outfile_resnet.write('split\tMHC\tPeptide\ty\ty_pred\tstd\n')
outfile_total = open(file2, 'w')
outfile_total.write('split\tMHC\tPeptide\ty\ty_pred\n')


def criterion(y, mu, std=None, normal_dist=True):
    if normal_dist:
        loss = -torch.distributions.Normal(loc=mu, scale=std).log_prob(y)
    else:
        loss = nn.functional.mse_loss(mu, y)
    return loss.mean()

for test_set in range(5):
    test_loader = torch.utils.data.DataLoader(MHC_dataset(data_path, test_set, BA_EL, MHC_dict, MHC_len),
                                              batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    for validation_set in range(5):

        t = time.process_time()
        if test_set == validation_set:
            continue
        k += 1
        best_val_MSE = np.inf

        # data loading: [N, Concat_length, Amino acids]
        train_set = list(All_data - set([test_set, validation_set]))
        train_loader = torch.utils.data.DataLoader(MHC_dataset(data_path, train_set, BA_EL, MHC_dict, MHC_len),
                                                   batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(MHC_dataset(data_path, validation_set, BA_EL, MHC_dict, MHC_len),
                                                 batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        if args.vae:
            VAE = VariationalAutoencoder().to(device)
            VAE_optimizer = optim.Adam(VAE.parameters(), lr=lr/10)
            model, optimizer = VAE.train_model(model=VAE,
                                         train_loader=train_loader,
                                         validation_loader=val_loader,
                                         optimizer=VAE_optimizer,
                                         save_dir=save_dir,
                                         crossvalsplit=k,
                                         n_epoch=10)

        net = ResidualNetwork(block_type=args.block_type, n_layers=args.n_reslayers, rezero=args.rezero).to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr)

        net, optimizer = train_epochs(args=args,
                                      model=net,
                                      loss_function=criterion,
                                      train_loader=train_loader,
                                      validation_loader=val_loader,
                                      optimizer=optimizer,
                                      save_dir=save_dir,
                                      model_name='resnet',
                                      crossvalsplit=k)

        performance_testing_print(
            data_path, test_set, BA_EL, MHC_dict, batch_size, MHC_len, Peptide_len, net, k, outfile_resnet)

        if args.lstm:
            # LSTM WITH FROZEN RESNET
            best_val_MSE = np.inf
            if args.fucking_raw:
                net2 = Resnet_Blosum_direct(block_type=args.block_type).to(device)
            else:
                net2 = Frozen_resnet(lstm_hidden=args.lstm_nhidden, lstm_layers=args.lstm_nlayers,
                                     full_lstm=args.full_lstm).to(device)
            optimizer2 = optim.Adam(net2.parameters(), lr=lr)

            net2, optimizer2 = train_epochs(args=args,
                                            model=net2,
                                            loss_function=criterion,
                                            train_loader=train_loader,
                                            validation_loader=val_loader,
                                            optimizer=optimizer,
                                            save_dir=save_dir,
                                            model_name='total',
                                            crossvalsplit=k,
                                            trained_model=net
                                            )

            performance_testing_print(
                data_path, test_set, BA_EL, MHC_dict,
                batch_size, MHC_len, Peptide_len, net, k, outfile_total,
                net2=net2, resnet=True)

outfile_resnet.close()
outfile_total.close()
