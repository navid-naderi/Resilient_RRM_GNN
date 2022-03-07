import torch_geometric
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import random
import time
import pylab as pl
from IPython import display
import numpy.matlib
import os
from channel_gen import create_channel_matrix_over_time
from utils import calc_rates, convert_channels, Data_modTxIndex, WirelessDataset, sample_selected_receivers, main_gnn
from data_creation import create_dataset
import argparse

folder_path = './'

def parse_option():

    parser = argparse.ArgumentParser('Resilient radio resource management')

    parser.add_argument('--m', type=int, default=8, help='Number of transmitters for evalation')
    parser.add_argument('--n', type=int, default=40, help='Number of receivers for evalation')
    parser.add_argument('--m_train', type=int, default=8, help='Number of transmitters for the trained model')
    parser.add_argument('--n_train', type=int, default=40, help='Number of receivers for the trained model')
    parser.add_argument('--T', type=int, default=200, help='Number of time slots for each configuration')
    parser.add_argument('--num_train_samples', type=int, default=256, help='Total number of training samples')
    parser.add_argument('--num_val_samples', type=int, default=128, help='Total number of validation samples')
    parser.add_argument('--num_test_samples', type=int, default=128, help='Total number of test samples')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[64] * 2, help='List of GNN hidden layer sizes')
    parser.add_argument('--beta_rate', type=float, default=5e-2, help='Exponential moving average parameter for receiver rates')
    parser.add_argument('--R', type=float, default=500, help='Network area side length')
    parser.add_argument('--min_D_TxTx', type=float, default=35, help='Minimum Tx-Tx distance')
    parser.add_argument('--min_D_TxRx', type=float, default=10, help='Minimum Tx-Rx distance')
    parser.add_argument('--shadowing', type=float, default=7, help='Shadowing standard deviation')
    parser.add_argument('--speed', type=float, default=1, help='Receiver speed (m/s)')
    parser.add_argument('--f_c', type=float, default=2.4e9, help='Carrier frequency (Hz)')
    parser.add_argument('--BW', type=float, default=1e7, help='Bandwidth (Hz)')
    parser.add_argument('--P_max_dBm', type=float, default=10, help='Maximum transmit power (dBm)')
    parser.add_argument('--Noise_PSD', type=float, default=-174, help='Noise power spectral density (dBm/Hz)')
    parser.add_argument('--tau', type=float, default=10, help='Temperature parameter for user selection distribution')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup time slots for each configuration')
    parser.add_argument('--random_seed', type=int, default=1234321, help='Random seed for reproducible results')

    opt = parser.parse_args()

    return opt

def main():

    args = parse_option()

    P_max = np.power(10, (args.P_max_dBm - 30) / 10) # Maximum transmit power in Watts
    noise_var = np.power(10, (args.Noise_PSD - 30 + 10 * np.log10(args.BW)) / 10) # Noise variance

    # set the random seed
    random_seed = args.random_seed
    os.environ['PYTHONHASHSEED']=str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # set the parameters
    m_train = args.m_train # number of transmitters for the trained model
    n_train = args.n_train # number of receivers for the trained model
    m = args.m # number of transmitters for evaluation
    n = args.n # number of receivers for evaluation
    
    T = args.T # number of time slots for each configuration
    num_samples = {'train': args.num_train_samples,
                   'val': args.num_val_samples,
                   'test': args.num_test_samples} # number of train/val/test samples
    batch_size = args.batch_size # batch size
    hidden_layers = args.hidden_layers # number of GNN features in different layers
    beta_rate = args.beta_rate # exponential moving average parameter for receiver rates
#     alpha_slack = args.alpha_slack # regularization coefficient for the slack norm in the objective function
    warmup_steps = args.warmup_steps # number of warmup time slots for each configuration

    assert warmup_steps >= n

    # create/load the evaluation dataset
    eval_experiment_name = 'm{}_n{}_T{}_train{}_val{}_test{}'.format(m, n, T, num_samples['train'], num_samples['val'], num_samples['test'])
    path = folder_path + 'data/data_{}.json'.format(eval_experiment_name)
    os.makedirs(folder_path + 'data', exist_ok=True)

    if os.path.exists(path):
        baseline_rates, data_list, locTx_all, locRx_all = torch.load(path)
    else:
        baseline_rates, data_list, locTx_all, locRx_all = create_dataset(args, num_samples, P_max, noise_var)
        torch.save([baseline_rates, data_list, locTx_all, locRx_all], path)

    # create the dataloaders
    loader = {}
    for phase in data_list:
        loader[phase] = DataLoader(WirelessDataset(data_list[phase]), batch_size=batch_size, shuffle=(phase == 'train'))

    # set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # initiate the parameters
    main_GNN = main_gnn([1] + hidden_layers, P_max, args.tau).to(device)
    
    # load the trained model
    train_experiment_name = 'm{}_n{}_T{}_train{}_val{}_test{}'.format(m_train, n_train, T, num_samples['train'], num_samples['val'], num_samples['test'])
    try:
        main_GNN.load_state_dict(torch.load(folder_path + 'results/models/{}.pt'.format(train_experiment_name)))
    except:
        raise Exception('Trained model {} not found!'.format(train_experiment_name))
    main_GNN.eval()
    
    # create folders to save model and results
    os.makedirs(folder_path + 'results/raw_results/eval', exist_ok=True)
    phase = 'test'

    all_variables = defaultdict(list)
    graph_index_start = 0
    for data, batch_idx in loader[phase]:

        data = data.to(device)

        y, edge_index_l, edge_weight_l, edge_index, edge_weight, a, init_long_term_avg_rates, transmitters_index = \
            data.y, data.edge_index_l, data.edge_weight_l, data.edge_index, data.edge_weight, \
            data.weighted_adjacency, data.init_long_term_avg_rates, data.transmitters_index

        with torch.set_grad_enabled(False):

            all_rates = []
            avg_rates = []
            min_rates = []
            sum_log_rate = []
            all_gammas = []
            all_sampled_gammas = []
            all_baselines = []

            # set initial node features to proportional-fairness (PF) ratios of the receivers
            initial_node_features_list = [] # list of initial node features (needed to trace the gradients)
            norm_init_long_term_avg_rates = torch.norm(init_long_term_avg_rates.view(-1, n), dim=1, keepdim=True).repeat(1, n).view(-1, 1)
            initial_node_features_list.append(init_long_term_avg_rates)
            unnormalized_long_term_avg_rate_list = [init_long_term_avg_rates.view(-1)]

            # pass the instantaneous fading arrays at each step into the main GNN to get RRM decisions
            for t in range(T-warmup_steps):

                # calculate receiver nominal rates (for PF ratio derivation)
                num_graphs = data.num_graphs
                nominal_rates = calc_rates(torch.ones(num_graphs * m).to(device), torch.ones_like(y), a[:, :, :, t], noise_var)

                # derive the RRM decisions
                p, gamma = main_GNN(nominal_rates / initial_node_features_list[-1], edge_index[t], edge_weight[t], transmitters_index)

                # select receivers based on sampling from the gamma distribution
                sampled_gamma = sample_selected_receivers(gamma, transmitters_index)

                # calculate the rates
                rates = calc_rates(p, sampled_gamma, a[:, :, :, t], noise_var)

                # update receiver initial node features for time t+1 to include exponential moving-average rates
                unnormalized_long_term_avg_rate_list_next_step = unnormalized_long_term_avg_rate_list[-1].clone().detach()

                unnormalized_long_term_avg_rate_list_next_step = \
                    (1 - beta_rate) * unnormalized_long_term_avg_rate_list_next_step + beta_rate * rates.detach().view(-1)
                norm_long_term_avg_rates = torch.norm(unnormalized_long_term_avg_rate_list_next_step.view(-1, n), dim=1, keepdim=True).repeat(1, n).view(-1)
                initial_node_features_next_step = unnormalized_long_term_avg_rate_list_next_step

                initial_node_features_list.append(initial_node_features_next_step.unsqueeze(1))
                unnormalized_long_term_avg_rate_list.append(unnormalized_long_term_avg_rate_list_next_step)

                # save the results
                all_rates.append(rates)
                all_gammas.append(gamma)
                all_sampled_gammas.append(sampled_gamma)

            # collect results from all time steps
            all_rates = torch.stack(all_rates, dim=0)

            avg_rates = torch.mean(all_rates, dim=0) # ergodic average rates
            
        # save the results within the epoch
        all_variables['rate'].extend(avg_rates.detach().cpu().numpy().tolist())

    rate_mean = np.mean(all_variables['rate'])
    rate_5th_percentile = np.percentile(all_variables['rate'], 5)

    # save the results
    torch.save([rate_mean, rate_5th_percentile], folder_path + 'results/raw_results/eval/{}_{}.json'.format(eval_experiment_name, train_experiment_name))

if __name__ == '__main__':
    main()            