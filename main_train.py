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

    parser.add_argument('--m', type=int, default=8, help='Number of transmitters')
    parser.add_argument('--n', type=int, default=40, help='Number of receivers')
    parser.add_argument('--T', type=int, default=200, help='Number of time slots for each configuration')
    parser.add_argument('--num_train_samples', type=int, default=256, help='Total number of training samples')
    parser.add_argument('--num_val_samples', type=int, default=128, help='Total number of validation samples')
    parser.add_argument('--num_test_samples', type=int, default=128, help='Total number of test samples')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=400, help='Number of training epochs')
    parser.add_argument('--lr_main', type=float, default=1e-3, help='Learning rate for main GNN parameters')
    parser.add_argument('--lr_slack', type=float, default=1, help='Learning rate for slack parameters')
    parser.add_argument('--lr_dual', type=float, default=1, help='Learning rate for dual parameters')
    parser.add_argument('--f_min', type=float, default=1, help='Minimum capacity constraint')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[64] * 2, help='List of GNN hidden layer sizes')
    parser.add_argument('--beta_rate', type=float, default=5e-2, help='Exponential moving average parameter for receiver rates')
    parser.add_argument('--alpha_slack', type=float, default=1e-2, help='Regularization coefficient for the slack norm in the objective function')
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
    m = args.m # number of transmitters
    n = args.n # number of receivers
    T = args.T # number of time slots for each configuration
    num_samples = {'train': args.num_train_samples,
                   'val': args.num_val_samples,
                   'test': args.num_test_samples} # number of train/val/test samples
    batch_size = args.batch_size # batch size
    hidden_layers = args.hidden_layers # number of GNN features in different layers
    num_epochs = args.num_epochs # number of epochs
    lr_main = args.lr_main # learning rate for main GNN
    lr_slack = args.lr_slack # learning rate for auxiliary GNN
    lr_dual = args.lr_dual # learning rate for dual GNN
    f_min = args.f_min # minimum capacity
    beta_rate = args.beta_rate # exponential moving average parameter for receiver rates
    alpha_slack = args.alpha_slack # regularization coefficient for the slack norm in the objective function
    warmup_steps = args.warmup_steps # number of warmup time slots for each configuration

    assert warmup_steps >= n

    # create/load the dataset
    experiment_name = 'm{}_n{}_T{}_train{}_val{}_test{}'.format(m, n, T, num_samples['train'], num_samples['val'], num_samples['test'])
    path = folder_path + 'data/data_{}.json'.format(experiment_name)
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
    mu_all = torch.zeros(num_samples['train'], n, requires_grad=True, device=device)
    z_all = torch.zeros(num_samples['train'], n, requires_grad=True, device=device)

    # create folders to save model and results
    os.makedirs(folder_path + 'results/raw_results', exist_ok=True)
    os.makedirs(folder_path + 'results/models', exist_ok=True)

    all_epoch_results = defaultdict(list)
    for epoch in tqdm(range(num_epochs)):

        for phase in ['train', 'val']:
            if phase == 'train':
                main_GNN.train()
            else:
                main_GNN.eval()

            all_variables = defaultdict(list)
            graph_index_start = 0
            for data, batch_idx in loader[phase]:

                main_GNN.zero_grad()

                data = data.to(device)

                y, edge_index_l, edge_weight_l, edge_index, edge_weight, a, init_long_term_avg_rates, transmitters_index = \
                    data.y, data.edge_index_l, data.edge_weight_l, data.edge_index, data.edge_weight, \
                    data.weighted_adjacency, data.init_long_term_avg_rates, data.transmitters_index

                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

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
                    all_gammas = torch.stack(all_gammas, dim=0)
                    all_sampled_gammas = torch.stack(all_sampled_gammas, dim=0)

                    avg_rates = torch.mean(all_rates, dim=0) # ergodic average rates
                    min_rates = torch.min(avg_rates.view(-1, n), dim=1)[0] # minimum rates per configuration
                    avg_gammas = torch.mean(all_sampled_gammas, dim=0)
                    min_gammas = torch.min(avg_gammas.view(-1, n), dim=1)[0] # minimum rates per configuration
                    sum_log_rate = torch.sum(torch.log(avg_rates).view(-1, n), dim=1) # sum-rate utility

                    if phase == 'train':

                        # calculate the Lagrangian

                        mu = mu_all[batch_idx].view(-1, 1)
                        z = z_all[batch_idx].view(-1, 1)

                        U = torch.sum(avg_rates.view(-1, n), dim=1) # sum-rate utility
                        Z_norm = (- alpha_slack / 2) * (torch.norm(z.view(-1, n), dim=1) ** 2)
                        f_min_constraint_term = - torch.sum((mu * (f_min - z - avg_rates)).view(-1, n), dim=1)

                        L = U + Z_norm + f_min_constraint_term

                        returns_for_PG = torch.sum(((1 + mu) * all_rates).view(T-warmup_steps, -1, n), dim=2).detach()
                        sampled_users_gammas = torch.prod((all_gammas ** all_sampled_gammas).view(T-warmup_steps, -1, n), dim=2)
                        gamma_policy_gradient_term = returns_for_PG.detach() * torch.log(1e-10 + sampled_users_gammas) # policy gradient term for user selection policy (gamma)ction policy (gamma)

                        # total objective used for backpropagation
                        L_total = (torch.mean(L) + torch.mean(gamma_policy_gradient_term))

                        # calculate the gradients
                        L_total.backward()

                        # perform gradient ascent/descent
                        with torch.no_grad():

                            # primal GNN parameters
                            for i, theta_main in enumerate(list(main_GNN.parameters())):
                                # theta_main += lr_main * torch.clamp(dtheta_main[i], min=-1, max=1)
                                if theta_main.grad is not None:
                                    # print('main', i)
                                    theta_main += lr_main * theta_main.grad

                            # slack and dual variables
                            z_all += lr_slack * z_all.grad
                            mu_all -= lr_dual * mu_all.grad

                            # ensure non-negativity
                            z_all.data.clamp_(0)
                            mu_all.data.clamp_(0)

                        # zero the gradients after updating
                        for theta_ in list(main_GNN.parameters()) + [mu_all, z_all]:
                            if theta_.grad is not None:
                                theta_.grad.zero_()

                # save the results within the epoch
                all_variables['z'].append(torch.mean((z)).item())
                all_variables['mu'].append(torch.mean(mu).item())
                all_variables['min_rate'].append(torch.mean(min_rates).item())
                all_variables['min_gamma'].append(torch.mean(min_gammas).item())
                all_variables['sum_log_rate'].append(torch.mean(sum_log_rate).item())
                all_variables['rate'].extend(avg_rates.detach().cpu().numpy().tolist())

            all_variables['mu_all'] = mu_all.detach().cpu().numpy()
            all_variables['z_all'] = z_all.detach().cpu().numpy()

            # save average epoch results
            for key in all_variables:
                if key == 'rate':
                    all_epoch_results[phase, 'rate_mean'].append(np.mean(all_variables['rate']))
                    all_epoch_results[phase, 'rate_5th_percentile'].append(np.percentile(all_variables['rate'], 5))
                elif '_all' in key:
                    all_epoch_results[phase, key] = all_variables[key]
                else:
                    all_epoch_results[phase, key].append(np.mean(all_variables[key]))

        # decay the learning rates every 50 epochs
        if (epoch + 1) % 50 == 0:
            lr_main *= 0.5
            lr_slack *= 0.5
            lr_dual *= 0.5

        # save the results
        torch.save(all_epoch_results, folder_path + 'results/raw_results/{}.json'.format(experiment_name))

        # save the model if best validation performance is achieved in this epoch
        if all_epoch_results['val', 'rate_5th_percentile'][-1] == np.max(all_epoch_results['val', 'rate_5th_percentile']):
            torch.save(main_GNN.state_dict(), folder_path + 'results/models/{}.pt'.format(experiment_name))

if __name__ == '__main__':
    main()            