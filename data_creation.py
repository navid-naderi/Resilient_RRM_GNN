import numpy as np
from collections import defaultdict
from tqdm import tqdm
from channel_gen import create_channel_matrix_over_time
import torch
from utils import convert_channels, calc_rates, Data_modTxIndex
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse
from baselines import ITLinQ, wmmse

# create PyTorch Geometric datasets and dataloaders
def create_dataset(args, num_samples, P_max, noise_var):#m, n, T, num_samples, warmup_steps):

    m = args.m
    n = args.n
    T = args.T
    warmup_steps = args.warmup_steps
    beta_rate = args.beta_rate

    # create datasets
    H = defaultdict(list)
    H_l = defaultdict(list)
    A = dict() # reshaped instantaneous weighted adjacency matrix
    A_l = dict() # reshaped large-scale weighted adjacency matrix
    associations = dict()
    locTx_all = defaultdict(list)
    locRx_all = defaultdict(list)
    for phase in num_samples:
        for _ in tqdm(range(num_samples[phase])):
            h, h_l, locTx, locRx = create_channel_matrix_over_time(args)
            H[phase].append(h)
            H_l[phase].append(h_l)
            locTx_all[phase].append(locTx)
            locRx_all[phase].append(locRx)
        H[phase] = np.stack(H[phase])
        H_l[phase] = np.stack(H_l[phase])
        associations[phase] = (H_l[phase] == np.max(H_l[phase], axis=1, keepdims=True))

        # reshape the channel matrices to get the weighted adjacency matrices as the basis for GNNs
        # instantaneous channel
        A[phase] = np.zeros((num_samples[phase], m+n, m+n, T))
        A[phase][:, :m, m:, :] = np.expand_dims(associations[phase], 3) * H[phase]
        A[phase][:, m:, :m, :] = np.transpose((np.expand_dims((1 - associations[phase]), 3) * H[phase]), (0, 2, 1, 3))
        # long-term channel
        A_l[phase] = np.zeros((num_samples[phase], m+n, m+n))
        A_l[phase][:, :m, m:] = associations[phase] * H_l[phase]
        A_l[phase][:, m:, :m] = np.transpose(((1 - associations[phase]) * H_l[phase]), (0, 2, 1))

    # create PyG graphs
    data_list = defaultdict(list)
    y = torch.ones(n, 1)

    for phase in H:
        for i in tqdm(range(num_samples[phase])):
            a, a_l, h, h_l = A[phase][i], A_l[phase][i], H[phase][i], H_l[phase][i]

            serving_transmitters = torch.Tensor(np.argmax(h_l, axis=0)).to(torch.long)

            weighted_adjacency = torch.Tensor(a).unsqueeze(0)
            gg = ((1 - associations[phase][i]) * h_l)[serving_transmitters] + np.eye(n) * h_l[serving_transmitters]
            normalized_log_channel_matrix = convert_channels(gg, P_max, noise_var)
            edge_index_l, edge_weight_l = from_scipy_sparse_matrix(sparse.csr_matrix(normalized_log_channel_matrix))
            all_edge_indices = []
            all_edge_weights = []
            long_term_avg_rates = 0
            for t in range(T):

                if t < warmup_steps:
                      p = P_max * torch.ones(m)
                      gamma = torch.zeros(n)
                      selected_rxs = []
                      for tx in range(m):
                          associated_receivers = np.where(weighted_adjacency[0, tx , m:, 0].detach().cpu().numpy() > 0)[0]
                          selected_receiver = associated_receivers[t % len(associated_receivers)]
                          selected_rxs.append(selected_receiver)
                      selected_rxs = np.array(selected_rxs)
                      gamma[selected_rxs] = 1
                      sampled_gamma = gamma
                      rates = calc_rates(p, sampled_gamma, weighted_adjacency[:, :, :, t], noise_var)
                      long_term_avg_rates = \
                        (1 - beta_rate) * long_term_avg_rates + beta_rate * rates.detach()

                else:
                    gg = ((1 - associations[phase][i]) * h[:, :, t])[serving_transmitters] + np.eye(n) * h[:, :, t][serving_transmitters]
                    normalized_log_channel_matrix = convert_channels(gg, P_max, noise_var)
                    edge_index_t, edge_weights = from_scipy_sparse_matrix(sparse.csr_matrix(normalized_log_channel_matrix))
                    all_edge_indices.append(edge_index_t)
                    all_edge_weights.append(edge_weights.float())

            data_list[phase].append(Data_modTxIndex( y=y,
                                                     edge_index_l=edge_index_l,
                                                     edge_weight_l=edge_weight_l.float(),
                                                     edge_index=all_edge_indices,
                                                     edge_weight=all_edge_weights,
                                                     weighted_adjacency=weighted_adjacency,
                                                     transmitters_index=serving_transmitters,
                                                     init_long_term_avg_rates=long_term_avg_rates,
                                                     num_nodes=n,
                                                     m=m,
                                                    )
                                  )

    # calculate baseline rates for val/test phases

    baseline_rates = defaultdict(list)
    for phase in ['test']:

        for alg in ['ITLinQ', 'FR', 'WMMSE']:
            print(alg)
            for i in tqdm(range(len(H[phase]))):
                a = A[phase][i]
                weighted_avg_rates = 1e-10 * np.ones(n)
                mean_rates = np.zeros(n)
                for t in range(T):
                    current_S = P_max * np.sum(a[:m, m:, t], axis=0)
                    current_I = P_max * np.sum(a[m:, :m, t], axis=1)
                    current_rates = np.log2(1 + current_S / (noise_var + current_I))
                    PFs = current_rates / weighted_avg_rates
                    selected_rxs = []
                    for tx in range(m):
                        if t < warmup_steps:
                            associated_receivers = np.where(associations[phase][i][tx, :] > 0)[0]
                            selected_receiver = associated_receivers[t % len(associated_receivers)]
                        else:
                            masked_PFs = (associations[phase][i][tx, :] > 0) * PFs
                            selected_receiver = np.argmax(masked_PFs)
                        selected_rxs.append(selected_receiver)
                    h = H[phase][i][:, selected_rxs, t]

                    if t < warmup_steps:
                        p = P_max * np.ones(m)
                    else:
                        if alg == 'ITLinQ':
                            p = ITLinQ(h, P_max, noise_var, PFs[selected_rxs])
                        elif alg == 'WMMSE':
                            p = wmmse(np.expand_dims(h, 0), P_max, noise_var)[0]
                        elif alg == 'FR':
                            p = P_max * np.ones(m)
                        else:
                            raise Exception

                    h_power_adjusted = np.expand_dims(p, 1) * h
                    S = np.diag(h_power_adjusted)
                    I = np.sum(h_power_adjusted, axis=0) - S
                    rates = np.zeros(n)
                    rates[selected_rxs] = np.log2(1 + S / (noise_var + I))
                    weighted_avg_rates = (1 - beta_rate) * weighted_avg_rates + beta_rate * rates
                    if t >= warmup_steps:
                        mean_rates += rates
                mean_rates /= (T - warmup_steps)
                baseline_rates[phase, alg].extend(mean_rates.tolist())

    return baseline_rates, data_list, locTx_all, locRx_all
