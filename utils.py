import numpy as np
from numpy import linalg as LA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import LEConv
from torch_geometric.utils import softmax
from torch_scatter import scatter
from collections import Counter, OrderedDict

def calc_rates(p, gamma, h, noise_var):
    """
    calculate rates for a batch of b networks, each with m transmitters and n recievers
    inputs:
        p: bm x 1 tensor containing transmit power levels
        gamma: bn x 1 tensor containing user scheduling decisions
        h: b x (m+n) x (m+n) weighted adjacency matrix containing instantaneous channel gains
        noise_var: scalar indicating noise variance
        training: boolean variable indicating whether the models are being trained or not; during evaluation,
        entries of gamma are forced to be integers to satisfy hard user scheudling constraints

    output:
        rates: bn x 1 tensor containing user rates
    """
    b = h.shape[0]
    p = p.view(b, -1, 1)
    gamma = gamma.view(b, -1, 1)
    m = p.shape[1]

    combined_p_gamma = torch.bmm(p, torch.transpose(gamma, 1, 2))
    signal = torch.sum(combined_p_gamma * h[:, :m, m:], dim=1)
    interference = torch.sum(combined_p_gamma * torch.transpose(h[:, m:, :m], 1, 2), dim=1)

    rates = torch.log2(1 + signal / (noise_var + interference)).view(-1, 1)

    return rates

def convert_channels(a, P_max, noise_var):
    a_flattened = a[a > 0]
    a_flattened_log = np.log(P_max * a_flattened / noise_var)
    a_norm = LA.norm(a_flattened_log)
    a_log = np.log(P_max * a / noise_var)
    a_log[a == 0] = 0
    return a_log / a_norm


class Data_modTxIndex(Data):
    def __init__(self,
                 y=None,
                 edge_index_l=None,
                 edge_weight_l=None,
                 edge_index=None,
                 edge_weight=None,
                 weighted_adjacency=None,
                 transmitters_index=None,
                 init_long_term_avg_rates=None,
                 num_nodes=None,
                 m=None):
        super().__init__()
        self.y = y
        self.edge_index_l = edge_index_l
        self.edge_weight_l = edge_weight_l
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.weighted_adjacency = weighted_adjacency
        self.transmitters_index = transmitters_index
        self.init_long_term_avg_rates = init_long_term_avg_rates
        self.num_nodes = num_nodes
        self.m = m

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'transmitters_index':
            return self.m
        else:
            return super().__inc__(key, value, *args, **kwargs)

class WirelessDataset(Dataset):
    def __init__(self, data_list):
        super().__init__(None, None, None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx], idx

# backbone GNN class
class gnn_backbone(torch.nn.Module):
    def __init__(self, num_features_list):
        super(gnn_backbone, self).__init__()
        num_layers = len(num_features_list)
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(LEConv(num_features_list[i], num_features_list[i + 1]))

    def forward(self, y, edge_index, edge_weight):
        for i, layer in enumerate(self.layers):
            y = layer(y, edge_index=edge_index, edge_weight=edge_weight)
            y = F.leaky_relu(y)

        return y

class main_gnn(torch.nn.Module):
    def __init__(self, num_features_list, P_max, tau):
        super(main_gnn, self).__init__()
        self.gnn_backbone = gnn_backbone(num_features_list)
        self.b_p = nn.Linear(num_features_list[-1], 1, bias=False)
        self.b_gamma = nn.Linear(num_features_list[-1], 1, bias=False)
        self.P_max = P_max
        self.tau = tau

    def forward(self, y, edge_index, edge_weight, transmitters_index):
        y = self.gnn_backbone(y, edge_index, edge_weight) # derive node embeddings
        gamma = softmax(self.b_gamma(y) / self.tau, transmitters_index) # soft scheduling decisions during training
        Tx_embeddings = scatter(y, transmitters_index, dim=0, reduce='mean')
        p = self.P_max * torch.sigmoid(self.b_p(Tx_embeddings)) # derive power levels for transmitters

        return p, gamma

def sample_selected_receivers(gamma, transmitters_index):
    """
    sample selected receivers from the user distribution given by gamma
    """
    sampled_gamma = torch.zeros_like(gamma.detach()).to(gamma.device)
    g = gamma.detach().squeeze(1).cpu().numpy()
    t = transmitters_index.detach().cpu().numpy()

    counter = OrderedDict(sorted(Counter(t).items()))
    max_num_associated_users = np.max(list(counter.values()))
    num_Txs = len(counter)
    index_increment = np.cumsum([0] + list(counter.values())[:-1]) #max_num_associated_users * torch.arange(num_Txs)

    g_sorted = g[np.argsort(t)]

    g_extended = np.zeros(num_Txs * max_num_associated_users)
    non_zero_positions = np.concatenate([i * max_num_associated_users + np.arange(num_users) for i, num_users in enumerate(list(counter.values()))])

    g_extended[non_zero_positions] = g_sorted
    g_extended = torch.tensor(g_extended.reshape(num_Txs, -1))
    selected_rxs = g_extended.multinomial(1).squeeze()
    selected_rxs_absolute = selected_rxs + index_increment
    selected_rxs_absolute_original_indices = np.argsort(t)[selected_rxs_absolute]
    sampled_gamma[selected_rxs_absolute_original_indices] = 1
    return sampled_gamma
