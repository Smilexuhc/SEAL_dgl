import scipy.sparse as ssp
import os.path as osp
from dgl import NID
from scipy.sparse.csgraph import shortest_path
import numpy as np
import torch
import argparse
from ogb.linkproppred import DglLinkPropPredDataset
from dgl.dataloading.negative_sampler import Uniform
from dgl import add_self_loop


def parse_arguments():
    """
    Parse arguments
    TODO: add arguments
    """
    parser = argparse.ArgumentParser(description='SEAL')
    parser.add_argument("--dataset", type=str)
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    print(args)
    return args


def load_ogb_dataset(dataset):
    """
    Load OGB dataset
    Args:
        dataset(str): name of dataset (ogbl-collab, ogbl-ddi, ogbl-citation)

    Returns:

    """
    dataset = DglLinkPropPredDataset(name=dataset)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]
    return graph, split_edge


def load_mat(data_name, path='./data/'):
    all_idx = np.loadtxt(osp.join(path, data_name))
    max_idx = np.max(all_idx)

    mat = ssp.csc_matrix(
        (np.ones(len(all_idx)), (all_idx[:, 0], all_idx[:, 1])),
        shape=(max_idx + 1, max_idx + 1)
    )
    mat[all_idx[:, 1], all_idx[:, 0]] = 1

    return mat


# def add_self_loop(edge_index,num_nodes):
#     loop_index = torch.arange(0, num_nodes, dtype=torch.long,
#                               device=edge_index.device)
#     loop_index = loop_index.unsqueeze(0).repeat(2, 1)
#     edge_index = torch.cat([edge_index, loop_index], dim=1)
#     return edge_index


def generate_pos_neg_edges(split_type, split_edge, g, neg_samples=1, subsample_ratio=1):
    """
    Generate positive and negative edges for model.
    Args:
        split_type(str): 'train', 'valid' or 'test'
        split_edge(dict):
        g(DGLGraph): the graph
        neg_samples(int, optional): the number of negative edges sampled for each positive edges
        subsample_ratio(float, optional): the ratio of subsampling

    Returns:

    """
    pos_edges = split_edge[split_edge]['edge'].t()

    if split_type == 'train':
        g = add_self_loop(g)
        neg_sampler = Uniform(neg_samples)
        neg_edges = neg_sampler(g, g.edges())
    else:
        neg_edges = split_edge[split_type]['edge_neg'].t()
    np.random.seed(123)
    num_pos = pos_edges.size(1)
    perm = np.random.permutation(num_pos)
    perm = perm[:int(subsample_ratio * num_pos)]
    pos_edges = pos_edges[:, perm]
    # subsample for neg_edge
    np.random.seed(123)
    num_neg = neg_edges.size(1)
    perm = np.random.permutation(num_neg)
    perm = perm[:int(subsample_ratio * num_neg)]
    neg_edges = neg_edges[:, perm]
    return pos_edges, neg_edges


def add_val_edges_as_train_collab(graph, split_edge):
    """
    According to OGB, this dataset allows including validation links in training when all the hyperparameters are
    finalized using the validation set. Thus, you should first tune your hyperparameters
    without "--use_valedges_as_input", and then append "--use_valedges_as_input" to your final command
    when all hyperparameters are determined. See https://github.com/snap-stanford/ogb/issues/84
    Args:
        graph:
        split_edge:

    Returns:

    """
    raise NotImplementedError


def drnl_node_labeling(subgraph, u, v):
    """
    Double Radius Node labeling
    d = r(i,u)+r(i,v)
    label = 1+ min(r(i,u),r(i,v))+ (d//2)*(d//2+d%2-1)
    Extreme large graph may cause memory error.

    Args:
        subgraph(DGLGraph): The graph
        u(int): node id of one of target nodes
        v(int): node id of one of target nodes
    Returns:
        z(Tensor): node labeling tensor
    """
    u_id = int((subgraph.ndata[NID] == u).nonzero())  # Each node should have unique node id in the new subgraph
    v_id = int((subgraph.ndata[NID] == v).nonzero())
    adj = subgraph.adj().to_dense().numpy()

    dist_u = shortest_path(adj, directed=False, unweighted=True, indices=u_id)  # todo: dgl shortest_path
    dist_v = shortest_path(adj, directed=False, unweighted=True, indices=v_id)

    dist_sum = dist_u + dist_v
    dist_div_2, dist_mod_2 = dist_sum // 2, dist_sum % 2

    z = 1 + np.min(dist_u, dist_v) + dist_div_2 * (dist_div_2 + dist_mod_2 - 1)
    z[u_id] = 1
    z[v_id] = 1
    z[np.isnan(z)] = 0

    return torch.from_numpy(z).to(torch.long)
