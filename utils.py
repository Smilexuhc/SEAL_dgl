import scipy.sparse as ssp
import os.path as osp
from dgl import NID
from scipy.sparse.csgraph import shortest_path
import numpy as np
import torch
import argparse


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


def load_mat(data_name, path='./data/'):
    all_idx = np.loadtxt(osp.join(path, data_name))
    max_idx = np.max(all_idx)

    mat = ssp.csc_matrix(
        (np.ones(len(all_idx)), (all_idx[:, 0], all_idx[:, 1])),
        shape=(max_idx + 1, max_idx + 1)
    )
    mat[all_idx[:, 1], all_idx[:, 0]] = 1

    return mat


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

    dist_u = shortest_path(adj, directed=False, unweighted=True, indices=u_id)
    dist_v = shortest_path(adj, directed=False, unweighted=True, indices=v_id)

    dist_sum = dist_u + dist_v
    dist_div_2, dist_mod_2 = dist_sum // 2, dist_sum % 2

    z = 1 + np.min(dist_u, dist_v) + dist_div_2 * (dist_div_2 + dist_mod_2 - 1)
    z[u_id] = 1
    z[v_id] = 1
    z[np.isnan(z)] = 0

    return torch.from_numpy(z).to(torch.long)
