import scipy.sparse as ssp
import os.path as osp
from scipy.sparse.csgraph import shortest_path
import numpy as np
import torch
import argparse
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator


def parse_arguments():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='SEAL')
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--gcn_type', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_units', type=int, default=32)
    parser.add_argument('--pooling', type=str, default='center')
    parser.add_argument('--dropout', type=str, default=0.5)
    parser.add_argument('--hits_k', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--neg_samples', type=int, default=1)
    # parser.add_argument('--train_subsample_ratio', type=float, default=1.0)
    # parser.add_argument('--val_subsample_ratio', type=float, default=1.0)
    # parser.add_argument('--test_subsample_ratio', type=float, default=1.0)
    parser.add_argument('--subsample_ratio', type=float, default=1.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--random_seed', type=int, default=2021)
    args = parser.parse_args()

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


def drnl_node_labeling(subgraph, u_id, v_id):
    """
    Double Radius Node labeling
    d = r(i,u)+r(i,v)
    label = 1+ min(r(i,u),r(i,v))+ (d//2)*(d//2+d%2-1)
    Isolated nodes in subgraph will be set as zero.
    Extreme large graph may cause memory error.

    Args:
        subgraph(DGLGraph): The graph
        u_id(int): node id of one of target nodes in new subgraph
        v_id(int): node id of one of target nodes in new subgraph
    Returns:
        z(Tensor): node labeling tensor
    """

    adj = subgraph.adj().to_dense().numpy()

    dist_u, dist_v = shortest_path(adj, directed=False, unweighted=True, indices=(u_id, v_id))

    dist_u = torch.from_numpy(dist_u)
    dist_v = torch.from_numpy(dist_v)
    dist_sum = dist_u + dist_v
    dist_div_2, dist_mod_2 = dist_sum // 2, dist_sum % 2

    z = 1 + torch.min(dist_u, dist_v) + dist_div_2 * (dist_div_2 + dist_mod_2 - 1)
    z[u_id] = 1
    z[v_id] = 1
    z[torch.isnan(z)] = 0

    return z.long()


def evaluate_hits(name, pos_pred, neg_pred, K):
    evaluator = Evaluator(name)
    evaluator.K = K
    hits = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })[f'hits@{K}']

    return hits*100
