from scipy.sparse.csgraph import shortest_path
import numpy as np
import torch
import argparse
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
import dgl
import pandas as pd


def parse_arguments():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='SEAL')
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--model', type=str, default='dgcnn')
    parser.add_argument('--gcn_type', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_units', type=int, default=32)
    parser.add_argument('--sort_k', type=int, default=30)
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--dropout', type=str, default=0.5)
    parser.add_argument('--hits_k', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--subsample_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--save_dir', type=str, default='./processed')
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


def coalesce_graph(graph, aggr_type='sum', copy_data=False):
    src, dst = graph.edges()
    graph_df = pd.DataFrame({'src': src, 'dst': dst})
    graph_df['edge_weight'] = graph.edata['edge_weight'].numpy()

    if aggr_type == 'sum':
        tmp = graph_df.groupby(['src', 'dst'])['edge_weight'].sum().reset_index()
    elif aggr_type == 'mean':
        tmp = graph_df.groupby(['src', 'dst'])['edge_weight'].mean().reset_index()
    else:
        raise ValueError("aggr type error")

    if copy_data:
        graph = dgl.to_simple(graph, copy_ndata=True, copy_edata=True)
    else:
        graph = dgl.to_simple(graph)

    src, dst = graph.edges()
    graph_df = pd.DataFrame({'src': src, 'dst': dst})
    graph_df = pd.merge(graph_df, tmp, how='left', on=['src', 'dst'])
    graph.edata['edge_weight'] = torch.from_numpy(graph_df['edge_weight'].values).unsqueeze(1)

    graph.edata.pop('count')
    return graph


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


def drnl_node_labeling(subgraph, src, dst):
    # Double-radius node labeling (DRNL).
    adj = subgraph.adj().to_dense().numpy()
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def evaluate_hits(name, pos_pred, neg_pred, K):
    """
    Compute hits
    Args:
        name(str): name of dataset
        pos_pred(Tensor): predict value of positive edges
        neg_pred(Tensor): predict value of negative edges
        K(int):

    Returns:


    """
    evaluator = Evaluator(name)
    evaluator.K = K
    hits = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })[f'hits@{K}']

    return hits
