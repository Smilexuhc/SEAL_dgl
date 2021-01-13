import torch
import dgl
from torch.utils.data import DataLoader, Dataset
from dgl import DGLGraph, NID
from utils import drnl_node_labeling
from dgl.dataloading.negative_sampler import Uniform
from dgl import add_self_loop
import numpy as np
import os.path as osp
from tqdm import tqdm


class GraphDataSet(Dataset):
    """
    GraphDataset for torch DataLoader
    """

    def __init__(self, graph_list, tensor1, tensor2):
        self.graph_list = graph_list
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, index):
        return (self.graph_list[index], self.tensor1[index], self.tensor2[index])


class SEALDataLoader(object):
    """
    Data Loader of SEAL
    Attributes:

        batch_size(int): size of batch

        num_workers(int):
        shuffle(bool):
        drop_last(bool):
        pin_memory(bool):
    """

    def __init__(self, graph_list, pair_nodes, labels, batch_size, num_workers=0, shuffle=True,
                 drop_last=False, pin_memory=False):
        dataset = GraphDataSet(graph_list, pair_nodes, labels)
        self.dataloader = DataLoader(dataset=dataset, collate_fn=self._collate, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=num_workers,
                                     drop_last=drop_last, pin_memory=pin_memory)

    def _collate(self, batch):
        batch_graphs = [item[0] for item in batch]
        batch_labels = [item[1] for item in batch]
        batch_pair_nodes = [item[2] for item in batch]

        batch_graphs = dgl.batch(batch_graphs)
        batch_labels = torch.stack(batch_labels)
        batch_pair_nodes = torch.stack(batch_pair_nodes)
        return batch_graphs, batch_pair_nodes, batch_labels

    def __len__(self):
        """Return the number of batches of the data loader."""
        return len(self.dataloader)

    def __iter__(self):
        return iter(self.dataloader)


class PosNegEdgesGenerator(object):
    """

    """

    def __init__(self, g, split_edge, neg_samples=1, subsample_ratio=1, return_type='combine'):
        self.neg_sampler = Uniform(neg_samples)
        self.subsample_ratio = subsample_ratio
        # self.random_seed = random_seed
        self.split_edge = split_edge
        self.g = g
        self.return_type = return_type

    def __call__(self, split_type):
        pos_edges = self.split_edge[split_type]['edge']
        if split_type == 'train':
            g = add_self_loop(self.g)
            eids = torch.from_numpy(np.arange(g.num_edges())).long()
            neg_edges = torch.stack(self.neg_sampler(g, eids), dim=1)
        else:
            neg_edges = self.split_edge[split_type]['edge_neg']
        pos_edges = self.subsample(pos_edges).long()
        neg_edges = self.subsample(neg_edges).long()

        if self.return_type == 'split':
            return pos_edges, torch.ones(pos_edges.size(0)), neg_edges, torch.zeros(neg_edges.size(0))
        elif self.return_type == 'combine':
            edges = torch.cat([pos_edges, neg_edges])
            labels = torch.cat([torch.ones(pos_edges.size(0)), torch.zeros(neg_edges.size(0))])
            return edges, labels

    def subsample(self, edges):

        num_pos = edges.size(0)
        perm = torch.randperm(num_pos)
        perm = perm[:int(self.subsample_ratio * num_pos)]
        edges = edges[perm]
        return edges


class SEALSampler(object):
    """
    Sampler for SEAL in paper(no-block version)
    The  strategy is to sample all the k-hop neighbors around the two target nodes.
    # todo: save subgraph
    Attributes:
        graph(DGLGraph): The graph
        hop(int): num of hop
        edges(Tensor):
        labels(Tensor):
        save_dir:
    """

    def __init__(self, graph, hop, prefix=None, save_dir=None, print_fn=print()):
        self.graph = graph
        self.hop = hop
        # self.use_node_label = use_node_label
        self.prefix = prefix or ''
        self.save_dir = save_dir
        self.print_fn = print_fn

    def sample_subgraph(self, target_nodes):
        """

        Args:
            target_nodes(Tensor): Tensor of two target nodes
        Returns:
            subgraph(DGLGraph): subgraph

        """
        sample_nodes = [target_nodes]
        frontiers = target_nodes

        for i in range(self.hop):
            frontiers = self.graph.in_edges(frontiers)[0]
            frontiers = torch.unique(frontiers)
            sample_nodes.append(frontiers)

        sample_nodes = torch.cat(sample_nodes)
        sample_nodes = torch.unique(sample_nodes)
        subgraph = dgl.node_subgraph(self.graph, sample_nodes)
        # Each node should have unique node id in the new subgraph
        # set as_tuple to prevent warning(torch 1.6.0)
        u_id = int(torch.nonzero(subgraph.ndata[NID] == int(target_nodes[0]), as_tuple=False))
        v_id = int(torch.nonzero(subgraph.ndata[NID] == int(target_nodes[1]), as_tuple=False))

        z = drnl_node_labeling(subgraph, u_id, v_id)
        subgraph.ndata['z'] = z

        return subgraph, (u_id, v_id)

    def sample(self, edges):
        subgraph_list = []
        pair_nodes_list = []

        for pair_nodes in tqdm(edges):
            subgraph, pair_nodes = self.sample_subgraph(pair_nodes)
            subgraph_list.append(subgraph)
            pair_nodes_list.append(pair_nodes)

        return subgraph_list, torch.LongTensor(pair_nodes_list)

    def __call__(self, split_type, edges=None, labels=None):
        file_name = '{}_{}_{}-hop.bin'.format(self.prefix, split_type, self.hop)
        path = osp.join(self.save_dir or '', file_name)
        if self.save_dir is not None and osp.exists(path):
            self.print_fn("Load preprocessed subgraph from {}".format(path))
            subgraph_list, data = dgl.load_graphs(path)
            labels = data['y']
            pair_nodes = data['pair_nodes']
        else:
            self.print_fn("Start sampling subgraph.")
            labels = labels
            subgraph_list, pair_nodes = self.sample(edges)
            if self.save_dir is not None:
                self.print_fn("Save preprocessed subgraph to {}".format(path))
                dgl.save_graphs(path, subgraph_list, {'y': labels, 'pair_nodes': pair_nodes})

        return subgraph_list, pair_nodes, labels
