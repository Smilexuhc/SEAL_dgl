import torch
import dgl
from torch.utils.data import DataLoader, TensorDataset, Dataset
from dgl import DGLGraph, NID
from utils import drnl_node_labeling
from dgl.dataloading.negative_sampler import Uniform
from dgl import add_self_loop
import numpy as np


class SEALDataLoader(object):
    """
    Data Loader of SEAL
    Attributes:
        dataset(TensorDataset): TensorDataset(edges, labels)
        batch_size(int): size of batch
        sampler(SEALSampler): sampler
        num_workers(int):
        shuffle(bool):
        drop_last(bool):
        pin_memory(bool):
    """

    def __init__(self, dataset, batch_size, sampler, num_workers=0, shuffle=True,
                 drop_last=False, pin_memory=False):
        self.sampler = sampler

        self.dataloader = DataLoader(dataset=dataset, collate_fn=self._collate, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=num_workers,
                                     drop_last=drop_last, pin_memory=pin_memory)

    def _collate(self, batch):
        # todo: adjust collate func

        edges = [item[0] for item in batch]
        batch_labels = [item[1] for item in batch]

        batch_graphs, batch_pair_nodes = self.sampler(edges)
        batch_labels = torch.stack(batch_labels)

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
        pos_edges = self.split_edge[self.split_edge]['edge']
        if split_type == 'train':
            g = add_self_loop(self.g)
            eids = torch.from_numpy(np.arange(g.num_edges())).long()
            neg_edges = torch.stack(self.neg_sampler(g, eids), dim=1)
        else:
            neg_edges = self.split_edge[split_type]['edge_neg']
        pos_edges = torch.from_numpy(self.subsample(pos_edges)).long()
        neg_edges = torch.from_numpy(self.subsample(neg_edges)).long()

        if self.return_type == 'split':
            return pos_edges, torch.ones(pos_edges.size(0)), neg_edges, torch.zeros(neg_edges.size(0))
        elif self.return_type == 'combine':
            edges = torch.cat([pos_edges, neg_edges])
            labels = torch.cat([torch.ones(pos_edges.size(0)), torch.zeros(neg_edges.size(0))])
            return TensorDataset(edges, labels)

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
    """

    def __init__(self, graph, hop):
        self.graph = graph
        self.hop = hop
        # self.use_node_label = use_node_label

    def __sample_subgraph__(self, target_nodes):
        """

        Args:
            target_nodes(Tensor): Tensor of two target nodes
        Returns:
            subgraph(DGLGraph): subgraph

        """
        sample_nodes = [target_nodes]
        frontiers = target_nodes

        for i in range(len(self.hop)):
            frontiers = self.graph.in_edges(frontiers)[0]
            frontiers = torch.unique(frontiers)
            sample_nodes.append(frontiers)

        sample_nodes = torch.cat(sample_nodes)
        subgraph = dgl.node_subgraph(self.graph, sample_nodes)
        # Each node should have unique node id in the new subgraph
        u_id = int((subgraph.ndata[NID] == int(target_nodes[0])).nonzero())
        v_id = int((subgraph.ndata[NID] == int(target_nodes[1])).nonzero())

        z = drnl_node_labeling(subgraph, u_id, v_id)
        subgraph.ndata['z'] = z

        return subgraph, (u_id, v_id)

    def __call__(self, edges):
        subgraph_list = []
        pair_nodes_list = []

        for pair_nodes in edges:
            subgraph, pair_nodes = self.__sample_subgraph__(pair_nodes)
            subgraph_list.append(subgraph)
            pair_nodes_list.append(pair_nodes_list)

        return dgl.batch(subgraph_list),  torch.LongTensor(pair_nodes_list)


