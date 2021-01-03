import torch
import dgl
from torch.utils.data import DataLoader, TensorDataset
from dgl import DGLGraph
from utils import drnl_node_labeling


class SEALDataLoader(object):
    """
    Data Loader of SEAL
    Attributes:
        eids(Tensor): The edge set in graph. [2,N]
        labels(Tensor): Tensor of labels
        batch_size(int): size of batch
        sampler(SEALSampler): sampler
        num_workers(int):
        shuffle(bool):
        drop_last(bool):
        pin_memory(bool):
    """

    def __init__(self, eids, labels, batch_size, sampler, num_workers=0, shuffle=True,
                 drop_last=False, pin_memory=False):
        self.sampler = sampler
        dataset = TensorDataset(eids, labels)
        self.dataloader = DataLoader(dataset=dataset, collate_fn=self._collate, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=num_workers,
                                     drop_last=drop_last, pin_memory=pin_memory)

    def _collate(self, batch):
        batch_graphs = dgl.batch(self.sampler.sample(batch[0]))
        batch_labels = batch[1]
        return batch_graphs, batch_labels

    def __len__(self):
        """Return the number of batches of the data loader."""
        return len(self.dataloader)

    def __iter__(self):
        return iter(self.dataloader)


class SEALSampler(object):
    """
    Sampler for SEAL in paper(no-block version)
    The  strategy is to sample all the k-hop neighbors around the two target nodes.
    Attributes:
        graph(DGLGraph): The graph
        hop(int): num of hop
        use_node_label(bool, optional): set 'True' to generate node labeling for each subgraph
    """

    def __init__(self, graph, hop, use_node_label=True):
        self.graph = graph
        self.hop = hop
        self.use_node_label = use_node_label

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

        if self.use_node_label:
            z = drnl_node_labeling(subgraph, int(target_nodes[0]), int(target_nodes[1]))
            subgraph.ndata['z'] = z

        return subgraph

    def sample(self, batch):
        subgraphs = []
        for pair_nodes in batch:
            subgraphs.append(self.__sample_subgraph__(pair_nodes))
        return subgraphs
