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
from multiprocessing import Pool
from copy import deepcopy


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

    def __init__(self, data, batch_size, num_workers=1, shuffle=True,
                 drop_last=False, pin_memory=False):

        if isinstance(data, dict):
            graph_list = data['graph_list']
            pair_nodes = data['pair_nodes']
            labels = data['labels']
            dataset = GraphDataSet(graph_list, pair_nodes, labels)
        elif isinstance(data, list):
            raise NotImplementedError
        else:
            raise ValueError("data type error")

        self.dataloader = DataLoader(dataset=dataset, collate_fn=self._collate, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=num_workers,
                                     drop_last=drop_last, pin_memory=pin_memory)

    def _collate(self, batch):
        # batch_graphs = [item[0] for item in batch]
        # batch_pair_nodes = [item[1] for item in batch]
        # batch_labels = [item[2] for item in batch]

        batch_graphs, batch_pair_nodes, batch_labels = map(list, zip(*batch))

        batch_graphs = dgl.batch(batch_graphs)
        batch_pair_nodes = torch.stack(batch_pair_nodes)
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

    def __init__(self, g, split_edge, neg_samples=1, subsample_ratio=1, shuffle=True, return_type='combine'):
        self.neg_sampler = Uniform(neg_samples)
        self.subsample_ratio = subsample_ratio
        # self.random_seed = random_seed
        self.split_edge = split_edge
        self.g = g
        self.return_type = return_type
        self.shuffle = shuffle

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
            if self.shuffle:
                perm = torch.randperm(edges.size(0))
                edges = edges[perm]
                labels = labels[perm]
            return edges, labels

    def subsample(self, edges):

        num_pos = edges.size(0)
        perm = torch.randperm(num_pos)
        perm = perm[:int(self.subsample_ratio * num_pos)]
        edges = edges[perm]
        return edges


class EdgeDataSet(Dataset):
    """
    Assistant Dataset for speeding up the SEALSampler
    """

    def __init__(self, edges, transform):
        self.edges = edges
        self.transform = transform

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index):
        graphs, pair_nodes = self.transform(self.edges[index])
        return (graphs, torch.LongTensor(pair_nodes))


class SEALSampler(object):
    """
    Sampler for SEAL in paper(no-block version)
    The  strategy is to sample all the k-hop neighbors around the two target nodes.
    # todo: save subgraph
    Attributes:
        graph(DGLGraph): The graph
        hop(int): num of hop
        save_dir:
    """

    def __init__(self, graph, hop=1, prefix=None, num_workers=32, save_dir=None,
                 num_parts=1, print_fn=print()):
        self.graph = graph
        self.hop = hop
        # self.use_node_label = use_node_label
        # self.save_dir = save_dir
        self.print_fn = print_fn
        self.num_workers = num_workers
        self.num_parts = num_parts

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

    def _collate(self, items):

        return dgl.batch([item[0] for item in items]), torch.stack([item[1] for item in items])

    def sample(self, edges):
        subgraph_list = []
        pair_nodes_list = []
        edge_dataset = EdgeDataSet(edges, transform=self.sample_subgraph)

        sampler = DataLoader(edge_dataset, batch_size=3 * self.num_workers, num_workers=self.num_workers,
                             shuffle=False, collate_fn=self._collate)
        for subgraph, pair_nodes in tqdm(sampler, ncols=70):
            subgraph = dgl.unbatch(subgraph)
            pair_nodes_copy = deepcopy(pair_nodes)
            del pair_nodes
            subgraph_list += subgraph
            pair_nodes_list.append(pair_nodes_copy)

        return subgraph_list, torch.cat(pair_nodes_list)

    def __call__(self, split_type, path, edges=None, labels=None):

        if split_type == 'train':
            num_parts = self.num_parts
        else:
            num_parts = 1

        # if num_parts != 1:
        #     path = ['{}_{}_{}-hop-part{}.bin'.format(self.prefix, split_type, self.hop, i) for i in
        #             range(num_parts)]
        #     path = [osp.join(self.save_dir or '', p) for p in path]
        # else:
        #     path = [osp.join(self.save_dir or '', '{}_{}_{}-hop.bin'.format(self.prefix, split_type, self.hop))]

        self.print_fn("Start sampling subgraph.")
        self.print_fn('Using {} workers in sampling job.'.format(self.num_workers))

        if num_parts == 1:
            data = dict()
        batch_size = len(edges) // num_parts + 1
        for i in range(num_parts):

            if osp.exists(path[i]):
                self.print_fn('Part {} exists.'.format(i))
            else:

                batch_labels = labels[i * batch_size:i * batch_size + batch_size]
                subgraph_list, pair_nodes = self.sample(edges[i * batch_size:i * batch_size + batch_size])

                dgl.save_graphs(path[i], subgraph_list, {'labels': batch_labels, 'pair_nodes': pair_nodes})
                if num_parts == 1:
                    data['graph_list'] = subgraph_list
                    data['pair_nodes'] = pair_nodes
                    data['labels'] = labels
        self.print_fn("Save preprocessed subgraph to {}".format(path))

        if num_parts == 1:
            return data
        else:
            return path


class ParallelSEALSampler(object):
    """
    Sampler for SEAL in paper(no-block version)
    The  strategy is to sample all the k-hop neighbors around the two target nodes.
    Exist bugs
    Attributes:
        graph(DGLGraph): The graph
        hop(int): num of hop
        prefix(str):
        num_workers(int):
        save_dir(str):
    """

    def __init__(self, graph, hop, prefix=None, num_workers=1, save_dir=None, print_fn=print):
        self.graph = graph
        self.hop = hop
        # self.use_node_label = use_node_label
        self.prefix = prefix or ''
        self.save_dir = save_dir
        self.print_fn = print_fn
        self.num_workers = num_workers

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

    def batch_sample(self, edges, labels):
        batch_graphs = []
        batch_pair_nodes = []

        for edge in edges:
            subgraph, pair_node = self.sample_subgraph(edge)
            batch_graphs.append(subgraph)
            batch_pair_nodes.append(pair_node)

        return (batch_graphs, torch.LongTensor(batch_pair_nodes), labels)

    def parallel_sample(self, edges, labels):
        self.print_fn("Input edges: {}".format(len(edges)))
        if self.hop >= 2:
            self.print_fn("Sampling hop {} >2, will take server hours.".format(self.hop))

        pool = Pool(self.num_workers)

        batch_size = 128
        num_batch = len(edges) // batch_size + 1
        workers = []
        for i in range(num_batch):
            batch_start = i * batch_size
            workers.append(
                pool.apply_async(func=self.batch_sample,
                                 kwds={'edges': edges[batch_start: batch_start + batch_size],
                                       'labels': labels[batch_start: batch_start + batch_size]}))

        graph_list = []
        pair_nodes_list = []
        labels_list = []
        for worker in tqdm(workers):
            batch_graph, batch_pair_nodes, batch_labels = worker.get()
            graph_list.append(batch_graph)
            pair_nodes_list.append(batch_pair_nodes)
            labels_list.append(batch_labels)

        pool.close()
        pool.join()

        return graph_list, torch.cat(pair_nodes_list), torch.cat(labels_list)

    def __call__(self, edges, split_type, labels):
        file_name = '{}_{}_{}-hop.bin'.format(self.prefix, split_type, self.hop)
        path = osp.join(self.save_dir or '', file_name)

        if self.save_dir is not None and osp.exists(path):
            self.print_fn("Load preprocessed subgraph from {}".format(path))
            subgraph_list, data = dgl.load_graphs(path)
            pair_nodes = data['pair_nodes']
            labels = data['y']
        else:
            self.print_fn("Start sampling subgraph.")
            subgraph_list, pair_nodes, labels = self.parallel_sample(edges, labels)
            if self.save_dir is not None:
                self.print_fn("Save preprocessed subgraph to {}".format(path))
                dgl.save_graphs(path, subgraph_list, {'y': labels, 'pair_nodes': pair_nodes})

        return subgraph_list, pair_nodes, labels


class SEALData(object):
    """
    1. generate positive and negative samples
    2. Subgraph sampling, support saving and loading processed graphs.
    Attributes:
        g(dgl.DGLGraph):
    """

    def __init__(self, g, split_edge, hop=1, neg_samples=1, subsample_ratio=1, prefix=None, save_dir=None,
                 num_workers=32, shuffle=True, print_fn=print):
        self.g = g
        self.hop = hop
        self.subsample_ratio = subsample_ratio
        self.prefix = prefix
        self.save_dir = save_dir
        self.print_fn = print_fn
        if self.hop > 1:
            self.num_parts = 10
        else:
            self.num_parts = 1

        self.ndata = {k: v for k, v in self.g.ndata.items()}
        self.edata = {k: v for k, v in self.g.edata.items()}
        self.g.ndata.clear()
        self.g.edata.clear()
        self.print_fn("Save ndata and edata in class.")
        self.print_fn("Clear ndata and edata in graph.")

        self.generator = PosNegEdgesGenerator(g=self.g,
                                              split_edge=split_edge,
                                              neg_samples=neg_samples,
                                              subsample_ratio=subsample_ratio,
                                              shuffle=shuffle,
                                              return_type='combine')

        self.sampler = SEALSampler(graph=self.g,
                                   hop=hop,
                                   num_workers=num_workers,
                                   num_parts=self.num_parts,
                                   print_fn=print_fn)

    def __call__(self, split_type):

        if split_type == 'train':
            num_parts = self.num_parts
        else:
            num_parts = 1

        if num_parts != 1:
            path = ['{}_{}_{}-hop_{}-subsample-part{}.bin'.format(self.prefix, split_type, self.hop,
                                                                  self.subsample_ratio, i) for i in
                    range(self.num_parts)]
            path = [osp.join(self.save_dir or '', p) for p in path]
        else:
            path = [osp.join(self.save_dir or '', '{}_{}_{}-hop_{}-subsample.bin'.format(self.prefix, split_type,
                                                                                         self.hop,
                                                                                         self.subsample_ratio))]

        if all([osp.exists(p) for p in path]):
            self.print_fn("{} processed files exist".format(split_type.capitalize()))
            if num_parts == 1:
                tmp = dgl.load_graphs(path[0])
                data = {'graph_list': tmp[0]}
                for k, v in tmp[1].items():
                    data[k] = v
                return data
            else:
                return path
        else:

            self.print_fn("Processed files not exist.")

            edges, labels = self.generator(split_type)
            self.print_fn("Generate {} edges totally.".format(edges.size(0)))
            if num_parts > 1:
                print("{}-hop subgraph too large, partition to {} files".format(self.hop, num_parts))
            return self.sampler(split_type, path, edges, labels)
