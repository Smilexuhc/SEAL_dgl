import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv, SortPooling, SumPooling


class GCN(nn.Module):
    """
    GCN Model

    Attributes:
        num_layers(int): num of gcn layers
        hidden_units(int): num of hidden units
        gcn_type(str): type of gcn layer, 'gcn' for GraphConv and 'sage' for SAGEConv
        pooling_type(str): type of graph pooling to get subgraph representation
                           'sum' for sum pooling and 'center' for center pooling.
        attribute_dim(int): dimension of nodes' attributes
        node_embedding(Tensor, optional): pre-trained node embedding
        use_embedding(bool, optional): whether to use node embedding. Note that if 'use_embedding' is set True
                             and 'node_embedding' is None, will automatically randomly initialize node embedding.
        num_nodes(int, optional): num of nodes
        dropout(float, optional): dropout rate
        max_z(int, optional): default max vocab size of node labeling, default 1000.

    """

    def __init__(self, num_layers, hidden_units, gcn_type='gcn', pooling_type='center', attribute_dim=None,
                 node_embedding=None, use_embedding=False, num_nodes=None, dropout=0.5, max_z=1000):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling_type = pooling_type
        self.use_attribute = False if attribute_dim is None else True
        self.use_embedding = use_embedding

        self.z_embedding = nn.Embedding(max_z, hidden_units)

        if node_embedding is not None:
            self.node_embedding = nn.Embedding.from_pretrained(node_embedding)
        elif use_embedding:
            self.node_embedding = nn.Embedding(num_nodes, hidden_units)

        initial_dim = hidden_units

        if self.use_attribute:
            initial_dim += attribute_dim
        if self.use_embedding:
            initial_dim += self.node_embedding.embedding_dim

        self.layers = nn.ModuleList()

        if gcn_type == 'gcn':
            self.layers.append(GraphConv(initial_dim, hidden_units, weight=False))
            for _ in range(num_layers - 1):
                self.layers.append(GraphConv(hidden_units, hidden_units, weight=False))
        elif gcn_type == 'sage':
            self.layers.append(SAGEConv(initial_dim, hidden_units, aggregator_type='gcn',))
            for _ in range(num_layers - 1):
                self.layers.append(SAGEConv(hidden_units, hidden_units, aggregator_type='gcn'))
        else:
            raise ValueError('Gcn type error.')

        self.linear_1 = nn.Linear(hidden_units, hidden_units)
        self.linear_2 = nn.Linear(hidden_units, 1)

        if pooling_type == 'sum':
            self.pooling = SumPooling()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, z, pair_nodes=None, x=None, node_id=None, edge_weight=None):
        """

        Args:
            g(DGLGraph): the graph
            z(Tensor): node labeling tensor, shape [N, 1]
            pair_nodes(Tensor): id of two target nodes used in center pooling
            x(Tensor, optional): node attribute tensor, shape [N, dim]
            node_id(Tensor, optional): node id tensor, shape [N, 1]
            edge_weight(Tensor, optional): edge weight tensor [E, dim]

        Returns:
            x(Tensor)

        """

        z_emb = self.z_embedding(z)

        # if z_emb.ndim == 3:  # in case z has multiple integer labels
        #     z_emb = z_emb.sum(dim=1)

        if self.use_attribute and x is not None:
            x = torch.cat([z_emb, x], 1)
        else:
            x = z_emb

        if self.use_embedding:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)

        for layer in self.layers[:-1]:
            x = layer(g, x, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](g, x, edge_weight)

        if self.pooling_type == 'center':
            x_u = x[pair_nodes[0]]
            x_v = x[pair_nodes[1]]
            x = (x_u * x_v)

        elif self.pooling_type == 'sum':
            x = self.pooling(g, x)

        x = F.relu(self.linear_1(x))
        F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_2(x)

        return x


class DGCNN(nn.Module):
    """
    An end-to-end deep learning architecture for graph classification.
    paper link: https://muhanzhang.github.io/papers/AAAI_2018_DGCNN.pdf
    todo: rewrite the conv part

    Attributes:
        num_layers(int): num of gcn layers
        hidden_units(int): num of hidden units
        k(int, optional): The number of nodes to hold for each graph in SortPooling.
        gcn_type(str): type of gcn layer, 'gcn' for GraphConv and 'sage' for SAGEConv
        attribute_dim(int): dimension of nodes' attributes
        node_embedding(Tensor, optional): pre-trained node embedding
        use_embedding(bool, optional): whether to use node embedding. Note that if 'use_embedding' is set True
                             and 'node_embedding' is None, will automatically randomly initialize node embedding.
        num_nodes(int, optional): num of nodes
        dropout(float, optional): dropout rate
        max_z(int, optional): default max vocab size of node labeling, default 1000.
    """

    def __init__(self, num_layers, hidden_units, k=10, gcn_type='gcn', attribute_dim=None,
                 node_embedding=None, use_embedding=False, num_nodes=None, dropout=0.5, max_z=1000):
        super(DGCNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attribute = False if attribute_dim is None else True
        self.use_embedding = use_embedding

        self.z_embedding = nn.Embedding(max_z, hidden_units)

        if node_embedding is not None:
            self.node_embedding = nn.Embedding.from_pretrained(node_embedding)
        elif use_embedding:
            self.node_embedding = nn.Embedding(num_nodes, hidden_units)

        initial_dim = hidden_units

        if self.use_attribute:
            initial_dim += attribute_dim
        if self.use_embedding:
            initial_dim += self.node_embedding.embedding_dim

        self.layers = nn.ModuleList()

        if gcn_type == 'gcn':
            self.layers.append(GraphConv(initial_dim, hidden_units, weight=False))
            for _ in range(num_layers - 1):
                self.layers.append(GraphConv(hidden_units, hidden_units, weight=False))
        elif gcn_type == 'sage':
            self.layers.append(SAGEConv(initial_dim, hidden_units, aggregator_type='gcn'))
            for _ in range(num_layers - 1):
                self.layers.append(SAGEConv(hidden_units, hidden_units, aggregator_type='gcn'))
        else:
            raise ValueError('Gcn type error.')

        self.pooling = SortPooling(k=k)
        conv1d_channels = [16, 32]
        total_latent_dim = hidden_units * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv_1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                                conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv_2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1],
                                conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.linear_1 = nn.Linear(dense_dim, 128)
        self.linear_2 = nn.Linear(128, 1)

    def forward(self, g, z, pair_nodes=None, x=None, node_id=None, edge_weight=None):
        z_emb = self.z_embedding(z)

        if self.use_attribute and x is not None:
            x = torch.cat([z_emb, x], 1)
        else:
            x = z_emb

        if self.use_embedding:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], 1)

        xs = [x]

        for layer in self.layers:
            xs += [torch.tanh(layer(g, xs[-1], edge_weight))]

        x = torch.cat(xs[1:], dim=-1)

        # SortPooling
        x = self.pooling(g, x)
        x = F.relu(self.conv_1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv_2(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.linear_1(x))
        F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear_2(x)

        return x
