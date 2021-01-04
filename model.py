import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv
import dgl


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
            initial_dim += node_embedding.embedding_dim

        self.layers = nn.ModuleList()

        if gcn_type == 'gcn':
            self.layers.append(GraphConv(initial_dim, hidden_units))
            for _ in range(num_layers - 1):
                self.layers.append(GraphConv(hidden_units, hidden_units))
        elif gcn_type == 'sage':
            self.layers.append(SAGEConv(initial_dim, hidden_units, aggregator_type='gcn'))
            for _ in range(num_layers - 1):
                self.layers.append(SAGEConv(hidden_units, hidden_units, aggregator_type='gcn'))
        else:
            raise ValueError('Gcn type error.')

        self.linear_1 = nn.Linear(hidden_units, hidden_units)
        self.linear_2 = nn.Linear(hidden_units, 1)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, z, pair_nodes, x=None, node_id=None, edge_weight=None):
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

        if self.use_attribute:
            x = torch.cat([z_emb, x], 1)

        if self.use_embedding:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb])

        for layer in self.layers[:-1]:
            x = layer(g, x, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](g, x, edge_weight)

        with g.local_scope():
            g.ndata['x'] = x
            if self.pooling_type == 'center':
                x_u = x[pair_nodes[0]]
                x_v = x[pair_nodes[1]]
                x = (x_u * x_v)
                x = F.relu(self.linear_1(x))
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.linear_2(x)

            elif self.pooling_type == 'sum':
                x = dgl.mean_nodes(g, 'x')
                x = F.relu(self.linear_1(x))
                F.dropout(x, p=self.dropout, training=self.training)
                x = self.linear_2(x)
            else:
                raise ValueError("Pooling type error.")

            return x


# class BaseRGCN(nn.Module):
#     def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
#                  num_hidden_layers=1, dropout=0,
#                  use_self_loop=False, use_cuda=False):
#         super(BaseRGCN, self).__init__()
#         self.num_nodes = num_nodes
#         self.h_dim = h_dim
#         self.out_dim = out_dim
#         self.num_rels = num_rels
#         self.num_bases = None if num_bases < 0 else num_bases
#         self.num_hidden_layers = num_hidden_layers
#         self.dropout = dropout
#         self.use_self_loop = use_self_loop
#         self.use_cuda = use_cuda
#
#         # create rgcn layers
#         self.build_model()
#
#     def build_model(self):
#         self.layers = nn.ModuleList()
#         # i2h
#         i2h = self.build_input_layer()
#         if i2h is not None:
#             self.layers.append(i2h)
#         # h2h
#         for idx in range(self.num_hidden_layers):
#             h2h = self.build_hidden_layer(idx)
#             self.layers.append(h2h)
#         # h2o
#         h2o = self.build_output_layer()
#         if h2o is not None:
#             self.layers.append(h2o)
#
#     def build_input_layer(self):
#         return None
#
#     def build_hidden_layer(self, idx):
#         raise NotImplementedError
#
#     def build_output_layer(self):
#         return None
#
#     def forward(self, g, h, r, norm):
#         for layer in self.layers:
#             h = layer(g, h, r, norm)
#         return h
#
# class RelGraphEmbedLayer(nn.Module):
#     r"""Embedding layer for featureless heterograph.
#     Parameters
#     ----------
#     dev_id : int
#         Device to run the layer.
#     num_nodes : int
#         Number of nodes.
#     node_tides : tensor
#         Storing the node type id for each node starting from 0
#     num_of_ntype : int
#         Number of node types
#     input_size : list of int
#         A list of input feature size for each node type. If None, we then
#         treat certain input feature as an one-hot encoding feature.
#     embed_size : int
#         Output embed size
#     embed_name : str, optional
#         Embed name
#     """
#     def __init__(self,
#                  dev_id,
#                  num_nodes,
#                  node_tids,
#                  num_of_ntype,
#                  input_size,
#                  embed_size,
#                  sparse_emb=False,
#                  embed_name='embed'):
#         super(RelGraphEmbedLayer, self).__init__()
#         self.dev_id = th.device(dev_id if dev_id >= 0 else 'cpu')
#         self.embed_size = embed_size
#         self.embed_name = embed_name
#         self.num_nodes = num_nodes
#         self.sparse_emb = sparse_emb
#
#         # create weight embeddings for each node for each relation
#         self.embeds = nn.ParameterDict()
#         self.num_of_ntype = num_of_ntype
#         self.idmap = th.empty(num_nodes).long()
#
#         for ntype in range(num_of_ntype):
#             if input_size[ntype] is not None:
#                 input_emb_size = input_size[ntype].shape[1]
#                 embed = nn.Parameter(th.Tensor(input_emb_size, self.embed_size))
#                 nn.init.xavier_uniform_(embed)
#                 self.embeds[str(ntype)] = embed
#
#         self.node_embeds = th.nn.Embedding(node_tids.shape[0], self.embed_size, sparse=self.sparse_emb)
#         nn.init.uniform_(self.node_embeds.weight, -1.0, 1.0)
#
#     def forward(self, node_ids, node_tids, type_ids, features):
#         """Forward computation
#         Parameters
#         ----------
#         node_ids : tensor
#             node ids to generate embedding for.
#         node_ids : tensor
#             node type ids
#         features : list of features
#             list of initial features for nodes belong to different node type.
#             If None, the corresponding features is an one-hot encoding feature,
#             else use the features directly as input feature and matmul a
#             projection matrix.
#         Returns
#         -------
#         tensor
#             embeddings as the input of the next layer
#         """
#         tsd_ids = node_ids.to(self.node_embeds.weight.device)
#         embeds = th.empty(node_ids.shape[0], self.embed_size, device=self.dev_id)
#         for ntype in range(self.num_of_ntype):
#             if features[ntype] is not None:
#                 loc = node_tids == ntype
#                 embeds[loc] = features[ntype][type_ids[loc]].to(self.dev_id) @ self.embeds[str(ntype)].to(self.dev_id)
#             else:
#                 loc = node_tids == ntype
#                 embeds[loc] = self.node_embeds(tsd_ids[loc]).to(self.dev_id)
#
#         return embeds
