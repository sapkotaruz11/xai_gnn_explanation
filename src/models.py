

import torch.nn as nn
import torch as th
import torch.nn.functional as F

from src.layer import RelGraphConvLayer


class EntityClassify(nn.Module):
    def __init__(
            self,
            g,
            h_dim,
            out_dim,
            num_bases,
            num_hidden_layers=1,
            dropout=0,
            use_self_loop=False,
    ):
        super(EntityClassify, self).__init__()
        self.g = g
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = sorted(set(g.etypes))
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            embed = nn.Parameter(th.Tensor(g.num_nodes(ntype), self.h_dim))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
            self.embeds[ntype] = embed
        # self.embed_layer = self.embeds
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(
            RelGraphConvLayer(
                self.h_dim,
                self.h_dim,
                self.rel_names,
                num_bases=self.num_bases,
                activation=F.relu,
                self_loop=self.use_self_loop,
                dropout=self.dropout
            )
        )
        # h2h
        for _ in range(self.num_hidden_layers):
            self.layers.append(
                RelGraphConvLayer(
                    self.h_dim,
                    self.h_dim,
                    self.rel_names,
                    num_bases=self.num_bases,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout,
                )
            )
        # h2o
        self.layers.append(
            RelGraphConvLayer(
                self.h_dim,
                self.out_dim,
                self.rel_names,
                num_bases=self.num_bases,
                activation=None,
                self_loop=self.use_self_loop,
            )
        )

    def forward(self, graph=None, feat=None,h=None, blocks=None,eweight =None,explain_node=False):
        if h is None:
            # full graph training
            # h = self.embeds
            h = feat
        blocks = None
        if blocks is None:
            # full graph training
            for layer in self.layers:
                h = layer(graph, h)
        else:
            # minibatch training
            for layer, block in zip(self.layers, blocks):
                h = layer(block, h)

        if not self.training and not explain_node:
            return h['d']
        return h

import dgl.function as fn
import dgl

class Model(nn.Module):
    def __init__(self, in_dim, num_classes, canonical_etypes):
        super(Model, self).__init__()
        self.etype_weights = nn.ModuleDict({
            '_'.join(c_etype): nn.Linear(in_dim, num_classes)
            for c_etype in canonical_etypes
        })

    def forward(self, graph, feat, eweight=None):
        with graph.local_scope():
            c_etype_func_dict = {}
            for c_etype in graph.canonical_etypes:
                src_type, etype, dst_type = c_etype
                wh = self.etype_weights['_'.join(c_etype)](feat[src_type])
                graph.nodes[src_type].data[f'h_{c_etype}'] = wh
                if eweight is None:
                    c_etype_func_dict[c_etype] = (fn.copy_u(f'h_{c_etype}', 'm'),
                        fn.mean('m', 'h'))
                else:
                    graph.edges[c_etype].data['w'] = eweight[c_etype]
                    c_etype_func_dict[c_etype] = (
                        fn.u_mul_e(f'h_{c_etype}', 'w', 'm'), fn.mean('m', 'h'))
            graph.multi_update_all(c_etype_func_dict, 'sum')
            hg = 0
            for ntype in graph.ntypes:
                if graph.num_nodes(ntype):
                    hg = hg + dgl.mean_nodes(graph, 'h', ntype=ntype)
            return hg
