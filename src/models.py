import torch.nn as nn
import torch as th
import torch.nn.functional as F

from src.layer import RelGraphConvLayer


class NodeClassifier(nn.Module):
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
        """
        Node classifier module for graph-based node classification.

        Parameters:
        -----------
        g : dgl.DGLGraph
            Input graph.
        h_dim : int
            Dimensionality of the input node features and hidden node features.
        out_dim : int
            Dimensionality of the output node features (final prediction).
        num_bases : int
            Number of bases for the weight matrix parameterization in RelGraphConvLayer.
        num_hidden_layers : int, optional
            Number of hidden layers in the model. Default is 1.
        dropout : float, optional
            Dropout rate applied to the node features during training. Default is 0.
        use_self_loop : bool, optional
            Flag indicating whether to include self-loop messages during message passing. Default is False.
        """
        
        super(NodeClassifier, self).__init__()
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

    def forward(self, *args, **kwargs):
        g = kwargs.get("graph", self.g)
        h = kwargs.get("feat", self.embeds)
        # full graph training
        for layer in self.layers:
            h = layer(g, h)

        if not self.training and not kwargs.get('explain_node', False):
            return h['d']
        return h
