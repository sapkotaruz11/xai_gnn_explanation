"""
--- Created by Aashish Prajapati
--- Date: 24/06/2023 
"""

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
from dgl.nn.pytorch import HeteroGNNExplainer
import torch.nn.functional as F


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
            return graph.ndata['h']


input_dim = 5
num_classes = 2
g = dgl.heterograph({
    ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1])})
g.nodes['user'].data['h'] = th.randn(g.num_nodes('user'), input_dim)
g.nodes['game'].data['h'] = th.randn(g.num_nodes('game'), input_dim)

transform = dgl.transforms.AddReverse()
g = transform(g)

# define and train the model
model = Model(input_dim, num_classes, g.canonical_etypes)
feat = g.ndata['h']
optimizer = th.optim.Adam(model.parameters())
for epoch in range(10):
    logits = model(g, feat)['user']
    loss = F.cross_entropy(logits, th.tensor([1, 1, 1]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Explain the prediction for node 0 of type 'user'
explainer = HeteroGNNExplainer(model, num_hops=1)
new_center, sg, feat_mask, edge_mask = explainer.explain_node('user', 0, g, feat)
print(new_center)
print(sg)
print(feat_mask)
print(edge_mask)

new_center, sg, feat_mask, edge_mask = explainer.explain_node('user', 0, g, feat)
print(new_center)
print(sg)
print(feat_mask)
print(edge_mask)