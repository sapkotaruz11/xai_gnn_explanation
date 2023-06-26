"""
--- Created by Aashish Prajapati
--- Date: 24/06/2023 
"""
from dgl.nn.pytorch import HeteroGNNExplainer
import torch.nn as nn
from torch_geometric.explain import fidelity, groundtruth_metrics

from src.trainer import gnn_trainer
import torch as th


def explain_model(gnn_model, graph):
    explainer = HeteroGNNExplainer(gnn_model, num_hops=1, num_epochs=100)
    embeds = nn.ParameterDict()
    for ntype in graph.ntypes:
        embed = nn.Parameter(th.Tensor(graph.num_nodes(ntype), 16))
        nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
        graph.nodes[ntype].data['h'] = embed
        embeds[ntype] = embed
    # graph.ndata['h'] = embeds
    feat = graph.ndata['h']
    explanation = explainer.explain_graph(graph=graph, feat=feat, **{'explain_node': False})
    feat_mask, edge_mask = explanation
    # metrics = groundtruth_metrics()
    # fidel = fidelity(explainer, explanation)
    # print(fidel)

    print("EXPLAIN GRAPH FEAT MASK", feat_mask)
    print("EXPLAIN GRAPH EDGE MASK", edge_mask)
    print("EXPLAIN GRAPH GRAPH NODES", graph.nodes)
    print()
    print()
    # new_center, sg, feat_mask, edge_mask = explainer.explain_node('d', 0, graph, feat, **{'explain_node': True})
    # print("EXPLAIN NODES NEW center", new_center)
    # print("EXPLAIN NODES SG", sg)
    # print("EXPLAIN NODES featmask", feat_mask)
    # print("EXPLAIN NODES Edge mask", edge_mask)


def gnn_explainer(args):
    gnn_model, graph = gnn_trainer(args)
    if args.explain:
        explain_model(gnn_model, graph)
