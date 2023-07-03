import os.path
import random

import dgl
import torch.nn as nn

from src.dglnn_local.gnnexplainer import HeteroGNNExplainer
from src.trainer import gnn_trainer
import torch as th
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx


def flatten_list(input_list, flattened_list):
    for sublist in input_list:
        if isinstance(sublist, list):
            flattened_list = flatten_list(sublist, flattened_list)
        else:
            flattened_list.append(sublist)
    return flattened_list


def get_combination_matrix(edge_dict):
    import numpy as np
    sources = set()
    destinations = set()

    for keys in edge_dict.keys():
        s, p, o = keys
        sources.add(s)
        destinations.add(o)
    sources = list(sources)
    destinations = list(destinations)

    # Create empty matrix
    num_sources = len(sources)
    num_destinations = len(destinations)
    matrixs = {}
    for i in range(num_sources):
        matrixs[sources[i]] = {}
        for j in range(num_destinations):
            matrixs[sources[i]][sources[j]] = 0

    for key in edge_dict.keys():
        s, p, o = key
        matrixs[s][o] += 1

    return matrixs


def get_mapping(sg):
    etype_dict = {}
    # Add edge indices to the HeteroData object
    numer_dict = {}
    for etype in sg.canonical_etypes:
        src, dst = sg.all_edges(form='uv', etype=etype)
        if src.numel() == 0 or dst.numel() == 0:
            continue
        edge_index = th.stack([src, dst], dim=0)
        etype_dict[etype] = edge_index
        s, p, o = etype
        if numer_dict.get(s, None) is None:
            numer_dict[s] = []
        # if edge_index[0] not in numer_dict[s]:
        numer_dict[s].extend(edge_index[0].tolist())
        if numer_dict.get(o, None) is None:
            numer_dict[o] = []
        # if edge_index[1] not in numer_dict[o]:
        numer_dict[o].append(edge_index[1].tolist())
        # hetero_data[etype].edge_index = edge_index
    for k, v in numer_dict.items():
        numer_dict[k] = list(set(flatten_list(v, [])))
    return numer_dict, etype_dict


def explain_model(gnn_model, graph):
    explainer = HeteroGNNExplainer(gnn_model, num_hops=1, num_epochs=10)
    embeds = nn.ParameterDict()
    # Generate Node features
    for ntype in graph.ntypes:
        embed = nn.Parameter(th.Tensor(graph.num_nodes(ntype), 16))
        nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
        graph.nodes[ntype].data['h'] = embed
        embeds[ntype] = embed
    # graph.ndata['h'] = embeds
    feat = graph.ndata['h']
    # EXPLAIN GRAPH
    explanation = explainer.explain_graph(graph=graph, feat=feat, **{'explain_node': False})

    graph_feat_mask, graph_edge_mask = explanation

    print("EXPLAIN GRAPH FEAT MASK", graph_feat_mask)
    print("EXPLAIN GRAPH EDGE MASK", graph_edge_mask)
    print("EXPLAIN GRAPH GRAPH NODES", graph.nodes)
    print()
    print()
    # Explain NODE SECTION
    output_dir = "data"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    store_dict = {}
    for idx in range(1, 10):
        i = random.randint(1, 9500)
        print("Selected node index", i)
        new_center, sg, feat_mask, edge_mask = explainer.explain_node('d', i, graph, feat, **{'explain_node': True})
        print("EXPLAIN NODE FEAT MASK", feat_mask)
        print("EXPLAIN NODE EDGE MASK", edge_mask)
        print("EXPLAIN NODE GRAPH NODES", graph.nodes)
        print()

        # Converted to homogenous because networkx doesnt support heterogenous graph
        sg_homo = dgl.to_homogeneous(sg)
        G = dgl.to_networkx(sg_homo)

        # TODO Try to map
        node_dict, edge_dict = get_mapping(sg)
        print("Node dict", node_dict)
        print("Edge dict", edge_dict)
        combination = get_combination_matrix(edge_dict)
        plt.figure(figsize=[15, 7])

        nodes = G.nodes()
        edges = G.edges()
        node_colors = ['b' for node in nodes]
        edge_colors = ['r' if edge in edge_mask else 'b' for edge in edges]

        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos=pos, node_color=node_colors, edge_color=edge_colors, with_labels=True)
        plt.savefig("{}/path_{}.png".format(output_dir,i))
        # plt.show()
        store_dict[i] = {'feat_mask': feat_mask, 'edge_mask': edge_mask, "sub_graph": sg, "sub_graph_nodes": node_dict,
                         "sub_graph_edge": edge_dict, "combination": combination, 'graph_feat_mask': graph_feat_mask,
                         'graph_edge_mask': graph_edge_mask}

        print(store_dict)

    th.save(store_dict, '{}/explanation.pt'.format(output_dir))


def gnn_explainer(args):
    gnn_model, graph = gnn_trainer(args)
    explain_model(gnn_model, graph)


def load_pt(file_path):
    return th.load(file_path)
