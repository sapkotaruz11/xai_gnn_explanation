import os
import random

import dgl
import torch.nn as nn
from torch_geometric.explain import groundtruth_metrics

from src.dglnn_local.gnnexplainer import HeteroGNNExplainer
from src.trainer import gnn_trainer
import torch as th
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx


def flatten_list(input_list, flattened_list):
    """
    Flatten a nested list.

    Parameters:
    - input_list: The nested list to be flattened.
    - flattened_list: The list to store the flattened elements.

    Returns:
    - flattened_list: The flattened list.

    """
    for sublist in input_list:
        if isinstance(sublist, list):
            flattened_list = flatten_list(sublist, flattened_list)
        else:
            flattened_list.append(sublist)
    return flattened_list


def get_combination_matrix(edge_dict):
    """
    Create a combination matrix based on the given edge dictionary.

    Parameters:
    - edge_dict: A dictionary containing the edges in the form of (source, predicate, target).

    Returns:
    - matrixs: A dictionary representing the combination matrix.

    """
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
    """
    Generate node and edge mappings for the given subgraph.

    Parameters:
    - sg: The subgraph.

    Returns:
    - numer_dict: A dictionary mapping node types to their corresponding node indices.
    - etype_dict: A dictionary mapping edge types to their corresponding edge indices.

    """
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


def get_prediction(gnn_model, graph, feat, category, **kwargs):
    """
        Get the predicted label for a given graph and category using a graph neural network model.

        Parameters:
            - gnn_model (nn.Module): The graph neural network model used for prediction.
            - graph (dgl.DGLGraph): The input graph.
            - feat (torch.Tensor): The input node features.
            - category (str): The category or task for which the prediction is made.
            - **kwargs: Additional keyword arguments to be passed to the model.

        Returns:
            - pred_label (torch.Tensor): The predicted label for the given graph and category.

        """
    # Set to eval mode to get prediction
    gnn_model.eval()
    with th.no_grad():
        logits = gnn_model(graph=graph, feat=feat, **{'explain_node': True})[category]
        pred_label = logits.argmax(dim=-1)
    gnn_model.train()
    return pred_label


def explain_model(model, g, test_idx, labels, category, args):
    node_index = args.node_index
    print_metrics = args.print_metrics

    # Create an explainer object
    explainer = HeteroGNNExplainer(model, num_hops=1, num_epochs=10)

    # Initialize embedding parameters for each node type
    embeds = nn.ParameterDict()
    for ntype in g.ntypes:
        embed = nn.Parameter(th.Tensor(g.num_nodes(ntype), 16))
        nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
        g.nodes[ntype].data['h'] = embed
        embeds[ntype] = embed
    feat = g.ndata['h']

    # Explain the graph if required
    if args.explain_graph:
        explanation = explainer.explain_graph(g, feat, **{'explain_node': False})
        graph_feat_mask, graph_edge_mask = explanation
        print("EXPLAIN GRAPH FEAT MASK", graph_feat_mask)
        print("EXPLAIN GRAPH EDGE MASK", graph_edge_mask)

    # Get the model prediction
    prediction = get_prediction(model, g, feat, category)

    # Print metrics if required
    if print_metrics:
        target_mask = labels[test_idx]
        prediction_mask = th.tensor(prediction)[test_idx]
        metric = groundtruth_metrics(pred_mask=prediction_mask, target_mask=target_mask,
                                     metrics=['accuracy', 'precision', 'recall', 'f1_score'])
        metrics_dict = {'accuracy': metric[0], 'precision': metric[1], 'recall': metric[2], 'f1_score': metric[3]}
        print("Metrics:", metrics_dict)

    # Explain the node section
    store_dict = {}
    if node_index is not None:
        new_center, sg, feat_mask, edge_mask = explainer.explain_node(category, node_index, g, feat,
                                                                      **{'explain_node': True})
        print("EXPLAIN NODE FEAT MASK", feat_mask)
        print("EXPLAIN NODE EDGE MASK", edge_mask)

    else:
        for idx in range(10):

            # Randomly select a node index from the test data
            i = random.randint(th.min(test_idx), th.max(test_idx))
            print("Selected node index", i)
            node_prediction = prediction[i]
            print("PREDICTION :", node_prediction)

            # Explain the node
            new_center, sg, feat_mask, edge_mask = explainer.explain_node(category, i, g, feat,
                                                                          **{'explain_node': True})
            print("EXPLAIN NODE FEAT MASK", feat_mask)
            print("EXPLAIN NODE EDGE MASK", edge_mask)

            # Convert the subgraph to a homogeneous graph using DGL
            sg_homo = dgl.to_homogeneous(sg)
            G = dgl.to_networkx(sg_homo)

            # Visualize the subgraph using NetworkX and Matplotlib
            plt.figure(figsize=[15, 7])

            nodes = G.nodes()
            edges = G.edges()
            node_colors = ['b' for node in nodes]
            edge_colors = ['r' if edge in edge_mask else 'b' for edge in edges]

            pos = nx.spring_layout(G)
            nx.draw_networkx(G, pos=pos, node_color=node_colors, edge_color=edge_colors, with_labels=False)
            plt.savefig("data/path2_{}.png".format(i))


def gnn_explainer(args):
    try:
        if os.path.exists("data"):
            os.mkdir("data")
        model, g, test_idx, labels, category = gnn_trainer(args)
        explain_model(model, g, test_idx, labels, category, args)
    except Exception as e:
        raise e


