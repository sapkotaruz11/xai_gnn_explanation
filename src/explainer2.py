"""
--- Created by Aashish Prajapati
--- Date: 28/06/2023 
"""
import torch
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, CaptumExplainer

from src.trainer import gnn_trainer

  # A heterogeneous graph data object.

def gnn_explainer(args):
    model, graph = gnn_trainer(args)
    data = HeteroData()
    data.x = graph.ndata
    data.edge_index = graph.edata

    explainer = Explainer(
        model,  # It is assumed that model outputs a single tensor.
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config = dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='probs',  # Model returns probabilities.
        ),
    )

    # Generate batch-wise heterogeneous explanations for
    # the nodes at index `1` and `3`:
    hetero_explanation = explainer(
        graph.ndata,
        graph.edata,
        index=torch.tensor([1, 3]),
    )
    print(hetero_explanation.edge_mask_dict)
    print(hetero_explanation.node_mask_dict)

