#!/usr/bin/env python
# coding: utf-8
import json
import os
import random
from os import mkdir

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.explain import GNNExplainer, Explainer, fidelity, characterization_score

dataset = TUDataset(root="data", name="MUTAG")

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
# print(f'Has isolated nodes: {data.has_isolated_nodes()}')
# print(f'Has self-loops: {data.has_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train_model():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def model_test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 50):
    train_model()
    train_acc = model_test(train_loader)
    test_acc = model_test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')



explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',  # Model returns log probabilities.
    ),
)

store_dict = {}
for i in range(10):
    indx = random.randint(1,100)
    # Generate explanation for the node at index `10`:
    explanation = explainer(data.x, data.edge_index,batch=data.batch, index=i)
    output_dir = "data/{}".format(indx)
    if os.path.exists(output_dir):
        indx = random.randint(1,100)
        explanation = explainer(data.x, data.edge_index, batch=data.batch, index=i)
        output_dir = "data/{}".format(indx)
    mkdir(output_dir)


    explanation.visualize_feature_importance(top_k=3,path="data/{}/feature_importance".format(indx))

    explanation.visualize_graph(path="data/{}/graph".format(indx))

    from torch_geometric.explain import unfaithfulness

    metric = unfaithfulness(explainer, explanation)

    pos_fidel, neg_fidel = fidelity(explainer, explanation)
    try:
        char_score = characterization_score(pos_fidelity=pos_fidel, neg_fidelity=neg_fidel )
    except Exception as e:
        char_score = 0

    store_dict[indx]= {"metric":metric,"pos_fidel":pos_fidel,"neg_fidel":neg_fidel, "char_score":char_score}
json_object = json.dumps(store_dict, indent=4)
with open("data/metrics.json".format(indx), "w") as outfile:
    json.dump(store_dict, outfile)


