import os
import time

import numpy as np
import torch as th
from dgl.data import MUTAGDataset

from src.models import NodeClassifier
import torch.nn.functional as F


def gnn_trainer(args):
    """
    Train a Graph Neural Network (GNN) model for node classification.

    Parameters:
    - args: An object containing the training arguments and configuration.

    Returns:
    - gnn_model: The trained GNN model.
    - g: The input graph used for training.

    """

    if args.dataset == "mutag":
        dataset = MUTAGDataset()
    else:
        raise Exception("Dataset not supported")

    # Retrieve the input graph, category for prediction, and other information from the dataset
    g = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data.pop("train_mask")
    test_mask = g.nodes[category].data.pop("test_mask")
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop("labels")
    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    # # Split dataset into train, validate, and test sets based on the validation flag
    if args.validation:
        val_idx = train_idx[: len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # Check CUDA availability and move tensors to GPU if specified
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        g = g.to("cuda:%d" % args.gpu)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()

    # Create the GNN model
    model = NodeClassifier(
        g,
        args.n_hidden,
        num_classes,
        num_bases=args.n_bases,
        num_hidden_layers=args.n_layers - 2,
        dropout=args.dropout,
        use_self_loop=args.use_self_loop,
        category=category
    )

    if use_cuda:
        model.cuda()

    # Define the optimizer
    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2norm
    )

    # Training loop
    print("start training...")
    dur = []
    if os.path.exists(args.model_path):
        model.load_state_dict(th.load(args.model_path))
    else:
        model.train()
        for epoch in range(args.n_epochs):
            optimizer.zero_grad()
            t0 = time.time()
            logits = model()[category]
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()
            t1 = time.time()

            dur.append(t1 - t0)
            train_acc = th.sum(
                logits[train_idx].argmax(dim=1) == labels[train_idx]
            ).item() / len(train_idx)
            val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
            val_acc = th.sum(
                logits[val_idx].argmax(dim=1) == labels[val_idx]
            ).item() / len(val_idx)
            print(
                "Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".format(
                    epoch,
                    train_acc,
                    loss.item(),
                    val_acc,
                    val_loss.item(),
                    np.average(dur),
                )
            )
        print()
        if args.model_path is not None:
            th.save(model.state_dict(), args.model_path)

    return model, g, test_idx, labels, category
