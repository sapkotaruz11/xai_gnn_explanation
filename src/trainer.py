import torch as th

from src.data_loader import get_dataset
from src.models import NodeClassifier
import torch.nn.functional as F


def gnn_trainer(g, num_classes, train_idx, val_idx,category,labels, args):
    """
    Train a Graph Neural Network (GNN) model for node classification.

    Parameters:
    - args: An object containing the training arguments and configuration.

    Returns:
    - gnn_model: The trained GNN model.
    - g: The input graph used for training.

    """


    n_hidden = args.n_hidden
    num_bases = -1
    dropout = args.dropout
    num_hidden_layers = args.num_hidden_layers
    use_self_loop = False
    lr = args.lr
    l2norm = 5e-4
    n_epochs = args.n_epochs
    input_dim = n_hidden

    gnn_model = NodeClassifier(
        g,
        input_dim,
        n_hidden,
        num_classes,
        num_bases,
        num_hidden_layers,
        dropout,
        use_self_loop,
    )
    feat = gnn_model.embeds
    optimizer = th.optim.Adam(
        gnn_model.parameters(), lr=lr, weight_decay=l2norm
    )

    if args.mode == 'train':
        print("start training...")
        dur = []
        losses = []
        gnn_model.train()
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            logits = gnn_model()[category]
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()

            train_acc = th.sum(
                logits[train_idx].argmax(dim=1) == labels[train_idx]
            ).item() / len(train_idx)

            val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
            val_acc = th.sum(
                logits[val_idx].argmax(dim=1) == labels[val_idx]
            ).item() / len(val_idx)
            losses.append(loss)
            print(
                "Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} ".format(
                    epoch,
                    train_acc,
                    loss.item(),
                    val_acc,
                    val_loss.item(),
                )
            )

    return gnn_model
