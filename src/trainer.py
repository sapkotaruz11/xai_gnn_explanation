"""
--- Created by Aashish Prajapati
--- Date: 24/06/2023 
"""
import  torch as th

from src.data_loader import get_dataset
from src.models import EntityClassify
import torch.nn.functional as F



def gnn_trainer(args):
    g, num_classes, train_mask, test_mask, train_idx, val_idx, test_idx, labels, category_id, category = get_dataset(
        args)

    n_hidden = args.n_hidden
    num_bases = -1
    dropout = args.dropout
    num_hidden_layers = args.num_hidden_layers
    use_self_loop = False
    lr = args.lr
    l2norm = 5e-4
    n_epochs = args.n_epochs

    gnn_model = EntityClassify(
        g,
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
            logits = gnn_model(graph=g,feat=feat )[category]
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
    return gnn_model , g

