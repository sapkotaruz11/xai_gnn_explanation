"""
--- Created by Aashish Prajapati
--- Date: 24/06/2023 
"""
import dgl
import torch as th


def get_dataset(args):
    if args.dataset_name == "mutag":
        dataset = dgl.data.rdf.MUTAGDataset()
    else:
        raise Exception("Dataset undefined")
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

    val_idx = train_idx[: len(train_idx) // 5]
    train_idx = train_idx[len(train_idx) // 5:]
    return g, num_classes, train_mask, test_mask, train_idx, val_idx, test_idx, labels, category_id, category
