
import dgl
import torch as th
import torch_geometric.datasets


def get_dataset(args):
    if args.dataset_name == "mutag":
        dataset = dgl.data.rdf.MUTAGDataset()
        # dataset = torch_geometric.datasets.Entities(root=".",name="Mutag")

    else:
        raise Exception("Dataset undefined")
    g = dataset[0]
    print(g)
    # g_homo = dgl.to_homogeneous(g)


    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data["train_mask"]
    test_mask = g.nodes[category].data["test_mask"]
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    labels = g.nodes[category].data.pop("labels")

    # subgraph_idx = th.cat([train_idx, test_idx])
    # subgraph = dgl.node_subgraph(g, {'d': subgraph_idx})
    #
    #
    # # Update the train_idx and test_idx with the new indices
    # train_idx = train_idx - 9189
    # test_idx = test_idx - 9189
    #
    # # Update the labels based on the subgraph
    # subgraph.nodes[category].data["labels"] = labels[subgraph_idx]
    # labels = subgraph.nodes[category].data["labels"]
    # g = subgraph
    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    val_idx = train_idx[: len(train_idx) // 5]
    train_idx = train_idx[len(train_idx) // 5:]
    return g, num_classes, train_mask, test_mask, train_idx, val_idx, test_idx, labels, category_id, category
