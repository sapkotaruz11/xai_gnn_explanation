import dgl
import torch as th
import torch_geometric.datasets


def get_dataset(args):
    """Retrieve and preprocess the dataset based on the provided arguments.

    Parameters:
    -----------
    args : argparse.Namespace
        Arguments specifying the dataset to retrieve.

    Returns:
    --------
    g : dgl.DGLGraph
        The preprocessed graph dataset.
    num_classes : int
        The number of classes in the dataset.
    train_mask : torch.Tensor
        Mask indicating the training nodes.
    test_mask : torch.Tensor
        Mask indicating the testing nodes.
    train_idx : torch.Tensor
        Indices of the training nodes.
    val_idx : torch.Tensor
        Indices of the validation nodes.
    test_idx : torch.Tensor
        Indices of the testing nodes.
    labels : torch.Tensor
        Labels of the nodes in the dataset.
    category_id : int
        ID of the category node type.
    category : str
        Name of the category node type.
    """
    
    if args.dataset_name == "mutag":
        dataset = dgl.data.rdf.MUTAGDataset()
    else:
        raise Exception("Dataset undefined")
    g = dataset[0]

    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data["train_mask"]
    test_mask = g.nodes[category].data["test_mask"]
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
