
from src.explainer import gnn_explainer
import argparse


def get_args():
    '''
    Parses the command-line arguments using parser.parse_args(), which returns an args object containing the parsed arguments.
    
    Arguments include:
    dataset_name(str):         Specifies the dataset name (default: 'mutag').
    n_epochs(int):             Specifies the number of epochs for training (default: 100).
    lr(float):                 Specifies the learning rate for the optimizer (default: 0.01).
    dropout(float):            Specifies the dropout rate for regularization (default: 0.3).
    n_hidden(int):             Specifies the number of hidden units in the model (default: 16).
    num_hidden_layers(int):    Specifies the number of hidden layers in the model (default: 3).
    mode(str):                 Specifies the mode of operation (default: 'train').
    explain(bool):             Specifies whether to enable explanation of node classification (default: True).

    Returns the args object.'''

    parser = argparse.ArgumentParser(description="RGCN")
    parser.add_argument(
        "--dropout", type=float, default=0, help="dropout probability"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=16, help="number of hidden units"
    )
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--n-bases",
        type=int,
        default=-1,
        help="number of filter weight matrices, default: -1 [use all]",
    )
    parser.add_argument(
        "--n-layers", type=int, default=2, help="number of propagation rounds"
    )
    parser.add_argument(
        "-e",
        "--n-epochs",
        type=int,
        default=50,
        help="number of training epochs",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=False, help="dataset to use", default="mutag"
    )
    parser.add_argument(
        "--model_path", type=str, default='data/saved_model.pt', help="path for save the model"
    )
    parser.add_argument(
        "--explain_graph", type=bool, default=False, help="Set to True if graph explanation is needed"
    )
    parser.add_argument(
        "--print_metrics", type=bool, default=False, help="Set to True to print evaluation metrics"
    )
    parser.add_argument(
        "--node_index", type=int, default=None, help="Provide node index if a single node index output is expected"
    )
    parser.add_argument("--l2norm", type=float, default=0, help="l2 norm coef")
    parser.add_argument(
        "--use-self-loop",
        default=False,
        action="store_true",
        help="include self feature as a special relation",
    )
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument("--validation", dest="validation", action="store_true")
    fp.add_argument("--testing", dest="validation", action="store_false")
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    try:

        gnn_explainer(args)
    except Exception as e:
        print(e)
