from src.explainer import gnn_explainer
# from src.trainer import gnn_trainer


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
    
    import argparse

    parser = argparse.ArgumentParser(description="Main Arguments")

    # model paramteres
    parser.add_argument(
        '-n', '--dataset_name', default='mutag', type=str, required=False,
        help='mutag')
    parser.add_argument(
        '-e', '--n_epochs', default=10, type=int, required=False)
    parser.add_argument(
        '-lr', '--lr', default=0.01, type=float, required=False)
    parser.add_argument(
        '-d', '--dropout', default=0.3, type=float, required=False)
    parser.add_argument(
        '-nh', '--n_hidden', default=16, type=int, required=False)
    parser.add_argument(
        '-nhl', '--num_hidden_layers', default=3, type=int, required=False)
    parser.add_argument(
        '-m', '--mode', default='train', type=str, required=False)
    parser.add_argument(
        '-ex', '--explain', default=True, type=bool, required=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    gnn_explainer(args)

