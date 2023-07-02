from src.explainer import gnn_explainer
# from src.trainer import gnn_trainer


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Main Arguments")

    # model paramteres
    parser.add_argument(
        '-n', '--dataset_name', default='mutag', type=str, required=False,
        help='mutag')
    parser.add_argument(
        '-e', '--n_epochs', default=100, type=int, required=False)
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

