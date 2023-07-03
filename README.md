# Program Readme

This program is designed to perform node classification using a Graph Neural Network (GNN) model. It supports the R-GCN (Relational Graph Convolutional Network) architecture and provides options for training the model, performing graph explanation, and evaluating the performance metrics.

## Installation

To run this program, please follow these steps:

1. Clone the repository:

```
git clone <repository-url>
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Set up the dataset:

By default, the program uses the "mutag" dataset. If you want to use a different dataset, make sure to update the `--dataset` argument in the command-line or modify the default value in the code.

4. Run the program:

```
python main.py
```

## Command-line Arguments

The program accepts the following command-line arguments:

- `--dropout`: Dropout probability for regularization (default: 0).
- `--n-hidden`: Number of hidden units in the model (default: 16).
- `--gpu`: GPU device index to use. Set to -1 for CPU (default: -1).
- `--lr`: Learning rate for the optimizer (default: 0.01).
- `--n-bases`: Number of filter weight matrices. Use -1 to use all (default: -1).
- `--n-layers`: Number of propagation rounds (default: 2).
- `-e`, `--n-epochs`: Number of training epochs (default: 50).
- `-d`, `--dataset`: Dataset to use (default: "mutag").
- `--model_path`: Path to save the trained model (default: "data/saved_model.pt").
- `--explain_graph`: Set to True if graph explanation is needed (default: False).
- `--print_metrics`: Set to True to print evaluation metrics (default: False).
- `--node_index`: Provide node index if a single node index output is expected (default: None).
- `--l2norm`: L2 norm coefficient (default: 0).
- `--use-self-loop`: Include self-feature as a special relation (default: False).

## Example Usage

To train the model and evaluate its performance:

```
python main.py
```

To enable graph explanation:

```
python main.py --explain_graph True
```

To print evaluation metrics:

```
python main.py --print_metrics True
```

To print for a single node index only

```
python main.py --node_index 1
```

## License

This program is licensed under the [MIT License](LICENSE).