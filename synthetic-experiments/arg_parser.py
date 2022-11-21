import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
    )
    parser.add_argument(
        "--fastmode",
        action="store_true",
        default=False,
        help="Validate during training pass.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "gcn",
            "sgc",
            "mlp",
            "acmgcn",
            "acmsgc"
        ],
        help="Indicate the GNN model to use",
        default="mlp",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--epochs", type=int, default=5000, help="Number of epochs to train."
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 loss on parameters).",
    )
    parser.add_argument(
        "--hidden", type=int, default=32, help="Number of hidden units."
    )
    parser.add_argument(
        "--base_dataset",
        type=str,
        choices=[
            "chameleon",
            "film",
            "squirrel",
            "cora",
            "citeseer",
            "pubmed",
            "random",
        ],
        help="base dataset to generate dataset from",
        default="chameleon",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        help="regular or random graphs",
        default="random",
    )
    parser.add_argument(
        "--early_stopping",
        type=float,
        default=200,
        help="early stopping used in GPRGNN",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate (1 - keep probability).",
    )
    parser.add_argument(
        "--edge_homo",
        type=float,
        default=0.1,
        help="edge homophily level of the synthetic graph",
    )
    parser.add_argument(
        "--degree_intra",
        type=int,
        default=2,
        help="number of neighbors in the same class for each node",
    )
    parser.add_argument(
        "--num_graph",
        type=int,
        default=10,
        help="number of graphs to generate for each homophily level",
    )

    args = parser.parse_args()
    return args
