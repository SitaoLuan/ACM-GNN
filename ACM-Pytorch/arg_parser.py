import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
    )
    parser.add_argument(
        "--param_tunning",
        action="store_true",
        default=True,
        help="Parameter fine-tunning mode",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--epochs", type=int, default=5000, help="Number of epochs to train."
    )
    parser.add_argument(
        "--num_splits", type=int, help="number of training/val/test splits ", default=10
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "gcn",
            "sgc",
            "graphsage",
            "snowball",
            "gcnII",
            "acmgcn",
            "acmgcnp",
            "acmgcnpp",
            "acmsgc",
            "acmgraphsage",
            "acmsnowball",
            "mlp",
        ],
        help="name of the model",
        default="acmgcn",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="optimizer for large datasets (Adam, AdamW)",
        default="Adam",
    )
    parser.add_argument(
        "--early_stopping", type=float, default=200, help="early stopping used in GPRGNN"
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay (L2 loss on parameters).",
    )
    parser.add_argument("--hidden", type=int, default=64, help="Number of hidden units.")
    parser.add_argument("--hops", type=int, default=1, help="Number of hops we use, k= 1,2")
    parser.add_argument(
        "--layers", type=int, default=1, help="Number of hidden layers, i.e. network depth"
    )
    parser.add_argument(
        "--link_init_layers_X", type=int, default=1, help="Number of initial layer"
    )
    parser.add_argument("--dataset_name", type=str, help="Dataset name.", default="cornell")
    parser.add_argument(
        "--dropout", type=float, default=0, help="Dropout rate (1 - keep probability)."
    )
    parser.add_argument(
        "--fixed_splits",
        type=float,
        default=0,
        help="0 for random splits in GPRGNN, 1 for fixed splits in GeomGCN",
    )
    parser.add_argument(
        "--variant", type=float, default=0, help="Indicate ACM, GCNII variant models."
    )
    parser.add_argument(
        "--structure_info",
        type=int,
        default=0,
        help="1 for using structure information in acmgcnp, 0 for not",
    )

    args = parser.parse_args()
    return args

