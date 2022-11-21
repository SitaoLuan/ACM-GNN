from __future__ import division
from pathlib import Path

import click
import torch
import numpy as np

from utils import load_full_data
from logger import SyntheticExpLogger

BASE_DIR = "./synthetic_graphs"
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

logger = SyntheticExpLogger()


@click.command()
@click.option(
    "--num_node_total", type=int, default=2000, help="total number of nodes in graph"
)
@click.option(
    "--base_dataset",
    type=str,
    default="pubmed",
    help="base dataset to generate dataset from",
)
def generate_feature(
    num_node_total, base_dataset
):  # Generate features for balanced dataset
    data_dir = f"{BASE_DIR}/features/{base_dataset}"
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    for i in range(10):
        logger.log_init(f"Generating graph features {i} based on dataset {base_dataset}")
        base_features = generate_base_features(base_dataset, num_node_total)
        torch.save(
            base_features,
            f"{data_dir}/{base_dataset}_{i}.pt",
        )
    return 0


def generate_base_features(base_dataset, num_node_total):
    if base_dataset == "random":
        return torch.from_numpy(np.random.rand(num_node_total, 1433)).float()
    else:
        _, _, features, labels = load_full_data(base_dataset)
        nclass = labels.max().item() + 1
        column_idx = [np.where(labels == i % nclass)[0] for i in range(5)]
        idx = []
        for j in range(5):
            if column_idx[j].shape[0] > 400:
                idx = (
                    idx
                    + np.random.choice(column_idx[j], 400, replace=False).tolist()
                )
            else:
                idx = (
                    idx
                    + column_idx[j].tolist()
                    + np.random.choice(
                        column_idx[j], 400 - column_idx[j].shape[0], replace=False
                    ).tolist()
                )
        return features[np.array(idx), :]


if __name__ == "__main__":
    generate_feature()
