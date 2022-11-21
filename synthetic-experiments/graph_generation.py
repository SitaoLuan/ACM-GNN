import ast
from pathlib import Path
import random

import click
import torch
import numpy as np

from utils import generate_output_label
from logger import SyntheticExpLogger

BASE_DIR = "./synthetic_graphs"
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

logger = SyntheticExpLogger()


class PythonOption(click.Option):
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


@click.command()
@click.option("--num_class", type=int, default=5, help="number of class")
@click.option(
    "--num_node_total", type=int, default=2000, help="total number of nodes in graph"
)
@click.option(
    "--degree_intra",
    type=int,
    default=2,
    help="number of neighbors in the same class for each node",
)
@click.option(
    "--num_graph",
    type=int,
    default=10,
    help="number of graphs to generate for each homophily level",
)
@click.option(
    "--graph_type",
    type=click.Choice(["regular", "random"]),
    default="random",
    help="type of the output synthetic graph: "
    "regular - all nodes have the same number of neighbours or "
    "random -  number of neighbours of different nodes may vary",
)
@click.option(
    "--edge_homos",
    multiple=True,
    default=[
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ],
    help="edge homophily of the output graph, can input more than one value",
)
def generate_graph(
    num_class, num_node_total, degree_intra, num_graph, graph_type, edge_homos
):
    node_per_class = num_node_total // num_class
    base_data_dir = f"{BASE_DIR}/{graph_type}"
    Path(base_data_dir).mkdir(parents=True, exist_ok=True)

    if graph_type == "regular":
        for edge_homo in edge_homos:
            for graph_num in range(num_graph):
                logger.log_init(f"Generating regular graph {graph_num} with edge homophily: {edge_homo}")
                degree_inter = int(degree_intra / edge_homo - degree_intra)
                output_label = generate_output_label(num_class, node_per_class)
                adj_matrix = np.zeros((num_node_total, num_node_total))
                for i in range(num_class):
                    for j in range(i * node_per_class, (i + 1) * 400):
                        # generate inner class adjacency
                        adj_matrix[
                            j,
                            random.sample(
                                set(
                                    np.delete(
                                        np.arange(i * node_per_class, (i + 1) * 400),
                                        np.where(
                                            np.arange(i * node_per_class, (i + 1) * 400)
                                            == j
                                        ),
                                    )
                                ),
                                degree_intra,
                            ),
                        ] = 1
                        # generate cross class adjacency
                        adj_matrix[
                            j,
                            random.sample(
                                set(
                                    np.delete(
                                        np.arange(0, num_node_total),
                                        np.arange(i * node_per_class, (i + 1) * 400),
                                    )
                                ),
                                degree_inter,
                            ),
                        ] = 1

                degree_matrix = np.diag(np.sum(adj_matrix, axis=1))

                # save generated graph matrices
                save_graphs(base_data_dir, edge_homo, graph_num, adj_matrix, degree_matrix, output_label)

    else:
        for edge_homo in edge_homos:
            for graph_num in range(num_graph):
                logger.log_init(f"Generating regular graph {graph_num} with edge homophily: {edge_homo}")
                degree_matrix = np.zeros((num_node_total, num_node_total))
                output_label = generate_output_label(num_class, node_per_class)
                adj_matrix = np.zeros((num_node_total, num_node_total))
                for i in range(num_class):
                    # generate inner class adjacency
                    num_edge_same = degree_intra * 400
                    adj_in_class = np.zeros((node_per_class, node_per_class))
                    adj_up_elements = np.array(
                        [1] * (int(num_edge_same / 2))
                        + [0]
                        * (
                            int(
                                (node_per_class - 1) * node_per_class / 2
                                - num_edge_same / 2
                            )
                        )
                    )
                    np.random.shuffle(adj_up_elements)
                    adj_in_class[np.triu_indices(node_per_class, 1)] = adj_up_elements
                    adj_in_class = adj_in_class + adj_in_class.T
                    adj_matrix[
                        node_per_class * i: node_per_class * (i + 1),
                        node_per_class * i: node_per_class * (i + 1),
                    ] = adj_in_class

                    # generate cross class adjacency
                    if i != num_class - 1:
                        if i == 0:
                            node_out_class = (
                                round(num_edge_same * (1 - edge_homo) / edge_homo) + 1
                            )
                        else:
                            existing_out_class_edges = np.sum(
                                adj_matrix[
                                    node_per_class * i : node_per_class * (i + 1),
                                    0: node_per_class * (i),
                                ]
                            )
                            node_out_class = (
                                round(
                                    num_edge_same * (1 - edge_homo) / edge_homo
                                    - existing_out_class_edges
                                )
                                + 1
                            )
                        adj_out_elements = np.array(
                            [1] * (node_out_class)
                            + [0]
                            * (
                                (num_class - 1 - i) * node_per_class ** 2
                                - node_out_class
                            )
                        )
                        np.random.shuffle(adj_out_elements)
                        adj_out_elements = adj_out_elements.reshape(
                            node_per_class, (num_class - 1 - i) * node_per_class
                        )
                        adj_matrix[
                            node_per_class * i : node_per_class * (i + 1),
                            node_per_class * (i + 1): node_per_class * (num_class),
                        ] = adj_out_elements
                        adj_matrix[
                            node_per_class * (i + 1): node_per_class * (num_class),
                            node_per_class * i: node_per_class * (i + 1),
                        ] = adj_out_elements.T
                    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))

                # save generated graph matrices
                save_graphs(base_data_dir, edge_homo, graph_num, adj_matrix, degree_matrix, output_label)


def save_graphs(
    base_data_dir, edge_homo, graph_num, adj_matrix, degree_matrix, output_label
):
    adj_matrix = torch.tensor(adj_matrix).to_sparse()
    degree_matrix = torch.tensor(degree_matrix).to_sparse()
    output_label = torch.tensor(output_label).to_sparse()

    DATA_DIR = f"{base_data_dir}/{edge_homo}"
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    torch.save(
        adj_matrix,
        f"{DATA_DIR}/adj_{edge_homo}_{graph_num}.pt",
    )
    torch.save(
        degree_matrix,
        f"{DATA_DIR}/degree_{edge_homo}_{graph_num}.pt",
    )
    torch.save(output_label, f"{DATA_DIR}/label_{edge_homo}_{graph_num}.pt")


if __name__ == "__main__":
    generate_graph()
