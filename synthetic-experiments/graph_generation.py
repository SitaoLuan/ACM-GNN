import argparse
import random
from pathlib import Path

import numpy as np
import torch
from logger import SyntheticExpLogger
from utils import generate_output_label

BASE_DIR = "./synthetic_graphs"
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

logger = SyntheticExpLogger()


def generate_graph(args):
    node_per_class = args.num_node_total // args.num_class
    base_data_dir = f"{BASE_DIR}/{args.graph_type}"
    Path(base_data_dir).mkdir(parents=True, exist_ok=True)

    if args.graph_type == "regular":
        for edge_homo in args.edge_homos:
            for graph_num in range(args.num_graph):
                logger.log_init(f"Generating regular graph {graph_num} with edge homophily: {edge_homo}")
                degree_inter = int(args.degree_intra / edge_homo - args.degree_intra)
                output_label = generate_output_label(args.num_class, node_per_class)
                adj_matrix = np.zeros((args.num_node_total, args.num_node_total))
                for i in range(args.num_class):
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
                                args.degree_intra,
                            ),
                        ] = 1
                        # generate cross class adjacency
                        adj_matrix[
                            j,
                            random.sample(
                                set(
                                    np.delete(
                                        np.arange(0, args.num_node_total),
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
        for edge_homo in args.edge_homos:
            for graph_num in range(args.num_graph):
                logger.log_init(f"Generating regular graph {graph_num} with edge homophily: {edge_homo}")
                degree_matrix = np.zeros((args.num_node_total, args.num_node_total))
                output_label = generate_output_label(args.num_class, node_per_class)
                adj_matrix = np.zeros((args.num_node_total, args.num_node_total))
                for i in range(args.num_class):
                    # generate inner class adjacency
                    num_edge_same = args.degree_intra * 400
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
                    if i != args.num_class - 1:
                        if i == 0:
                            node_out_class = (
                                    round(num_edge_same * (1 - edge_homo) / edge_homo) + 1
                            )
                        else:
                            existing_out_class_edges = np.sum(
                                adj_matrix[
                                node_per_class * i: node_per_class * (i + 1),
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
                                    (args.num_class - 1 - i) * node_per_class ** 2
                                    - node_out_class
                            )
                        )
                        np.random.shuffle(adj_out_elements)
                        adj_out_elements = adj_out_elements.reshape(
                            node_per_class, (args.num_class - 1 - i) * node_per_class
                        )
                        adj_matrix[
                        node_per_class * i: node_per_class * (i + 1),
                        node_per_class * (i + 1): node_per_class * (args.num_class),
                        ] = adj_out_elements
                        adj_matrix[
                        node_per_class * (i + 1): node_per_class * (args.num_class),
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=5, help='Number of classes')
    parser.add_argument('--num_node_total', type=int, default=2000, help='Total number of nodes in graph')
    parser.add_argument('--degree_intra', type=int, default=2,
                        help='Number of neighbors in the same class for each node')
    parser.add_argument('--num_graph', type=int, default=10,
                        help='Number of graphs to generate for each homophily level')
    parser.add_argument('--graph_type',
                        type=str,
                        default='random',
                        choices=['regular', 'random'],
                        help='Type of the output synthetic graph: '
                             'regular - all nodes have the same number of neighbours or '
                             '"random -  number of neighbours of different nodes may vary"')
    parser.add_argument('--edge_homos',
                        type=int,
                        nargs='+',
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
                        help='Edge homophily level of the output graph, can input more than one value')
    args = parser.parse_args()

    generate_graph(args)
