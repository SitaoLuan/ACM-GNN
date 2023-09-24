import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import torch

from abc import ABC, abstractmethod
from os import path


if torch.cuda.is_available():
    import scipy.io
    from torch_geometric.utils import to_undirected


class _BaseDataset(ABC):
    """
    Base class for Dataset
    """
    def __init__(self, name):
        self.name = name
        self.base_dir = path.dirname(path.abspath(__file__)) + "/data/"

    @property
    @abstractmethod
    def adj(self):
        pass

    @property
    @abstractmethod
    def features(self):
        pass

    @property
    @abstractmethod
    def labels(self):
        pass

    @abstractmethod
    def _load(self):
        pass


class DeezerDataset(_BaseDataset):
    def __init__(self, name="deezer-europe"):
        super().__init__(name)
        self.dataset = self._load()

    @property
    def adj(self):
        unprocessed_adj = self.dataset["A"]

        edge_index = torch.tensor(unprocessed_adj.nonzero(), dtype=torch.long)
        row, col = to_undirected(edge_index)

        num_nodes = self.labels.shape[0]
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(num_nodes, num_nodes))
        return adj

    @property
    def features(self):
        unprocessed_features = self.dataset["label"]
        features = torch.tensor(unprocessed_features.todense(), dtype=torch.float)

        return features

    @property
    def labels(self):
        unprocessed_labels = self.dataset["label"]
        labels = torch.tensor(unprocessed_labels, dtype=torch.long).squeeze()

        return labels

    def _load(self):
        dataset = scipy.io.loadmat(f"{self.base_dir}deezer-europe.mat")
        return dataset


class HeterophilicDataset(_BaseDataset):
    def __init__(self, name):
        super().__init__(name)
        self.dataset = self._load()

    @property
    def adj(self):
        return nx.adjacency_matrix(self.dataset, sorted(self.dataset.nodes()))

    @property
    def features(self):
        features = np.array(
            [
                features
                for _, features in sorted(self.dataset.nodes(data="features"), key=lambda x: x[0])
            ]
        )
        return features

    @property
    def labels(self):
        labels = np.array(
            [label for _, label in sorted(self.dataset.nodes(data="label"), key=lambda x: x[0])]
        )
        return labels

    def _load(self):
        graph = nx.DiGraph().to_undirected()

        feature_label_path = os.path.join(
            "../new_data", self.name, "out1_node_feature_label.txt"
        )
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if self.name == "film":
            with open(feature_label_path) as feature_label_fp:
                feature_label_fp.readline()
                for line in feature_label_fp:
                    line = line.rstrip().split("\t")
                    assert len(line) == 3
                    assert (
                        int(line[0]) not in graph_node_features_dict
                        and int(line[0]) not in graph_labels_dict
                    )
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(","), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(feature_label_path) as feature_label_fp:
                feature_label_fp.readline()
                for line in feature_label_fp:
                    line = line.rstrip().split("\t")
                    assert len(line) == 3
                    assert (
                        int(line[0]) not in graph_node_features_dict
                        and int(line[0]) not in graph_labels_dict
                    )
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(","), dtype=np.uint8
                    )
                    graph_labels_dict[int(line[0])] = int(line[2])

        adj_path = os.path.join(
            "../new_data", self.name, "out1_graph_edges.txt"
        )
        with open(adj_path) as adj_fp:
            adj_fp.readline()
            for line in adj_fp:
                line = line.rstrip().split("\t")
                assert len(line) == 2
                if int(line[0]) not in graph:
                    graph.add_node(
                        int(line[0]),
                        features=graph_node_features_dict[int(line[0])],
                        label=graph_labels_dict[int(line[0])],
                    )
                if int(line[1]) not in graph:
                    graph.add_node(
                        int(line[1]),
                        features=graph_node_features_dict[int(line[1])],
                        label=graph_labels_dict[int(line[1])],
                    )
                graph.add_edge(int(line[0]), int(line[1]))

        return graph


class HomophilicDataset(_BaseDataset):
    def __init__(self, name):
        super().__init__(name)
        self.dataset = self._load()
        self.test_index_list = self._load_test_index_list()
        self.test_index_range = np.sort(self.test_index_list)

    @property
    def adj(self):
        return nx.adjacency_matrix(nx.from_dict_of_lists(self.dataset["graph"]))

    @property
    def features(self):
        features = sp.vstack((self.dataset["allx"], self.dataset["tx"])).tolil()
        features[self.test_index_list, :] = features[self.test_index_range, :]
        features = features.todense()

        return features

    @property
    def labels(self):
        labels = np.vstack((self.dataset["ally"], self.dataset["ty"]))
        labels[self.test_index_list, :] = labels[self.test_index_range, :]
        labels = np.argmax(labels, axis=-1)

        return labels

    def _load(self):
        graph = dict.fromkeys(["x", "y", "tx", "ty", "allx", "ally", "graph"])
        for graph_component in graph.keys():
            with open("../data/ind.{}.{}".format(self.name, graph_component), "rb") as f:
                if sys.version_info > (3, 0):
                    graph[graph_component] = pkl.load(f, encoding="latin1")
                else:
                    graph[graph_component] = pkl.load(f)

        if self.name == "citeseer":
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position

            test_idx_range_full = range(min(self.test_index_list), max(self.test_index_list) + 1)

            tx_extended = sp.lil_matrix((len(test_idx_range_full), graph["x"].shape[1]))
            tx_extended[self.test_index_range - min(self.test_index_range), :] = graph["tx"]
            graph["tx"] = tx_extended

            ty_extended = np.zeros((len(test_idx_range_full), graph["y"].shape[1]))
            ty_extended[self.test_index_range - min(self.test_index_range), :] = graph["ty"]
            graph["ty"] = ty_extended

        return graph

    def _load_test_index_list(self):
        index = []
        for line in open("../data/ind.{}.test.index".format(self.name)):
            index.append(int(line.strip()))

        return index

