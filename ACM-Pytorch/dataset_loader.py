import numpy as np
import torch

from datasets import DeezerDataset, HomophilicDataset, HeterophicDataset


class DatasetLoader:
    def __init__(self, name, device):
        if name == "deezer-europe":
            self.dataset = DeezerDataset()
        if name in {"cora", "citeseer", "pubmed"}:
            self.dataset = HomophilicDataset(name)
        else:
            self.dataset = HeterophicDataset(name)

        self.device = device

    def load_dataset(self):
        adj, features, labels = self.dataset.adj, self.dataset.features, self.dataset.labels

        features = torch.FloatTensor(features).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        adj = self._sparse_mx_to_torch_sparse_tensor(adj)  # .to(device)
        return adj, features, labels

    @staticmethod
    def _sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """
        Convert a scipy sparse matrix to a torch sparse tensor.
        """
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
