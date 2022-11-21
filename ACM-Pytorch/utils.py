import os
from os import path
import pickle as pkl
import sys

from google_drive_downloader import GoogleDriveDownloader as gdd
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy
import scipy.io
import scipy.sparse
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize as sk_normalize
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    import scipy.io
    from torch_geometric.utils import add_self_loops, to_undirected

    from torch_sparse import SparseTensor


DATA_PATH = path.dirname(path.abspath(__file__)) + "/data/"
SPLITS_DRIVE_URL = {
    "snap-patents": "12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N",
    "pokec": "1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_",
}

device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
sys.setrecursionlimit(99999)


class NCDataset(object):
    def __init__(self, name, root=f"{DATA_PATH}"):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """
        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None


def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def data_split(idx, dataset_name):
    splits_file_path = "splits/" + dataset_name + "_split_0.6_0.2_" + str(idx) + ".npz"
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file["train_mask"]
        val_mask = splits_file["val_mask"]
        test_mask = splits_file["test_mask"]
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    return train_mask, val_mask, test_mask


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


@torch.no_grad()
def evaluate(output, labels, split_idx, eval_func):
    acc = eval_func(labels[split_idx], output[split_idx])
    return acc


def eval_acc(y_true, y_pred):
    if y_true.dim() > 1:
        acc_list = []
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct)) / len(correct))

        return sum(acc_list) / len(acc_list)
    else:
        preds = y_pred.max(1)[1].type_as(y_true)
        correct = preds.eq(y_true).double()
        correct = correct.sum()
        return correct / len(y_true)


def eval_rocauc(y_true, y_pred):
    """
    adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py
    """
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            "No positively labeled data available. Cannot compute ROC-AUC."
        )

    return sum(rocauc_list) / len(rocauc_list)


def even_quantile_labels(vals, nclasses, verbose=True):
    """
    partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on
    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print("Class Label Intervals:")
        for class_idx, interval in enumerate(interval_lst):
            print(f"Class {class_idx}: [{interval[0]}, {interval[1]})]")
    return label


def get_adj_high(adj_low):
    adj_high = -adj_low + sp.eye(adj_low.shape[0])
    return adj_high


def gen_normalized_adjs(dataset):
    """
    returns the normalized adjacency matrix
    """
    dataset.graph["edge_index"] = add_self_loops(dataset.graph["edge_index"])[0]
    row, col = dataset.graph["edge_index"]
    N = dataset.graph["num_nodes"]
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float("inf")] = 0
    DAD = D_isqrt.view(-1, 1) * adj * D_isqrt.view(1, -1)
    DA = D_isqrt.view(-1, 1) * D_isqrt.view(-1, 1) * adj
    AD = adj * D_isqrt.view(1, -1) * D_isqrt.view(1, -1)
    return DAD, DA, AD


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return adj, features, labels


def load_deezer_dataset():
    filename = "deezer-europe"
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f"{DATA_PATH}deezer-europe.mat")
    A, label, features = deezer["A"], deezer["label"], deezer["features"]
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]
    dataset.graph = {
        "edge_index": edge_index,
        "edge_feat": None,
        "node_feat": node_feat,
        "num_nodes": num_nodes,
    }
    dataset.label = label
    return dataset


def load_fixed_splits(dataset, sub_dataset):
    """
    loads saved fixed splits for dataset
    """
    name = dataset
    if sub_dataset and sub_dataset != "None":
        name += f"-{sub_dataset}"

    if not os.path.exists(f"./splits/{name}-splits.npy"):
        assert dataset in SPLITS_DRIVE_URL.keys()
        gdd.download_file_from_google_drive(
            file_id=SPLITS_DRIVE_URL[dataset],
            dest_path=f"./splits/{name}-splits.npy",
            showsize=True,
        )

    splits_lst = np.load(f"./splits/{name}-splits.npy", allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst


def load_full_data(dataset_name):
    if dataset_name in {"cora", "citeseer", "pubmed"}:
        adj, features, labels = load_data(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
    elif dataset_name == "deezer-europe":
        dataset = load_deezer_dataset()
        dataset.graph["edge_index"] = to_undirected(dataset.graph["edge_index"])
        row, col = dataset.graph["edge_index"]
        N = dataset.graph["num_nodes"]
        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph["node_feat"], dataset.label

    else:
        graph_adjacency_list_file_path = os.path.join(
            "../new_data", dataset_name, "out1_graph_edges.txt"
        )
        graph_node_features_and_labels_file_path = os.path.join(
            "../new_data", dataset_name, "out1_node_feature_label.txt"
        )

        G = nx.DiGraph().to_undirected()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == "film":
            with open(
                graph_node_features_and_labels_file_path
            ) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
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
            with open(
                graph_node_features_and_labels_file_path
            ) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
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

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split("\t")
                assert len(line) == 2
                if int(line[0]) not in G:
                    G.add_node(
                        int(line[0]),
                        features=graph_node_features_dict[int(line[0])],
                        label=graph_labels_dict[int(line[0])],
                    )
                if int(line[1]) not in G:
                    G.add_node(
                        int(line[1]),
                        features=graph_node_features_dict[int(line[1])],
                        label=graph_labels_dict[int(line[1])],
                    )
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))

        features = np.array(
            [
                features
                for _, features in sorted(G.nodes(data="features"), key=lambda x: x[0])
            ]
        )
        labels = np.array(
            [label for _, label in sorted(G.nodes(data="label"), key=lambda x: x[0])]
        )

    features = torch.FloatTensor(features).to(device)
    labels = torch.LongTensor(labels).to(device)
    adj = sparse_mx_to_torch_sparse_tensor(adj)  # .to(device)
    return adj, features, labels


def normalize(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = np.array(mx.sum(1))
    if eqvar:
        r_inv = np.power(rowsum, -1 / eqvar).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    else:
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_tensor(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx


def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0) * 1 + row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def random_disassortative_splits(labels, num_classes):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing
    labels, num_classes = labels.cpu(), num_classes.cpu().numpy()
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    percls_trn = int(round(0.6 * (labels.size()[0] / num_classes)))
    val_lb = int(round(0.2 * labels.size()[0]))
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=labels.size()[0])
    val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
    test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])

    return train_mask.to(device), val_mask.to(device), test_mask.to(device)


def preprocess_features(features):
    """
    Row-normalize feature matrix and convert to tuple representation
    """
    rowsum = np.array(features.sum(1))
    r_inv = (1.0 / rowsum).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def rand_train_test_idx(label, train_prop=0.6, valid_prop=0.2, ignore_negative=True):
    """
    randomly splits label into train/valid/test splits
    """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num : train_num + valid_num]
    test_indices = perm[train_num + valid_num :]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def row_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sk_normalize(adj, norm="l1", axis=1)
    return sp.coo_matrix(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
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


def train_model(
    model,
    optimizer,
    adj_low,
    adj_high,
    adj_low_unnormalized,
    features,
    labels,
    idx_train,
    criterion,
    dataset_name,
):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_low, adj_high, adj_low_unnormalized)
    if dataset_name == "deezer-europe":
        output = F.log_softmax(output, dim=1)
        loss_train = criterion(output[idx_train], labels.squeeze(1)[idx_train])
        acc_train = eval_acc(labels[idx_train], output[idx_train])
    else:
        output = F.log_softmax(output, dim=1)
        loss_train = criterion(output[idx_train], labels[idx_train])
        acc_train = accuracy(labels[idx_train], output[idx_train])

    loss_train.backward()
    optimizer.step()

    return 100 * acc_train.item(), loss_train.item()


def train_prep(logger, args):
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    split_idx_lst = None

    # Training settings
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Run info
    model_info = {
        "model": args.model,
        "structure_info": args.structure_info,
        "dataset_name": args.dataset_name,
        "hidden": args.hidden,
        "init_layers_X": args.link_init_layers_X,
        "variant": args.variant,
        "layers": args.layers,
        "hop": args.hops,
    }

    run_info = {
        "result": 0,
        "std": 0,
        "lr": None,
        "weight_decay": None,
        "dropout": None,
        "runtime_average": None,
        "epoch_average": None,
        "split": None,
    }

    logger.log_init("Done Proccessing...")
    adj_low_unnormalized, features, labels = load_full_data(args.dataset_name)
    if (args.model == "acmgcnp" or args.model == "acmgcnpp") and (
        args.structure_info == 1
    ):
        pass
    else:
        features = normalize_tensor(features)

    nnodes = labels.shape[0]
    if args.structure_info:
        adj_low = normalize_tensor(torch.eye(nnodes) + adj_low_unnormalized.to_dense())
        adj_high = (torch.eye(nnodes) - adj_low).to(device).to_sparse()
        adj_low = adj_low.to(device)
        adj_low_unnormalized = adj_low_unnormalized.to(device)
    else:
        adj_low = normalize_tensor(torch.eye(nnodes) + adj_low_unnormalized.to_dense())
        adj_high = (torch.eye(nnodes) - adj_low).to(device).to_sparse()
        adj_low = adj_low.to(device)
        adj_low_unnormalized = None

    if (args.model == "acmsgc") and (args.hops > 1):
        A_EXP = adj_low.to_dense()
        for _ in range(args.hops - 1):
            A_EXP = torch.mm(A_EXP, adj_low.to_dense())
        adj_low = A_EXP.to_sparse()
        del A_EXP
        adj_low = adj_low.to(device).to_sparse()

    if args.dataset_name == "deezer-europe":
        args.num_splits = 5
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1)
        labels = labels.to(device)
        split_idx_lst = load_fixed_splits(args.dataset_name, "")

    return (
        device,
        model_info,
        run_info,
        adj_high,
        adj_low,
        adj_low_unnormalized,
        features,
        labels,
        split_idx_lst,
    )
