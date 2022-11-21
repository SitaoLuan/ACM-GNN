import numpy as np
import torch

# from torch_scatter import scatter_add
# from torch_geometric.utils import remove_self_loops


def edge_homophily(adj, label):
    """
    gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    """
    adj = (adj > 0).float()
    adj = adj - torch.diag(torch.diag(adj))
    label_adj = torch.mm(label, label.transpose(0, 1))
    edge_hom = torch.sum(label_adj * adj) / torch.sum(adj)

    return edge_hom


def compat_matrix(A, labels):
    """
    c x c compatibility matrix, where c is number of classes
    H[i,j] is proportion of endpoints that are class j
    of edges incident to class i nodes
    See Zhu et al. 2020
    """
    c = len(np.unique(labels))
    H = np.zeros((c, c))
    src_node, targ_node = A.nonzero()
    for i in range(len(src_node)):
        src_label = labels[src_node[i]]
        targ_label = labels[targ_node[i]]
        H[src_label, targ_label] += 1
    H = H / np.sum(H, axis=1, keepdims=True)
    return H


def node_homophily(A, labels):
    """average of homophily for each node"""

    A = A - torch.diag(torch.diag(A))
    src_node, targ_node = A.nonzero()[:, 0], A.nonzero()[:, 1]
    edge_idx = torch.tensor(
        np.vstack((src_node, targ_node)), dtype=torch.long
    ).contiguous()
    labels = torch.tensor(labels)
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)


def node_homophily_edge_idx(edge_index, labels, num_nodes):
    """edge_idx is 2 x(number edges)"""
    # edge_index = remove_self_loops(edge_idx)[0]
    hs = torch.zeros(num_nodes)
    degs = torch.bincount(edge_index[0, :]).float()
    matches = (labels[edge_index[0, :]] == labels[edge_index[1, :]]).float()
    hs = hs.scatter_add(0, edge_index[0, :], matches) / degs
    return hs[degs != 0].mean()


def compat_matrix_edge_idx(edge_index, labels):
    """
    c x c compatibility matrix, where c is number of classes
    H[i,j] is proportion of endpoints that are class j
    of edges incident to class i nodes
    "Generalizing GNNs Beyond Homophily"
    treats negative labels as unlabeled
    """
    # edge_index = remove_self_loops(edge_idx)[0]
    src_node, targ_node = edge_index[:, 0], edge_index[:, 1]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max() + 1
    H = torch.zeros((c, c)).to(edge_index.device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    label_idx = torch.cat((src_label.unsqueeze(0), targ_label.unsqueeze(0)), axis=0)
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        H[k, :].scatter_add_(
            src=torch.ones_like(add_idx).to(H.dtype), dim=-1, index=add_idx
        )
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H


def class_homophily(A, label):
    """
    our measure \hat{h}
    treats negative labels as unlabeled
    """
    A = A - torch.diag(torch.diag(A))
    A = A + torch.diag((torch.sum(A, 1) == 0).float())
    edge_index = A.nonzero()
    label = label.squeeze()
    c = label.max() + 1
    H = compat_matrix_edge_idx(edge_index, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k, k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c - 1
    return val


def aggregation_homophily(features, adj, label, modified=True):
    inner_prod = torch.mm(
        torch.mm(adj, features), torch.mm(adj, features).transpose(0, 1)
    )
    labels = torch.argmax(label, 1)
    weight_matrix = torch.zeros(
        adj.clone().detach().size(0), labels.clone().detach().max() + 1
    )
    for i in range(labels.max() + 1):
        weight_matrix[:, i] = torch.mean(inner_prod[:, labels == i], 1)
    return torch.mean(torch.argmax(weight_matrix, 1).eq(labels).float())
