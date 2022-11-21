from __future__ import division
from __future__ import print_function

import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from arg_parser import arg_parser
from baseline_models.models import GCN
from logger import SyntheticExpLogger
from utils import accuracy, load_synthetic_data, normalize, random_disassortative_splits


logger = SyntheticExpLogger()


# Training settings
args = arg_parser()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.pi = torch.acos(torch.zeros(1)).item() * 2

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

num_edge_same = args.degree_intra * 400

# Log info
model_info = {
    "model_type": args.model_type,
    "graph_type": args.graph_type,
    "num_edge_same": num_edge_same,
    "edge_homo": args.edge_homo,
    "base_dataset": args.base_dataset,
    "lr": args.lr,
    "weight_decay": args.weight_decay,
    "dropout": args.dropout,
}

run_info = {
    "result": 0,
    "graph_idx": 0
}

record_info = {
    "test_result": 0,
    "test_std": 0,
    "graph_idx": 0
}

# Load data
t_total = time.time()

best_result = 0
best_std = 0
best_dropout = None
best_weight_decay = None

result = np.zeros(args.num_graph)
for sample in range(args.num_graph):
    run_info["graph_idx"] = sample

    adj, labels, degree, features = load_synthetic_data(
        args.graph_type, sample, args.edge_homo, args.base_dataset
    )

    nnodes = adj.shape[0]
    adj_dense = adj
    adj_dense[adj_dense != 0] = 1
    adj_dense = adj_dense - torch.diag(torch.diag(adj_dense))
    adj_low = torch.tensor(normalize(adj_dense + torch.eye(nnodes)))
    adj_high = torch.eye(nnodes) - adj_low
    adj_low = adj_low.to_sparse()
    adj_high = adj_high.to_sparse()

    if args.cuda:
        features = features.cuda()
        adj_low = adj_low.cuda()
        adj_high = adj_high.cuda()
        labels = labels.cuda()

    def test():  # isolated_mask
        model.eval()
        output = model(features, adj_low, adj_high)
        output = F.log_softmax(output, dim=1)
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test

    # Train model
    idx_train, idx_val, idx_test = random_disassortative_splits(
        labels, labels.max() + 1
    )

    model = GCN(
        nfeat=features.shape[1],
        nhid=args.hidden,
        nclass=labels.max().item() + 1,
        dropout=args.dropout,
        model_type=args.model_type,
    )
    if args.cuda:
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        model.cuda()

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_training_loss = None
    best_val_acc = 0
    best_val_loss = float("inf")
    val_loss_history = torch.zeros(args.epochs)
    best_test = 0

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_low, adj_high)
        output = F.log_softmax(output, dim=1)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj_low, adj_high)
            output = F.log_softmax(output, dim=1)

        val_loss = F.nll_loss(output[idx_val], labels[idx_val])
        val_acc = accuracy(output[idx_val], labels[idx_val])

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_test = test()  # isolated_mask
            best_training_loss = loss_train

        if epoch >= 0:
            val_loss_history[epoch] = val_loss.detach()
        if args.early_stopping > 0 and epoch > args.early_stopping:
            tmp = torch.mean(
                val_loss_history[epoch - args.early_stopping : epoch]
            )
            if val_loss > tmp:
                break

    run_info["result"] = best_test
    logger.log_run(run_info)

    # Testing
    result[sample] = best_test
    del model, optimizer
    if args.cuda:
        torch.cuda.empty_cache()

    if np.mean(result) > best_result:
        record_info["result"] = np.mean(result)
        record_info["std"] = np.std(result)
        record_info["graph_idx"] = sample

logger.log_record(model_info, record_info)