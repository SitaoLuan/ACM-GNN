from __future__ import division
from __future__ import print_function

import itertools
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
}
run_info = {
    "result": 0,
    "loss": 0,
    "graph_idx": 0,
    "weight_decay": None,
    "dropout": None
}
best_result_info = {
    "result": 0,
    "std": 0,
    "weight_decay": None,
    "dropout": None
}

# Hyperparameter search range
weight_decay = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3]

if (
        args.model_type == "sgc"
):
    dropout = [0]
else:
    dropout = [0.1, 0.3, 0.5, 0.7, 0.9]


# Load data
t_total = time.time()

num_edge_same = args.degree_intra * 400

best_std = 0
best_dropout = None
best_weight_decay = None

logger.log_init("Start hyperparameter Searching...")

for curr_weight_decay, curr_dropout in itertools.product(
        weight_decay, dropout
):
    result = np.zeros(args.num_graph)
    curr_result = 0

    run_info["weight_decay"] = curr_weight_decay
    run_info["dropout"] = curr_dropout

    for sample in range(args.num_graph):
        run_info["graph_idx"] = sample

        adj, labels, degree, features = load_synthetic_data(
            args.graph_type, sample, args.edge_homo, args.base_dataset
        )

        nnodes = adj.shape[0]

        adj_dense = adj  # adj.to_dense() ##

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
            dropout=curr_dropout,
            model_type=args.model_type,
        )  # , isolated_mask = isolated_mask
        if args.cuda:
            idx_train = idx_train.cuda()
            idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()
            model.cuda()

        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=curr_weight_decay
        )
        best_val_acc = 0
        best_val_loss = float("inf")
        val_loss_history = torch.zeros(args.epochs)
        best_test = 0
        best_training_loss = None

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

                run_info["loss"] = loss_train
                run_info["result"] = best_test

            if epoch >= 0:
                val_loss_history[epoch] = val_loss.detach()
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.mean(
                    val_loss_history[epoch - args.early_stopping: epoch]
                )
                if val_loss > tmp:
                    break

        # Testing
        result[sample] = best_test
        del model, optimizer
        if args.cuda:
            torch.cuda.empty_cache()
        logger.log_param_tune(model_info, run_info)

    if np.mean(result) > best_result_info["result"] :
        curr_result = np.mean(result)
        best_result_info["result"] = np.mean(result)
        best_result_info["std"] = np.std(result)
        best_result_info["dropout"] = curr_dropout
        best_result_info["weight_decay"] = curr_weight_decay

logger.log_best_result(model_info, best_result_info)
