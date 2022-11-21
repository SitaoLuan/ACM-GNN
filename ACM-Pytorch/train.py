from __future__ import division
from __future__ import print_function

import time

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from arg_parser import arg_parser
from logger import ACMPythorchLogger
from models.models import GCN
from utils import (
    evaluate,
    eval_acc,
    data_split,
    random_disassortative_splits,
    train_model,
    train_prep,
)

logger = ACMPythorchLogger()
args = arg_parser()

(
    device,
    model_info,
    run_info,
    adj_high,
    adj_low,
    adj_low_unnormalized,
    features,
    labels,
    split_idx_lst,
) = train_prep(logger, args)

criterion = nn.NLLLoss()
eval_func = eval_acc

t_total = time.time()
epoch_total = 0
result = np.zeros(args.num_splits)

run_info["lr"] = args.lr
run_info["weight_decay"] = args.weight_decay
run_info["dropout"] = args.dropout

for idx in range(args.num_splits):
    run_info["split"] = idx
    model = GCN(
        nfeat=features.shape[1],
        nhid=args.hidden,
        nclass=labels.max().item() + 1,
        nlayers=args.layers,
        nnodes=features.shape[0],
        dropout=args.dropout,
        model_type=args.model,
        structure_info=args.structure_info,
        variant=args.variant,
        init_layers_X=args.link_init_layers_X,
    ).to(device)
    if args.dataset_name == "deezer-europe":
        idx_train, idx_val, idx_test = (
            split_idx_lst[idx]["train"].to(device),
            split_idx_lst[idx]["valid"].to(device),
            split_idx_lst[idx]["test"].to(device),
        )
        args.epochs = 500
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        if args.fixed_splits == 0:
            idx_train, idx_val, idx_test = random_disassortative_splits(
                labels, labels.max() + 1
            )
        else:
            idx_train, idx_val, idx_test = data_split(idx, args.dataset_name)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    if args.cuda:
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        model.cuda()

    curr_res = 0
    curr_training_loss = None
    best_val_loss = float("inf")
    val_loss_history = torch.zeros(args.epochs)

    for epoch in range(args.epochs):
        t = time.time()
        acc_train, loss_train = train_model(
            model,
            optimizer,
            adj_low,
            adj_high,
            adj_low_unnormalized,
            features,
            labels,
            idx_train,
            criterion,
            dataset_name=args.dataset_name,
        )

        model.eval()
        output = model(features, adj_low, adj_high, adj_low_unnormalized)
        if args.dataset_name == "deezer-europe":
            output = F.log_softmax(output, dim=1)
            val_loss, val_acc = criterion(
                output[idx_val], labels.squeeze(1)[idx_val]
            ), evaluate(output, labels, idx_val, eval_func)
        else:
            output = F.log_softmax(output, dim=1)
            val_loss, val_acc = criterion(output[idx_val], labels[idx_val]), evaluate(
                output, labels, idx_val, eval_func
            )
        if args.dataset_name == "deezer-europe":
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                curr_res = evaluate(output, labels, idx_test, eval_func)
                curr_training_loss = loss_train
        else:
            if val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                curr_res = evaluate(output, labels, idx_test, eval_func)
                curr_training_loss = loss_train
            if epoch >= 0:
                val_loss_history[epoch] = val_loss.detach()
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.mean(val_loss_history[epoch - args.early_stopping : epoch])
                if val_loss > tmp:
                    break

    epoch_total = epoch_total + epoch

    # Testing
    result[idx] = curr_res
    del model, optimizer
    if args.cuda:
        torch.cuda.empty_cache()

    total_time_elapsed = time.time() - t_total
    runtime_average = total_time_elapsed / args.num_splits
    epoch_average = total_time_elapsed / epoch_total * 1000

    run_info["runtime_average"] = runtime_average
    run_info["epoch_average"] = epoch_average
    run_info["result"] = np.mean(result)
    run_info["std"] = np.std(result)

    logger.log_time(f"{time.time() - t_total:.4f}s")
    logger.log_run(model_info, run_info)
