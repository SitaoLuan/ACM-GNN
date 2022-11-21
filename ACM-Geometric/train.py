import torch
import argparse
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import time
from parse import parser_add_main_args, parse_method
from dataset import load_nc_dataset
from data_utils import load_fixed_splits, to_sparse_tensor
from data_utils import eval_acc, eval_rocauc
from data_utils import evaluate_acmgcn
from utils import normalize_tensor, sparse_mx_to_torch_sparse_tensor
from logger import Logger
from torch_geometric.utils import to_undirected
from torch_geometric.utils.convert import to_scipy_sparse_matrix


parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

### Load and preprocess data ###
dataset = load_nc_dataset(args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

dataset.label = dataset.label.to(device)

if args.rand_split or args.dataset in ['ogbn-proteins', 'wiki']:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.num_splits)]
else:
    split_idx_lst = load_fixed_splits(args.dataset, args.sub_dataset)

# only for ogbn-proteins (can delete this part since we do not test ogbn-proteins)
if args.dataset == 'ogbn-proteins':
    # if args.method == 'mlp' or args.method == 'cs':
    #     dataset.graph['node_feat'] = scatter(dataset.graph['edge_feat'], dataset.graph['edge_index'][0],
    #                                          dim=0, dim_size=dataset.graph['num_nodes'], reduce='mean')
    # else:
    dataset.graph['edge_index'] = to_sparse_tensor(dataset.graph['edge_index'],
                                                    dataset.graph['edge_feat'], dataset.graph['num_nodes'])
    dataset.graph['node_feat'] = dataset.graph['edge_index'].mean(dim=1)
    dataset.graph['edge_index'].set_value_(None)
    dataset.graph['edge_feat'] = None


n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot labels
c = dataset.label.max().item() + 1
d = dataset.graph['node_feat'].shape[1]
print(f"num nodes {n} | num classes {c} | num node feats {d}")

# whether or not to symmetrize matters a lot!! pay attention to this
# e.g. directed edges are temporally useful in arxiv-year,
# so we usually do not symmetrize, but for label prop symmetrizing helps
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

if (args.method == 'acmgcnp' or args.method == 'acmgcnpp') and (args.structure_info == 1) :
    pass
else:
    dataset.graph['node_feat'] = normalize_tensor(dataset.graph['node_feat'])
    dataset.graph['node_feat'] = torch.tensor(dataset.graph['node_feat'].todense())

x = dataset.graph['node_feat'].to(device)
adj_low_unnormalized = to_scipy_sparse_matrix(dataset.graph['edge_index'])
adj_low = normalize_tensor(sp.identity(n) + adj_low_unnormalized)
adj_high = sp.identity(n) - adj_low
adj_low = sparse_mx_to_torch_sparse_tensor(adj_low).to(device)
adj_high = sparse_mx_to_torch_sparse_tensor(adj_high).to(device)
adj_low_unnormalized = sparse_mx_to_torch_sparse_tensor(adj_low_unnormalized).to(device)

if not args.structure_info:
   adj_low_unnormalized = None

# using rocauc as the eval function
if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc
else:
    criterion = nn.NLLLoss()
    eval_func = eval_acc


### Training loop ###

if args.rand_split:
    splits = args.num_splits
else:
    splits = len(split_idx_lst)

logger = Logger(splits, args)

total_time_list = []
avg_epoch_time_list = []
last_time = time.time()
for run in range(splits):  
    total_time = 0  
    split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model = parse_method(args, n, c, d, device)
    if args.adam:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, adj_low, adj_high, adj_low_unnormalized)
        if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
            if dataset.label.shape[1] == 1:
                # change -1 instances to 0 for one-hot transform
                # dataset.label[dataset.label==-1] = 0
                true_label = F.one_hot(
                    dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx].to(torch.int64))
        loss.backward()
        optimizer.step()
        result = evaluate_acmgcn(model, x, adj_low, adj_high, adj_low_unnormalized, 
                                dataset, split_idx, eval_func)               
        logger.add_result(run, result[:-1])
        if result[1] > best_val:
            best_val = result[1]
            if args.dataset != 'ogbn-proteins':
                best_out = F.softmax(result[-1], dim=1)
            else:
                best_out = result[-1]
        now_time = time.time()
        time_elapsed = now_time - last_time
        last_time = now_time
        total_time += time_elapsed
        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Time: {time_elapsed:.4f}')
    logger.print_statistics(run)
    total_time_list.append(total_time)
    avg_epoch_time_list.append(total_time/args.epochs)


best_val, best_test = logger.print_statistics()
filename = f'results/{args.dataset}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{args.method}, " + f"{sub_dataset}, " +
                    f"{best_val.mean():.3f}, {best_val.std():.3f}," +
                    f"{best_test.mean():.3f}, {best_test.std():.3f}, " +
                    f"{np.mean(avg_epoch_time_list) * 1000:.2f}ms/{np.mean(total_time_list):.2f}s, " +
                    f"{args}\n")