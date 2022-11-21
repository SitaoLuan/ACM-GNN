#!/bin/bash
cd <your-working-directory>/Adaptive-Channel-Mixing-GNN/ACM-Geometric

python train.py --adam --dataset $1 --sub_dataset "" --method $2 --lr $3 --variant $4 --structure_info $5 --weight_decay $6 --dropout $7