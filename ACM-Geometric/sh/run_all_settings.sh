#!/bin/bash
for data in "twitch-gamer" "pokec" "snap-patents" "arxiv-year" "genius" "Penn94"
do
    for dropout in 0 0.1 0.3 0.5 0.7
    do
        for lr in 0.002 0.01 0.05
        do
            for wd in 0 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2
            do
                for variant in 0 1
                do 
                    for structure_info in 0 1
                    do
                        for method in "acmgcnp" "acmgcnpp"
                        do
                            <your-working-directory>/Adaptive-Channel-Mixing-GNN/ACM-Geometric/sh/run_train.sh $data $method $lr $variant $structure_info $wd $dropout
                            <your-working-directory>/Adaptive-Channel-Mixing-GNN/ACM-Geometric/sh/run_train_adam.sh $data $method $lr $variant $structure_info $wd $dropout
                        done
                    done
                done
            done
        done
    done
done


