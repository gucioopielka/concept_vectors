#!/bin/bash
n_train_list=(3 2 1)
cd ../

for n_train in "${n_train_list[@]}"
do
    echo "Running Script for: ${n_train} training examples"
    python3 compute_ICL_FVs.py --n_train ${n_train} --batch_size 3
done