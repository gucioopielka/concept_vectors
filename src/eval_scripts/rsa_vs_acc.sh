#!/bin/bash
trap "echo 'Exiting script...'; exit" SIGINT # Exit by ctrl + c
cd ../

model_names=('meta-llama/Meta-LLama-3.1-8B' 'meta-llama/Meta-LLama-3.1-70B')

for model_name in ${model_names[@]}; do
    for n_train in $(seq 1 10); do
        while true; do
            echo "Running script for model ${model_name}..."
            python3 compute_cv_simmat.py \
                --model ${model_name} \
                --n_train ${n_train} \
                --dataset_size 50 \
                --prompt_batch_size 50 \
                --remote_run \
                --seed 42 \
                --output_dir rsa_vs_acc/$(basename ${model_name})/ \
                --indicators_file ../data/task_attributes/indicators.json
            
            if [ $? -eq 0 ]; then
                echo "Script finished successfully for model ${model_name}."
                break
            else
                echo "Script failed for model ${model_name}. Retrying in 10 seconds..."
                sleep 10
            fi
        done
    done
done