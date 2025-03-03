#!/bin/bash
trap "echo 'Exiting script...'; exit" SIGINT # Exit by ctrl + c
cd ../

model_names=('meta-llama/Meta-LLama-3.1-8B' 'meta-llama/Meta-LLama-3.1-70B')

for model_name in ${model_names[@]}; do
    while true; do
        echo "Running script for model ${model_name}..."
        python3 generalization.py \
            --model ${model_name} \
            --dataset_ids 'next_item' 'prev_item' 'successor_letter' 'successor_word' 'predecessor_letter' 'predecessor_word' \
            --dataset_size 20 \
            --batch_size 20 \
            --n_train 5 \
            --remote_run \
            --seq_len 20 \
            --seed 42 \
            --output_dir generalization/
        
        if [ $? -eq 0 ]; then
            echo "Script finished successfully for model ${model_name}."
            break
        else
            echo "Script failed for model ${model_name}. Retrying in 10 seconds..."
            sleep 10
        fi
    done
done