#!/bin/bash
trap "echo 'Exiting script...'; exit" SIGINT # Exit by ctrl + c
cd ../

model_names='meta-llama/Meta-LLama-3.1-70B'

for index in ${!model_names[@]}; do
    model_name=${model_names[index]}
    while true; do
        echo "Running script for model ${model_name}..."
        python3 intervention.py \
            --model ${model_name} \
            --interleaved_datasets antonym_eng english_french \
            --extract_datasets antonym_eng antonym_fr antonym_eng-mc \
            --layer_batch_size 20 \
            --dataset_extract_size 50 \
            --dataset_eval_size 50 \
            --prompt_batch_size 50 \
            --n_train 10 \
            --remote_run \
            --seed 42 \
            --output_dir causal/ \
            --sleep_time 5
        
        if [ $? -eq 0 ]; then
            echo "Script finished successfully for model ${model_name}."
            break
        else
            echo "Script failed for model ${model_name}. Retrying in 10 seconds..."
            sleep 10
        fi
    done
done