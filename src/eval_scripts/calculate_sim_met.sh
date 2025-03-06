#!/bin/bash
trap "echo 'Exiting script...'; exit" SIGINT # Exit by ctrl + c
cd ../

model_names=('meta-llama/Meta-LLama-3.1-8B' 'meta-llama/Meta-LLama-3.1-70B')
batch_sizes=(2 1) # Corresponding batch sizes for each model

for index in ${!model_names[@]}; do
    model_name=${model_names[index]}
    layer_batch_size=${batch_sizes[index]}
    while true; do
        echo "Running script for model ${model_name}..."
        python3 calculate_sim_met.py \
            --model ${model_name} \
            --abstract_relations predecessor successor \
            --seq_len 20 \
            --dataset_size 50 \
            --layer_batch_size ${layer_batch_size} \
            --prompt_batch_size 50 \
            --n_train 5 \
            --remote_run \
            --seed 42 \
            --output_dir sim_mat/ \
            --indicators_file ../data/task_attributes.json
        
        if [ $? -eq 0 ]; then
            echo "Script finished successfully for model ${model_name}."
            break
        else
            echo "Script failed for model ${model_name}. Retrying in 10 seconds..."
            sleep 10
        fi
    done
done