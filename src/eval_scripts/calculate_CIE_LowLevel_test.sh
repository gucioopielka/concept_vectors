#!/bin/bash
trap "echo 'Exiting script...'; exit" SIGINT # Exit by ctrl + c
cd ../

datasets=(
    'antonym_eng-mc' 
    'categorical_eng-mc'
    'english_spanish-mc' 
)
model_names=('meta-llama/Meta-LLama-3.1-8B')
batch_sizes=(32) # Corresponding batch sizes for each model

for index in ${!model_names[@]}; do
    model_name=${model_names[index]}
    base_layer_batch_size=${batch_sizes[index]}        
    while true; do
        echo "Running script for model ${model_name}..."
        python3 calculate_CIE.py \
            --model ${model_name} \
            --datasets ${datasets[@]} \
            --dataset_size 50 \
            --layer_batch_size ${base_layer_batch_size} \
            --prompt_batch_size 50 \
            --n_train 5 \
            --remote_run \
            --seed 42 \
            --output_dir CIE_LowLevel_test/
        
        if [ $? -eq 0 ]; then
            echo "Script finished successfully for model ${model_name}."
            break
        else
            echo "Script failed for model ${model_name}. Retrying in 10 seconds..."
            sleep 10
        fi
    done
done