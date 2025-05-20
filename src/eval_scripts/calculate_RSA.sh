#!/bin/bash
trap "echo 'Exiting script...'; exit" SIGINT # Exit by ctrl + c
cd ../

datasets=(
    'antonym_eng' 
    'antonym_fr' 
    'antonym_eng-mc' 
    'categorical_eng' 
    'categorical_fr' 
    'categorical_eng-mc'
    'translation_eng_es'
    'translation_de_fr'
    'translation_eng_es-mc'
    'presentPast_eng'
    'presentPast_fr'
    'presentPast_eng-mc'
    'singularPlural_eng'
    'singularPlural_fr'
    'singularPlural_eng-mc'
    'synonym_eng'
    'synonym_fr'
    'synonym_eng-mc'    
)
model_names=('meta-llama/Meta-LLama-3.1-8B')
batch_sizes=(10 20) # Corresponding batch sizes for each model

for index in ${!model_names[@]}; do
    model_name=${model_names[index]}
    layer_batch_size=${batch_sizes[index]}
    while true; do
        echo "Running script for model ${model_name}..."
        python3 calculate_RSA.py \
            --model ${model_name} \
            --datasets ${datasets[@]} \
            --dataset_size 50 \
            --prompt_batch_size 50 \
            --layer_batch_size ${layer_batch_size} \
            --n_train 5 \
            --remote_run \
            --seed 42 \
        
        if [ $? -eq 0 ]; then
            echo "Script finished successfully for model ${model_name}."
            break
        else
            echo "Script failed for model ${model_name}. Retrying in 10 seconds..."
            sleep 10
        fi
    done
done