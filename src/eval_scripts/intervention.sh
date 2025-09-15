#!/bin/bash
trap "echo 'Exiting script...'; exit" SIGINT # Exit by ctrl + c
cd ../

model_names='meta-llama/Meta-LLama-3.1-70B'

for index in ${!model_names[@]}; do
    model_name=${model_names[index]}
    while true; do
        echo "Running script for model ${model_name}..."
        python3 intervention_cache.py \
            --model ${model_name} \
            --intervention_type zeroshot \
            --extract_size 50 \
            --intervene_size 100 \
            --n_heads 5 \
            --concepts antonym categorical causal presentPast singularPlural synonym \
            --weights 1 5 10 \
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