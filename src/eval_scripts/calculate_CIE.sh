#!/bin/bash
trap "echo 'Exiting script...'; exit" SIGINT # Exit by ctrl + c
cd ../

datasets=('antonym' 'capitalize' 'country-capital' 'english-french' 'present-past' 'singular-plural' 'person-instrument' 'person-sport' 'product-company' 'landmark-country')
model_names=('meta-llama/Meta-LLama-3.1-8B' 'meta-llama/Meta-LLama-3.1-70B')

for model_name in ${model_names[@]}; do
    while true; do
        python3 calculate_CIE.py \
            --model ${model_name} \
            --datasets ${datasets[@]} \
            --dataset_size 50 \
            --layer_batch_size 2 \
            --prompt_batch_size 50 \
            --n_train 5 \
            --remote_run \
            --seed 42 \
            --dataset_dir ../../data/todd_et_al/ \
            --output_dir CIE/
        
        if [ $? -eq 0 ]; then
            echo "Script finished successfully for model ${model_name}."
            break
        else
            echo "Script failed for model ${model_name}. Retrying in 10 seconds..."
            sleep 10
        fi
    done
done