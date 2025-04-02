import argparse
import json
import os
import time
from functools import wraps
import numpy as np
import sys
import pandas as pd

from utils.query_utils import get_completions, get_avg_summed_vec, intervene_with_vec
from utils.ICL_utils import ICLDataset, DatasetConstructor
from utils.model_utils import ExtendedLanguageModel
from utils.eval_utils import accuracy_completions, batch_process_layers
from utils.globals import RESULTS_DIR


if __name__ == "__main__":
    # check if we're running from the command line
    if len(sys.argv) > 2:
        parser = argparse.ArgumentParser(description="Compute function vectors for a given model.")
        
        parser.add_argument("--model", type=str, help="Name of the model to use.")
        parser.add_argument("--interleaved_datasets", nargs="+", type=str, help="List of datasets to intervene on.")
        parser.add_argument("--extract_datasets", nargs="+", type=str, help="List of datasets to extract vectors from.")
        parser.add_argument("--layer_batch_size", type=int, help="Number of layers to process at once.", default=4)
        parser.add_argument("--dataset_extract_size", type=int, help="Size of the dataset to extract vectors from.", default=50)
        parser.add_argument("--dataset_eval_size", type=int, help="Size of the dataset to evaluate on.", default=50)
        parser.add_argument("--prompt_batch_size", type=int, help="Number of prompts to process at once.", default=50)
        parser.add_argument('--n_train', type=int, help="Number of training examples to use.", default=5)
        parser.add_argument("--remote_run", type=bool, help="Whether to run the script on a remote server.", default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument("--seed", type=int, help="Random seed to use.", default=42)
        parser.add_argument("--output_dir", type=str, help="Path to save the output CSV file.", default='data/causal')
        parser.add_argument("--sleep_time", type=int, help="Time to sleep between remote runs.", default=10)

        args = parser.parse_args()
    else:
        args = {
            'model': 'meta-llama/Meta-LLama-3.1-70B',
            'interleaved_datasets': ['antonym_eng', 'english_french'],
            'extract_datasets': ['antonym_eng', 'antonym_fr', 'antonym_eng-mc'],
            'layer_batch_size': 4,
            'dataset_extract_size': 50,
            'dataset_eval_size': 50,
            'prompt_batch_size': 50,
            'n_train': 5,
            'remote_run': True,
            'sleep_time': 5,
            'seed': 42,
            'output_dir': 'intervention/'
        }
        args = argparse.Namespace(**args)

    def sleep_after_call(sleep_time):
        """Decorator to sleep after calling the function."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                time.sleep(sleep_time)
                return result
            return wrapper
        return decorator
    
    # Sleep after each call
    intervene_with_vec = sleep_after_call(args.sleep_time)(intervene_with_vec)
    get_completions = sleep_after_call(args.sleep_time)(get_completions)
    get_avg_summed_vec = sleep_after_call(args.sleep_time)(get_avg_summed_vec)


    # Load the model 
    model = ExtendedLanguageModel(args.model, cv_heads_n=3, fv_heads_n=3)
    n_heads = model.config['n_heads']
    n_layers = model.config['n_layers']
    

    ### Calculate the Relation and Function Vectors ###

    fvs = {}
    rvs = {}
    for dataset in args.extract_datasets:
        print(f"\nExtracting dataset {dataset}...")
        dataset_extract = DatasetConstructor(
            dataset_ids=dataset, 
            dataset_size=args.dataset_extract_size, 
            n_train=args.n_train,
            batch_size=args.prompt_batch_size,
            #tokenizer=model.lm.tokenizer,
            seed=args.seed
        )

        # Compute FVs and RVs
        print('Computing FVs...')
        fvs[dataset] = get_avg_summed_vec(model, dataset_extract, model.fv_heads, remote=args.remote_run)
        
        print('Computing RVs...')
        rvs[dataset] = get_avg_summed_vec(model, dataset_extract, model.cv_heads, remote=args.remote_run)


    ### Find the best layer to intervene with ###
    
    # Load the dataset to intervene on 
    dataset_intervene = ICLDataset(
        dataset=args.interleaved_datasets,
        size=args.dataset_eval_size, 
        n_train=args.n_train, 
        seed=args.seed, 
        batch_size=args.prompt_batch_size
    )

    int_layer_path = os.path.join(RESULTS_DIR, args.output_dir, f"{model.nickname}_intervene_layers.json")
    if not os.path.exists(int_layer_path):
        print('Finding the best layer to intervene with...')
        accs = []
        for layers in batch_process_layers(n_layers, batch_size=args.layer_batch_size):
            print(f"Processing layers: {', '.join(map(str, layers))} of {n_layers} ...")
            inter_cmpl = intervene_with_vec(model, dataset_intervene, rvs[args.extract_datasets[0]]*10, layers=layers, remote=args.remote_run)
            accs.extend([accuracy_completions(model, inter_cmpl[layer], dataset_intervene.completions) for layer in layers])
        json.dump(accs, open(int_layer_path, "w"), indent=4)     
    else:
        accs = json.load(open(int_layer_path))
        rv_intervention_layer = int(np.argmax(accs))
        print(f"Best layer to intervene with: {rv_intervention_layer}")


    # Default layer to intervene with for FVs
    fv_intervention_layer = n_layers // 3

    
    ### Zero-shot intervention ###
    print('\nIntervening zero-shot...\n')

    # Initialize results
    results = {}

    # Load the dataset to intervene on
    dataset_zero_shot = ICLDataset(
        dataset=args.interleaved_datasets[0],
        size=args.dataset_eval_size, 
        n_train=0, 
        seed=args.seed, 
        batch_size=args.prompt_batch_size
    )

    # Get the original completions
    print('\nComputing original completions...\n')
    org_cmpl = get_completions(model, dataset_zero_shot, remote=args.remote_run)
    accuracy_org = accuracy_completions(model, org_cmpl, dataset_zero_shot.completions)

    # Intervene
    print('\nIntervening zero-shot with FVs...\n')
    fv_inter_cmpl = intervene_with_vec(model, dataset_zero_shot, fvs[args.extract_datasets[0]], layers=fv_intervention_layer, remote=args.remote_run)
    accuracy_fv = accuracy_completions(model, fv_inter_cmpl[fv_intervention_layer], dataset_zero_shot.completions)

    print('\nIntervening zero-shot with RVs...\n')
    rv_inter_cmpl = intervene_with_vec(model, dataset_zero_shot, rvs[args.extract_datasets[0]]*10, layers=rv_intervention_layer, remote=args.remote_run)
    accuracy_rv = accuracy_completions(model, rv_inter_cmpl[rv_intervention_layer], dataset_zero_shot.completions)

    results['zero_shot'] = {
        'accuracy_org': accuracy_org,
        'accuracy_fv': accuracy_fv,
        'accuracy_rv': accuracy_rv
    }


    ### Intervene with function/relation vectors ###

    # Get the original completions
    print('\nComputing original completions...\n')
    org_cmpl = get_completions(model, dataset_intervene, remote=args.remote_run)
    accuracy_org = accuracy_completions(model, org_cmpl, dataset_intervene.completions)

    dfs = []
    # Intervene
    for dataset_name in args.extract_datasets:
        print(f"\n{'- *'*5}\nIntervening with vectors extracted from {dataset_name} ...\n{'* -'*5}")

        print('\nIntervening with FVs...\n')
        fv_inter_cmpl = intervene_with_vec(model, dataset_intervene, fvs[dataset_name]*10, layers=[fv_intervention_layer], remote=args.remote_run)
        accuracy_fv = accuracy_completions(model, fv_inter_cmpl[fv_intervention_layer], dataset_intervene.completions)
        print(f"Accuracy: {accuracy_fv}")
        
        print('\nIntervening with RVs...\n')
        rv_inter_cmpl = intervene_with_vec(model, dataset_intervene, rvs[dataset_name]*10, layers=[rv_intervention_layer], remote=args.remote_run)
        accuracy_rv = accuracy_completions(model, rv_inter_cmpl[rv_intervention_layer], dataset_intervene.completions)
        print(f"Accuracy: {accuracy_rv}")

        results[dataset_name] = {
            'accuracy_fv': accuracy_fv,
            'accuracy_rv': accuracy_rv
        }
    
    results['original'] = accuracy_org

    # Save results
    results_path = os.path.join(RESULTS_DIR, args.output_dir, f"{model.nickname}_intervene_results.json")
    json.dump(results, open(results_path, "w"), indent=4)