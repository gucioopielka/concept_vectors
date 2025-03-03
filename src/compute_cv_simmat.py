import argparse
from typing import *
import os
import json
import pickle

import numpy as np

from utils.query_utils import get_completions, get_summed_vec_per_item, compute_similarity_matrix
from utils.ICL_utils import DatasetConstructor
from utils.model_utils import ExtendedLanguageModel
from utils.eval_utils import accuracy_completions, create_design_matrix, rsa
from utils.globals import RESULTS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute function vectors for a given model.")
    parser.add_argument("--model", type=str, help="Name of the model to use.")
    parser.add_argument("--indicators_file", type=str, help="Path to the indicators file.")
    parser.add_argument("--dataset_size", type=int, help="Size of the dataset to use.", default=20)
    parser.add_argument("--prompt_batch_size", type=int, help="Number of prompts to process at once.", default=20)
    parser.add_argument("--seq_len", type=int, help="Length of the sequence to use.", default=20)
    parser.add_argument('--n_train', type=int, help="Number of training examples to use.", default=5)
    parser.add_argument("--remote_run", type=bool, help="Whether to run the script on a remote server.", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, help="Random seed to use.", default=42)
    parser.add_argument("--output_dir", type=str, help="Path to save the output CSV file.", default='data/rsa_vs_acc/')

    args = parser.parse_args()


    # Load the model
    model = ExtendedLanguageModel(args.model)

    # Load the indicators
    indicators = json.load(open(args.indicators_file, 'r'))
    dataset_ids = list(indicators.keys())

    
    ### Create the dataset ###
    dataset_constructor = DatasetConstructor(
        dataset_ids=dataset_ids, 
        dataset_size=args.dataset_size, 
        n_train=args.n_train,
        batch_size=args.prompt_batch_size,
        tokenizer=model.lm.tokenizer,
        seq_len=args.seq_len, 
        seed=args.seed
    )


    ### Compute model completions for each prompt ###
    print(f"Computing completions for {len(dataset_constructor.prompts)} prompts ...")
    completions = get_completions(model, dataset_constructor, remote=args.remote_run)

    # Calculate accuracy for each dataset
    results = {'dataset_names': dataset_ids, 'acc': [], 'acc_bool': []}
    for i, dataset_id in enumerate(dataset_ids):
        # Split completions by dataset
        completions_task = completions[i * args.dataset_size: (i + 1) * args.dataset_size]
        ys_task = dataset_constructor.completions[i * args.dataset_size: (i + 1) * args.dataset_size]
        acc, acc_bool = accuracy_completions(model, completions_task, ys_task, return_correct=True)
        results['acc'].append(acc)
        results['acc_bool'].append(acc_bool)


    ### Compute concept vector sim_mat ###
    print('Computing CVs...')
    rv = get_summed_vec_per_item(model, dataset_constructor, model.rv_heads, remote=args.remote_run)
    rv_simmat = compute_similarity_matrix(rv)
    fv_path = os.path.join(RESULTS_DIR, args.output_dir, f'fv_simmat_{model.nickname}_{args.n_train}.pkl')
    pickle.dump(rv_simmat, open(fv_path, 'wb'))
    
        
    ### Compute RSA
    tasks = np.repeat(dataset_ids, args.dataset_size)
    info_list = [indicators[t]['relation'] for t in tasks]
    design_matrix = create_design_matrix(info_list)
    results['rsa'] = rsa(X=rv_simmat.numpy(), Y=design_matrix)
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, args.output_dir, f'rsa_{model.nickname}_{args.n_train}.json')
    json.dump(results, open(results_path, 'w'), indent=4)