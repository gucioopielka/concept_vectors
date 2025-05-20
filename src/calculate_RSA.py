import argparse
from typing import *
import torch
import os
import json
import pickle
import pandas as pd
import numpy as np

from utils.query_utils import get_completions, get_summed_vec_per_item, compute_similarity_matrix, get_rsa
from utils.ICL_utils import DatasetConstructor
from utils.model_utils import ExtendedLanguageModel
from utils.eval_utils import accuracy_completions, create_design_matrix, batch_process_layers
from utils.globals import RESULTS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute function vectors for a given model.")
    parser.add_argument("--model", type=str, help="Name of the model to use.")
    parser.add_argument("--datasets", nargs="+", type=str, help="List of datasets to use.")
    parser.add_argument("--dataset_size", type=int, help="Size of the dataset to use.", default=20)
    parser.add_argument("--layer_batch_size", type=int, help="Number of layers to process at once.", default=None)
    parser.add_argument("--prompt_batch_size", type=int, help="Number of prompts to process at once.", default=20)
    parser.add_argument("--seq_len", type=int, help="Length of the sequence to use.", default=20)
    parser.add_argument('--n_train', type=int, help="Number of training examples to use.", default=5)
    parser.add_argument("--remote_run", type=bool, help="Whether to run the script on a remote server.", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, help="Random seed to use.", default=42)
    parser.add_argument("--output_dir", type=str, help="Path to save the output files.", default=RESULTS_DIR + '/RSA')

    args = parser.parse_args()

    # Load the model
    model = ExtendedLanguageModel(args.model)

    n_heads = model.config['n_heads']
    n_layers = model.config['n_layers']
    
    ### Create the dataset ###
    dataset_constructor = DatasetConstructor(
        dataset_ids=args.datasets, 
        dataset_size=args.dataset_size, 
        n_train=args.n_train,
        batch_size=args.prompt_batch_size,
        seq_len=args.seq_len, 
        seed=args.seed
    )

    ### Compute model completions for each prompt ###
    
    # completion_results_path = os.path.join(RESULTS_DIR, args.output_dir, f'completions_{model.nickname}.json')
    # if not os.path.exists(completion_results_path):
    #     # Compute completions 
    #     print(f"Computing completions for {len(dataset_constructor.prompts)} prompts ...")
    #     completions = get_completions(model, dataset_constructor, remote=args.remote_run)

    #     # Calculate accuracy for each dataset
    #     completion_results = {'dataset_names': args.datasets, 'acc': [], 'acc_bool': []}
    #     for i, dataset_id in enumerate(args.datasets):
    #         # Split completions by dataset
    #         completions_task = completions[i * args.dataset_size: (i + 1) * args.dataset_size]
    #         ys_task = dataset_constructor.completions[i * args.dataset_size: (i + 1) * args.dataset_size]
    #         acc, acc_bool = accuracy_completions(model, completions_task, ys_task, return_correct=True)
    #         completion_results['acc'].append(acc)
    #         completion_results['acc_bool'].append(acc_bool)

    #     # Save results
    #     json.dump(completion_results, open(completion_results_path, 'w'), indent=4)
    # print(f"Completions computed for {len(dataset_constructor.prompts)} prompts.")


    ### Compute RSA for each layer and head

    # Prepare data
    concepts = [dataset.split('_')[0] for dataset in args.datasets]
    concepts = np.repeat(concepts, args.dataset_size)
    design_matrix = torch.tensor(create_design_matrix(concepts))

    # # Load intermediate results if they exist
    intermediate_results_path = os.path.join(args.output_dir, f'rsa_{model.nickname}_temp.pkl')
    if os.path.exists(intermediate_results_path):
        rsa_vals = pickle.load(open(intermediate_results_path, 'rb'))
        start = int(rsa_vals.any(dim=1).sum()) # Calculate number of layers with nonzero values
        if start == n_layers:
            print(f"All layers have been processed.")
        else:
            print(f"Resuming from layer {start}...")
    else:
        rsa_vals = torch.zeros(n_layers, n_heads)
        start = 0
        
    # Process and compute RSA for each batch of layers
    for layer_batch in batch_process_layers(n_layers, args.layer_batch_size):
        print(f"Computing RSA for layers {layer_batch[0]} - {layer_batch[-1]} ...")
        rsa_vals_batch = get_rsa(model, dataset_constructor, design_matrix, layers=layer_batch, remote=args.remote_run)
        rsa_vals[layer_batch[0]:layer_batch[-1]+1] = rsa_vals_batch
        pickle.dump(rsa_vals, open(intermediate_results_path, 'wb'))
    
    rsa_df = pd.DataFrame([(layer, head, rsa.item()) 
                          for layer, layer_rsa in enumerate(rsa_vals)
                          for head, rsa in enumerate(layer_rsa)],
                         columns=['layer', 'head', 'rsa'])
    rsa_path = os.path.join(args.output_dir, f'rsa_{model.nickname}.csv')
    rsa_df.to_csv(rsa_path, index=False)

    # ### Compute Concept and Function Vectors ###
    # print('Computing Concept and Function Vectors...')
    # model = ExtendedLanguageModel(args.model, rsa_heads_n=args.n_heads, fv_heads_n=args.n_heads, cie_path=args.cie_path)
    # with model.lm.session(remote=args.remote_run) as sess:
    #     cv = [get_summed_vec_per_item(model, prompts, model.cv_heads, remote=args.remote_run) for prompts, _ in dataset_constructor]
    #     fv = [get_summed_vec_per_item(model, prompts, model.fv_heads, remote=args.remote_run) for prompts, _ in dataset_constructor]
    #     cv_simmat = compute_similarity_matrix(torch.concatenate(cv)).save()
    #     fv_simmat = compute_similarity_matrix(torch.concatenate(fv)).save()
        
    # cv_path = os.path.join(RESULTS_DIR, args.output_dir, f'cv_simmat_{model.nickname}.pkl')
    # fv_path = os.path.join(RESULTS_DIR, args.output_dir, f'fv_simmat_{model.nickname}.pkl')
    # pickle.dump(cv_simmat.to(torch.float32).numpy(), open(cv_path, 'wb'))
    # pickle.dump(fv_simmat.to(torch.float32).numpy(), open(fv_path, 'wb'))