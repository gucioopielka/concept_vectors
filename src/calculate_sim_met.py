import argparse
from typing import *
import torch
import os
import json
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.query_utils import get_simmats, get_completions, get_summed_vec_per_item, compute_similarity_matrix
from utils.ICL_utils import DatasetConstructor
from utils.model_utils import ExtendedLanguageModel
from utils.eval_utils import accuracy_completions, create_design_matrix, rsa, batch_process_layers
from utils.globals import RESULTS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute function vectors for a given model.")
    parser.add_argument("--model", type=str, help="Name of the model to use.")
    parser.add_argument("--dataset_size", type=int, help="Size of the dataset to use.", default=20)
    parser.add_argument("--layer_batch_size", type=int, help="Number of layers to process at once.", default=4)
    parser.add_argument("--prompt_batch_size", type=int, help="Number of prompts to process at once.", default=20)
    parser.add_argument("--seq_len", type=int, help="Length of the sequence to use.", default=20)
    parser.add_argument('--n_train', type=int, help="Number of training examples to use.", default=5)
    parser.add_argument("--remote_run", type=bool, help="Whether to run the script on a remote server.", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, help="Random seed to use.", default=42)
    parser.add_argument("--output_dir", type=str, help="Path to save the output files.")
    parser.add_argument("--abstract_relations", nargs="+", type=str, default=None, help="List of abstract relations to use. The RSA relation will be then split to verbal and abstract. Defaults to None.")
    parser.add_argument("--indicators_file", type=str, help="Path to the indicators file.", default='../data/task_attributes/indicators.json')

    args = parser.parse_args()

    # Load the model
    model = ExtendedLanguageModel(args.model)
    n_heads = model.config['n_heads']
    n_layers = model.config['n_layers']

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
    
    completion_results_path = os.path.join(RESULTS_DIR, args.output_dir, f'completions_{model.nickname}.json')
    if not os.path.exists(completion_results_path):
        # Compute completions 
        print(f"Computing completions for {len(dataset_constructor.prompts)} prompts ...")
        completions = get_completions(model, dataset_constructor, remote=args.remote_run)

        # Calculate accuracy for each dataset
        completion_results = {'dataset_names': dataset_ids, 'acc': [], 'acc_bool': []}
        for i, dataset_id in enumerate(dataset_ids):
            # Split completions by dataset
            completions_task = completions[i * args.dataset_size: (i + 1) * args.dataset_size]
            ys_task = dataset_constructor.completions[i * args.dataset_size: (i + 1) * args.dataset_size]
            acc, acc_bool = accuracy_completions(model, completions_task, ys_task, return_correct=True)
            completion_results['acc'].append(acc)
            completion_results['acc_bool'].append(acc_bool)

        # Save results
        json.dump(completion_results, open(completion_results_path, 'w'), indent=4)
    print(f"Completions computed for {len(dataset_constructor.prompts)} prompts.")


    ### Compute similarity matrices for each layer and head ###
        
    # Load intermediate results if they exist
    intermediate_results_path = os.path.join(RESULTS_DIR, args.output_dir, f'simmat_{model.nickname}.pkl')
    if os.path.exists(intermediate_results_path):
        simmat_tensor = pickle.load(open(intermediate_results_path, 'rb'))
        start = int(simmat_tensor.any(dim=(1,2,3)).sum()) # Calculate the total number of layers processed so far
        if start == n_layers:
            print(f"All layers have been processed.")
        else:
            print(f"Resuming from layer {start}...")
    else:
        simmat_tensor = torch.zeros(n_layers, n_heads, len(dataset_constructor.prompts), len(dataset_constructor.prompts))
        start = 0
    
    # Process and compute similarity matrices for each batch of layers
    for layers in batch_process_layers(n_layers, batch_size=args.layer_batch_size, start=start):
        print(f"Processing layers: {', '.join(map(str, layers))} of {n_layers} ...")
        
        # Compute similarity matrices for the current batch of layers
        batch_simmat = get_simmats(model=model, dataset=dataset_constructor, layers=layers, remote=args.remote_run)
        
        # Populate the tensor with results
        for (layer, head), mat in batch_simmat.items():
            simmat_tensor[layer, head] = mat
        
        # Save progress after each batch
        pickle.dump(simmat_tensor, open(intermediate_results_path, 'wb'))
    print(f"Similarity matrices computed for {n_layers} layers and {n_heads} heads.")


    ### Compute function vectors over the dataset ###

    fv_path = os.path.join(RESULTS_DIR, args.output_dir, f'fv_simmat_{model.nickname}.pkl')
    if not os.path.exists(fv_path):
        print('Computing FVs...')
        fv = get_summed_vec_per_item(model, dataset_constructor, model.fv_heads, remote=args.remote_run)
        fv_simmat = compute_similarity_matrix(fv)
        pickle.dump(fv_simmat, open(fv_path, 'wb'))
    print('Function vectors computed.')

        
    ### Compute RSA for each layer and head

    rsa_path = os.path.join(RESULTS_DIR, 'RSA', f'rsa_{model.nickname}.csv')
    if not os.path.exists(rsa_path):
        # Prepare data
        tasks = np.repeat(dataset_ids, args.dataset_size)
        rels = [indicators[t]['relation'] for t in tasks]
        info_types = list(indicators[list(indicators.keys())[-1]].keys())
        
        # Precompute design matrices for each information type
        design_matrices = {}
        for info_type in info_types:
            info_list = [indicators[t][info_type] for t in tasks]
            design_matrices[info_type] = create_design_matrix(info_list)

        # Split into verbal and abstract tasks and get their indices and design matrices
        if args.abstract_relations:
            rels_abstract_idx = np.array([idx for idx, rel in enumerate(rels) if rel in args.abstract_relations])
            rels_verbal_idx = np.array([idx for idx, rel in enumerate(rels) if rel not in args.abstract_relations])
            for rel_type, rel_idx in [('relation_verbal', rels_verbal_idx), ('relation_abstract', rels_abstract_idx)]:
                info_list = [indicators[t]['relation'] for t in np.array(tasks)[rel_idx]]
                design_matrices[rel_type] = create_design_matrix(info_list)

        # Compute RSA for each layer and head
        print(f"Computing RSA for {n_layers} layers and {n_heads} heads ...")
        data = []    
        for layer in tqdm(range(n_layers)):
            for head in range(n_heads):
                # Initialize row data
                row_data = {}
                row_data['layer'] = layer
                row_data['head'] = head

                # Compute RSA for each information type
                for info_type in info_types:
                    row_data[info_type] = rsa(
                        X=simmat_tensor[layer, head].numpy(),
                        Y=design_matrices[info_type]
                    )
                
                # Compute RSA for abstract and verbal relations
                if args.abstract_relations:
                    for rel_type, idx_group in [('relation_verbal', rels_verbal_idx), ('relation_abstract', rels_abstract_idx)]:
                        row_data[rel_type] = rsa(
                            X=simmat_tensor[layer, head].numpy()[idx_group, :][:, idx_group], # Slice the similarity matrix
                            Y=design_matrices[rel_type]
                        )

                data.append(row_data)
        
        # Save results to CSV
        rsa_df = pd.DataFrame(data)
        rsa_df.to_csv(rsa_path, index=False)
    print('RSA computed.')


    ### Compute Relation Vector Over the Dataset ###

    rv_path = os.path.join(RESULTS_DIR, args.output_dir, f'rv_simmat_{model.nickname}.pkl')
    if not os.path.exists(rv_path):
        print('Computing RVs...')
        rv = get_summed_vec_per_item(model, dataset_constructor, model.rv_heads, remote=args.remote_run)
        rv_simmat = compute_similarity_matrix(rv)
        pickle.dump(rv_simmat, open(rv_path, 'wb'))