import pandas as pd
import json
import argparse
from typing import *
import os
import pickle

from utils.ICL_utils import DatasetConstructor
from utils.model_utils import ExtendedLanguageModel
from utils.eval_utils import accuracy_completions
from utils.query_utils import get_completions, get_summed_vec_per_item, compute_similarity_matrix
from utils.globals import RESULTS_DIR

def construct_prompt(item: str, alphabet: str, target: str=None) -> str:
    def split_item_elements(item: str) -> List[str]:
        train, test = item.split('<br>')
        elements = train.split(' &nbsp ') + [test.split(' &nbsp ')[0]]
        return [e.replace('[', '').replace(']', '') for e in elements]
    prompt_format = 'Q: [ {} ]\nA: [ {} ]\n\nQ: [ {} ]\nA: ['
    prompt = alphabet + '\n\n' + prompt_format.format(*split_item_elements(item))
    return prompt + target if target else prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, help="Name of the model to use.")
    parser.add_argument("--dataset_ids", nargs="+", type=str, help="List of dataset IDs to use.")
    parser.add_argument("--dataset_size", type=int, help="Size of the dataset to use.", default=20)
    parser.add_argument("--batch_size", type=int, help="Number of prompts to process at once.", default=20)
    parser.add_argument('--n_train', type=int, help="Number of training examples to use.", default=5)
    parser.add_argument("--remote_run", type=bool, help="Whether to run the script on a remote server.", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--seq_len", type=int, help="Length of the sequence to use.", default=20)
    parser.add_argument("--dir", type=str, help="Path to the directory with the data.", default='data/lewis_mitchell')
    parser.add_argument("--output_dir", type=str, help="Path to save the output CSV file.", default='results/rsa_vs_acc/')

    args = parser.parse_args()


    ### Load data from Lewis & Mitchell (2024) ###
    df = pd.read_csv(f'{args.dir}/gpt_human_data.csv')
    df['prob_ind'] = df['prob_ind'].astype(int)
    items = json.load(open(f'{args.dir}/all_prob_all_permuted_7_human.json'))

    prompts = {'succ': {}, 'pred': {}}
    ys = {'succ': {}, 'pred': {}}

    # Iterate over the N permutations
    for perm in list(items.keys()):
        prompts['succ'][perm] = []
        prompts['pred'][perm] = []
        ys['succ'][perm] = []
        ys['pred'][perm] = []

        # Iterate over the alphabets (create one list of prompts and ys for each permutation)
        for alph in list(items[perm].keys()):
            n_items = len(items[perm][alph]['succ'])
            
            # Successor
            filter_df = df[(df['nperms'] == perm) & (df['alph'] == alph) & (df['prob_type'] == 'succ')]
            y_succ = [filter_df[filter_df['prob_ind'] == i]['correct_answer'].values[0] for i in range(n_items)]
            y_succ = [eval(y) for y in y_succ]
            
            # Predecessor
            filter_df = df[(df['nperms'] == perm) & (df['alph'] == alph) & (df['prob_type'] == 'pred')]
            y_pred = [filter_df[filter_df['prob_ind'] == i]['correct_answer'].values[0] for i in range(n_items)]
            y_pred = [eval(y) for y in y_pred]

            # Get y (successor: last element, predecessor: first element)
            ys['succ'][perm].extend(' '+y[-1] for y in y_succ) # Add space to match the format of the model
            ys['pred'][perm].extend(' '+y[0] for y in y_pred)

            # Get x
            shuffled_alphabet = '[ ' + ' '.join(items[perm][alph]['shuffled_alphabet']) + ' ]'
            items_succ = [p['prompt'] for p in items[perm][alph]['succ']]
            items_pred = [p['prompt'] for p in items[perm][alph]['pred']]
            target_succ = [' ' + ' '.join(i[:3]) for i in y_succ] # Append 3 first letters to the successsor prompt
            prompts['succ'][perm].extend([construct_prompt(item, shuffled_alphabet, target) for item, target in zip(items_succ, target_succ)])
            prompts['pred'][perm].extend([construct_prompt(item, shuffled_alphabet) for item in items_pred])
        
    
    # Load the model
    model = ExtendedLanguageModel(args.model)

    # Create the dataset
    dataset = DatasetConstructor(
        dataset_ids=args.dataset_ids, 
        dataset_size=args.dataset_size,
        n_train=args.n_train,
        batch_size=args.batch_size,
        tokenizer=model.lm.tokenizer,
        seq_len=args.seq_len,
        seed=args.seed
    )

    # Add Lewis & Mitchell data to the dataset
    for concept in ['succ', 'pred']:
        for perm in list(items.keys()):
            dataset.prompts.extend(prompts[concept][perm][:args.dataset_size])
            dataset.completions.extend(ys[concept][perm][:args.dataset_size])
    dataset.dataset_ids.extend([f'{concept}_{perm}' for concept in ['succ', 'pred'] for perm in list(items.keys())])
    

    # Compute accuracy
    completion_results_path = os.path.join(RESULTS_DIR, args.output_dir, f'acc_{model.nickname}.json')
    if not os.path.exists(completion_results_path):
        # Compute completions 
        print(f"Computing completions for {len(dataset.prompts)} prompts ...")
        completions = get_completions(model, dataset, remote=args.remote_run)

        # Calculate accuracy for each dataset
        completion_results = {'dataset_names': dataset.dataset_ids, 'acc': [], 'acc_bool': []}
        for i, dataset_id in enumerate(dataset.dataset_ids):
            # Split completions by dataset
            completions_task = completions[i * args.dataset_size: (i + 1) * args.dataset_size]
            ys_task = dataset.completions[i * args.dataset_size: (i + 1) * args.dataset_size]
            acc, acc_bool = accuracy_completions(model, completions_task, ys_task, return_correct=True)
            completion_results['acc'].append(acc)
            completion_results['acc_bool'].append(acc_bool)

        # Save results
        json.dump(completion_results, open(completion_results_path, 'w'), indent=4)
    
    print(f"Completions computed for {len(dataset.prompts)} prompts.")

    ### Compute Relation Vector Over the Dataset ###
    rv_path = os.path.join(RESULTS_DIR, args.output_dir, f'rv_simmat_{model.nickname}.pkl')
    if not os.path.exists(rv_path):
        print('Computing RVs...')
        rv = get_summed_vec_per_item(model, dataset, model.rv_heads, remote=args.remote_run)
        rv_simmat = compute_similarity_matrix(rv)
        pickle.dump(rv_simmat, open(rv_path, 'wb'))
    
    print(f"RVs computed for {len(dataset.prompts)} prompts.")

    ### Compute Function Vector Over the Dataset ###
    fv_path = os.path.join(RESULTS_DIR, args.output_dir, f'fv_simmat_{model.nickname}.pkl')
    if not os.path.exists(fv_path):
        print('Computing FVs...')
        fv = get_summed_vec_per_item(model, dataset, model.fv_heads, remote=args.remote_run)
        fv_simmat = compute_similarity_matrix(fv)
        pickle.dump(fv_simmat, open(fv_path, 'wb'))