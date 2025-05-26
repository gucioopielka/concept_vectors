import os
import argparse
from typing import *
from itertools import batched

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from utils.query_utils import get_att_simmats
from utils.ICL_utils import DatasetConstructor
from utils.model_utils import ExtendedLanguageModel
from utils.eval_utils import rsa, create_design_matrix
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
    model = ExtendedLanguageModel(args.model, remote_run=args.remote_run)
    n_heads = model.config['n_heads']
    n_layers = model.config['n_layers']
    
    # Create the dataset
    dataset_constructor = DatasetConstructor(
        dataset_ids=args.datasets, 
        dataset_size=args.dataset_size, 
        n_train=args.n_train,
        batch_size=args.prompt_batch_size,
        seq_len=args.seq_len, 
        seed=args.seed
    )

    # Load intermediate results if they exist
    intermediate_results_path = os.path.join(args.output_dir, f'rsa_{model.nickname}_temp.pkl')
    if os.path.exists(intermediate_results_path):
        simmat_tensor = torch.load(intermediate_results_path)
        start = (~torch.all(simmat_tensor == 0, dim=(1, 2, 3))).sum() # Calculate number of layers with nonzero values
        if start == n_layers:
            print(f"All layers have been processed.")
            exit()
        else:
            print(f"Resuming from layer {start}...")
    else:
        simmat_tensor = torch.zeros(n_layers, n_heads, len(dataset_constructor.prompts), len(dataset_constructor.prompts))
        start = 0

    # Compute similarity matrices for each layer and head
    for layer_batch in batched(range(start, n_layers), args.layer_batch_size):
        print(f"Computing RSA for layers {layer_batch[0]} - {layer_batch[-1]} ...")
        simmat_tensor[layer_batch[0]:layer_batch[-1]+1] = get_att_simmats(model, dataset_constructor, layers=layer_batch)
        torch.save(simmat_tensor, intermediate_results_path)

    # Create design matrix
    concepts = [dataset.split('_')[0] for dataset in args.datasets]
    concepts = np.repeat(concepts, args.dataset_size)
    design_matrix = torch.tensor(create_design_matrix(concepts))

    # Compute RSA for each layer and head
    print('Computing RSA...')
    rsa_df = pd.DataFrame([(layer, head, rsa(simmat_tensor[layer, head], design_matrix)) 
                          for layer in tqdm(range(n_layers), desc='Computing RSA')
                          for head in range(n_heads)],
                         columns=['layer', 'head', 'rsa'])
    rsa_path = os.path.join(args.output_dir, f'rsa_{model.nickname}.csv')
    
    # Save results and remove intermediate results
    rsa_df.to_csv(rsa_path, index=False)
    os.remove(intermediate_results_path)