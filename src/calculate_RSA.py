import os
import pickle
import argparse
from typing import *
from itertools import batched

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from utils.query_utils import get_att_simmats, get_summed_vec_per_item, compute_similarity_matrix
from utils.ICL_utils import DatasetConstructor
from utils.model_utils import ExtendedLanguageModel
from utils.eval_utils import rsa, create_design_matrix, SimilarityMatrix
from utils.globals import RESULTS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute function vectors for a given model.")
    parser.add_argument("--model", type=str, help="Name of the model to use.", required=True)
    parser.add_argument("--datasets", nargs="+", type=str, help="List of datasets to use.", required=True)
    parser.add_argument("--dataset_size", type=int, help="Size of the dataset to use.", default=20)
    parser.add_argument("--layer_batch_size", help="Number of layers to process at once. If None, all layers will be processed at once.", default=None, type=lambda x: None if x == 'None' else int(x))
    parser.add_argument("--prompt_batch_size", type=int, help="Number of prompts to process at once.", default=20)
    parser.add_argument('--n_train', type=int, help="Number of training examples to use.", default=5)
    parser.add_argument("--remote_run", type=bool, help="Whether to run the script on a remote server.", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_simmats", type=bool, help="Whether to save the similarity matrices.", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, help="Random seed to use.", default=42)
    parser.add_argument("--rsa_quantile", type=float, help="Quantile to use to get the RSA heads.", default=0.99)
    parser.add_argument("--output_dir", type=str, help="Path to save the output files.", required=True)

    args = parser.parse_args()

    # Load the model
    model = ExtendedLanguageModel(args.model, remote_run=args.remote_run)
    n_heads = model.config['n_heads']
    n_layers = model.config['n_layers']
    layer_batch_size = args.layer_batch_size if args.layer_batch_size is not None else n_layers

    # Output directory
    output_dir = os.path.join(RESULTS_DIR, args.output_dir, model.nickname)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the dataset
    dataset_constructor = DatasetConstructor(
        dataset_ids=args.datasets, 
        dataset_size=args.dataset_size, 
        n_train=args.n_train,
        batch_size=args.prompt_batch_size,
        seed=args.seed
    )

    # # Load intermediate results if they exist
    simmats_file = os.path.join(output_dir, 'simmats.pkl')
    if os.path.exists(simmats_file):
        simmat_tensor = torch.load(simmats_file)
        start = (~torch.all(simmat_tensor == 0, dim=(1, 2))).sum() # Calculate number of layers with nonzero values
        if start == n_layers:
            print(f"All layers have been processed.")
        else:
            print(f"Resuming from layer {start}...")
    else:
        condensed_vec_size = torch.triu_indices(len(dataset_constructor.prompts), len(dataset_constructor.prompts), offset=1).shape[1]
        simmat_tensor = torch.zeros(n_layers, n_heads, condensed_vec_size)
        start = 0

    # Compute similarity matrices for each layer and head
    for layer_batch in batched(range(start, n_layers), layer_batch_size):
        print(f"Computing RSA for layers {layer_batch[0]+1} - {layer_batch[-1]+1} ...")
        simmat_tensor[layer_batch[0]:layer_batch[-1]+1] = get_att_simmats(model, dataset_constructor, layers=list(layer_batch))
        torch.save(simmat_tensor, simmats_file)

    # Create design matrix
    concepts = [dataset.split('_')[0] for dataset in args.datasets]
    concepts = np.repeat(concepts, args.dataset_size)
    design_matrix = torch.tensor(create_design_matrix(concepts))

    # # Compute RSA for each layer and head
    print('Computing RSA...')
    rsa_df = pd.DataFrame([(layer, head, rsa(simmat_tensor[layer, head], design_matrix)) 
                          for layer in tqdm(range(n_layers), desc='Computing RSA')
                          for head in range(n_heads)],
                         columns=['layer', 'head', 'rsa'])
    
    # Save results and remove intermediate results
    rsa_file = os.path.join(output_dir, 'rsa.csv')
    rsa_df.to_csv(rsa_file, index=False)
    if not args.save_simmats:
        os.remove(simmats_file)

    # # Get CV similarity matrices
    # print('Getting CV similarity matrices...')
    # cv_heads = model.get_heads(rsa_df, n=5) #quantile=args.rsa_quantile)
    # cvs = get_summed_vec_per_item(model, dataset_constructor, cv_heads)
    # cv_simmat = compute_similarity_matrix(cvs).cpu().to(torch.float32).numpy()
    # pickle.dump(cv_simmat, open(os.path.join(output_dir, 'cv_simmat.pkl'), 'wb'))

    # # Plot CV similarity matrices
    # print('Plotting CV similarity matrices...')
    # SimilarityMatrix(
    #     sim_mat=cv_simmat,
    #     tasks=args.datasets,
    #     attribute_list=concepts
    # ).plot(
    #     bounding_boxes=True,
    #     save_path=os.path.join(output_dir, 'cv_simmat.png')
    # )