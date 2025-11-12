import os
import argparse
import pandas as pd
import numpy as np

from utils.ICL_utils import DatasetConstructor
from utils.model_utils import ExtendedLanguageModel
from utils.query_utils import get_summed_vec_simmat
from utils.eval_utils import create_design_matrix, rsa
from utils.globals import RESULTS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute function vectors for a given model.")
    parser.add_argument("--model", type=str, help="Name of the model to use.", required=True)
    parser.add_argument("--datasets", nargs="+", type=str, help="List of datasets to use.", required=True)
    parser.add_argument("--dataset_size", type=int, help="Size of the dataset to use.", default=20)
    parser.add_argument('--n_train', type=int, help="Number of training examples to use.", default=5)
    parser.add_argument("--prompt_batch_size", type=int, help="Number of prompts to process at once.", default=20)
    parser.add_argument("--heads", nargs="+", type=int, help="List of heads to use.", default=list(range(1, 100)))
    parser.add_argument("--save_simmats", type=bool, help="Whether to save the similarity matrices.", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--remote_run", type=bool, help="Whether to run the script on a remote server.", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, help="Random seed to use.", default=42)
    parser.add_argument("--output_dir", type=str, help="Path to save the output files.", required=True)

    args = parser.parse_args()

    # Load the model
    model = ExtendedLanguageModel(args.model, remote_run=args.remote_run)

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

    # Design matrix
    concepts = [d.split('_')[0] for d in args.datasets]
    concept_design_matrix = create_design_matrix(np.repeat(concepts, args.dataset_size))
    q_types = [d.split('-')[1] for d in args.datasets]
    q_type_dm = create_design_matrix(np.repeat(q_types, args.dataset_size))

    # RSA
    print("Computing RSA scores...")
    rsa_scores = []
    rsa_simmats = [[], []]
    for head_i, n_heads in enumerate(args.heads):
        print(f"K Heads: {n_heads} | {head_i+1}/{len(args.heads)}")
        results = {'n_heads': n_heads}
        for metric_i, metric in enumerate(['RSA', 'CIE']):
            if metric_i == 0:
                results[f'concept_cv'] = 0
                results[f'q_type_cv'] = 0
                continue
            # Get simmat
            heads = model.get_top_heads(metric, n=n_heads, to_dict=True)
            simmats = get_summed_vec_simmat(model, dataset_constructor, heads, logging=False)
            rsa_simmats[metric_i].append(simmats)
            # Get RSA score
            vec = 'cv' if metric_i == 0 else 'fv'
            results[f'concept_{vec}'] = rsa(concept_design_matrix, simmats)
            results[f'q_type_{vec}'] = rsa(q_type_dm, simmats)
        rsa_scores.append(results)

    df = pd.DataFrame(rsa_scores)
    df.to_csv(os.path.join(output_dir, f"rsa_n_heads_1.csv"), index=False)
    if args.save_simmats:
        np.save(os.path.join(output_dir, f"rsa_simmats.npy"), np.array(rsa_simmats))

    # Compute simmats for CIEs
    print("Computing simmats for CIEs...")
    #n = df.sort_values(by='concept_cv', ascending=False)['n_heads'].iloc[0]
    n = 5
    print(f"Using {n} heads for CIEs")
    cie_simmats = []
    for metric in ['CIE', 'CIE_eng', 'CIE_fr', 'CIE_mc']:
        print(f"Computing simmats for {metric}...")
        heads = model.get_top_heads(metric, n=n, to_dict=True)
        cie_simmats.append(get_summed_vec_simmat(model, dataset_constructor, heads, logging=False))

    top_rsa_simmat = rsa_simmats[0][args.heads.index(n)]
    all_simmats = np.stack([top_rsa_simmat] + cie_simmats)
    np.save(os.path.join(output_dir, f"simmats_all_metrics.npy"), all_simmats)