from utils.query_utils import get_att_out_proj_input, get_att_output_per_item
from utils.model_utils import ExtendedLanguageModel
from utils.ICL_utils import DatasetConstructor, ICLDataset
from utils.query_utils import convert_bfloat, no_grad, compute_similarity_matrix, condense_matrix
from utils.globals import RESULTS_DIR

from typing import Dict, List, Tuple
import torch
import nnsight
import pandas as pd
import os
from collections import defaultdict
import sys

def get_datasets(dataset_names, size, n_train, seed, n_batches_oe, n_batches_mc):
    """Load datasets for the analysis."""
    datasets_oe = []
    for dataset in dataset_names:
        datasets_oe.append(f'{dataset}_eng_es-oe' if dataset == 'translation' else f'{dataset}_eng-oe')
        datasets_oe.append(f'{dataset}_de_fr-oe' if dataset == 'translation' else f'{dataset}_fr-oe')

    datasets_mc = []
    for dataset in dataset_names:
        datasets_mc.append(f'{dataset}_eng_es-mc' if dataset == 'translation' else f'{dataset}_eng-mc')

    datasets = datasets_oe + datasets_mc
    batch_sizes_oe = [int((len(datasets_oe)*size)/n_batches_oe)] * n_batches_oe 
    batch_sizes_mc = [int((len(datasets_mc)*size)/n_batches_mc)] * n_batches_mc
    print(f'Batch sizes: {batch_sizes_oe} + {batch_sizes_mc}')

    return DatasetConstructor(
        dataset_ids=datasets, 
        dataset_size=size, 
        n_train=n_train,
        batch_size= batch_sizes_oe + batch_sizes_mc,
        seed=seed
    )

@no_grad
@convert_bfloat
def get_k_summed_vec_simmats(
    model: ExtendedLanguageModel,
    dataset: ICLDataset | DatasetConstructor,
    top_heads: List[Tuple[int, int]],
    token: int = -1,
) -> List[torch.Tensor]:

    k_values = list(range(1, len(top_heads) + 1))
    heads_by_layer = defaultdict(list)
    for layer, head in top_heads:
        heads_by_layer[layer].append(head)
    
    with model.lm.session(remote=model.remote_run) as sess:
        # Dictionary to store all head outputs: {layer: {head: [batch_outputs]}}
        all_head_outputs = {layer: {head: [] for head in heads} for layer, heads in heads_by_layer.items()}
        
        for idx, (batched_prompts, _) in enumerate(dataset):
            sess.log(f"Batch: {idx+1} / {len(dataset)}")
            
            with model.lm.trace(batched_prompts) as t:
                # Extract all needed head outputs in one forward pass
                for layer, head_list in heads_by_layer.items():
                    for head in head_list:
                        att_out = get_att_output_per_item(model, layer, [head], token)
                        att_out = att_out if model.remote_run else att_out.cpu()
                        all_head_outputs[layer][head].append(att_out)
                    
        # Concatenate all batches for each head
        for layer in all_head_outputs:
            for head in all_head_outputs[layer]:
                all_head_outputs[layer][head] = torch.cat(all_head_outputs[layer][head], dim=0)
        
        # Now compute similarity matrices for each k
        sess.log(f"Computing similarity matrix for top {len(k_values)} heads...")
        simmats = []
        for k in k_values:
            # Get the top k heads
            k_heads = top_heads[:k]
            
            # Sum the outputs for these k heads
            summed_vector = 0
            for layer, head in k_heads:
                summed_vector += all_head_outputs[layer][head]
            
            # Compute similarity matrix
            simmat = compute_similarity_matrix(summed_vector.to(model.device))
            simmat_condensed = condense_matrix(simmat, n=len(dataset.prompts))
            simmats.append(simmat_condensed.cpu().save())
    
    return simmats

# Load model and data
model_name = sys.argv[1]
model = ExtendedLanguageModel(model_name)

LUMI_DIR = os.path.join(RESULTS_DIR, 'LUMI')
df_metrics = pd.read_csv(os.path.join(LUMI_DIR, 'RSA', model.nickname, f'metrics.csv'))

# Get top 100 heads sorted by RSA
top_100_heads = df_metrics.sort_values(by='RSA', ascending=False).head(100)[['layer', 'head']].values.tolist()

# Load dataset
dataset_names = ['antonym', 'categorical', 'causal', 'synonym', 'translation', 'presentPast', 'singularPlural']
size = 50
n_train = 5
seed = 42
n_batches_oe = 4
n_batches_mc = 5
dataset = get_datasets(dataset_names, size, n_train, seed, n_batches_oe, n_batches_mc)

# Check if results already exist
progressive_simmats_path = os.path.join(LUMI_DIR, f'{model.nickname}_progressive_simmats_1_to_100.pkl')

print("Computing progressive similarity matrices...")
# Compute similarity matrices for each k
simmats = get_k_summed_vec_simmats(model, dataset, top_100_heads)

# Save results
torch.save(torch.stack(simmats), progressive_simmats_path)
print(f"Saved progressive similarity matrices to {progressive_simmats_path}")
