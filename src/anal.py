import os
import pickle
import torch
import json
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import tabulate
from utils.model_utils import ExtendedLanguageModel
from utils.eval_utils import SimilarityMatrix, create_design_matrix, rsa
from utils.globals import PLOTS_DIR, RESULTS_DIR, DATASET_DIR
from utils.ICL_utils import DatasetConstructor
from utils.query_utils import get_summed_vec_simmat
mpl.rcParams['figure.dpi'] = 200
torch.manual_seed(42)

def get_top_heads(metric, n=5):
    head_df = df.sort_values(by=metric, ascending=False).head(n)
    head_dict = defaultdict(set)
    for layer, head in zip(head_df['layer'], head_df['head']):
        head_dict[layer].add(head)
    return dict(head_dict)

def calculate_overlap(metric1, metric2, k_values):
    """Calculate overlap between top k heads for two metrics"""

    def get_top_heads_as_tuples(metric, n=5):
        head_df = df.sort_values(by=metric, ascending=False).head(n)
        return set(zip(head_df['layer'], head_df['head']))
    
    overlaps = []
    for k in k_values:
        heads1 = get_top_heads_as_tuples(metric1, k)
        heads2 = get_top_heads_as_tuples(metric2, k)
        
        intersection = len(heads1.intersection(heads2))
        
        overlaps.append({
            'k': k,
            'intersection': intersection,
            'overlap_ratio': intersection / k if k > 0 else 0
        })
    
    return overlaps

def get_datasets(dataset_names, size, n_train, seed, n_batches_oe, n_batches_mc):
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

def get_random_layer_head_pairs(n_heads=5):
    num_layers = model.config['n_layers']
    num_heads = model.config['n_heads']
    rand_indices = torch.randint(0, num_layers * num_heads, (n_heads,))
    layer_head_pairs = [(idx // num_heads, idx % num_heads) for idx in rand_indices.tolist()]
    layer_head_dict = defaultdict(set)
    for layer, head in layer_head_pairs:
        layer_head_dict[layer].add(head)
    return dict(layer_head_dict)

def sort_by_concept(dataset_ids, concepts):
    order_index = {key: i for i, key in enumerate(concepts)}
    return sorted(dataset_ids, key=lambda x: order_index.get(x.split('_')[0]))

# Load model
model_name = 'meta-llama/Meta-Llama-3.1-70B'
model = ExtendedLanguageModel(model_name, remote_run=True)

# Load CIE and RSA results
LUMI_DIR = os.path.join(RESULTS_DIR, 'LUMI')
cie_mc = pickle.load(open(os.path.join(LUMI_DIR, 'CIE_mc', f'cie_{model.nickname}.pkl'), 'rb'))
cie_mc = torch.stack(cie_mc).to(torch.float32)
cie_oe = pickle.load(open(os.path.join(LUMI_DIR, 'CIE_oe', f'cie_{model.nickname}.pkl'), 'rb'))
cie_oe = torch.stack(cie_oe).to(torch.float32)
cie = torch.concat([cie_oe, cie_mc])

# Create dataframe with CIE and RSA results
df = pd.read_csv(os.path.join(LUMI_DIR, 'RSA', model.nickname, f'rsa.csv'))
df.rename(columns={'rsa': 'RSA'}, inplace=True)
df['CIE'] = cie.mean(dim=0).flatten()
df['CIE_eng'] = cie_oe[::2].mean(dim=0).flatten()
df['CIE_fr'] = cie_oe[1::2].mean(dim=0).flatten()
df['CIE_mc'] = cie_mc.mean(dim=0).flatten()
df_path = os.path.join(LUMI_DIR, 'RSA', model.nickname, f'metrics.csv')
if not os.path.exists(df_path): df.to_csv(df_path, index=False)

# Load datasets
dataset_names = ['antonym', 'categorical', 'causal', 'synonym', 'translation', 'presentPast', 'singularPlural']
size = 50
n_train = 5
seed = 42
n_batches_oe = 4
n_batches_mc = 5
dataset = get_datasets(dataset_names, size, n_train, seed, n_batches_oe, n_batches_mc)
concepts = [d.split('_')[0] for d in dataset.dataset_ids]

# Get CV similarity matrices
n_heads = 5
simmats_path = os.path.join(LUMI_DIR, f'{model.nickname}_simmats_{n_heads}.pkl')
metrics = ['RSA', 'CIE', 'CIE_eng', 'CIE_fr', 'CIE_mc']
if not os.path.exists(simmats_path):
    simmats = []
    for metric in metrics:
        heads = get_top_heads(metric, n_heads)
        simmats.append(get_summed_vec_simmat(model, dataset, heads))
    pickle.dump(torch.stack(simmats), open(simmats_path, 'wb'))
else:
    simmats = pickle.load(open(simmats_path, 'rb'))

# Get random layer head pairs
random_pairs_path = os.path.join(LUMI_DIR, f'{model.nickname}_random_layer_head_pairs_{n_heads}.pkl')
if not os.path.exists(random_pairs_path):
    random_simmats = []
    for i in range(10):
        random_layer_head_pairs = get_random_layer_head_pairs(n_heads=n_heads)
        random_simmats.append(get_summed_vec_simmat(model, dataset, random_layer_head_pairs))
    pickle.dump(torch.stack(random_simmats), open(random_pairs_path, 'wb'))
else:
    random_simmats = pickle.load(open(random_pairs_path, 'rb'))

prompt_format_dm = create_design_matrix(np.repeat([d.split('-')[1] for d in dataset.dataset_ids], size))
concept_dm = create_design_matrix(np.repeat(concepts, size))

results = []
for i, metric in enumerate(['CV', 'FV']):
    for dm_name, dm in zip(['Prompt Format', 'Concept'], [prompt_format_dm, concept_dm]):
        result = rsa(simmats[i], dm)
        results.append([metric, dm_name, round(result, 2)])
print(tabulate.tabulate(results, headers=['Vec', 'Task Attribute', 'RSA']))


names_sorted = sort_by_concept(dataset.dataset_ids, dataset_names)
labels = [c.replace('_', ' ').title().replace('Eng-Mc', 'MC').replace('-Oe', '') for c in names_sorted]

prev_color = 'black'
label_colors = []
for i, attr in enumerate(names_sorted):
    if i == 0:
        label_colors.append('black')
    else:
        if attr.split('_')[0] != names_sorted[i-1].split('_')[0]:
            label_colors.append('grey' if prev_color == 'black' else 'black')
            prev_color = label_colors[-1]
        else:
            label_colors.append(prev_color)

fig, axs = plt.subplots(1, 3, figsize=(15, 15))
min_sim, max_sim = simmats.min(), simmats.max()
plt.subplots_adjust(wspace=0.5)  # Add horizontal space between subplots
for i, (simmat, metric) in enumerate(zip(simmats[2:], metrics[2:])):
    sm=SimilarityMatrix(
        sim_mat=simmat,
        tasks=dataset.dataset_ids,
        attribute_list=concepts
    )
    sm.relocate_tasks(names_sorted)
    sm.plot(
        #norm=(min_sim, max_sim),
        bounding_boxes=True,
        labels=labels,
        axis=axs[i],
        title=metric,
        bounding_box_color='black',
        #cmap='viridis',
        label_colors=label_colors
    )

fig, axs = plt.subplots(10, 1, figsize=(15, 50))
plt.subplots_adjust(wspace=0.5)  # Add horizontal space between subplots
min_sim, max_sim = simmats.min(), simmats.max()
names_sorted = sort_by_concept(dataset.dataset_ids, dataset_names)
labels = [c.replace('_', ' ').title().replace('Eng-Mc', 'MC').replace('-Oe', '') for c in names_sorted]
for i, simmat in enumerate(random_simmats):
    sm=SimilarityMatrix(
        sim_mat=simmat,
        tasks=dataset.dataset_ids,
        attribute_list=concepts
    )
    sm.relocate_tasks(names_sorted)
    sm.plot(
        #norm=(min_sim, max_sim),
        bounding_boxes=True,
        axis=axs[i],
        title=f'Sum of {n_heads} random heads #{i}',
        label_colors=label_colors
    )
    axs[i].set_xticks([])





corr = 'pearson'
plt.imshow(df[['RSA', 'CIE_eng', 'CIE_fr', 'CIE_mc']].corr(corr))
plt.xticks(range(4), ['RSA', 'CIE_eng', 'CIE_fr', 'CIE_mc'], rotation=90)
plt.yticks(range(4), ['RSA', 'CIE_eng', 'CIE_fr', 'CIE_mc'])
plt.title(f'{corr} correlation across {model.config["n_layers"]*model.config["n_heads"]} heads') 
plt.colorbar()
plt.show()

# Calculate overlap between RSA and CIE heads
print("Calculating overlap between RSA and CIE heads...")
k_values = [3, 5, 10, 20, 50, 100, 200, 500, 1000]
overlap_results = calculate_overlap('RSA', 'CIE', k_values)

# Print results
print("\nOverlap between RSA and CIE heads:")
print("k\tIntersection\tOverlap_Ratio")
print("-" * 40)
for result in overlap_results:
    print(f"{result['k']}\t{result['intersection']}\t\t{result['overlap_ratio']:.4f}")

# Also calculate overlap with other CIE variants
cie_variants = ['CIE_fr', 'CIE_mc']
for variant in cie_variants:
    print(f"\nOverlap between CIE_eng and {variant} heads:")
    print("k\tIntersection\tOverlap_Ratio")
    print("-" * 50)
    variant_overlaps = calculate_overlap('CIE_eng', variant, k_values)
    for result in variant_overlaps:
        print(f"{result['k']}\t{result['intersection']}\t\t{result['overlap_ratio']:.4f}")





plt.figure(figsize=(8, 5))
cie_cols = ['CIE_eng', 'CIE_fr', 'CIE_mc']
cie_labels = ['English', 'French', 'Multiple Choice']
data = [df[col] for col in cie_cols]

parts = plt.violinplot(
    data, 
    #showmeans=True, 
    #showmedians=True, 
    widths=0.8,
    bw_method=0.3
)

# Set colors for violins
colors = sns.color_palette("Set2", n_colors=len(cie_cols))
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.8)

# Style the mean and median lines
#parts['cmeans'].set_color('black')
#parts['cmeans'].set_linewidth(2)
#parts['cmedians'].set_color('red')
#parts['cmedians'].set_linewidth(2)

plt.xticks(range(1, len(cie_cols) + 1), cie_labels, fontsize=13)
plt.xlabel('Prompts', fontsize=14)
plt.ylabel('CIE', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)
sns.despine()
plt.tight_layout()
plt.show()





simmats_all = torch.load(open(os.path.join(LUMI_DIR, 'RSA', model.nickname, 'simmats.pkl'), 'rb'))

def create_concept_design_matrix(info_list, concept):
    n_items = len(info_list)
    m = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(i+1, n_items):
            # Only mark pairs where BOTH are the concept
            if info_list[i] == concept and info_list[j] == concept:
                m[i, j] = 1
                m[j, i] = 1
    # Diagonal: only mark as 1 if the item is the concept itself
    for i in range(n_items):
        m[i, i] = 1 if info_list[i] == concept else 0
    return m

rsa_ = np.zeros((model.config['n_layers'], model.config['n_heads']))
dm_antonym = create_concept_design_matrix(np.repeat(concepts_, size), 'antonym')
plt.imshow(dm_antonym)

for n_layer in range(model.config['n_layers']):
    print(n_layer)
    for n_head in range(model.config['n_heads']):
        rsa_[n_layer, n_head] = rsa(simmats_all[n_layer, n_head], dm_antonym)


df_rsa = pd.DataFrame(
    {'RSA': rsa_.flatten(), 'Layer': np.repeat(range(model.config['n_layers']), model.config['n_heads']), 'Head': np.tile(range(model.config['n_heads']), model.config['n_layers'])}
)

df_rsa.groupby('Layer')['RSA'].mean().plot(kind='bar')

for rsa_, layer, head in df.sort_values(by='RSA', ascending=False).head(10)[['RSA', 'layer', 'head']].values:
    print(rsa_, layer, head)
    SimilarityMatrix(
        sim_mat=simmats_all[int(layer), int(head)],
        tasks=names_sorted,
    ).plot(
        bounding_boxes=True,
        title=f'RSA: {rsa_:.2f}',
    )




top_simmats = torch.load(os.path.join(LUMI_DIR, f'{model.nickname}_progressive_simmats_1_to_100 (2).pkl')).to(torch.float32)
rsa_top_k = []
for i, simmat in enumerate(top_simmats):
    rsa_top_k.append(rsa(simmat, concept_dm))

i = 4
s=SimilarityMatrix(
    sim_mat=top_simmats[i],
    tasks=dataset.dataset_ids,
)
#s.relocate_tasks(names_sorted)
s.plot(
    title=f'RSA: {rsa_top_k[i]:.2f}',
)



