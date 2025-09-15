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
mpl.rcParams['figure.dpi'] = 300
torch.manual_seed(42)
LUMI_DIR = os.path.join(RESULTS_DIR, 'LUMI')

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

def get_datasets(concepts, size, n_train, seed):
    datasets = []
    for concept in concepts:
        if concept == 'translation':
            datasets.append(f'{concept}_eng_es-oe')
            datasets.append(f'{concept}_de_fr-oe')
            datasets.append(f'{concept}_eng_es-mc')
        else:
            datasets.append(f'{concept}_eng-oe')
            datasets.append(f'{concept}_fr-oe')
            datasets.append(f'{concept}_eng-mc')
    return DatasetConstructor(
        dataset_ids=datasets, 
        dataset_size=size, 
        n_train=n_train,
        batch_size=size,
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
model_name = 'Qwen/Qwen2.5-72b'
model = ExtendedLanguageModel(model_name)
df = model.metrics
plt.rcParams['font.family'] = 'Times New Roman'


top_k = 50
cie_max = df.sort_values(by='CIE', ascending=False)['CIE'].head(top_k).values[-1]
rsa_max = df.sort_values(by='RSA', ascending=False)['RSA'].head(top_k).values[-1]

cie_mask = (df['CIE'].values >= cie_max).reshape(model.config['n_layers'], model.config['n_heads']).T
rsa_mask = (df['RSA'].values >= rsa_max).reshape(model.config['n_layers'], model.config['n_heads']).T

# Apply masks to show only the top values
cie_values = df['CIE'].values.reshape(model.config['n_layers'], model.config['n_heads']).T
rsa_values = df['RSA'].values.reshape(model.config['n_layers'], model.config['n_heads']).T

# Set non-masked values to NaN so they don't show in the heatmap
cie_masked = np.where(cie_mask, cie_values, np.nan)
rsa_masked = np.where(rsa_mask, rsa_values, np.nan)

cmap = 'bwr'
plt.rcParams['figure.dpi'] = 300


fig, axes = plt.subplots(1, 2, figsize=(10, 20))
im0 = axes[0].imshow(
    cie_values,
    cmap=cmap,
    #vmin=0,
    #vmax=2
)
cbar0 = plt.colorbar(im0, ax=axes[0], shrink=0.15)
cbar0.ax.tick_params(labelsize=20)
# Position label below and to the left of colorbar
cbar0.ax.text(2, -0.12, 'CIE', fontsize=20, ha='center', va='top', transform=cbar0.ax.transAxes)
axes[0].set_xlabel('Layer', fontsize=22)
axes[0].set_ylabel('Head Index', fontsize=22)
axes[0].tick_params(axis='both', labelsize=20)  # Increase x and y ticks

im1 = axes[1].imshow(
    rsa_values,
    cmap=cmap,
    #vmin=0,
    #vmax=2
)
cbar1 = plt.colorbar(im1, ax=axes[1], shrink=0.15)
cbar1.ax.tick_params(labelsize=20)
cbar1.ax.text(2, -0.12, 'RSA', fontsize=20, ha='center', va='top', transform=cbar1.ax.transAxes)
axes[1].set_xlabel('Layer', fontsize=22)
axes[1].set_yticks([])
axes[1].tick_params(axis='both', labelsize=20)  # Increase x and y ticks

#plt.tight_layout()
#plt.savefig(os.path.join(PLOTS_DIR, 'rsa_cie_heatmap.svg'), bbox_inches='tight')
plt.show()


# Create a combined mask for both CIE and RSA heads
combined_mask = cie_mask | rsa_mask

# Create arrays for plotting - use different values for CIE vs RSA
plot_data = np.zeros_like(cie_values)
plot_data[cie_mask & ~rsa_mask] = 1  # Only CIE heads
plot_data[~cie_mask & rsa_mask] = 2  # Only RSA heads  
plot_data[cie_mask & rsa_mask] = 3   # Both CIE and RSA heads

# Create custom colormap for the 3 states (0=neither, 1=CIE, 2=RSA, 3=Both)
colors = ['white', '#c65f5b', '#22b2b2', '#F0C354']
plt.rcParams['figure.dpi'] = 300
cmap = plt.cm.colors.ListedColormap(colors)

fig, ax = plt.subplots(1, 1, figsize=(10, 12))

# Plot the combined data
to_plot = plot_data.T#[list(range(16, 49))]
im = ax.imshow(
    to_plot,
    cmap=cmap,
    vmin=0,
    vmax=3
)

# Create categorical legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#22b2b2', label='CIE'),
    Patch(facecolor='#c65f5b', label='RSA'),
    Patch(facecolor='#bfa536', label='Both')
]
#ax.legend(handles=legend_elements, loc='upper right', fontsize=20, frameon=True)

#ax.set_yticks([3, 5, 10])
ax.set_yticklabels([int(i + 16 )for i in ax.get_yticks()])
ax.set_xlabel('Head Index', fontsize=34)
ax.set_ylabel('Layer', fontsize=34)
ax.tick_params(axis='both', labelsize=30, length=0, which='major')
ax.tick_params(axis='both', length=0, which='minor')

# Add grid (only minor lines for cell boundaries)
ax.set_xticks(np.arange(-0.5, to_plot.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, to_plot.shape[0], 1), minor=True)
ax.grid(True, which='minor', color='black', linewidth=0.5, alpha=0.3)

# Add title
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'rsa_cie_combined_heatmap.svg'), bbox_inches='tight')
plt.show()




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
            'overlap_ratio': intersection / k,
            'iou': intersection / (len(heads1) + len(heads2) - intersection),
        })
    
    return overlaps


models =[
    'Qwen/Qwen2.5-72b',
    'meta-llama/Meta-Llama-3.1-70B',
    'meta-llama/Meta-Llama-3.1-8B',
    # 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    # 'meta-llama/Meta-Llama-3.1-70B-Instruct',
]
model_colors = ['darkred', 'darkblue']
overlap_results = {}
for model_name in models:
    model = ExtendedLanguageModel(model_name)
    n = model.config['n_layers'] * model.config['n_heads']
    df = model.metrics
    k_values = range(1, n)
    overlap_results[model_name] = calculate_overlap('RSA', 'CIE', k_values)

n_show = 1000
model_name = 'Qwen/Qwen2.5-72B'

plt.rcParams['font.family'] = 'sans-serif'
plt.figure(figsize=(7, 5))
model_names = ['Qwen 72B', 'Llama 70B']
models = ['Qwen/Qwen2.5-72b', 'meta-llama/Meta-Llama-3.1-70B']
for model_, model_name, color in zip(models, model_names, model_colors):
    model = ExtendedLanguageModel(model_)
    df = model.metrics
    n = model.config['n_layers'] * model.config['n_heads']
    k_values = range(1, n)
    # Plot overlap as percentage of k
    percent_values = [max(r['overlap_ratio'] * 100.0, 0.1) for r in overlap_results[model_][:n_show]]
    plt.plot(
        [k for k in k_values[:n_show]],
        percent_values,
        linewidth=4,
        color=color,
        label=model_name
    )
    #plt.ylim(0, 1)
    #plt.ylim(, 100)
    plt.xlim(1, n_show)
    plt.xscale('log')
    #plt.yscale('log')

# Set custom tick locations for integer display
x_ticks = [1, 10, 100, 1000]
plt.xticks(x_ticks, [str(x) for x in x_ticks], fontsize=26)
plt.yticks(fontsize=26)
plt.xlabel(r'$K$ Heads', fontsize=30)
plt.ylabel('Overlap (%)', fontsize=30)

# Calculate and plot chance level
total_heads = model.config['n_layers'] * model.config['n_heads']
k_values = range(1, n_show)
# Chance overlap ratio in % is (k / total_heads) * 100
chance_level = [(k / total_heads) * 100.0 for k in k_values]
plt.plot(k_values, chance_level, 'g--', alpha=0.7, linewidth=3, zorder=5, label='Chance')
plt.legend(fontsize=26, frameon=True, facecolor='white', loc='upper left', fancybox=False, edgecolor='white')
#plt.savefig(os.path.join(PLOTS_DIR, 'rsa_cie_overlap.svg'), bbox_inches='tight')
plt.show()


sm=torch.stack(pickle.load(open('/Users/gustaw/Documents/concept_vectors/results/LUMI/CIE_oe/cie_qwen2.5-72b.pkl', 'rb')))
plt.hist(sm[3].flatten().to(torch.float32))

sm=torch.stack(pickle.load(open('/Users/gustaw/Documents/concept_vectors/results/LUMI/CIE_mc/cie_qwen2.5-72b.pkl', 'rb')))
plt.hist(sm[1].flatten().to(torch.float32))


# Print results
print("\nOverlap between RSA and CIE heads:")
print("k\tIntersection\tOverlap_Ratio")
print("-" * 40)
# for result in overlap_results:
#     print(f"{result['k']}\t{result['intersection']}\t\t{result['overlap_ratio']:.4f}")




# Plot CIE
fig, axes = plt.subplots(1, 4, figsize=(12, 5))
for i, metric in enumerate(['CIE', 'CIE_eng', 'CIE_fr', 'CIE_mc']):
    axes[i].imshow(
        df[metric].values.reshape(model.config['n_layers'], model.config['n_heads']).T,
        cmap='coolwarm',
        # vmin=df['CIE_eng'].min(),
        # vmax=df['CIE_eng'].max()
    )
    axes[i].set_title(metric)
    axes[i].set_xlabel('Head')
    axes[i].set_ylabel('Layer')
plt.tight_layout()
#plt.colorbar(im0, ax=axes[-1], shrink=0.5)
plt.show()


# Load datasets
dataset_names = ['antonym', 'categorical', 'causal', 'synonym', 'translation', 'presentPast', 'singularPlural']
size = 50
n_train = 5
seed = 42
dataset = get_datasets(dataset_names, size, n_train, seed)
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

# # Get random layer head pairs
# random_pairs_path = os.path.join(LUMI_DIR, f'{model.nickname}_random_layer_head_pairs_{n_heads}.pkl')
# if not os.path.exists(random_pairs_path):
#     random_simmats = []
#     for i in range(10):
#         random_layer_head_pairs = get_random_layer_head_pairs(n_heads=n_heads)
#         random_simmats.append(get_summed_vec_simmat(model, dataset, random_layer_head_pairs))
#     pickle.dump(torch.stack(random_simmats), open(random_pairs_path, 'wb'))
# else:
#     random_simmats = pickle.load(open(random_pairs_path, 'rb'))

prompt_format_dm = create_design_matrix(np.repeat([d.split('-')[1] for d in dataset.dataset_ids], size))
concept_dm = create_design_matrix(np.repeat(concepts, size))
language_dm = create_design_matrix(np.repeat([d.split('_')[-1].split('-')[0] for d in dataset.dataset_ids if 'mc' not in d], size))

for model_name in [
    'meta-llama/Meta-Llama-3.1-8B',
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'Qwen/Qwen2.5-7B',
    'meta-llama/Meta-Llama-3.1-70B',
]:
    model = ExtendedLanguageModel(model_name)
    df = model.metrics
    simmats = np.load(os.path.join(LUMI_DIR, 'metrics', model.nickname, f'simmats_all_metrics.npy'))
    # SimilarityMatrix(
    #     sim_mat=simmats[0],
    #     tasks=dataset.dataset_ids,
    #     attribute_list=concepts
    # ).plot()

    results = []
    for i, metric in enumerate(['CV', 'FV']):
        for dm_name, dm in zip(['Prompt Format', 'Concept', 'Language'], [prompt_format_dm, concept_dm, language_dm]):
            if dm_name == 'Language':
                simmmat_lang = SimilarityMatrix(
                    sim_mat=simmats[i],
                    tasks=dataset.dataset_ids,
                    attribute_list=concepts
                )
                simmmat_lang.filter_tasks([d for d in dataset.dataset_ids if 'mc' not in d])
                result = rsa(simmmat_lang.matrix, dm)
            else:
                result = rsa(simmats[i], dm)
            results.append([metric, dm_name, round(result, 2)])
        SimilarityMatrix(
                    sim_mat=simmats[i],
                    tasks=dataset.dataset_ids,
                    attribute_list=concepts
                ).plot()
    print(model_name)
    print(tabulate.tabulate(results, headers=['Vec', 'Task Attribute', 'RSA']))


plt.imshow(language_dm)

rsa_simmats = np.load(os.path.join(LUMI_DIR, 'metrics', 'meta-llama-3.1-70B', f'rsa_simmats.npy'))
simmat_top = SimilarityMatrix(
    sim_mat=rsa_simmats[0],
    tasks=dataset.dataset_ids,
    attribute_list=concepts
)
simmat_top.filter_tasks([name for name in dataset.dataset_ids if name.split('_')[0] == 'antonym' or name.split('_')[0] == 'categorical'])

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
simmat_top.plot(
    labels=['']*6,
    axis=ax
)
#plt.savefig(os.path.join(PLOTS_DIR, 'rsa_simmat_top.svg'), bbox_inches='tight')
plt.show()

indices=[0, 51, 101, 152, 200, 250]
simmat_top.matrix[np.ix_(indices, indices)]


fig, ax = plt.subplots(1, 1, figsize=(4, 4))
simmat_ex = SimilarityMatrix(
    sim_mat=simmat_top.matrix[np.ix_(indices, indices)],
    tasks=list(range(6)),
    attribute_list=['antonym', 'antonym', 'antonym', 'categorical', 'categorical', 'categorical']
)
simmat_ex.plot(
    labels=['']*6,
    fade_upper_diag=True,
    upper_diag_fade_alpha=0.15,
    axis=ax
)
for i in ax.spines:
    ax.spines[i].set_visible(False)

#plt.savefig(os.path.join(PLOTS_DIR, 'rsa_simmat_ex.svg'), bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
SimilarityMatrix(
    sim_mat=simmat_ex.design_matrix,
    tasks=list(range(6)),
    attribute_list=['antonym', 'antonym', 'antonym', 'categorical', 'categorical', 'categorical']
).plot(
    labels=['']*6,
    fade_upper_diag=True,
    upper_diag_fade_alpha=0.15,
    axis=ax
)
for i in ax.spines:
    ax.spines[i].set_visible(False)
#plt.savefig(os.path.join(PLOTS_DIR, 'rsa_simmat_ex_dm.svg'), bbox_inches='tight')
plt.show()




rsa_simmats[0].shape

df = ExtendedLanguageModel('meta-llama/Meta-Llama-3.1-70B').metrics
get_top_heads('RSA', 1)

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

for i, (simmat, metric) in enumerate(zip(simmats[[1, 0]], np.array(metrics)[[1, 0]])):
    fig, axs = plt.subplots(1, 1, figsize=(4, 4))
    min_sim, max_sim = simmats.min(), simmats.max()
    sm=SimilarityMatrix(
        sim_mat=simmat,
        tasks=dataset.dataset_ids,
        attribute_list=concepts
    )
    sm.relocate_tasks(names_sorted)
    sm.plot(
        #norm=(min_sim, max_sim),
        bounding_boxes=True,
        axis=axs,
        labels=['']*len(concepts),
        #title='Concept Vector' if i == 1 else 'Function Vector',
        bounding_box_color='black',
        #cmap='viridis',
        label_colors=label_colors,
        bounding_box_width=1.1,
    )
    plt.savefig(os.path.join(PLOTS_DIR, f'simmats_{metric}.svg'), bbox_inches='tight')

sm.design_matrix

sm=SimilarityMatrix(
    sim_mat=simmats[0],
    tasks=dataset.dataset_ids,
    attribute_list=concepts
)
sm.relocate_tasks(names_sorted)
sm.filter_tasks([name for name in names_sorted if name.split('_')[0] == 'antonym' or name.split('_')[0] == 'categorical'])

mpl.rcParams['figure.dpi'] = 1200
fig, axs = plt.subplots(1, 1, figsize=(1.2, 1.2))
sm.plot(
    #norm=(min_sim, max_sim),
    bounding_boxes=False,
    labels=['']*6,
    bounding_box_color='black',
    #cmap='viridis',
    axis=axs
)
plt.savefig(os.path.join(PLOTS_DIR, 'cv_simmat.svg'), bbox_inches='tight')
plt.savefig(os.path.join(PLOTS_DIR, 'cv_simmat.png'))

plt.imshow(sm.design_matrix)
sm=SimilarityMatrix(
    sim_mat=simmat,
    tasks=dataset.dataset_ids,
    attribute_list=concepts
)
dm = SimilarityMatrix(
    sim_mat=sm.design_matrix,
    tasks=dataset.dataset_ids,
    attribute_list=concepts
)
dm.relocate_tasks(names_sorted)
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
dm.plot(
    labels=['']*len(concepts),
    bounding_box_color='black',
    axis=axs,
)
plt.savefig(os.path.join(PLOTS_DIR, 'concept_dm.svg'), bbox_inches='tight')

prompt_formats = [d.split('-')[1] for d in dataset.dataset_ids]
sm=SimilarityMatrix(
    sim_mat=simmat,
    tasks=dataset.dataset_ids,
    attribute_list=prompt_formats
)
dm = SimilarityMatrix(
    sim_mat=sm.design_matrix,
    tasks=dataset.dataset_ids,
    attribute_list=prompt_formats
)
dm.relocate_tasks(names_sorted)
fig, axs = plt.subplots(1, 1, figsize=(4, 4))
dm.plot(
    labels=['']*len(prompt_formats),
    bounding_box_color='black',
    axis=axs,
)
plt.savefig(os.path.join(PLOTS_DIR, 'prompt_format_dm.svg'), bbox_inches='tight')

dm = SimilarityMatrix(
    sim_mat=np.random.rand(*sm.design_matrix.shape),
    tasks=names_sorted,
    attribute_list=concepts
)
dm.plot(
    labels=labels,
    bounding_box_color='black',
    label_colors=label_colors
)

cax = plt.imshow(dm.matrix, cmap='coolwarm', interpolation='nearest', norm=plt.Normalize(0, 1))
cbar = plt.colorbar(cax)
cbar.set_ticks([0, 0.5, 1])
cbar.ax.tick_params(labelsize=16)
plt.show()


#plt.savefig(os.path.join(PLOTS_DIR, 'concept_vector_rsa.pdf'), bbox_inches='tight')

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



df = ExtendedLanguageModel('Qwen/Qwen2.5-72B').metrics

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



df['CIE'].quantile(0.999)


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



