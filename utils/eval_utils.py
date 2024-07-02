import numpy as np
import pandas as pd
import torch

from sklearn.metrics import pairwise_distances
from typing import *
from utils.model_utils import ExtendedLanguageModel

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize


def get_RDM(mat):
    mat_flattened = np.reshape(mat, (mat.shape[0], -1))
    return pairwise_distances(mat_flattened, metric='cosine')

def get_unique_indices(s: pd.Series) -> Dict:
    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    s.reset_index(drop=True, inplace=True)

    result_dict = {}
    for value in s.unique():
        # Find the first and last index of the current value
        first_idx = s[s == value].index[0]
        last_idx = s[s == value].index[-1]
        
        # Store the result in the dictionary
        result_dict[value] = (first_idx, last_idx)
    
    return result_dict

def get_rule_sim_diagonal(rdm, rel_idx, sort=True):
    rule_similarity = []
    for relation, indices in rel_idx.items():
        rule_similarity.append(np.mean(rdm[indices[0]:indices[1], indices[0]:indices[1]]))
    
    sim_df = pd.DataFrame({'Concept' : rel_idx.keys(), 'Similarity' : rule_similarity})
    if sort:
        return sim_df.sort_values(by='Similarity', ascending=False)
    else:
        return sim_df
    
def within_task_similarity(rdm, rel_idx):
    within_task_similarities = []

    for task, (start_idx, end_idx) in rel_idx.items():
        task_rdm = rdm[start_idx:end_idx+1, start_idx:end_idx+1]
        lower_triangle = np.tril(task_rdm, -1)  # Only take elements below the diagonal
        mean_similarity = np.mean(lower_triangle[lower_triangle != 0])  # Ignore zeros which represent unused parts
        within_task_similarities.append(mean_similarity)

    return np.mean(within_task_similarities)

def between_task_similarity(rdm, rel_idx):
    between_task_similarities = []

    for i, (task1, (start1, end1)) in enumerate(rel_idx.items()):
        for j, (task2, (start2, end2)) in enumerate(rel_idx.items()):
            if i > j:  # This ensures that we only consider each pair once
                between_rdm = rdm[start1:end1+1, start2:end2+1]
                mean_similarity = np.mean(between_rdm)
                between_task_similarities.append(mean_similarity)

    return np.mean(between_task_similarities)

def decode_to_vocab(prob_dist, tokenizer, k=5):
    """
    Decodes and returns the top K words of a probability distribution

    Parameters:
    prob_dist: torch tensor of model logits (distribution over the vocabulary)
    tokenizer: huggingface model tokenizer
    k: number of vocabulary words to include

    Returns:
    list of top K decoded vocabulary words in the probability distribution as strings, along with their probabilities (float)
    """
    if not isinstance(prob_dist, torch.Tensor):
        prob_dist = torch.tensor(prob_dist)

    # Compute softmax and then get the top k predictions once
    softmax = torch.softmax(prob_dist, dim=-1)
    topk = torch.topk(softmax, k=k, dim=-1)

    # Prepare the list of decoded words and their probabilities
    topk_words = [tokenizer.decode([idx]) for idx in topk.indices]
    topk_probs = [round(prob.item(), 5) for prob in topk.values]

    return list(zip(topk_words, topk_probs))

def fv_to_vocab(function_vector, model, model_config, tokenizer, n_tokens=10):
    """
    Decodes a provided function vector into the model's vocabulary embedding space.

    Parameters:
    function_vector: torch vector extracted from ICL contexts that represents a particular function
    model: huggingface model
    model_config: dict with model information - n_layers, n_heads, etc.
    tokenizer: huggingface tokenizer
    n_tokens: number of top tokens to include in the decoding

    Returns:
    decoded_tokens: list of tuples of the form [(token, probability), ...]
    """

    if 'gpt-j' in model_config['name_or_path']:
        decoder = torch.nn.Sequential(model.transformer.ln_f, model.lm_head, torch.nn.Softmax(dim=-1))
    elif 'llama' in model_config['name_or_path']:
        decoder = torch.nn.Sequential(model.model.norm, model.lm_head, torch.nn.Softmax(dim=-1))
    else:
        raise ValueError("Model not yet supported")
    
    d_out = decoder(function_vector.reshape(1,1,model_config['resid_dim']).to(model.device))

    vals, inds = torch.topk(d_out, k=n_tokens,largest=True)
    decoded_tokens = [(tokenizer.decode(x),round(y.item(), 4)) for x,y in zip(inds.squeeze(), vals.squeeze())]
    return decoded_tokens

def accuracy_completions(model: ExtendedLanguageModel, data: dict) -> float:
    '''
    Calculate the accuracy of the model on the completions

    Args:
        model (ExtendedLanguageModel): The model object
        data (dict): The data dictionary containing the completions and the correct completions
        Expected keys: completions, Ys
    '''
    assert 'completions' in data.keys(), 'The data dictionary must contain model completions'
    assert 'Ys' in data.keys(), 'The data dictionary must contain expected Ys'

    correct = []
    for completion, y in zip(data['completions'], data['Ys']):
        correct_completion_first_token = model.tokenizer.tokenize(y)[0].replace('▁', '')
        correct.append(completion == correct_completion_first_token)
    return np.mean(correct)

class SimilarityMatrix:
    def __init__(self, 
        model: ExtendedLanguageModel, 
        data: np.ndarray,
        tasks: List[str] = None
    ):
        self.model = model
        self.data = data
        self.matrix = get_RDM(data)
        self.metric = 'dissimilarity'
        if tasks and len(data) != len(tasks):   
            # If the number of tasks is not equal to the number of data points, repeat the tasks
            assert len(data) % len(tasks) == 0, 'The number of tasks must be equal to the number of data points or divisible by the number of data points'
            self.tasks = np.repeat(tasks, len(data))
        self.tasks_idx = get_unique_indices(self.tasks) if tasks is not None else None

    def toggle_similarity(self):
        '''
        Toggle between the similarity matrix and the dissimilarity matrix
        '''
        self.matrix = 1 - self.matrix
        self.metric = 'similarity' if self.metric == 'dissimilarity' else 'dissimilarity'

    def plot_similarity(
        self,
        rel_indices=None,
        rel_ticks=True, 
        title=None,
        metric='similarity',
        norm=Normalize(vmin=0, vmax=1),
        axis=None, 
    ):
        rel_indices = self.tasks_idx
        
        # Create a new figure and axis if none is provided
        if axis is None:
            fig, ax = plt.subplots()
        else:
            ax = axis

        if self.metric != metric:
            self.toggle_similarity() 
        
        # Plot the RDM on the specified (or new) axes object
        cax = ax.imshow(self.matrix, cmap='coolwarm', norm=norm)
        if title:
            ax.set_title(title, fontsize=14)

        if rel_indices:
            for relation, (start, end) in rel_indices.items():
                width = height = end - start

                # Draw a rectangle on the specified axes
                rect = patches.Rectangle((start, start), width, height, linewidth=0.8, edgecolor='purple', facecolor='none')
                ax.add_patch(rect)

            midpoints = [(start + end) / 2 for start, end in rel_indices.values()]

            if rel_ticks:
                ax.set_xticks(midpoints)
                ax.set_yticks(midpoints)
                ax.set_xticklabels(rel_indices.keys(), rotation=90, fontsize=10)
                ax.set_yticklabels(rel_indices.keys(), fontsize=10)

        # Add colorbar for the axes
        if axis is None:
            cbar = plt.colorbar(cax, ax=ax)
            cbar.ax.set_ylabel('Similarity', fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        # Adjust layout
        plt.tight_layout()
        if axis is None:
            plt.show()