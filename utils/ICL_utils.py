import os
from typing import *
import json
import numpy as np
import torch

from utils.model_utils import ExtendedLanguageModel

class ICLSequence:
    '''
    Class to store a single antonym sequence.

    Uses the default template "Q: {x}\nA: {y}" (with separate pairs split by "\n\n").
    '''
    def __init__(
        self, 
        word_pairs: List[List[str]],
        padded_space: bool = True
    ):
        self.word_pairs = word_pairs
        self.padded_space = padded_space
        self.x, self.y = zip(*word_pairs)

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx: int):
        return self.word_pairs[idx]

    def prompt(self):
        '''Returns the prompt, which contains all but the second element in the last word pair.'''
        p = "\n\n".join([f"Q: {x}\nA: {y}" for x, y in self.word_pairs])
        return p[:-len(self.completion())]

    def completion(self):
        '''Returns the second element in the last word pair.'''
        return " " + self.y[-1] if self.padded_space else self.y[-1]

    def __str__(self):
        '''Prints a readable string representation of the prompt & completion (indep of template).'''
        return f"{', '.join([f'({x}, {y})' for x, y in self[:-1]])}, {self.x[-1]} ->".strip(", ")
    
class ICLDataset:
    '''
    Dataset to create antonym pair prompts, in ICL task format. We use random seeds for consistency
    between the corrupted and clean datasets.

    Inputs:
        word_pairs:
            list of ICL task, e.g. [["old", "young"], ["top", "bottom"], ...] for the antonym task
        size:
            number of prompts to generate
        n_prepended:
            number of antonym pairs before the single-word ICL task
        bidirectional:
            if True, then we also consider the reversed antonym pairs
        corrupted:
            if True, then the second word in each pair is replaced with a random word
        seed:
            random seed, for consistency & reproducibility
    '''

    def __init__(
        self,
        dataset: str,
        size: int,
        n_prepended: int,
        bidirectional: bool = True,
        seed: int = 0,
        corrupted: bool = False,
        padded_space: bool = True,
        batch_size: int = None,
        root_data_dir: str = 'data/ICL/abstractive'
    ):
        d_path = os.path.join(root_data_dir, f'{dataset}.json')
        raw_data = json.load(open(d_path, 'r'))
        self.word_pairs = [[i['input'], i['output']] for i in raw_data]        

        self.word_list = [word for word_pair in self.word_pairs for word in word_pair]
        self.size = size
        self.n_prepended = n_prepended
        self.bidirectional = bidirectional
        self.corrupted = corrupted
        self.seed = seed

        self.seqs = []
        self.prompts = []
        self.completions = []

        # Generate the dataset (by choosing random word pairs, and constructing `ICLSequence` objects)
        for n in range(size):
            np.random.seed(seed + n)
            random_pairs = np.random.choice(len(self.word_pairs), n_prepended+1, replace=False)
            # Randomize the order of each word pair (x, y). If not bidirectional, we always have x -> y not y -> x
            random_orders = np.random.choice([1, -1], n_prepended+1)
            if not(bidirectional): random_orders[:] = 1
            word_pairs = [self.word_pairs[pair][::order] for pair, order in zip(random_pairs, random_orders)]
            # If corrupted, then replace y with a random word in all (x, y) pairs except the last one
            if corrupted:
                for i in range(len(word_pairs) - 1):
                    word_pairs[i][1] = np.random.choice(self.word_list)
            seq = ICLSequence(word_pairs, padded_space=padded_space)

            self.seqs.append(seq)
            self.prompts.append(seq.prompt())
            self.completions.append(seq.completion())

        self.batch_size = batch_size if batch_size else size
        self.num_batches = self.calculate_n_batches(self.batch_size)

    def create_corrupted_dataset(self):
        '''Creates a corrupted version of the dataset (with same random seed).'''
        return ICLDataset(self.word_pairs, self.size, self.n_prepended, self.bidirectional, corrupted=True, seed=self.seed)
    
    def calculate_n_batches(self, batch_size):
        return self.size // batch_size + (0 if self.size % batch_size == 0 else 1)

    def __len__(self):
        return self.calculate_n_batches(self.batch_size)

    def __getitem__(self, idx: int):
        if idx >= self.num_batches:
            raise IndexError("Batch index out of range")
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.size)
        batch_prompts = self.prompts[start_idx:end_idx]
        batch_completions = self.completions[start_idx:end_idx]
        return batch_prompts, batch_completions
    

def get_FVs_and_completions(model: ExtendedLanguageModel, prompts: List[str]) -> torch.Tensor:
    '''
    Get the relation vector per item by summing the output of the most influential attention heads for the 
    model's output tokens given a prompt or list of prompts 

    Args:
        model (ExtendedLanguageModel): The model object
        prompts (List[str]): The prompt or list of prompts to run the model on

    Returns:
        torch.Tensor: The relation vectors for the model's output tokens (shape: [batch_size, resid_dim])
    '''
    D_MODEL = model.config['resid_dim']
    N_HEADS = model.config['n_heads']
    D_HEAD = D_MODEL // N_HEADS # dimension of each head
    B = len(prompts)
    T = -1  # values taken from last token
    head_dict = model.top_heads # get the top heads for each layer

    relation_vec_list = []
    with model.trace(remote=True) as runner:
        with runner.invoke(prompts) as invoker:
            for layer, head_list in head_dict.items():
        
                # Get the projection layer output
                out_proj = model.config['out_proj'](layer)

                # Output projection input
                z = out_proj.input[0][0][:, T]

                # Reshape output projection input into heads
                z_ablated = z.view(B, N_HEADS, D_HEAD).clone()
                
                # Zero-ablate all heads which aren't in our list of infuential heads
                heads_to_ablate = list(set(range(N_HEADS)) - head_list)
                z_ablated[:, heads_to_ablate, :] = 0.0

                # Concatanate the heads back into the residual dimension
                z_ablated = z_ablated.view(B, -1) 

                # Get the projection (if more than one head is in the list of heads to keep, 
                # the output will be the sum of those heads)
                out_proj_output = out_proj(z_ablated)

                relation_vec_list.append(out_proj_output.save())

                # Save the completions
                token_ids = model.lm_head.output[:,T,:].argmax(dim=-1).save()

    # Sum all the attention heads per item
    relation_vecs = torch.sum(torch.stack(relation_vec_list),dim=0)
    # Decode the completions
    completions = model.tokenizer.batch_decode(token_ids)

    return relation_vecs.numpy(), completions    