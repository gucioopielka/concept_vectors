import os
import pickle
import time
import argparse

import numpy as np
import torch

from utils.eval_utils import decode_to_vocab
from utils.model_utils import ExtendedLanguageModel

model = ExtendedLanguageModel('meta-llama/Llama-2-70b-hf')


antonym = pickle.load(open('data/ICL/results/Llama-2-70b-hf__5_n.pkl','rb'))['antonym']['FVs']
fv = np.mean(antonym, axis=0)

with model.trace(remote=True) as runner:
    with runner.invoke('') as invoker:
        pre = model.lm_head.output[0][0].save()
    
    with runner.invoke('') as invoker:
        model.lm_head.input[0][0][0,0,:] = torch.tensor(antonym[0]).unsqueeze(0)
        post = model.lm_head.output[0][0].save()

decode_to_vocab(torch.tensor(pre), model.tokenizer)
decode_to_vocab(torch.tensor(post), model.tokenizer)


import torch

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

# Usage example
# decode_to_vocab(your_probability_distribution_tensor, your_model_tokenizer)
decode_to_vocab(torch.tensor(post), model.tokenizer)

torch.tensor(fv).reshape(1,1,model.config['resid_dim']).to(model.device)

with model.trace(remote=True) as runner:
    with runner.invoke('') as invoker:
        e = model.lm_head(torch.tensor(fv).reshape(1,1,model.config['resid_dim']).to(model.device)).save()

