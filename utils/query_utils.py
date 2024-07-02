import numpy as np
import torch
from typing import *
from utils.model_utils import ExtendedLanguageModel


def get_logits(model:ExtendedLanguageModel, prompt:List[str]|str, token:int = -1) -> torch.Tensor:
    '''
    Get the logits for the model's output tokens given a prompt or list of prompts

    Args:
        model (ExtendedLanguageModel): The model object
        prompt (List[str]|str): The prompt or list of prompts to run the model on
        token (int): The token index to save the logits for (default: -1 for the last token)
    
    Returns:
        torch.Tensor: The logits for the model's output tokens (shape: [batch_size, vocab_size])
    '''
    with model.trace(remote=True) as runner:
        with runner.invoke(prompt) as invoker:
            return model.lm_head.output[:,token,:].save()

def get_response_opt_probs(model:ExtendedLanguageModel, probs: torch.Tensor, mc_to_opt: Dict[str, str]) -> Dict[str, float]:
    '''
    Convert the model's multiple choice token probabilities to response option probabilities for a given item

    Args: 
        model (ExtendedLanguageModel): The model object
        probs (torch.Tensor): The model's token probabilities for a given item (shape: [vocab_size])
        mc_to_opt (dict): A mapping of the multiple choice tokens to the response options
    
    Returns:
        dict: A mapping of the response options to their probabilities {D: 0.2, Ans1: 0.1, ...}
    '''
    # index of multiple choice tokens in the model's vocab
    mc_probs = probs[model.mc_tokens_ids] 
    # map the probs to the response options
    mc_probs = {['a', 'b', 'c', 'd', 'e'][idx]:prob.item() for idx, prob in enumerate(mc_probs)} # map to abcde
    opt_probs = {mc_to_opt[mc]:prob for mc, prob in mc_probs.items()} # map back to respone options (D, Ans1, etc.)
    return opt_probs

def get_completion(model:ExtendedLanguageModel, probs: torch.Tensor) -> Tuple[str, np.ndarray]:
    '''
    Get the next token prediction and probability

    Args:
        model (ExtendedLanguageModel): The model object
        probs (torch.Tensor): The model's token probabilities for a given item (shape: [batch, vocab_size])
    
    Returns:
        str: The predicted token(s)
        float: The probability of the predicted token(s)
    '''
    probs = probs.unsqueeze(0) if probs.dim() == 1 else probs
    token_idx = torch.argmax(probs, dim=-1)
    completion = model.tokenizer.batch_decode(token_idx) if len(token_idx) > 1 else model.tokenizer.decode(token_idx)
    return completion, probs[torch.arange(probs.shape[0]), token_idx].numpy()

def get_FVs(model: ExtendedLanguageModel, prompts: List[str], completion=False) -> torch.Tensor|Tuple[torch.Tensor, List[str]]:
    '''
    Get the relation vector per item by summing the output of the most influential attention heads for the 
    model's output tokens given a prompt or list of prompts 

    Args:
        model (ExtendedLanguageModel): The model object
        prompts (List[str]): The prompt or list of prompts to run the model on
        completion (bool): Whether to return the completions for the model's output tokens (default: False)

    Returns:
        torch.Tensor: The relation vectors for the model's output tokens (shape: [batch_size, resid_dim])
        List[str]: The completions for the model's output tokens (if completion=True)
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

                if completion:
                    # Save the completions
                    token_ids = model.lm_head.output[:,T,:].argmax(dim=-1).save()

    # Sum all the attention heads per item
    relation_vecs = torch.sum(torch.stack(relation_vec_list),dim=0)

    if completion:
        # Decode the completions
        completions = model.tokenizer.batch_decode(token_ids)
        return relation_vecs.numpy(), completions

    return relation_vecs.numpy()