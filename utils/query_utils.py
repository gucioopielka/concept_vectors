import numpy as np
import torch
from typing import Tuple, List, Dict
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