from collections import defaultdict
from typing import *
import functools

import einops
import torch
import torch.nn.functional as F
import nnsight

from .model_utils import ExtendedLanguageModel
from .eval_utils import spearman_rho_torch
from .ICL_utils import ICLDataset, DatasetConstructor

def no_grad(func):
    """Decorator to run a function with torch.no_grad() context."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper

def flush_torch_ram(func):
    """Decorator to flush torch RAM after function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    return wrapper

def convert_bfloat(func):
    """Decorator to convert bfloat tensors to their corresponding float types."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor) and 'bfloat' in str(result.dtype):
            target_dtype = str(result.dtype).split('.')[-1].replace('bfloat', 'float')
            return result.to(getattr(torch, target_dtype))
        return result
    return wrapper

def condense_matrix(X, n=None):
    '''
    Condense a square matrix into a condensed vector
    '''
    n = X.size(0) if n is None else n
    inds = torch.triu_indices(n, n, offset=1)
    return X[inds[0], inds[1]]
    
@no_grad
def get_completions(
    model: ExtendedLanguageModel, 
    dataset: ICLDataset | DatasetConstructor,
    logging: bool = True,
) -> List[str]:
    with model.lm.session(remote=model.remote_run) as sess:
        completion_ids = []
        for idx, (prompts, _) in enumerate(dataset):
            if logging:
                sess.log(f"Batch: {idx+1} / {len(dataset)}")
            with model.lm.trace(prompts) as t:
                logits = model.lm.lm_head.output[:, -1]
                completion_ids.extend([logits.log_softmax(dim=-1).argmax(dim=-1).save()])
    return model.lm.tokenizer.batch_decode(torch.cat(completion_ids))

@no_grad
def intervene_with_vec(
    model: ExtendedLanguageModel,
    dataset: ICLDataset | DatasetConstructor,
    vector: torch.Tensor,
    layers: List[int] = None,
    token: int = -1,
    remote: bool = True,
    logging: bool = True
) -> Dict[int, List[str]]:
    T = token
    with model.lm.session(remote=remote) as sess:
        intervention_top_ind = {layer: [] for layer in layers}
        for prompts, _ in dataset:
            for layer in layers:
                if logging:
                    sess.log(f'Layer: {layer}')
                with model.lm.trace(prompts) as t:
                    # Add the vector to the residual stream, at the last sequence position
                    hidden_states = model.lm.model.layers[layer].output[0]
                    hidden_states[:, T] += vector
                    # Get correct logprobs
                    logits = model.lm.lm_head.output[:, -1]
                    intervention_top_ind[layer].extend([logits.log_softmax(dim=-1).argmax(dim=-1).save()])
    
    # Decode the completions
    return {layer: model.lm.tokenizer.batch_decode(torch.stack(ind).squeeze()) for layer, ind in intervention_top_ind.items()}

@no_grad
def intervene_and_get_probs(
    model: ExtendedLanguageModel,
    dataset: ICLDataset | DatasetConstructor,
    vector: torch.Tensor,
    layer: int,
    token: int = -1,
    remote: bool = True,
    logging: bool = True
) -> Tuple[List[str], List[float]]:
    T = token
    with model.lm.session(remote=remote) as sess:
        completion_ids = nnsight.list().save()
        y_logits = nnsight.list().save()
        for prompts, y in dataset:
            y_ids = model.config['get_first_token_ids'](y)
            with model.lm.trace(prompts) as t:
                # Add the vector to the residual stream, at the last sequence position
                hidden_states = model.lm.model.layers[layer].output[0]
                hidden_states[:, T] += vector
                # Get correct logprobs
                logits = model.lm.lm_head.output[:, -1].log_softmax(dim=-1)
                completion_ids.extend(logits.argmax(dim=-1).tolist())
                y_logits.extend(logits[torch.arange(len(prompts)), y_ids].tolist())

    completions = model.lm.tokenizer.batch_decode(torch.tensor(completion_ids))
    y_probs = torch.tensor(y_logits).exp().tolist()
    return completions, y_probs

def get_att_out_proj_input(
    model: ExtendedLanguageModel,
    layer: int,
    token: int = -1,
) -> torch.Tensor:
    out_proj = model.config['out_proj'](layer).inputs[0][0][:, token]
    return out_proj.view(out_proj.shape[0], model.config['n_heads'], model.config['d_head'])

def get_avg_att_output(
    model: ExtendedLanguageModel,
    layer: int,
    heads: List[int],
    token: int = -1,
) -> torch.Tensor:
    heads_to_ablate = list(set(range(model.config['n_heads'])) - set(heads))
    z = get_att_out_proj_input(model, layer, token)
    z_ablated = z.mean(dim=0)
    z_ablated[heads_to_ablate, :] = 0.0
    z_ablated = z_ablated.view(-1)
    return model.config['out_proj'](layer)(z_ablated) # If multiple heads are in the list of heads to keep, the output will be the sum of those heads

def get_att_output_per_item(
    model: ExtendedLanguageModel,
    layer: int,
    heads: List[int],
    token: int = -1,
) -> torch.Tensor:
    heads_to_ablate = list(set(range(model.config['n_heads'])) - set(heads))
    z = get_att_out_proj_input(model, layer, token)
    z_ablated = z.clone()
    z_ablated[:, heads_to_ablate, :] = 0.0
    z_ablated = z_ablated.view(z.shape[0], -1)
    return model.config['out_proj'](layer)(z_ablated) # If multiple heads are in the list of heads to keep, the output will be the sum of those heads

@no_grad
def get_summed_vec_per_item(
    model: ExtendedLanguageModel,
    dataset: ICLDataset | DatasetConstructor,
    heads: Dict[int, List[int]],
    token: int = -1,
    logging: bool = True
) -> torch.Tensor:
    with model.lm.session(remote=model.remote_run) as sess:
        all_head_outputs = nnsight.list().save()
        
        for idx, (batched_prompts, _) in enumerate(dataset):
            if logging:
                sess.log(f"Batch: {idx+1} / {len(dataset)}")
            
            with model.lm.trace(batched_prompts) as t:
                batch_head_outputs = []
                for layer, head_list in heads.items():
                    out_proj_output = get_att_output_per_item(model, layer, head_list, token=token)
                    batch_head_outputs.append(out_proj_output)
                
                # Stack and sum for this batch
                batch_summed = torch.stack(batch_head_outputs).sum(dim=0)
                batch_summed = batch_summed if model.remote_run else batch_summed.cpu()
                all_head_outputs.append(batch_summed)

    return torch.cat(all_head_outputs, dim=0).to(model.device)

@no_grad
@convert_bfloat
def get_summed_vec_simmat(
    model: ExtendedLanguageModel,
    dataset: ICLDataset | DatasetConstructor,
    heads: Dict[int, List[int]],
    token: int = -1,
    logging: bool = True
) -> torch.Tensor:
    with model.lm.session(remote=model.remote_run) as sess:
        all_head_outputs = nnsight.list()
        
        for idx, (batched_prompts, _) in enumerate(dataset):
            if logging:
                sess.log(f"Batch: {idx+1} / {len(dataset)}")
            
            with model.lm.trace(batched_prompts) as t:
                batch_head_outputs = []
                for layer, head_list in heads.items():
                    out_proj_output = get_att_output_per_item(model, layer, head_list, token=token)
                    batch_head_outputs.append(out_proj_output)
                
                # Stack and sum for this batch
                batch_summed = torch.stack(batch_head_outputs).sum(dim=0)
                batch_summed = batch_summed if model.remote_run else batch_summed.cpu()
                all_head_outputs.append(batch_summed)
        
        simmat = compute_similarity_matrix(torch.cat(all_head_outputs, dim=0))
        simmat_condensed = condense_matrix(simmat, n=len(dataset.prompts)).save()

    return simmat_condensed.cpu()

@no_grad
def get_avg_summed_vec(
    model: ExtendedLanguageModel, 
    dataset: ICLDataset | DatasetConstructor,
    heads: List[Tuple[int, int, float]],
    token: int = -1,
    remote: bool = True,
    logging: bool = True
) -> torch.Tensor:
    '''
    Get the relation vector by summing the output of the most influential attention heads for the 
    model's output tokens given a prompt or list of prompts. Either per item or averaged across the batch.

    Args:
        model (ExtendedLanguageModel): The model object
        prompts (List[str]): The prompt or list of prompts to run the model on
        average_heads (bool): Whether to average across the heads in a batch or get a relation vector per item (default: False)
    Returns:
        torch.Tensor: The relation vectors for the model's output tokens (shape: [batch_size, resid_dim])
    '''
    # Turn head_list into a dict of {layer: heads we need in this layer}
    head_dict = defaultdict(set)
    for layer, head in heads:
        head_dict[layer].add(head)
    head_dict = dict(head_dict)

    relation_vec_list = []
    with model.lm.session(remote=remote) as sess:
        with model.lm.trace(dataset.prompts) as runner:
            for layer, head_list in head_dict.items():        
                out_proj_output = get_avg_att_output(model, layer, head_list, token=token)
                relation_vec_list.append(out_proj_output.save())

    # Sum all the attention heads per item
    relation_vec_tensor = torch.stack(relation_vec_list) # (len(head_list), D_MODEL) if average_heads else (len(head_list), B, D_MODEL)
    relation_vecs = torch.sum(relation_vec_tensor, dim=0)

    return relation_vecs

@flush_torch_ram
def compute_similarity_matrix(vectors: torch.Tensor) -> torch.Tensor:
    norm_v = F.normalize(vectors, p=2, dim=1)
    return torch.matmul(norm_v, torch.transpose(norm_v, 0, 1))

@flush_torch_ram
@no_grad 
@convert_bfloat
def get_att_simmats(
    model: ExtendedLanguageModel,
    dataset: ICLDataset | DatasetConstructor,
    layers: List[int] = None,
    heads: List[List[int]] = None,
    token: int = -1,
    logging: bool = True
) -> torch.Tensor:
    layers = list(range(model.config['n_layers'])) if (layers is None) else layers
    heads = list(range(model.config['n_heads']))

    with model.lm.session(remote=model.remote_run) as sess:

        sess.log(f"Getting hidden states ...")
        simmat_dict = {(layer, head): [] for layer in range(len(layers)) for head in range(len(heads))}
        for idx, (batched_prompts, _) in enumerate(dataset):
            if logging:
                sess.log(f"Batch: {idx+1} / {len(dataset)}")
            
            # Collect the hidden states for each head
            with model.lm.trace(batched_prompts) as t:
                for layer_idx, layer in enumerate(layers):
                    att_out = get_att_out_proj_input(model, layer, token)
                    for head_idx, head in enumerate(heads):
                        act = att_out[:, head]
                        act = act if model.remote_run else act.cpu()
                        simmat_dict[(layer_idx, head_idx)].append(act)

        sess.log(f"Computing similarity matrices ...")
        simmats = nnsight.list([[[] for _ in range(len(heads))] for _ in range(len(layers))]).save()
        for (layer, head), v in simmat_dict.items():
            v = torch.concat(v).to(model.device)
            simmat = compute_similarity_matrix(v)
            simmats[layer][head] = condense_matrix(simmat, n=len(dataset.prompts)).cpu()
    
    return torch.stack([torch.stack(simmats[layer]) for layer in range(len(layers))])

@no_grad
def get_rsa(
    model: ExtendedLanguageModel,
    dataset: ICLDataset | DatasetConstructor,
    design_matrix: torch.Tensor,
    layers: Optional[List[int]] = None,
    token: int = -1,
    remote: bool = True,
    logging: bool = True
) -> torch.Tensor:
    layers = range(model.config['n_layers']) if (layers is None) else layers
    heads = range(model.config['n_heads'])
    N_HEADS = model.config['n_heads']
    D_HEAD = model.config['resid_dim'] // N_HEADS

    with model.lm.session(remote=remote) as sess:        
            
        sess.log(f"Extracting hidden states ...")
        simmat_dict = {(layer, head): [] for layer in layers for head in heads}
        for batched_prompts, _ in dataset:            
            # Collect the hidden states for each head
            with model.lm.trace(batched_prompts) as t:
                for layer in layers:
                    # Get hidden states, reshape to get head dimension, store the mean tensor
                    out_proj = model.config['out_proj'](layer)
                    z = out_proj.inputs[0][0][:, token]
                    z_reshaped = z.reshape(len(batched_prompts), N_HEADS, D_HEAD)
                    for head in heads:
                        z_head = z_reshaped[:, head]
                        simmat_dict[(layer, head)].extend([z_head])

        sess.log(f"Computing similarity matrices ...")
        for k, v in simmat_dict.items():
            simmat_dict[k] = compute_similarity_matrix(torch.concat(v))
        
        # Get the upper triangular indices of the similarity matrix
        n = len(dataset.prompts)
        inds = torch.triu_indices(n, n, offset=1)
        design_matrix_condensed = design_matrix[inds[0], inds[1]]

        sess.log(f"Computing RSA ...")
        rsa_vals = nnsight.list([[] for _ in layers]).save()
        for i, layer in enumerate(layers):
            for head in heads:
                v_condensed = simmat_dict[(layer, head)][inds[0], inds[1]]
                rho = spearman_rho_torch(v_condensed, design_matrix_condensed)
                rsa_vals[i].append(rho)
    
    return torch.tensor(rsa_vals)

@flush_torch_ram
@no_grad
def calculate_CIE(
    model: ExtendedLanguageModel,
    dataset: ICLDataset,
    layers: Optional[List[int]] = None,
) -> torch.Tensor:
    '''
    Returns a tensor of shape (layers, heads), containing the CIE for each head.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the function vector (we'll also create a
            corrupted version of this dataset for interventions)
        layers: Optional[List[int]]
            the layers which this function will calculate the score for (if None, we assume all layers)
    '''
    corrupted_dataset = dataset.create_corrupted_dataset()
    layers = range(model.config['n_layers']) if (layers is None) else layers
    heads = range(model.config['n_heads'])

    N_HEADS = model.config['n_heads']
    D_HEAD = model.config['resid_dim'] // N_HEADS
    T = -1 # values taken from last token
        
    with model.lm.session(remote=model.remote_run) as sess:
        z_dict = {(layer, head): [] for layer in layers for head in heads}
        correct_logprobs_corrupted = []
        
        sess.log("Original prompts ...") 
        for (prompts, completions), (prompts_corrupted, _) in zip(dataset, corrupted_dataset):
            correct_completion_ids = model.config['get_first_token_ids'](completions)
            # Run a forward pass on corrupted prompts, where we don't intervene or store activations (just so we can
            # get the correct-token logprobs to compare with our intervention)  
            with model.lm.trace(prompts_corrupted) as t:
                logits = model.lm.lm_head.output[:, -1]
                correct_logprobs_corrupted.extend([logits.log_softmax(dim=-1)[torch.arange(len(prompts)), correct_completion_ids].save()])

            # Run a forward pass on clean prompts, where we store attention head outputs
            with model.lm.trace(prompts) as t:
                for layer in layers:
                    # Get hidden states, reshape to get head dimension, store the mean tensor
                    out_proj = model.config['out_proj'](layer)
                    z = out_proj.inputs[0][0][:, T]
                    z_reshaped = z.reshape(len(prompts), N_HEADS, D_HEAD)
                    for head in heads:
                        z_head = z_reshaped[:, head]
                        z_dict[(layer, head)].extend([z_head])

        # Get the mean of the head activations
        z_dict = {
            k: torch.stack(v).squeeze().mean(dim=0)
            for k, v in z_dict.items()
        }
        
        # For each head, run a forward pass on corrupted prompts (here we need multiple different forward passes
        correct_logprobs_dict = {}
        for (prompts, completions), (prompts_corrupted, _) in zip(dataset, corrupted_dataset):
            correct_completion_ids = model.config['get_first_token_ids'](completions)
            # For each head, run a forward pass on corrupted prompts (here we need multiple different forward passes,
            # because we're doing different interventions each time)
            for layer in layers:
                sess.log(f"Dataset: {dataset.dataset} | Layer: {layer}")
                for head in heads:
                    with model.lm.trace(prompts_corrupted) as t:
                        # Get hidden states, reshape to get head dimension, then set it to the a-vector
                        out_proj = model.config['out_proj'](layer)
                        z = out_proj.inputs[0][0][:, T]
                        z.reshape(len(prompts), N_HEADS, D_HEAD)[:, head] = z_dict[(layer, head)]
                        # Get logprobs at the end, which we'll compare with our corrupted logprobs
                        logits = model.lm.lm_head.output[:, -1]
                        correct_logprobs = logits.log_softmax(dim=-1)[torch.arange(len(prompts)), correct_completion_ids]
                        correct_logprobs = correct_logprobs if model.remote_run else correct_logprobs.cpu()
                        correct_logprobs_dict[(layer, head)] = correct_logprobs.save()

    # Get difference between intervention logprobs and corrupted logprobs, and take mean over batch dim
    all_correct_logprobs_intervention = einops.rearrange(
        torch.stack([v.value for v in correct_logprobs_dict.values()]),
        "(layers heads) batch -> layers heads batch",
        layers = len(layers),
    )
    correct_logprobs_corrupted = torch.stack(correct_logprobs_corrupted).squeeze() if model.remote_run else torch.stack(correct_logprobs_corrupted).squeeze().cpu()
    logprobs_diff = all_correct_logprobs_intervention - correct_logprobs_corrupted # shape [layers heads batch]

    # Return mean effect of intervention, over the batch dimension
    return logprobs_diff.mean(dim=-1)