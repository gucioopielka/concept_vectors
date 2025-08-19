import nnsight
from nnsight.intervention.graph.proxy import InterventionProxy
from dataclasses import dataclass
from rich.console import Console
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy

from utils.eval_utils import accuracy_completions
from utils.ICL_utils import ICLDataset
from utils.query_utils import get_avg_att_output, flush_torch_ram

import nnsight
from nnsight.intervention.graph.proxy import InterventionProxy

class InterventionResults:
    def __init__(self, use_nnsight=True):
        self.completion_ids = nnsight.list().save() if use_nnsight else []
        self.completion_probs = nnsight.list().save() if use_nnsight else []
        self.y_probs_1 = nnsight.list().save() if use_nnsight else []
        self.y_probs_2 = nnsight.list().save() if use_nnsight else []
        self.top_delta_ids = nnsight.list().save() if use_nnsight else []
        self.top_delta_probs = nnsight.list().save() if use_nnsight else []
        self.bottom_delta_ids = nnsight.list().save() if use_nnsight else []
        self.bottom_delta_probs = nnsight.list().save() if use_nnsight else []
        self.y_probs_1_fr = nnsight.list().save() if use_nnsight else []
        self.y_probs_1_es = nnsight.list().save() if use_nnsight else []

    def save(self):
        for key, value in self.__dict__.items():
            if isinstance(value, InterventionProxy):
                self.__dict__[key] = value.value


def get_probs_by_indices(probs, indices):
    return probs[torch.arange(probs.shape[0]), indices].tolist()

@flush_torch_ram
def get_summed_vector(model, heads_dict, token=-1):
    head_outputs = []
    for layer, head_list in heads_dict.items():
        out_proj_output = get_avg_att_output(model, layer, head_list, token=token)
        head_outputs.append(out_proj_output)
    return torch.stack(head_outputs).sum(dim=0)



def perform_intervention(model, prompts, intervention_vector, intervention_layer, results, org_probs, y_1_ids, y_2_ids=None, y_1_fr_ids=None, y_1_es_ids=None, token=-1, steer_weight=10):
    with model.lm.trace(prompts) as t:
        # Intervene
        hidden_states = model.lm.model.layers[intervention_layer].output[0]
        hidden_states[:, token] += intervention_vector * steer_weight

        # Get probabilities
        logits = model.lm.lm_head.output[:, -1].log_softmax(dim=-1)
        probs = logits.exp()
        max_probs = probs.max(dim=-1)
        
        # Save results
        results.completion_ids.append(max_probs.indices.tolist())
        results.completion_probs.append(max_probs.values.tolist())
        results.y_probs_1.append(get_probs_by_indices(probs, y_1_ids))
        if y_2_ids is not None:
            results.y_probs_2.append(get_probs_by_indices(probs, y_2_ids))
        if y_1_fr_ids is not None:
            results.y_probs_1_fr.append(get_probs_by_indices(probs, y_1_fr_ids))
        if y_1_es_ids is not None:
            results.y_probs_1_es.append(get_probs_by_indices(probs, y_1_es_ids))

        # Calculate and save deltas
        top_delta_probs = (probs - org_probs).topk(5, dim=-1)
        results.top_delta_probs.append(top_delta_probs.values.tolist())
        results.top_delta_ids.append(top_delta_probs.indices.tolist())
        
        bottom_delta_probs = (probs - org_probs).topk(5, dim=-1, largest=False)
        results.bottom_delta_probs.append(bottom_delta_probs.values.tolist())
        results.bottom_delta_ids.append(bottom_delta_probs.indices.tolist())

        return logits.save()

# ============================================================
# Pure PyTorch / HuggingFace implementation (KV-cache) for
# local experiments where NN-Sight is not desired. The original
# `perform_intervention` function above remains unchanged and
# relies on NN-Sight tracing. Use the new function when
# `model.remote_run` is False.
# ============================================================

def perform_intervention_kv(
    model,
    intervention_vector: torch.Tensor,
    intervention_layer: int,
    results,
    org_probs: torch.Tensor,
    y_1_ids,
    y_2_ids=None,
    y_1_fr_ids=None,
    y_1_es_ids=None,
    token: int = -1,
    steer_weight: int = 10,
    cache=None,
    last_ids_batch=None,
    attention_mask=None,
):
    """Intervene in the residual stream using a forward hook and KV cache.

    This version is fully independent of NN-Sight – it works directly with
    the underlying Hugging Face model stored in ``model.lm.model``. It adds
    ``steer_weight * intervention_vector`` to the hidden state at
    ``intervention_layer`` for the position ``token`` (default last) across
    all prompts, then logs the resulting probabilities in ``results``.
    """

    # Convenience handles
    hf_model = model.hf_model
    device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")

    vec = intervention_vector * steer_weight

    def patch_hidden(module, module_input, module_output):
        hidden = module_output[0]
        patched = hidden.clone()
        patched[:, token, :] += vec.to(hidden.device)
        return (patched,) + module_output[1:]

    handle = hf_model.model.layers[intervention_layer].register_forward_hook(patch_hidden)

    with torch.no_grad():
        past_key_values = copy.deepcopy(cache)
        out = hf_model(last_ids_batch, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, return_dict=True)
        logits = out.logits[:, -1].log_softmax(dim=-1)
        probs = logits.exp()

    handle.remove()

    # Record metrics
    max_probs = probs.max(dim=-1)
    results.completion_ids.append(max_probs.indices.tolist())
    results.completion_probs.append(max_probs.values.tolist())
    results.y_probs_1.append(get_probs_by_indices(probs, y_1_ids))
    if y_2_ids is not None:
        results.y_probs_2.append(get_probs_by_indices(probs, y_2_ids))
    if y_1_fr_ids is not None:
        results.y_probs_1_fr.append(get_probs_by_indices(probs, y_1_fr_ids))
    if y_1_es_ids is not None:
        results.y_probs_1_es.append(get_probs_by_indices(probs, y_1_es_ids))

    # Delta statistics
    top_delta = (probs - org_probs).topk(5, dim=-1)
    results.top_delta_probs.append(top_delta.values.tolist())
    results.top_delta_ids.append(top_delta.indices.tolist())

    bottom_delta = (probs - org_probs).topk(5, dim=-1, largest=False)
    results.bottom_delta_probs.append(bottom_delta.values.tolist())
    results.bottom_delta_ids.append(bottom_delta.indices.tolist())

    return logits

class InterventionEvaluation:
    def __init__(self, tokenizer, dataset: ICLDataset, extract_datasets: list, org_results: InterventionResults, fv_results: InterventionResults, cv_results: InterventionResults, random_results: InterventionResults=None):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.dataset_names = extract_datasets
        
        # Store results directly
        self.org_results = org_results
        self.fv_results = fv_results
        self.cv_results = cv_results
        self.random_results = random_results

        # Calculate delta probs
        self.delta_probs = self.get_delta_probs(completion = '1')

        # Original completions
        self.expected_completions = self.dataset.completions_1 if hasattr(self.dataset, 'completions_1') else self.dataset.completions

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def tokenize_completion(self, completion):
        return self.tokenizer.tokenize(completion)[0].replace('Ġ', ' ')

    def is_correct_completion(self, completion, y):
        return completion == self.tokenize_completion(y)

    @property
    def completions_org(self):
        return self.tokenizer.batch_decode(self.org_results.completion_ids)

    @property
    def completions_fv(self):
        return [self.tokenizer.batch_decode(ids) for ids in self.fv_results.completion_ids]

    @property
    def completions_cv(self):
        return [self.tokenizer.batch_decode(ids) for ids in self.cv_results.completion_ids]

    @property
    def top_delta_completions_fv(self):
        return [[self.tokenizer.batch_decode(ids) for ids in item] for item in self.fv_results.top_delta_ids]

    @property
    def top_delta_completions_cv(self):
        return [[self.tokenizer.batch_decode(ids) for ids in item] for item in self.cv_results.top_delta_ids]

    @property
    def bottom_delta_completions_fv(self):
        return [[self.tokenizer.batch_decode(ids) for ids in item] for item in self.fv_results.bottom_delta_ids]

    @property
    def bottom_delta_completions_cv(self):
        return [[self.tokenizer.batch_decode(ids) for ids in item] for item in self.cv_results.bottom_delta_ids]

    def get_yprobs_arr(self, completion=1):
        # if self.random_results is not None:
        #     return np.stack([
        #         np.stack(eval(f'self.fv_results.y_probs_{str(completion)}')), 
        #         np.stack(eval(f'self.cv_results.y_probs_{str(completion)}')), 
        #         np.stack(eval(f'self.random_results.y_probs_{str(completion)}'))
        #     ])
        # else:
        return np.stack([
            np.stack(eval(f'self.fv_results.y_probs_{str(completion)}')), 
            np.stack(eval(f'self.cv_results.y_probs_{str(completion)}'))
        ])

    def get_delta_probs(self, completion='1'):
        delta = self.get_yprobs_arr(completion) - eval(f'self.org_results.y_probs_{str(completion)}')
        return delta.transpose(2, 1, 0)

    @property
    def acc_org(self):
        return accuracy_completions(self.tokenizer, self.expected_completions, self.completions_org)

    @property
    def acc_fv(self):
        return [accuracy_completions(self.tokenizer, self.expected_completions, d) for d in self.completions_fv]

    @property
    def acc_cv(self):
        return [accuracy_completions(self.tokenizer, self.expected_completions, d) for d in self.completions_cv]

    def plot_acc(self):
        x = np.arange(len(self.dataset_names))
        width = 0.35
        plt.bar(x - width/2, self.acc_fv, width, label='FV')
        plt.bar(x + width/2, self.acc_cv, width, label='CV')
        plt.axhline(self.acc_org, color='black', linestyle='--', label='Original')
        plt.ylabel('Accuracy')
        plt.xticks(x, self.dataset_names)
        plt.legend()
        plt.show()
    
    def plot_delta_probs(self, completion='1'):
        x = np.arange(len(self.dataset_names))
        width = 0.35    
        m = self.delta_probs.mean(axis=0)
        se = self.delta_probs.std(axis=0) / np.sqrt(self.delta_probs.shape[0])
        plt.bar(x-width/2, m[:,0], yerr=se[:,0], width=width, label='FV') 
        plt.bar(x+width/2, m[:,1], yerr=se[:,1], width=width, label='CV') 
        plt.ylabel(r'$\Delta$ Probability')
        plt.xticks(x, self.dataset_names)
        plt.legend()
        plt.show()

    def plot_delta_probs_per_item(self, completion='1'):
        fig, ax = plt.subplots(3, 1, figsize=(8, 14))
        for i in range(3):
            ax[i].plot(self.delta_probs[:,i,0], label='+ FV')
            ax[i].plot(self.delta_probs[:,i,1], label='+ CV')
            ax[i].set_xlabel('Item')
            ax[i].set_ylabel(r'$\Delta$ Probability')
            ax[i].set_title(self.dataset_names[i])
        ax[i].legend()
        plt.show()

    def print_top_k_delta(self, item_idx, dataset_idx, vec='fv', k=5, bottom=False):
        s = ''
        if bottom:
            completions = self.bottom_delta_completions_fv if vec == 'fv' else self.bottom_delta_completions_cv
            probs = self.fv_results.bottom_delta_probs if vec == 'fv' else self.cv_results.bottom_delta_probs
        else:
            completions = self.top_delta_completions_fv if vec == 'fv' else self.top_delta_completions_cv
            probs = self.fv_results.top_delta_probs if vec == 'fv' else self.cv_results.top_delta_probs
        for i in range(k):
            completion = completions[dataset_idx][item_idx][i]
            prob = probs[dataset_idx][item_idx][i]
            color = 'green' if prob > 0 else 'red'
            sign = '+' if prob > 0 else '-'
            if self.is_correct_completion(completion, self.expected_completions[item_idx]):
                s += f"[u]{completion}[/u] ([{color}]{sign}{prob*100:.1f}%[/{color}])"
            else:
                s += f"{completion} ([{color}]{sign}{prob*100:.1f}%[/{color}])"
        return s

    def __getitem__(self, idx):
        console = Console(highlight=False)
        s = ''
        s = '[b]Item[/b]\n'
        s += '-'*50 + '\n'
        for i, (x, y) in enumerate(zip(self.dataset.seqs[idx].x, self.dataset.seqs[idx].y)):
            if i == len(self.dataset.seqs[idx]) - 1:
                s += f"Q: {x}\n[b]A[/]: [b cyan]?[/]\n"
            else:
                s += f"Q: {x}\n[b]A: {y}[/]\n\n\n"
        s += f'{self.dataset.dataset_name[0].split("_")[0].capitalize()} -> {self.expected_completions[idx]}\n'
        if hasattr(self.dataset, 'completions_2'):
            s += f'{self.dataset.dataset_name[1].split("_")[0].capitalize()} -> {self.dataset.completions_2[idx]}\n'
        s += '\n\n'
        s += '[bold]Without Intervention[/bold]\n'
        s += '-'*50 + '\n'
        s += f"Correct Token: [u]{self.tokenize_completion(self.expected_completions[idx])}[/u]     Probability: {self.org_results.y_probs_1[idx]*100:.1f}%\n"
        correct_org = self.is_correct_completion(self.completions_org[idx], self.expected_completions[idx])
        if correct_org:
            s += f"Original Predicted Token: [u]{self.completions_org[idx]}[/u]   Probability: {self.org_results.completion_probs[idx]*100:.1f}%    "
        else:
            s += f"Original Predicted Token: {self.completions_org[idx]}   Probability: {self.org_results.completion_probs[idx]*100:.1f}%    "
        s += "[green]Correct[/green]" if correct_org else "[red]Incorrect[/red]"
        s += '\n' + '-'*50 + '\n'
        for dataset_idx in range(len(self.dataset_names)):
            s += f"\n\n[b]+ {self.dataset_names[dataset_idx]}[/b]\n"
            s += '-'*50 + '\n'
            if self.is_correct_completion(self.completions_fv[dataset_idx][idx], self.expected_completions[idx]):
                s += f"Predicted Token [b]FV[/b]: [u]{self.completions_fv[dataset_idx][idx]}[/u]     Probability: {self.fv_results.completion_probs[dataset_idx][idx]*100:.1f}%     [green]Correct[/green]\n"
            else:
                s += f"Predicted Token [b]FV[/b]: {self.completions_fv[dataset_idx][idx]}     Probability: {self.fv_results.completion_probs[dataset_idx][idx]*100:.1f}%     [red]Incorrect[/red]\n"
            if self.is_correct_completion(self.completions_cv[dataset_idx][idx], self.expected_completions[idx]):
                s += f"Predicted Token [b]CV[/b]: [u]{self.completions_cv[dataset_idx][idx]}[/u]     Probability: {self.cv_results.completion_probs[dataset_idx][idx]*100:.1f}%     [green]Correct[/green]\n"
            else:
                s += f"Predicted Token [b]CV[/b]: {self.completions_cv[dataset_idx][idx]}     Probability: {self.cv_results.completion_probs[dataset_idx][idx]*100:.1f}%     [red]Incorrect[/red]\n"
            s += f"\nΔ Correct Token: [b]FV[/b]: "
            delta_probs = self.get_delta_probs('1')
            s += f"[green]+{delta_probs[idx][dataset_idx][0]*100:.1f}%[/green]" if delta_probs[idx][dataset_idx][0] > 0 else f"[red]{delta_probs[idx][dataset_idx][0]*100:.1f}%[/red]"
            s += f" [b]CV[/b]: "
            s += f"[green]+{delta_probs[idx][dataset_idx][1]*100:.1f}%[/green]" if delta_probs[idx][dataset_idx][1] > 0 else f"[red]{delta_probs[idx][dataset_idx][1]*100:.1f}%[/red]"
            s += f"\n\n"
            s += f"Top Δ Tokens FV: "
            s += self.print_top_k_delta(idx, dataset_idx, 'fv', 5)
            s += f"\n"
            s += f"Top Δ Tokens CV: "
            s += self.print_top_k_delta(idx, dataset_idx, 'cv', 5)
            s += f"\n\n"
            s += f"Bottom Δ Tokens FV: "
            s += self.print_top_k_delta(idx, dataset_idx, 'fv', 5, bottom=True)
            s += f"\n"
            s += f"Bottom Δ Tokens CV: "
            s += self.print_top_k_delta(idx, dataset_idx, 'cv', 5, bottom=True)
            s += '\n' + '-'*50 + '\n\n'
        console.print(s)