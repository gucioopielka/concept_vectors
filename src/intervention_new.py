import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from rich.console import Console
import pickle
import deepl
from utils.model_utils import ExtendedLanguageModel
from utils.ICL_utils import ICLDataset, DatasetConstructor
from utils.query_utils import get_avg_att_output
from utils.eval_utils import SimilarityMatrix, accuracy_completions, InterventionResults
from utils.globals import RESULTS_DIR

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

torch.manual_seed(42)

def get_probs_by_indices(probs, indices):
    return probs[torch.arange(probs.shape[0]), indices].tolist()

def get_summed_vector(heads_dict, token=-1):
    head_outputs = []
    for layer, head_list in heads_dict.items():
        out_proj = model.config['out_proj'](layer)
        out_proj_output = get_avg_att_output(model, out_proj, head_list, token=token)
        head_outputs.append(out_proj_output)
    return torch.stack(head_outputs).sum(dim=0)

def perform_intervention(prompts, intervention_vector, intervention_layer, results, org_probs, y_1_ids, y_2_ids=None, y_1_fr_ids=None, y_1_es_ids=None, token=-1, steer_weight=10):
    with model.lm.trace(prompts) as t:
        # Intervene
        hidden_states = model.lm.model.layers[intervention_layer].output[0]
        hidden_states[:, token] += intervention_vector * steer_weight

        # Get probabilities
        probs = model.lm.lm_head.output[:, -1].log_softmax(dim=-1).exp()
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
    
def translate(text, source_lang, target_lang, model_type='quality_optimized'):
    translator = deepl.Translator(os.environ.get('DEEPL_API_KEY'))
    return translator.translate_text(text, source_lang=source_lang, target_lang=target_lang, model_type=model_type).text

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

if __name__ == '__main__':
    intervention_type = 'ambiguous'
    assert intervention_type in ['zeroshot', 'ambiguous']
    translations = False
    n_heads = 5
    model_name = 'meta-llama/Meta-Llama-3.1-70B'
    low_level_cie_path = os.path.join(RESULTS_DIR, 'CIE_LowLevel', model_name.split('/')[-1] + '.csv')
    model = ExtendedLanguageModel(model_name)
    cv_heads = model.get_rsa_heads(task_attribute='relation_verbal')
    fv_heads = model.get_fv_heads()

    # Extract
    extract_datasets = ['antonym_eng', 'antonym_fr', 'antonym_es', 'antonym_eng-mc']

    # Intervene
    steer_weight = 10 # TODO: Validate this
    if intervention_type == 'zeroshot':
        dataset_intervene = ICLDataset(
            dataset='antonym_eng',
            size=50, 
            n_train=0, 
            seed=42, 
            batch_size=50,
        )
    elif intervention_type == 'ambiguous':
        dataset_intervene = ICLDataset(
            dataset=['translation_eng_fr', 'antonym_eng'],
            size=50, 
            n_train=5, 
            seed=42, 
            batch_size=50,
            ambiguous_draws=[0, 0, 0, 1, 1, 1],
        )

    prompts = dataset_intervene.prompts
    if hasattr(dataset_intervene, 'completions_2'):
        y_1 = dataset_intervene.completions_1
        y_2 = dataset_intervene.completions_2
        y_1_ids = model.config['get_first_token_ids'](y_1)
        y_2_ids = model.config['get_first_token_ids'](y_2)
    else:
        y_1 = dataset_intervene.completions
        y_1_ids = model.config['get_first_token_ids'](y_1)
        y_2_ids = None  
    
    with open('translations_fr.txt', 'r') as f:
        y_1_fr = f.readlines()
    y_1_fr = [' '+line.strip() for line in y_1_fr]

    with open('translations_es.txt', 'r') as f:
        y_1_es = f.readlines()
    y_1_es = [' '+line.strip() for line in y_1_es]

    # if translations:
    #     print('Translating...')
    #     y_1_fr = [translate(x, 'EN', 'FR') for x in y_1]
    #     y_1_es = [translate(x, 'EN', 'ES') for x in y_1]
    #     y_1_fr_ids = model.config['get_first_token_ids'](y_1_fr)
    #     y_1_es_ids = model.config['get_first_token_ids'](y_1_es)
    # else:
    #     y_1_fr_ids = None
    #     y_1_es_ids = None

    layers = range(model.config['n_layers'])
    #layers = [18, 31]
    with model.lm.session(remote=True) as sess:
        # Get the FVs and CVs
        sess.log('Getting FVs and CVs')
        fvs = {}
        cvs = {}
        for dataset_name in extract_datasets:
            dataset_extract = DatasetConstructor(
                dataset_ids=dataset_name, 
                dataset_size=50, 
                n_train=5, 
                batch_size=50, 
                seed=42
            )
            
            # Get FVs and CVs
            with model.lm.trace(dataset_extract.prompts) as t:
                fvs[dataset_name] = get_summed_vector(fv_heads)
                cvs[dataset_name] = get_summed_vector(cv_heads)

        # Get original probabilities
        org_results = InterventionResults()
        with model.lm.trace(dataset_intervene.prompts) as t:
            org_probs = model.lm.lm_head.output[:, -1].log_softmax(dim=-1).exp()
            org_results.y_probs_1 = get_probs_by_indices(org_probs, y_1_ids).save()
            if y_2_ids is not None:
                org_results.y_probs_2 = get_probs_by_indices(org_probs, y_2_ids).save()
            if y_1_fr_ids is not None:
                org_results.y_probs_1_fr = get_probs_by_indices(org_probs, y_1_fr_ids).save()
            if y_1_es_ids is not None:
                org_results.y_probs_1_es = get_probs_by_indices(org_probs, y_1_es_ids).save()
            max_probs = org_probs.max(dim=-1)
            org_results.completion_ids = max_probs.indices.tolist().save()
            org_results.completion_probs = max_probs.values.tolist().save()
            
        # Perform interventions for each layer
        results = {
            'org': org_results,
            'fv': [],
            'cv': [],
            'random': []
        }
        for layer in layers:
            sess.log(f'Intervening layer {layer}')

            # Perform interventions for each dataset
            fv_results = InterventionResults()
            cv_results = InterventionResults()
            for dataset_name in extract_datasets:
                perform_intervention(dataset_intervene.prompts, fvs[dataset_name], layer, fv_results, org_probs, y_1_ids, y_2_ids, y_1_fr_ids, y_1_es_ids)
                perform_intervention(dataset_intervene.prompts, cvs[dataset_name], layer, cv_results, org_probs, y_1_ids, y_2_ids, y_1_fr_ids, y_1_es_ids)
            
            results['fv'].append(fv_results)
            results['cv'].append(cv_results)

    # Evaluate results
    delta_probs = []
    delta_probs_lang = []
    layer_evals = []
    for layer_idx, layer in enumerate(layers):
        results['org'].save()
        results['fv'][layer_idx].save()
        results['cv'][layer_idx].save()
        layer_eval = InterventionEvaluation(
            tokenizer=model.lm.tokenizer,
            dataset=dataset_intervene, 
            extract_datasets=extract_datasets,
            org_results=results['org'],
            fv_results=results['fv'][layer_idx],
            cv_results=results['cv'][layer_idx],
        )
        layer_evals.append(layer_eval)
        delta_intervention_probs_1 = layer_eval.get_delta_probs('1')
        if y_2_ids is not None:
            delta_intervention_probs_2 = layer_eval.get_delta_probs('2')
            delta_probs.append(np.stack([delta_intervention_probs_1, delta_intervention_probs_2]))
        else:
            delta_probs.append(delta_intervention_probs_1)

        if y_1_fr_ids and y_1_es_ids:
            delta_probs_lang.append(np.stack([layer_eval.get_delta_probs('1_fr'), layer_eval.get_delta_probs('1_es')]))


    # Save results
    file_name = 'zeroshot' if intervention_type == 'zeroshot' else 'ambiguous'
    pickle.dump(layer_evals, open(os.path.join(RESULTS_DIR, 'intervention_new', f"{model.nickname}_layer_evals_{file_name}.pkl"), 'wb'))
    np.save(os.path.join(RESULTS_DIR, 'intervention_new', f"{model.nickname}_delta_probs_{file_name}.npy"), np.stack(delta_probs))
    if len(delta_probs_lang) > 0:
        np.save(os.path.join(RESULTS_DIR, 'intervention_new', f"{model.nickname}_delta_probs_{file_name}_lang.npy"), np.stack(delta_probs_lang))



    dataset_intervene_tokenized = ICLDataset(
        dataset=['antonym_eng', 'synonym_eng'],
        size=50, 
        n_train=5, 
        seed=42, 
        batch_size=50,
        tokenizer=model.lm.tokenizer,
        ambiguous_draws=[0, 0, 0, 1, 1, 1],
    )
    y_1 = dataset_intervene_tokenized.completions_1
    y_1_ids = model.config['get_first_token_ids'](y_1)
    n_tokens = len(model.lm.tokenizer.batch_encode_plus(dataset_intervene_tokenized.prompts)['input_ids'][0])
    cv_layer = 31
    fv_layer = 24
    with model.lm.session(remote=True) as sess:
        # Get the FVs and CVs
        sess.log('Getting FVs and CVs')
        dataset_extract = DatasetConstructor(
            dataset_ids='antonym_eng', 
            dataset_size=50, 
            n_train=5, 
            batch_size=50, 
            seed=42
        )
        
        # Get FVs and CVs
        with model.lm.trace(dataset_extract.prompts) as t:
            fv = get_summed_vector(fv_heads)
            cv = get_summed_vector(cv_heads)

        # Get original probabilities
        org_results = InterventionResults()
        with model.lm.trace(dataset_intervene_tokenized.prompts) as t:
            org_probs = model.lm.lm_head.output[:, -1].log_softmax(dim=-1).exp()
            org_results.y_probs_1 = get_probs_by_indices(org_probs, y_1_ids).save()
            max_probs = org_probs.max(dim=-1)
            org_results.completion_ids = max_probs.indices.tolist().save()
            org_results.completion_probs = max_probs.values.tolist().save()
            
        # Perform interventions for each layer
        results = {
            'org': org_results,
            'fv': [],
            'cv': [],
        }
        for token in range(n_tokens):
            sess.log(f'Intervening token {token}')
            fv_results = InterventionResults()
            cv_results = InterventionResults()
            perform_intervention(dataset_intervene_tokenized.prompts, fv, fv_layer, fv_results, org_probs, y_1_ids, token=token)
            perform_intervention(dataset_intervene_tokenized.prompts, cv, cv_layer, cv_results, org_probs, y_1_ids, token=token)
            
            results['fv'].append(fv_results)
            results['cv'].append(cv_results)

    
    # Evaluate results
    delta_probs = []
    layer_evals = []
    for token_idx, token in enumerate(range(n_tokens)):
        results['org'].save()
        results['fv'][token_idx].save()
        results['cv'][token_idx].save()
        layer_eval = InterventionEvaluation(
            tokenizer=model.lm.tokenizer,
            dataset=dataset_intervene_tokenized, 
            extract_datasets=extract_datasets,
            org_results=results['org'],
            fv_results=results['fv'][token_idx],
            cv_results=results['cv'][token_idx],
        )
        delta_probs.append(layer_eval.delta_probs)

    np.stack(delta_probs).mean(axis=1).shape