import numpy as np
import pandas as pd
import torch
import os
import copy
import matplotlib.pyplot as plt
from rich.console import Console
import pickle
from tqdm import tqdm
from transformers import StaticCache
import deepl
import argparse
from utils.model_utils import ExtendedLanguageModel
from utils.ICL_utils import ICLDataset, DatasetConstructor
from utils.query_utils import no_grad, flush_torch_ram
from utils.globals import RESULTS_DIR, DATASET_DIR
from utils.intervention_utils import (
    InterventionResults,
    InterventionEvaluation,
    get_summed_vector,
    perform_intervention_kv,
    get_probs_by_indices,
)

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

torch.manual_seed(42)
    
def translate(text, source_lang, target_lang, model_type='quality_optimized'):
    translator = deepl.Translator(os.environ.get('DEEPL_API_KEY'))
    return translator.translate_text(text, source_lang=source_lang, target_lang=target_lang, model_type=model_type).text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3.1-70B')
    parser.add_argument('--intervention_type', type=str, default='ambiguous')
    args = parser.parse_args()
    
    intervention_type = args.intervention_type
    assert intervention_type in ['zeroshot', 'ambiguous']
    n_heads = 5
    model_name = args.model_name
    model = ExtendedLanguageModel(model_name, load_metrics=True)
    cv_heads = model.get_top_heads('RSA', n_heads, to_dict=True)
    fv_heads = model.get_top_heads('CIE', n_heads, to_dict=True)

    for concept in ['antonym', 'categorical', 'causal', 'presentPast', 'singularPlural', 'synonym']:

        OUTPUT_DIR = os.path.join(RESULTS_DIR, 'LUMI', 'intervention_kv', intervention_type, concept, model.nickname)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Extract
        extract_datasets = [f'{concept}_eng', f'{concept}_fr', f'{concept}_es', f'{concept}_eng-mc']

        # Intervene
        if intervention_type == 'zeroshot':
            dataset_intervene = ICLDataset(
                dataset=f'{concept}_eng',
                size=50, 
                n_train=0, 
                seed=42, 
                batch_size=50,
            )
        elif intervention_type == 'ambiguous':
            dataset_intervene = ICLDataset(
                dataset=[f'{concept}_eng', 'translation_eng_fr'],
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
        
        # Translate to French and Spanish
        if intervention_type == 'ambiguous':
            path_fr = os.path.join(DATASET_DIR, 'ambigous_translations', f'{concept}_fr.txt')
            path_es = os.path.join(DATASET_DIR, 'ambigous_translations', f'{concept}_es.txt')
            
            if not os.path.exists(path_fr):
                y_1_fr = [translate(x, 'EN', 'FR') for x in y_1]
            else:
                with open(path_fr, 'r') as f:
                    y_1_fr = [' '+line.strip() for line in f.readlines()]

            if not os.path.exists(path_es):
                y_1_es = [translate(x, 'EN', 'ES') for x in y_1]
            else:
                with open(path_es, 'r') as f:
                    y_1_es = [' '+line.strip() for line in f.readlines()]
            
            y_1_fr_ids = model.config['get_first_token_ids'](y_1_fr)
            y_1_es_ids = model.config['get_first_token_ids'](y_1_es)
        else:
            y_1_fr_ids = None
            y_1_es_ids = None

        layers = range(model.config['n_layers'])
        with torch.no_grad():
            with model.lm.session(remote=model.remote_run) as sess:
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
                        fvs[dataset_name] = get_summed_vector(model, fv_heads).save()
                        cvs[dataset_name] = get_summed_vector(model, cv_heads).save()

        # Prepare prompts and cache
        org_results = InterventionResults(use_nnsight=False)
        tokenizer = model.lm.tokenizer
        hf_model = model.hf_model
        device = model.device
        
        tokenized_prompts = tokenizer(dataset_intervene.prompts, return_tensors="pt", padding=True)
        ids_batch = tokenized_prompts.input_ids.to(device)
        attention_mask = tokenized_prompts.attention_mask.to(device)
        
        prefix_ids_batch = ids_batch[:, :-1]
        prefix_attention_mask = attention_mask[:, :-1]
        last_ids_batch = ids_batch[:, -1:].to(device)

        cache = StaticCache(config=hf_model.config, max_batch_size=len(ids_batch), max_cache_len=ids_batch.shape[1], device=device)
        
        # Get baseline probabilities
        with torch.no_grad():
            cache = hf_model(prefix_ids_batch, attention_mask=prefix_attention_mask, use_cache=True, return_dict=True).past_key_values
            past_key_values = copy.deepcopy(cache)
            last_out = hf_model(last_ids_batch, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True, return_dict=True)
            logits_batch = last_out.logits[:, -1]

        org_probs = logits_batch.log_softmax(dim=-1).exp()

        # Store baseline metrics
        org_results.y_probs_1 = get_probs_by_indices(org_probs, y_1_ids)
        if y_2_ids is not None:
            org_results.y_probs_2 = get_probs_by_indices(org_probs, y_2_ids)
        if y_1_fr_ids is not None:
            org_results.y_probs_1_fr = get_probs_by_indices(org_probs, y_1_fr_ids)
        if y_1_es_ids is not None:
            org_results.y_probs_1_es = get_probs_by_indices(org_probs, y_1_es_ids)

        max_probs = org_probs.max(dim=-1)
        org_results.completion_ids = max_probs.indices.tolist()
        org_results.completion_probs = max_probs.values.tolist()
        torch.cuda.empty_cache()

        # Intervene
        for steer_weight in range(1, 16):
            results = {
                'org': org_results,
                'fv': [],
                'cv': [],
            }
        
            print(f'Intervening... steer_weight = {steer_weight}')
            for layer in tqdm(layers):

                # Perform interventions for each dataset
                fv_results = InterventionResults(use_nnsight=False)
                cv_results = InterventionResults(use_nnsight=False)
                
                for dataset_name in extract_datasets:
                    for vec, res in [(fvs[dataset_name], fv_results), (cvs[dataset_name], cv_results)]:
                        perform_intervention_kv(
                            model=model,
                            intervention_vector=vec,
                            intervention_layer=layer,
                            steer_weight=steer_weight,
                            results=res,
                            org_probs=org_probs,
                            y_1_ids=y_1_ids,
                            y_2_ids=y_2_ids,
                            y_1_fr_ids=y_1_fr_ids,
                            y_1_es_ids=y_1_es_ids,
                            cache=cache,
                            last_ids_batch=last_ids_batch,
                            attention_mask=attention_mask,
                        )
                
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
            pickle.dump(layer_evals, open(os.path.join(OUTPUT_DIR, f"layer_evals_x{steer_weight}.pkl"), 'wb'))
            np.save(os.path.join(OUTPUT_DIR, f"delta_probs_x{steer_weight}.npy"), np.stack(delta_probs))
            if intervention_type == 'ambiguous':
                np.save(os.path.join(OUTPUT_DIR, f"delta_probs_lang_x{steer_weight}.npy"), np.stack(delta_probs_lang))