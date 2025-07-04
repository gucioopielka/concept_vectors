import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from rich.console import Console
import pickle
# import deepl
import argparse
from utils.model_utils import ExtendedLanguageModel
from utils.ICL_utils import ICLDataset, DatasetConstructor
from utils.query_utils import no_grad, flush_torch_ram
from utils.globals import RESULTS_DIR
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
    
# def translate(text, source_lang, target_lang, model_type='quality_optimized'):
#     translator = deepl.Translator(os.environ.get('DEEPL_API_KEY'))
#     return translator.translate_text(text, source_lang=source_lang, target_lang=target_lang, model_type=model_type).text

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

    OUTPUT_DIR = os.path.join(RESULTS_DIR, 'LUMI', 'intervention_kv', intervention_type, model.nickname)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            dataset=['antonym_eng', 'translation_eng_fr'],
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
    

    if intervention_type == 'ambiguous':
        with open('/scratch/project_465001574/cv/src/translations_fr.txt', 'r') as f:
            y_1_fr = f.readlines()
        y_1_fr = [' '+line.strip() for line in y_1_fr]

        with open('/scratch/project_465001574/cv/src/translations_es.txt', 'r') as f:
            y_1_es = f.readlines()
        y_1_es = [' '+line.strip() for line in y_1_es]
        # print('Translating...')
        # y_1_fr = [translate(x, 'EN', 'FR') for x in y_1]
        # y_1_es = [translate(x, 'EN', 'ES') for x in y_1]
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
                    fvs[dataset_name] = get_summed_vector(model, fv_heads)
                    cvs[dataset_name] = get_summed_vector(model, cv_heads)

            # -------------------------
            # Baseline probabilities (no intervention) using pure HF model
            # -------------------------
            org_results = InterventionResults()
            tokenizer = model.lm.tokenizer
            hf_model = model.lm.model
            device = model.device

            ids_batch = tokenizer(dataset_intervene.prompts, return_tensors="pt", padding=True).input_ids.to(device)
            prefix_ids_batch = ids_batch[:, :-1]
            last_ids_batch = ids_batch[:, -1:].to(device)  # shape (B,1)

            with torch.no_grad():
                prefix_out = hf_model(prefix_ids_batch, use_cache=True)
                pkv_batch = prefix_out.past_key_values
                last_out = hf_model(last_ids_batch, past_key_values=pkv_batch)
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

            results = {
                'org': org_results,
                'fv': [],
                'cv': [],
                'random': []
            }
            for layer in layers:
                sess.log(f'Intervening layer {layer+1}')

                # Perform interventions for each dataset
                fv_results = InterventionResults()
                cv_results = InterventionResults()
                
                for dataset_name in extract_datasets:
                    perform_intervention_kv(model, dataset_intervene.prompts, fvs[dataset_name], layer, fv_results, org_probs, y_1_ids,
                        y_2_ids,
                        y_1_fr_ids,
                        y_1_es_ids,
                        past_key_values_batch=pkv_batch,
                        last_ids_batch=last_ids_batch,
                    )
                    perform_intervention_kv(
                        model,
                        dataset_intervene.prompts,
                        cvs[dataset_name],
                        layer,
                        cv_results,
                        org_probs,
                        y_1_ids,
                        y_2_ids,
                        y_1_fr_ids,
                        y_1_es_ids,
                        past_key_values_batch=pkv_batch,
                        last_ids_batch=last_ids_batch,
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
    pickle.dump(layer_evals, open(os.path.join(OUTPUT_DIR, f"layer_evals.pkl"), 'wb'))
    np.save(os.path.join(OUTPUT_DIR, "delta_probs.npy"), np.stack(delta_probs))
    if intervention_type == 'ambiguous':
        np.save(os.path.join(OUTPUT_DIR, "delta_probs_lang.npy"), np.stack(delta_probs_lang))