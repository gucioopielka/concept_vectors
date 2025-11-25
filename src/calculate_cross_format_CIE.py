import argparse
from typing import *
import torch
import os
import pickle
import itertools
from itertools import batched

from utils.query_utils import calculate_cross_format_CIE
from utils.ICL_utils import DatasetConstructor
from utils.model_utils import ExtendedLanguageModel
from utils.globals import RESULTS_DIR

def save_to_csv(tensor: torch.Tensor, output_path: str):
    # Get indices for all elements
    layers, heads = torch.meshgrid(
        torch.arange(tensor.size(0)),
        torch.arange(tensor.size(1)),
        indexing='ij'
    )
    
    # Create list of tuples (layer, head, weight)
    data = list(zip(
        layers.flatten().cpu().tolist(),
        heads.flatten().cpu().tolist(),
        tensor.flatten().cpu().tolist()
    ))
    
    # Sort by weight in descending order
    data.sort(key=lambda x: x[2], reverse=True)
    
    # Create CSV string
    csv_lines = ['Layer,Head,Weight']
    csv_lines.extend(f'{layer},{head},{weight:.4f}' for layer, head, weight in data)
    csv_string = '\n'.join(csv_lines)
    
    # Save to file 
    output_path = output_path + '.csv' if not output_path.endswith('.csv') else output_path
    with open(output_path, 'w') as f:
        f.write(csv_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cross-format CIE.")
    parser.add_argument("--model", type=str, help="Name of the model to use.")
    parser.add_argument("--concepts", help="List of concepts to use.", nargs="+", type=str, 
                        default=['antonym', 'categorical', 'causal', 'presentPast', 'singularPlural', 'synonym', 'translation_eng_es', 'translation_eng_fr'])
    parser.add_argument("--dataset_size", type=int, help="Size of the dataset to use.", default=20)
    parser.add_argument("--layer_batch_size", type=int, help="Number of layers to process at once.", default=4)
    parser.add_argument("--prompt_batch_size", type=int, help="Number of prompts to process at once.", default=20)
    parser.add_argument('--n_train', type=int, help="Number of training examples to use.", default=5)
    parser.add_argument("--remote_run", type=bool, help="Whether to run the script on a remote server.", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, help="Random seed to use.", default=42)
    parser.add_argument("--output_dir", type=str, help="Path to save the files.", default="CIE_cross_format")

    args = parser.parse_args()

    # Load the model
    model = ExtendedLanguageModel(args.model, remote_run=args.remote_run)
    n_heads = model.config['n_heads']
    n_layers = model.config['n_layers']

    output_base_dir = os.path.join(RESULTS_DIR, args.output_dir, model.nickname)
    os.makedirs(output_base_dir, exist_ok=True)

    for concept in args.concepts:
        print(f"\nProcessing concept: {concept}")
        
        # Define formats
        if concept.startswith('translation'):
            # Translations usually have specific files like translation_eng_fr.json
            # So formats are translation_eng_fr-oe, translation_eng_fr-mc
            # But "Open-Ended French" for translation_eng_fr is weird.
            # The project structure suggests translation datasets are pairs.
            # Let's stick to simple formats for translations or just OE/MC if applicable.
            # Actually, calculate_CIE.py uses dataset_ids directly.
            # Let's assume standard variants exist.
            formats = [f"{concept}-oe", f"{concept}-mc"]
        else:
            formats = [f"{concept}_eng-oe", f"{concept}_fr-oe", f"{concept}_eng-mc"]

        # Generate all pairs
        format_pairs = list(itertools.product(formats, formats))
        
        for clean_fmt, target_fmt in format_pairs:
            # Skip if same format
            if clean_fmt == target_fmt:
                continue 

            pair_name = f"{clean_fmt}_to_{target_fmt}"
            output_path = os.path.join(output_base_dir, pair_name)
            
            if os.path.exists(output_path + '.csv'):
                print(f"Skipping {pair_name}, already exists.")
                continue

            print(f"Running {pair_name} ...")
            
            try:
                # Construct datasets
                # For clean dataset, n_train depends on format
                clean_n_train = 3 if clean_fmt.endswith('mc') else args.n_train
                clean_dataset_wrapper = DatasetConstructor(
                    dataset_ids=clean_fmt,
                    dataset_size=args.dataset_size,
                    n_train=clean_n_train,
                    seed=args.seed,
                    batch_size=args.prompt_batch_size
                )
                
                target_n_train = 3 if target_fmt.endswith('mc') else args.n_train
                target_dataset_wrapper = DatasetConstructor(
                    dataset_ids=target_fmt,
                    dataset_size=args.dataset_size,
                    n_train=target_n_train,
                    seed=args.seed,
                    batch_size=args.prompt_batch_size
                )
                
                if not clean_dataset_wrapper.datasets or not target_dataset_wrapper.datasets:
                    print(f"Could not load datasets for {pair_name}")
                    continue

                clean_ds = clean_dataset_wrapper.datasets[0]
                target_ds = target_dataset_wrapper.datasets[0]

                # Intermediate results handling
                intermediate_path = output_path + '_temp.pkl'
                if os.path.exists(intermediate_path):
                     results = torch.load(intermediate_path)
                     start = sum(batch.size(0) for batch in results)
                     print(f'Resuming from layer {start}...')
                else:
                    results = []
                    start = 0

                # Compute CIE
                for layers in batched(range(start, n_layers), args.layer_batch_size):
                    print(f"Processing layers: {layers}")
                    try:
                        batch_cie = calculate_cross_format_CIE(
                            model=model,
                            clean_dataset=clean_ds,
                            target_dataset=target_ds,
                            layers=layers
                        )
                        results.append(batch_cie)
                        torch.save(results, intermediate_path)
                    except Exception as e:
                        print(f"Error processing layers {layers}: {e}")
                        # If it's a file not found (dataset missing), we might want to skip
                        if "No such file" in str(e):
                            print("Dataset file missing, skipping pair.")
                            if os.path.exists(intermediate_path): os.remove(intermediate_path)
                            results = None
                            break
                        # Otherwise re-raise
                        raise e
                
                if results is not None:
                    cie = torch.cat(results, dim=0)
                    pickle.dump(cie.mean(dim=0), open(output_path + '.pkl', 'wb'))
                    #save_to_csv(cie, output_path)
                    if os.path.exists(intermediate_path):
                        os.remove(intermediate_path)

            except Exception as e:
                print(f"Failed to run {pair_name}: {e}")
                continue

