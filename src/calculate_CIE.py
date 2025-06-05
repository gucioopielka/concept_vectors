import argparse
from typing import *
import torch
import os
import pickle
import signal
import functools
from itertools import batched

from utils.query_utils import calculate_CIE as original_calculate_CIE
from utils.ICL_utils import DatasetConstructor
from utils.model_utils import ExtendedLanguageModel
from utils.globals import RESULTS_DIR

def timeout(seconds: int = 4000) -> Callable:
    """
    Decorator that raises a TimeoutError if the function takes longer than specified seconds.
    """
    def decorator(func: Callable) -> Callable:
        def _handle_timeout(signum, frame):
            raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Set the signal handler and a 1-hour alarm
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            
            return result

        return wrapper
    return decorator

# Wrap the original calculate_CIE with timeout
calculate_CIE = timeout()(original_calculate_CIE)

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
    parser = argparse.ArgumentParser(description="Compute function vectors for a given model.")
    parser.add_argument("--model", type=str, help="Name of the model to use.")
    parser.add_argument("--datasets", help="List of datasets to use.", nargs="+", type=str)
    parser.add_argument("--dataset_size", type=int, help="Size of the dataset to use.", default=20)
    parser.add_argument("--layer_batch_size", type=int, help="Number of layers to process at once.", default=4)
    parser.add_argument("--prompt_batch_size", type=int, help="Number of prompts to process at once.", default=20)
    parser.add_argument('--n_train', type=int, help="Number of training examples to use.", default=5)
    parser.add_argument("--remote_run", type=bool, help="Whether to run the script on a remote server.", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, help="Random seed to use.", default=42)
    parser.add_argument("--dataset_dir", type=str, help="Path to the directory with the data.", default=RESULTS_DIR)
    parser.add_argument("--output_dir", type=str, help="Path to save the files.")

    args = parser.parse_args()

    # Load the model
    model = ExtendedLanguageModel(args.model)
    n_heads = model.config['n_heads']
    n_layers = model.config['n_layers']

    # Load intermediate results if they exist
    intermediate_results_path = os.path.join(args.dataset_dir, args.output_dir, f'cie_{model.nickname}.pkl')
    if os.path.exists(intermediate_results_path):
        print('Loading intermediate results...')
        cie = pickle.load(open(intermediate_results_path, 'rb'))
        datasets = args.datasets[len(cie):] # Only process the remaining datasets
        if not datasets:
            print(f"All datasets have been processed. Exiting")
            exit()
    else:
        cie = []
        datasets = args.datasets
    
    # Compute the CIE for each dataset
    for dataset_name in datasets:
        print(f"\n{'- *'*5}\nProcessing dataset: {dataset_name} ...\n{'* -'*5}")
        
        # Load intermediate results if they exist
        intermediate_results_dataset_path = os.path.join(args.dataset_dir, args.output_dir, f'cie_{model.nickname}_{dataset_name}.pkl')
        if os.path.exists(intermediate_results_dataset_path):
            results = pickle.load(open(intermediate_results_dataset_path, 'rb'))
            start = sum(batch.size(0) for batch in results) # Calculate the total number of layers processed so far
        else:
            results = []
            start = 0

        # Set the number of training examples
        if dataset_name.endswith('-mc'):
            n_train = 3 # For mc datasets, use 3 training examples
        else:
            n_train = args.n_train

        # Load the dataset
        dataset = DatasetConstructor(
            dataset_ids=dataset_name, 
            dataset_size=args.dataset_size,
            n_train=n_train, 
            seed=args.seed, 
            batch_size=args.prompt_batch_size
        )

        # Layer batch size for 70B model and mc datasets
        if args.model.endswith('70B') and dataset_name.endswith('-mc'):
            layer_batch_size = 27
        else:
            layer_batch_size = args.layer_batch_size

        # Compute CIE for each batch of layers
        for layers in batched(range(n_layers), layer_batch_size):
            print("Processing layers: ", layers)
            try:
                batch_cie = calculate_CIE(model=model, dataset=dataset.datasets[0], layers=layers, remote=args.remote_run)
                results.append(batch_cie)
                torch.save(results, intermediate_results_dataset_path) # Save intermediate results
            except TimeoutError:
                print(f"\nTimeout occurred while processing layers {layers}. Saving intermediate results and exiting...")
                torch.save(results, intermediate_results_dataset_path)
                exit(1)
        cie.append(torch.cat(results, dim=0))

        # Delete intermediate dataset results
        os.remove(intermediate_results_dataset_path)

        # Save intermediate CIE results
        pickle.dump(cie, open(intermediate_results_path, 'wb'))

    # Average across datasets
    cie = torch.stack(cie, dim=0)
    aie = cie.mean(dim=0)

    # Save to CSV
    output_path = os.path.join(args.dataset_dir, args.output_dir, f'{model.nickname}.csv')
    save_to_csv(aie, output_path)