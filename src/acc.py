import os
import argparse
import pandas as pd

from utils.ICL_utils import DatasetConstructor
from utils.model_utils import ExtendedLanguageModel
from utils.eval_utils import accuracy_completions
from utils.query_utils import get_completions
from utils.globals import RESULTS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute function vectors for a given model.")
    parser.add_argument("--model", type=str, help="Name of the model to use.", required=True)
    parser.add_argument("--datasets", nargs="+", type=str, help="List of datasets to use.", required=True)
    parser.add_argument("--dataset_size", type=int, help="Size of the dataset to use.", default=20)
    parser.add_argument('--n_train', type=int, help="Number of training examples to use.", default=5)
    parser.add_argument("--prompt_batch_size", type=int, help="Number of prompts to process at once.", default=20)
    parser.add_argument("--remote_run", type=bool, help="Whether to run the script on a remote server.", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, help="Random seed to use.", default=42)
    parser.add_argument("--output_dir", type=str, help="Path to save the output files.", required=True)

    args = parser.parse_args()

    # Load the model
    model = ExtendedLanguageModel(args.model, remote_run=args.remote_run)

    # Output directory
    output_dir = os.path.join(RESULTS_DIR, args.output_dir, model.nickname)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the dataset
    dataset_constructor = DatasetConstructor(
        dataset_ids=args.datasets, 
        dataset_size=args.dataset_size, 
        n_train=args.n_train,
        batch_size=args.prompt_batch_size,
        seed=args.seed
    )

    # Get the completions
    completions = get_completions(model, dataset_constructor)

    # Compute the accuracy and store results
    results = []
    for idx, dataset in enumerate(dataset_constructor.datasets):
        dataset_completions = completions[dataset.size * idx : dataset.size * (idx + 1)]
        acc = accuracy_completions(model, dataset_completions, dataset.completions)
        
        results.append({
            'concept': dataset.dataset_name.split('_')[0],
            'dataset': dataset.dataset_name,
            'accuracy': acc,
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f"results_ntrain{args.n_train}.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Results saved to: {output_file}")
    print(df)
