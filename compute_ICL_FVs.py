import os
import pickle
import time
import argparse

from utils.ICL_utils import *
from utils.model_utils import ExtendedLanguageModel

RESULTS_DIR = 'data/ICL/results'

datasets = [
    'antonym', 'capitalize', 'country-capital', 'english-french', 
    'present-past', 'singular-plural', 'person-instrument',
    'person-sport', 'product-company', 'landmark-country'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ICL completions on a model')

    parser.add_argument('--model_name', type=str, help='The model name to use for completions', default='meta-llama/Llama-2-70b-hf')
    parser.add_argument('--n_train', type=int, help='The number of training examples to use', default=5)
    parser.add_argument('--size', type=int, help='The number of prompts to generate', default=100)
    parser.add_argument('--batch_size', type=int, help='The batch size to use for completions', default=100)

    args = parser.parse_args()

    model_name = args.model_name
    n_train = args.n_train
    size = args.size
    batch_size = args.batch_size

    # load model and data
    dataset_config = dict(
        size=size, 
        n_prepended=n_train, 
        batch_size=batch_size,
        bidirectional=False,
        padded_space=False,
        seed=42
    )

    model = ExtendedLanguageModel('meta-llama/Llama-2-70b-hf')

    # load existing results or create new results file
    results_path = f'{RESULTS_DIR}/{model.nickname}__{n_train}_n.pkl'
    if not os.path.exists(results_path):
        data = {}
    else:
        # load existing results and filter out completed datasets
        data = pickle.load(open(results_path, 'rb'))
        remaining_datasets = [dataset for dataset in datasets if dataset not in data.keys()]
        datasets = remaining_datasets


    for dataset in datasets:
        print(f'Running completions on dataset: {dataset}\n')
        data_loader = ICLDataset(dataset=dataset, **dataset_config)

        FVs = np.empty((0, model.config['resid_dim']))
        completions = []
        Ys = []
        for idx, (prompts, y) in enumerate(data_loader):
            t0 = time.time()
            
            FVs_batch, completions_batch = get_FVs_and_completions(model, prompts)
            
            FVs = np.concatenate([FVs, FVs_batch])
            completions.extend(completions_batch)
            Ys.extend(y)
            print(f'Completed batch {idx+1}/{len(data_loader)} in {time.time()-t0:.2f} seconds\n')

        data[dataset] = {'FVs': FVs, 'completions': completions, 'Ys': Ys}
        pickle.dump(data, open(results_path, 'wb'))

