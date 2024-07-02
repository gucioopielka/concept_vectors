import os
import json, pickle
import time
import argparse
import numpy as np

from utils.ICL_utils import ICLDataset
from utils.model_utils import ExtendedLanguageModel
from utils.api_utils import APICallManager
from utils.query_utils import get_FVs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ICL completions on a model')

    parser.add_argument('--model_name', type=str, help='The model name to use for completions', default='meta-llama/Llama-2-70b-hf')
    parser.add_argument('--n_train', type=int, help='The number of training examples to use', default=5)
    parser.add_argument('--size', type=int, help='The number of prompts to generate', default=100)
    parser.add_argument('--batch_size', type=int, help='The batch size to use for completions', default=100)
    parser.add_argument('--seed', type=int, help='The random seed to use for the data loader', default=42)
    parser.add_argument('--response_type', type=str, help='The type of response to generate (multiple_choice or open_ended)', default='open_ended')
    parser.add_argument('--data_source', type=str, help='The source of the data: "abstractive" or "conceptnet"', default='abstractive')
    parser.add_argument('--data_dir', type=str, help='The directory to load the results from', default='data/ICL')
    parser.add_argument('--results_dir', type=str, help='The directory to save the results', default=None)

    args = parser.parse_args()

    model_name = args.model_name
    n_train = args.n_train
    size = args.size
    seed = args.seed
    batch_size = args.batch_size
    response_type = args.response_type
    data_source = args.data_source
    data_dir = f'{args.data_dir}/{data_source}'
    results_dir = f'{args.data_dir}/results/{data_source}/{response_type}' if args.results_dir is None else args.results_dir

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if data_source == 'abstractive':
        datasets = [
            'antonym', 'capitalize', 'country-capital', 'english-french', 
            'present-past', 'singular-plural', 'person-instrument',
            'person-sport', 'product-company', 'landmark-country'
        ]
    
    elif data_source == 'conceptnet':
        min_data = 15 #TODO
        # get all datasets with enough data
        relation_paths = [f'{data_dir}/{dataset}' for dataset in os.listdir(data_dir)]
        enough_data = [len(json.load(open(path,'rb'))) >= min_data for path in relation_paths] 
        datasets =  [rel.rsplit('.',1)[0] for i, rel in enumerate(os.listdir(data_dir)) if enough_data[i]]

    # load model and data
    dataset_config = dict(
        size=size, 
        n_prepended=n_train, 
        batch_size=batch_size,
        bidirectional=False,
        padded_space=False,
        response_type=response_type,
        data_source=data_source,
        seed=seed
    )

    model = ExtendedLanguageModel(model_name)

    # load existing results or create new results file
    results_file = f'{results_dir}/{model.nickname}__{n_train}_n.pkl'
    if not os.path.exists(results_file):
        data = {}
    else:
        # load existing results and filter out completed datasets
        data = pickle.load(open(results_file, 'rb'))
        remaining_datasets = [dataset for dataset in datasets if dataset not in data.keys()]
        datasets = remaining_datasets


    for dataset_idx, dataset in enumerate(datasets):
        print(f'Running completions on dataset: {dataset}\n')
        data_loader = ICLDataset(dataset=dataset, **dataset_config)

        FVs = np.empty((0, model.config['resid_dim']))
        completions = []
        Ys = []
        for idx, (prompts, y) in enumerate(data_loader):
            t0 = time.time()
            
            FVs_batch, completions_batch = get_FVs(model, prompts, completion=True)
            # api_manager = APICallManager(get_FVs, model=model, prompts=prompts, completion=True)
            # FVs_batch, completions_batch = api_manager()

            FVs = np.concatenate([FVs, FVs_batch])
            completions.extend(completions_batch)
            Ys.extend(y)
            print(f'Dataset: {dataset_idx+1}/{len(datasets)} --- Completed batch {idx+1}/{len(data_loader)} in {time.time()-t0:.2f} seconds\n')

        data[dataset] = {'FVs': FVs, 'completions': completions, 'Ys': Ys}
        pickle.dump(data, open(results_file, 'wb'))