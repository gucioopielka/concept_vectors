import os
import time
import argparse
import numpy as np

from utils.data_utils import AnalogiesDataLoader
from utils.model_utils import ExtendedLanguageModel
from utils.extract_utils import *

RESULTS_DIR = 'data/internals'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple choice analogy completions on a model')

    parser.add_argument('--model_name', type=str, help='The model name to use for completions', default='meta-llama/Llama-2-70b-hf')
    parser.add_argument('--dataset', type=str, help='The analogy dataset to use for completions', default='analogies_prowise_ENG')
    parser.add_argument('--batch_size', type=int, help='The batch size to use for completions', default=100)

    args = parser.parse_args()

    model_name = args.model_name
    dataset = args.dataset
    batch_size = args.batch_size

    # load model and data
    model = ExtendedLanguageModel(model_name)
    data_loader = AnalogiesDataLoader(dataset, batch_size=batch_size)

    # load existing results or create new results file
    results_path = f'{RESULTS_DIR}/{dataset}__{model.nickname}.npy'
    if not os.path.exists(results_path):
        vecs = np.empty((0, model.config['resid_dim']))
    else:
        # load existing results and filter out completed items
        vecs = np.load(results_path)
        remaining_items = [item for item in data_loader.data if item[0] not in list(range(vecs.shape[0]))]
        data_loader.set_data(remaining_items) # set the data loader to only load the remaining items


    for idx, (prompts, mc_to_opt, indices) in enumerate(data_loader):
        t0 = time.time()

        # run model on prompts
        batch_vecs = get_relation_vecs(model, prompts)
        vecs = np.concatenate([vecs, batch_vecs])

        # save output
        np.save(open(results_path, 'wb'), vecs)
        print(f'Completed batch {idx+1}/{len(data_loader)} in {time.time()-t0:.2f} seconds\n')