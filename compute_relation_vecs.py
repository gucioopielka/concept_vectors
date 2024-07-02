import os
import time
import argparse
import numpy as np

from utils.data_utils import AnalogiesDataLoader
from utils.model_utils import ExtendedLanguageModel
from utils.extract_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple choice analogy completions on a model')

    parser.add_argument('--model_name', type=str, help='The model name to use for completions', default='meta-llama/Llama-2-70b-hf')
    parser.add_argument('--prompt_type', type=str, help='The type of prompt to generate (multiple choice = "mc", open ended = "oe")', default='mc')
    parser.add_argument('--dataset', type=str, help='The analogy dataset to use for completions', default='analogies_prowise_ENG')
    parser.add_argument('--batch_size', type=int, help='The batch size to use for completions', default=100)
    parser.add_argument('--example_prompt', type=bool, help='Whether to use example prompts for the analogies', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--n_items', type=int, help='The number of items to run', default=None)
    parser.add_argument('--output_path', type=str, help='The directory to save the results', default=None)

    args = parser.parse_args()

    model_name = args.model_name
    prompt_type = args.prompt_type
    dataset = args.dataset
    batch_size = args.batch_size
    example_prompt = args.example_prompt
    n_items = args.n_items
    results_dir = args.output_path
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # load model and data
    model = ExtendedLanguageModel(model_name)
    data_loader = AnalogiesDataLoader(dataset, prompt_type=prompt_type, batch_size=batch_size, 
                                      example_prompt=example_prompt, n_items=n_items)

    # load existing results or create new results file
    results_path = f'{results_dir}/{dataset}__{model.nickname}.npy'
    if not os.path.exists(results_path):
        vecs = np.empty((0, model.config['resid_dim']))
    else:
        # load existing results and filter out completed items
        vecs = np.load(results_path)
        if n_items:
            # we don't care about item ids if we're only running a subset of the data
            remaining_items = data_loader.data[len(vecs):]
        else:
            remaining_items = [item for item in data_loader.data if item[0] not in list(range(vecs.shape[0]))]

        data_loader.set_data(remaining_items) # set the data loader to only load the remaining items


    for idx, data in enumerate(data_loader):
        t0 = time.time()
        
        prompts = data[0]

        # run model on prompts
        batch_vecs = get_relation_vecs(model, prompts)
        vecs = np.concatenate([vecs, batch_vecs])

        # save output
        np.save(open(results_path, 'wb'), vecs)
        print(f'Completed batch {idx+1}/{len(data_loader)} in {time.time()-t0:.2f} seconds\n')