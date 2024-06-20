import os, json
import time
import argparse
from torch.nn.functional import softmax

from utils.data_utils import AnalogiesDataLoader
from utils.model_utils import ExtendedLanguageModel
from utils.query_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple choice analogy completions on a model')

    parser.add_argument('--model_name', type=str, help='The model name to use for completions', default='meta-llama/Llama-2-70b-hf')
    parser.add_argument('--dataset', type=str, help='The analogy dataset to use for completions', default='analogies_prowise_ENG')
    parser.add_argument('--batch_size', type=int, help='The batch size to use for completions', default=100)

    args = parser.parse_args()

    RESULTS_DIR = 'data/completions'
    model_name = args.model_name
    dataset = args.dataset
    batch_size = args.batch_size

    # load model and data
    model = ExtendedLanguageModel(model_name)
    data_loader = AnalogiesDataLoader(dataset, batch_size=batch_size)

    # load existing results or create new results file
    results_path = f'{RESULTS_DIR}/{dataset}__{model.nickname}.json'
    if not os.path.exists(results_path):
        model_output = {}
    else:
        # load existing results and filter out completed items
        model_output = json.load(open(results_path, 'r'))
        remaining_items = [item for item in data_loader.data if str(item[0]) not in model_output.keys()]
        data_loader.set_data(remaining_items) # set the data loader to only load the remaining items

    
    for idx, (prompts, mc_to_opt, indices) in enumerate(data_loader):
        t0 = time.time()

        # run model on prompts
        logits = get_logits(model, prompts)
        probs = softmax(logits, dim=1)

        # iterate over all items in the batch
        for i in range(len(prompts)):
            # convert to response options probs
            opt_probs = get_response_opt_probs(model, probs[i], mc_to_opt[i])

            # get next token prediction
            token, prob = get_completion(model, probs[i])
            opt_probs['completion_token'] = token
            opt_probs['completion_prob'] = prob.item()

            # save output
            model_output[indices[i]] = opt_probs
        
        # save output
        json.dump(model_output, open(results_path, 'w'), indent=4)
        print(f'Completed batch {idx+1}/{len(data_loader)} in {time.time()-t0:.2f} seconds\n')