import os
from textwrap import dedent
import numpy as np
import pickle
import time
from datasets import load_dataset
import argparse

from utils.model_utils import ExtendedLanguageModel
from utils.query_utils import get_FVs

class MedicalDataset:
    def __init__(self, 
                 dataset_name='openlifescienceai/medmcqa',
                 size=100, 
                 batch_size=100,
                 example_prompt=True, 
                 seed=42):
        np.random.seed(seed)
        self.dataset = load_dataset(dataset_name)
        self.size = size
        self.batch_size = batch_size
        self.num_batches = self.calculate_n_batches(batch_size)
        self.seed = seed

        if example_prompt:
            self.example_prompt = self.construct_prompt(
                self.dataset['test']['question'][0],
                self.dataset['test']['opa'][0],
                self.dataset['test']['opb'][0],
                self.dataset['test']['opc'][0],
                self.dataset['test']['opd'][0],
                self.dataset['test']['cop'][0]
            )
        else:
            self.example_prompt = None

        self.dataset = self.dataset['train'][np.random.choice(len(self.dataset['train']), size)]
        self.ids = self.dataset['id']

        self.prompts = []
        self.completions = []
        for q in range(self.size):
            prompt = self.construct_prompt(
                self.dataset['question'][q],
                self.dataset['opa'][q],
                self.dataset['opb'][q],
                self.dataset['opc'][q],
                self.dataset['opd'][q],
            )
            prompt = self.construct_item(prompt, example_prompt=self.example_prompt)
            self.prompts.append(prompt)
            self.completions.append(['a', 'b', 'c', 'd'][self.dataset['cop'][q]])

    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise IndexError("Batch index out of range")
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.size)
        batch_prompts = self.prompts[start_idx:end_idx]
        batch_completions = self.completions[start_idx:end_idx]
        batch_ids = self.ids[start_idx:end_idx]
        return batch_prompts, batch_completions, batch_ids
        
    def calculate_n_batches(self, batch_size):
        return self.size // batch_size + (0 if self.size % batch_size == 0 else 1)
    
    def set_data(self, start_subset):
        self.prompts = self.prompts[start_subset:]
        self.completions = self.completions[start_subset:]
        self.size = len(self.prompts)
        self.num_batches = self.calculate_n_batches(self.batch_size)

    def construct_prompt(self, question, a, b, c, d, correct=None):
        prompt = f'''
        ### Instruction: {question}
        (a) {a}
        (b) {b}
        (c) {c}
        (d) {d}
        '''
        if correct:
            prompt += f'### Response ({['a', 'b', 'c', 'd'][correct]})\n\n'
        else:
            prompt += '### Response ('

        return dedent(prompt.lstrip('\n'))
        
    def construct_item(self, prompt, example_prompt=None):
        if example_prompt:
            prompt = self.example_prompt + prompt
        return prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple choice medical completions on a model')
    parser.add_argument('--model_name', type=str, help='The model name to use for completions', default='meta-llama/Llama-2-70b-hf')
    parser.add_argument('--size', type=int, help='The number of prompts to generate', default=100)
    parser.add_argument('--batch_size', type=int, help='The batch size to use for completions', default=20)
    parser.add_argument('--example_prompt', type=bool, help='Whether to use example prompts for the analogies', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--results_dir', type=str, help='The directory to save the results', default='data/medmcqa')

    args = parser.parse_args()
    model_name = args.model_name
    size = args.size
    batch_size = args.batch_size
    example_prompt = args.example_prompt
    RESULTS_DIR = args.results_dir

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    data_loader = MedicalDataset(batch_size=batch_size, size=size, example_prompt=example_prompt)
    print(data_loader.prompts[0])
    model = ExtendedLanguageModel(model_name)

    results_file = f'{RESULTS_DIR}/{model.nickname}.pkl'
    if not os.path.exists(results_file):
        FVs = np.empty((0, model.config['resid_dim']))
        completions = []
        Ys = []
        ids = []
    else:
        data = pickle.load(open(results_file, 'rb'))
        FVs = data['FVs']
        completions = data['completions']
        Ys = data['Ys']
        ids = data['ids']
        data_loader.set_data(len(FVs))

    for idx, (prompts, y, id) in enumerate(data_loader):
        t0 = time.time()
        FVs_batch, completions_batch = get_FVs(model, prompts, completion=True)
        FVs = np.concatenate([FVs, FVs_batch])
        completions.extend(completions_batch)
        Ys.extend(y)
        ids.extend(id)

        data = dict(
            FVs=FVs,
            completions=completions,
            Ys=Ys,
            ids=ids
        )
        pickle.dump(data, open(results_file, 'wb'))
        print(f'Completed batch {idx+1}/{len(data_loader)} in {time.time()-t0:.2f} seconds\n')





