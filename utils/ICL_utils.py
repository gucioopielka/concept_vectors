import os
from typing import *
import json
import numpy as np
from textwrap import dedent

class ICLSequence:
    '''
    Class to store a single antonym sequence.

    Uses the default template "Q: {x}\nA: {y}" (with separate pairs split by "\n\n").
    '''
    def __init__(
        self, 
        word_pairs: List[List[str]],
        padded_space: bool = True
    ):
        self.word_pairs = word_pairs
        self.padded_space = padded_space
        self.x, self.y = zip(*word_pairs)

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx: int):
        return self.word_pairs[idx]

    def prompt(self):
        '''Returns the prompt, which contains all but the second element in the last word pair.'''
        p = "\n\n".join([f"Q: {x}\nA: {y}" for x, y in self.word_pairs])
        return p[:-len(self.completion())]

    def completion(self):
        '''Returns the second element in the last word pair.'''
        return " " + self.y[-1] if self.padded_space else self.y[-1]

    def __str__(self):
        '''Prints a readable string representation of the prompt & completion (indep of template).'''
        return f"{', '.join([f'({x}, {y})' for x, y in self[:-1]])}, {self.x[-1]} ->".strip(", ")
    
class ICLMultipleChoice:
    def __init__(
        self, 
        word_pairs: List[List[str]], 
        all_word_pairs: List[List[str]], 
        seed: int = 0
    ):
        self.word_pairs = word_pairs
        self.y_list = [pair[1] for pair in all_word_pairs]
        self.x, self.y = zip(*word_pairs)

        self.options = []
        self.correct = []
        for idx, word in enumerate(self.y):
            np.random.seed(seed + idx*100)
            options = self.generate_unique_options(word)
            np.random.shuffle(options)
            self.options.append(options)
            self.correct.append(options.index(word))

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx: int):
        return self.word_pairs[idx]
    
    def generate_unique_options(self, correct):
        # Generate 3 random options from all possible Ys 
        # If all options are unique, then return
        while True:
            options = [correct] + [np.random.choice(self.y_list) for _ in range(3)]
            if len(set(options)) == 4:
                return options
    
    def prompt(self):
        self.prompt = ''
        for idx in range(len(self.word_pairs)):
            if idx == len(self.word_pairs) - 1:
                self.prompt += self.format_prompt(idx, correct=False)
            else:
                self.prompt += self.format_prompt(idx, correct=True)
        return self.prompt
    
    def completion(self):
        return ['a', 'b', 'c', 'd'][self.correct[-1]]
    
    def format_prompt(self, idx, correct=True):
        prompt = f'''
        ### Instruction: Q: {self.x[idx]} A: ?
        (a) {self.options[idx][0]}
        (b) {self.options[idx][1]}
        (c) {self.options[idx][2]}
        (d) {self.options[idx][3]}
        '''
        if correct:
            prompt += f'### Response: ({['a', 'b', 'c', 'd'][self.correct[idx]]})\n' 
        else:
            prompt += '### Response: ('

        return dedent(prompt.lstrip('\n'))
        
    
class ICLDataset:
    '''
    Dataset to create antonym pair prompts, in ICL task format. We use random seeds for consistency
    between the corrupted and clean datasets.

    Inputs:
        word_pairs:
            list of ICL task, e.g. [["old", "young"], ["top", "bottom"], ...] for the antonym task
        size:
            number of prompts to generate
        n_prepended:
            number of antonym pairs before the single-word ICL task
        bidirectional:
            if True, then we also consider the reversed antonym pairs
        corrupted:
            if True, then the second word in each pair is replaced with a random word
        seed:
            random seed, for consistency & reproducibility
    '''

    def __init__(
        self,
        dataset: str,
        size: int,
        n_prepended: int,
        response_type: str = 'open_ended',
        bidirectional: bool = True,
        seed: int = 0,
        corrupted: bool = False,
        padded_space: bool = True,
        batch_size: int = None,
        data_source: str = 'abstractive',
        root_data_dir: str = 'data/ICL'
    ):  
        # Load the data
        data_dir = f'{root_data_dir}/{data_source}'
        d_path = os.path.join(data_dir, f'{dataset}.json')
        raw_data = json.load(open(d_path, 'r'))
        self.word_pairs = [[i['input'], i['output']] for i in raw_data]        

        self.word_list = [word for word_pair in self.word_pairs for word in word_pair]
        self.size = size
        self.n_prepended = n_prepended
        self.bidirectional = bidirectional
        self.corrupted = corrupted
        self.seed = seed

        self.seqs = []
        self.prompts = []
        self.completions = []

        # Generate the dataset (by choosing random word pairs, and constructing `ICLSequence` objects)
        for n in range(size):
            np.random.seed(seed + n)
            random_pairs = np.random.choice(len(self.word_pairs), n_prepended+1, replace=False)
            # Randomize the order of each word pair (x, y). If not bidirectional, we always have x -> y not y -> x
            random_orders = np.random.choice([1, -1], n_prepended+1)
            if not(bidirectional): random_orders[:] = 1
            word_pairs = [self.word_pairs[pair][::order] for pair, order in zip(random_pairs, random_orders)]
            # If corrupted, then replace y with a random word in all (x, y) pairs except the last one
            if corrupted:
                for i in range(len(word_pairs) - 1):
                    word_pairs[i][1] = np.random.choice(self.word_list)
            if response_type == 'open_ended':
                seq = ICLSequence(word_pairs, padded_space=padded_space)
            elif response_type == 'multiple_choice':
                seq = ICLMultipleChoice(word_pairs, self.word_pairs, seed=seed+n)

            self.seqs.append(seq)
            self.prompts.append(seq.prompt())
            self.completions.append(seq.completion())

        self.batch_size = batch_size if batch_size else size
        self.num_batches = self.calculate_n_batches(self.batch_size)

    def create_corrupted_dataset(self):
        '''Creates a corrupted version of the dataset (with same random seed).'''
        return ICLDataset(self.word_pairs, self.size, self.n_prepended, self.bidirectional, corrupted=True, seed=self.seed)
    
    def calculate_n_batches(self, batch_size):
        return self.size // batch_size + (0 if self.size % batch_size == 0 else 1)

    def __len__(self):
        return self.calculate_n_batches(self.batch_size)

    def __getitem__(self, idx: int):
        if idx >= self.num_batches:
            raise IndexError("Batch index out of range")
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.size)
        batch_prompts = self.prompts[start_idx:end_idx]
        batch_completions = self.completions[start_idx:end_idx]
        return batch_prompts, batch_completions
    
    def multiple_choice(self, idx: int):
        '''Returns the multiple choice options for the idx-th prompt.'''
        x, y = self.seqs[idx].x[-1], self.seqs[idx].y[-1]
        options = [y] + [np.random.choice(self.word_list) for _ in range(3)]
        np.random.shuffle(options)
        return options