import os
from typing import *
import json
import hashlib
from textwrap import dedent

import numpy as np
from rich.console import Console

from .globals import DATASET_DIR


class ICLSequence:
    '''
    Class to store a single item sequence.

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
        seed: int,
        format_prompt: str = 'letter',
        padded_space: bool = True
    ):
        assert format_prompt in ['letter', 'word'], "format_prompt must be either 'letter' or 'word'"
        self.word_pairs = word_pairs
        self.y_list = [pair[1] for pair in all_word_pairs]
        self.x, self.y = zip(*word_pairs)
        self.format_prompt = format_prompt
        self.format_prompt_func = eval(f"self.format_prompt_{format_prompt}")
        self.padded_space = padded_space

        self.options = []
        self.correct = []
        for idx, word in enumerate(self.y):
            np.random.seed(seed + idx*100)
            options = self.generate_unique_options(word)
            np.random.shuffle(options)
            self.options.append(options)
            if self.format_prompt == 'letter':
                self.correct.append(options.index(word))
            elif self.format_prompt == 'word':
                self.correct.append(word)
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
                self.prompt += self.format_prompt_func(idx, correct=False)
            else:
                self.prompt += self.format_prompt_func(idx, correct=True)
        return self.prompt
    
    def completion(self):
        if self.format_prompt == 'letter':
            return ['a', 'b', 'c', 'd'][self.correct[-1]]
        elif self.format_prompt == 'word':
            return ' ' + self.y[-1] if self.padded_space else self.y[-1]
    
    def format_prompt_letter(self, idx, correct=True):
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

    def format_prompt_word(self, idx, correct=True):
        prompt = f'''
        ### Instruction: Q: {self.x[idx]} A: ?
        {self.options[idx][0]}
        {self.options[idx][1]}
        {self.options[idx][2]}
        {self.options[idx][3]}
        '''
        if correct:
            prompt += f'### Response: {self.correct[idx]}\n'
        else:
            prompt += '### Response: ' if self.padded_space else '### Response:'

        return dedent(prompt.lstrip('\n'))

class ICLAbstractSequence:
    def __init__(
        self, 
        concept: str,
        symbols: List[str],
        seq_len: int,
        seed: int = None
    ):
        assert concept in ['successor', 'predecessor'], "concept must be either 'successor' or 'predecessor"
        assert len(symbols) >= 4, "symbols must be at least 4 characters long"
        assert seq_len > 4, 'Sequence length must be longer than 4'
        if seed:
            np.random.seed(seed)

        if len(symbols) > 4:
            symbols = np.random.choice(symbols, 4, replace=False)
        
        self.x, self.y = self.generate_seq_task(concept, symbols, seq_len)

    def generate_seq_task(self, concept, symbols, seq_len): 
        # Generate the sequence with the letters at random positions 
        seq = ['.']*seq_len
        generate = True
        while generate:
            letter_pos = np.random.choice(seq_len, 4, replace=False)
            empty_pos = set(range(seq_len)) - set(letter_pos)
            # Ensure there is at least one empty position either after the last letter (successor) or before the first letter (predecessor)
            if concept == 'successor':
                generate = all([True if i > max(letter_pos) else False for i in empty_pos])
            if concept == 'predecessor':
                generate = all([True if i < min(letter_pos) else False for i in empty_pos])

        for letter, pos in zip(symbols, letter_pos):
            seq[pos] = letter

        # Randomly select position for the indicator '*'
        range_pos = range(min(letter_pos), seq_len) if concept == 'predecessor' else range(max(letter_pos)+1) # Predecessor only after the first letter and successor before the last letter
        valid_pos = [i for i in range_pos if i not in letter_pos] # Can't be at the position of the letters
        indicator_pos = np.random.choice(valid_pos)
        seq[indicator_pos] = '*'
        
        # Determine the closest letter position based on the concept
        if concept == 'predecessor':
            closest_letter_pos = max([pos for pos in letter_pos if pos < indicator_pos])
        elif concept == 'successor':
            closest_letter_pos = min([pos for pos in letter_pos if pos > indicator_pos])
        
        return " ".join(str(x) for x in seq), seq[closest_letter_pos]


class ICLDataset:
    '''
    Dataset to create antonym pair prompts, in ICL task format. We use random seeds for consistency
    between the corrupted and clean datasets.

    Inputs:
        word_pairs:
            list of ICL task, e.g. [["old", "young"], ["top", "bottom"], ...] for the antonym task
        size:
            number of prompts to generate
        n_train:
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
        n_train: int,
        response_type: str = 'open_ended',
        mc_format: str = 'letter',
        bidirectional: bool = False,
        seed: int = 0,
        seed_shuffle: int = 1,
        corrupted: bool = False,
        shuffle_prompts: str = None,
        symbols: str = 'letter',
        seq_len: int = 20,
        tokenizer: Any = None,
        padded_space: bool = True,
        batch_size: int = None,
        ambiguous_draws: List[int] = None,
        data_dir: str = DATASET_DIR,
    ):  
        self.dataset = dataset
        self.dataset_name = dataset
        self.response_type = response_type   
        self.size = size
        self.n_train = n_train
        self.bidirectional = bidirectional
        self.padded_space = padded_space
        self.corrupted = corrupted
        self.symbols = symbols
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.seed = seed
        self.shuffle_prompts = shuffle_prompts
        self.seed_shuffle = seed_shuffle
        self.mc_format = mc_format

        self.seqs = []
        self.prompts = []
        self.completions = []

        if self.dataset in ['predecessor', 'successor']:
            if symbols == 'word':
                # Create the word list from the datasets
                datasets = ['antonym', 'capitalize_first_letter', 'capitalize_last_letter', 'capitalize', 'english-french', 'synonym']
                word_list = []
                for dataset in datasets:
                    file_path = os.path.join(data_dir, '..', 'todd_et_al', f'{dataset}.json')
                    file = json.load(open(file_path, 'r'))
                    word_list.extend([i['input'] for i in file])
                self.symbol_list = list(set(word_list)) # Remove duplicates
                
                if tokenizer:
                    # Ensure that the words are tokenized as single tokens
                    self.symbol_list = [word for word in word_list if len(tokenizer.tokenize(word)) == 1]
            
            elif symbols == 'letter':
                self.symbol_list = ['a', 'b', 'c', 'd']
            
            self.create_seq_dataset()
            self.dataset_name = f"{self.dataset}_{symbols}"
        
        # elif isinstance(dataset, list):
        #     seqs = []
        #     for d in dataset:
        #         icl_d = ICLDataset(
        #             d,
        #             size,
        #             n_train // len(dataset),
        #             response_type=response_type,
        #             bidirectional=bidirectional,
        #             seed=generate_seed(d, seed),
        #             corrupted=corrupted,
        #             shuffle_prompts=shuffle_prompts,
        #             symbols=symbols,
        #             seq_len=seq_len,
        #             tokenizer=tokenizer,
        #             padded_space=padded_space,
        #             batch_size=batch_size,
        #             data_dir=data_dir
        #         )
        #         seqs.append(icl_d.seqs)
            
        #     for seq1, seq2 in zip(*seqs):
        #         # Concatenate the two arrays without the first element of the first sequence
        #         combined = np.concatenate((seq1[1:], seq2), axis=0)

        #         # Create a random permutation of indices for the combined array
        #         permuted_indices = np.random.permutation(len(combined))

        #         # Shuffle combined, and add the first element of the first sequence to the end
        #         shuffled = combined[permuted_indices]
        #         interleaved_array = np.concatenate((shuffled, seq1[:1]), axis=0)
                                
        #         # Create the interleaved sequence
        #         interleaved_seq = ICLSequence(interleaved_array.tolist(), padded_space=padded_space)

        #         self.seqs.append(interleaved_seq)
        #         self.prompts.append(interleaved_seq.prompt())
        #         self.completions.append(interleaved_seq.completion())
        
        elif isinstance(dataset, list):
            xs = []
            word_pairs = []
            for d in dataset:
                if 'translation' in d:
                    pairs = ICLDataset(d, size=1, n_train=0, data_dir=os.path.join(data_dir, '..', 'ambigous_translations')).word_pairs
                else:
                    pairs = ICLDataset(d, size=1, n_train=0).word_pairs
                if tokenizer: 
                    pairs_to_keep = []
                    for pair in pairs:
                        if len(tokenizer.tokenize(pair[0])) == 1 and len(tokenizer.tokenize(pair[1])) == 1:
                            pairs_to_keep.append(pair)
                    pairs = pairs_to_keep
                word_pairs.append(pairs)
                xs.append([pair[0] for pair in pairs])

            x_interesection = sorted(set(xs[0]).intersection(set(xs[1])))
            print(f'Number of pairs: {len(x_interesection)}')

            dataset_1_dict = {i[0]: i[1] for i in word_pairs[0]}
            dataset_2_dict = {i[0]: i[1] for i in word_pairs[1]}

            self.completions_1 = []
            self.completions_2 = []
            for i in range(size):
                x = np.random.choice(x_interesection, n_train+1)
                y_1 = np.array([dataset_1_dict[i] for i in x])
                y_2 = np.array([dataset_2_dict[i] for i in x])

                ys = np.stack([y_1, y_2]).transpose((1,0))
                if ambiguous_draws:
                    ys_chosen = np.array([y[draw] for y, draw in zip(ys, ambiguous_draws)])
                else:
                    ys_chosen = np.array([np.random.choice(i) for i in ys])

                icl_seq = np.stack([x, ys_chosen]).transpose((1,0))

                if response_type == 'multiple_choice':
                    prompt = ''
                    for i in range(n_train + 1):
                        options = np.random.permutation(2)

                        prompt += f"Q: {x[i]} A: ?\n"
                        prompt += f"a {ys[i, options[0]]}\n"
                        prompt += f"b {ys[i, options[1]]}\n"

                        # Find correct option
                        if ys_chosen[i] == ys[i, options[0]]:
                            correct_option = 'a'
                        elif ys_chosen[i] == ys[i, options[1]]:
                            correct_option = 'b'
                        if i == n_train:
                            prompt += "Response: ("
                        else:
                            prompt += f"Response: ({correct_option})\n\n"
                        
                    self.prompts.append(prompt)
                    self.completions_1.append(correct_option)
                    self.completions_2.append('a' if correct_option == 'b' else 'b')
                                            
                else:
                    interleaved_seq = ICLSequence(icl_seq, padded_space=padded_space)
                    self.seqs.append(interleaved_seq)
                    self.prompts.append(interleaved_seq.prompt())
                    self.completions_1.append(' ' + y_1[-1] if padded_space else y_1[-1])
                    self.completions_2.append(' ' + y_2[-1] if padded_space else y_2[-1])   
        else:
            # Load the data
            d_path = os.path.join(data_dir, f'{dataset}.json')
            raw_data = json.load(open(d_path, 'r'))
            self.word_pairs = [[i['input'], i['output']] for i in raw_data]   
            self.word_list = [word for word_pair in self.word_pairs for word in word_pair]
            self.create_json_dataset()
            if response_type == 'multiple_choice':
                self.dataset_name = f"{dataset}-mc"
                
        self.batch_size = batch_size if batch_size else size
        self.num_batches = self.calculate_n_batches(self.batch_size)
    
    def create_seq_dataset(self):
        for n in range(self.size):
            train_examples = []
            for i in range(self.n_train+1):
                abstract_seq = ICLAbstractSequence(self.dataset, self.symbol_list, self.seq_len, self.seed+i+n)
                train_examples.append([abstract_seq.x, abstract_seq.y])
            seq = ICLSequence(train_examples, padded_space=self.padded_space)
            self.seqs.append(seq)
            self.prompts.append(seq.prompt())
            self.completions.append(seq.completion())

    def create_json_dataset(self):
        # Generate the dataset (by choosing random word pairs, and constructing `ICLSequence` objects)
        word_pairs_list = []
        for n in range(self.size):
            np.random.seed(self.seed + n)
            random_pairs = np.random.choice(len(self.word_pairs), self.n_train+1, replace=False)
            # Randomize the order of each word pair (x, y). If not bidirectional, we always have x -> y not y -> x
            random_orders = np.random.choice([1, -1], self.n_train+1)
            if not(self.bidirectional): random_orders[:] = 1
            word_pairs = [self.word_pairs[pair][::order] for pair, order in zip(random_pairs, random_orders)]
            word_pairs_list.append(word_pairs)
        
        if self.shuffle_prompts:
            assert self.shuffle_prompts in ['input', 'output'], "shuffle_prompts must be either 'input' or 'output'"
            word_pairs_list = np.array(word_pairs_list)
            np.random.seed(self.seed_shuffle)
            indices = np.random.permutation(self.size)  # Create a shuffled index for the first dimension
            if self.shuffle_prompts == 'input':
                # Shuffle the first (m-1) elements (input) of the second dimension across the first dimension
                word_pairs_list[:, :-1, :] = word_pairs_list[indices, :-1, :] 
            elif self.shuffle_prompts == 'output':
                # Shuffle the last element (output) of the second dimension across the first dimension
                word_pairs_list[:, -1, :] = word_pairs_list[indices, -1, :]
            word_pairs_list = word_pairs_list.tolist()
        
        for n, word_pairs in enumerate(word_pairs_list):
            # If corrupted, then replace y with a random word in all (x, y) pairs except the last one
            if self.corrupted:
                for i in range(len(word_pairs) - 1):
                    word_pairs[i][1] = np.random.choice(self.word_list)
            if self.response_type == 'open_ended':
                seq = ICLSequence(word_pairs, padded_space=self.padded_space)
            elif self.response_type == 'multiple_choice':
                seq = ICLMultipleChoice(word_pairs, self.word_pairs, seed=self.seed+n, format_prompt=self.mc_format, padded_space=self.padded_space)

            self.seqs.append(seq)
            self.prompts.append(seq.prompt())
            self.completions.append(seq.completion())

    def create_corrupted_dataset(self):
        '''Creates a corrupted version of the dataset (with same random seed).'''
        return ICLDataset(
            self.dataset,
            self.size,
            self.n_train,
            response_type=self.response_type,
            bidirectional=self.bidirectional,
            seed=self.seed,
            corrupted=True,
            shuffle_prompts=self.shuffle_prompts,
            symbols=self.symbols,
            seq_len=self.seq_len,
            tokenizer=self.tokenizer,
            padded_space=self.padded_space,
            batch_size=self.batch_size
        )
    
    def calculate_n_batches(self, batch_size):
        return self.size // batch_size + (0 if self.size % batch_size == 0 else 1)

    def __len__(self):
        return self.calculate_n_batches(self.batch_size)

    def __getitem__(self, idx: int):
        if idx >= len(self.prompts) // self.batch_size:
            raise IndexError("Index out of range")
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.size)
        batch_prompts = self.prompts[start_idx:end_idx]
        batch_completions = self.completions[start_idx:end_idx]
        return batch_prompts, batch_completions
    
    def pretty_print_item(self, item = 0):
        '''Prints a single example from the dataset.'''
        if self.response_type == 'multiple_choice':
            s = f"[b u]{self.dataset_name}[/]\n\n"
            for idx, (x, y, options, correct) in enumerate(zip(self.seqs[item].x, self.seqs[item].y, self.seqs[item].options, self.seqs[item].correct)):
                s += f"### Instruction: Q: {x} A: \n\n"
                s += f"(a) {options[0]}\n(b) {options[1]}\n(c) {options[2]}\n(d) {options[3]}\n\n"
                if idx == len(self.seqs[item]) - 1:
                    s += f"### Response: ([b cyan]{['a', 'b', 'c', 'd'][correct]}[/]\n\n"
                else:
                    s += f"### Response: ([b]{['a', 'b', 'c', 'd'][correct]}[/])\n\n"
        else:
            s = f"[b u]{self.dataset_name}[/]\n\n"
            for idx, (x, y) in enumerate(zip(self.seqs[item].x, self.seqs[item].y)):
                if idx == len(self.seqs[item]) - 1:
                    s += f"Q: {x}\n[b]A[/]: [b cyan]{y}[/]\n"
                else:
                    s += f"Q: {x}\n[b]A: {y}[/]\n\n"
        Console().print(s, highlight=False)

def generate_seed(value, seed=None):
    '''Generate a hash seed for a given value'''
    hash_object = hashlib.md5(value.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    return (hash_int + (seed if seed else 0)) % (2**32 - 1) # Numpy seed must be between 0 and 2**32 - 1

class DatasetConstructor:
    def __init__(
        self, 
        dataset_ids: List[str] | str,
        dataset_size: int,
        n_train: int,
        batch_size: int | List[int] = None,
        tokenizer: Any=None,
        seq_len: int = 20,
        seed: int = 42,
        data_dir: str = DATASET_DIR
    ):
        self.seed = seed
        dataset_ids = dataset_ids if isinstance(dataset_ids, list) else [dataset_ids] # Ensure dataset_ids is a list

        # Create the datasets
        self.datasets = []
        self.prompts = []
        self.completions = []
        self.dataset_ids = []
        for dataset_id in dataset_ids:
            # Determine the type and update dataset_cfg accordingly
            if dataset_id.split('_')[0] in ['predecessor', 'successor']:
                # It's an abstract dataset
                dataset_cfg = {
                    'size': dataset_size,
                    'n_train': n_train,
                    'dataset': dataset_id.split('_')[0],
                    'symbols': dataset_id.split('_')[1],
                    'seq_len': seq_len,
                    'tokenizer': tokenizer,
                }
            elif dataset_id.split('-')[-1].startswith('mc'):
                # It's a multiple choice dataset
                dataset_cfg = {
                    'size': dataset_size,
                    'n_train': n_train,
                    'dataset': dataset_id.split('-')[0],
                    'response_type': 'multiple_choice',
                    'mc_format': 'word' if dataset_id.split('.')[-1] == 'word' else 'letter',
                    'padded_space': False if dataset_id.split('.')[-1] == 'word' else True
                }
            elif dataset_id.split('-')[-1].startswith('oe'):
                # It's an open ended dataset
                dataset_cfg = {
                    'size': dataset_size,
                    'n_train': n_train,
                    'dataset': dataset_id.split('-')[0],
                }
            elif dataset_id in [f.split('.')[0] for f in os.listdir(data_dir)]:
                # It's a JSON dataset
                dataset_cfg = {
                    'size': dataset_size,
                    'n_train': n_train,
                    'dataset': dataset_id,
                }
            else:
                raise ValueError(f"Dataset {dataset_id} not recognized.")
            
            # Create the dataset with a generated seed
            dataset = ICLDataset(
                **dataset_cfg,
                seed=generate_seed(dataset_id, self.seed),
            )
            
            # Append dataset and its associated prompts/completions in order
            self.datasets.append(dataset)
            self.prompts.extend(dataset.prompts)
            self.completions.extend(dataset.completions)
            self.dataset_ids.append(dataset_id)

        # Support batch_size as int or list
        if isinstance(batch_size, int):
            self.batch_sizes = [batch_size] * ((len(self.prompts) + batch_size - 1) // batch_size)
        else:
            assert sum(batch_size) == len(self.prompts), "Sum of batch sizes must equal number of prompts"
            self.batch_sizes = batch_size
        self.cum_batch_sizes = [0]
        for b in self.batch_sizes:
            self.cum_batch_sizes.append(self.cum_batch_sizes[-1] + b)

    def __getitem__(self, idx):
        if idx >= len(self.batch_sizes):
            raise IndexError("Index out of range")
        start_idx = self.cum_batch_sizes[idx]
        end_idx = self.cum_batch_sizes[idx + 1]
        return self.prompts[start_idx:end_idx], self.completions[start_idx:end_idx]

    def __len__(self):
        return len(self.batch_sizes)