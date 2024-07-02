import pandas as pd
import random
from textwrap import dedent
from typing import List, Tuple

DATA_DIR = 'data/analogies'

def load_analogies_data(file_path:str) -> List[list]:
    '''
    Load the analogies CSV data from the given file path and return it as a list of lists

    Args:
        file_path (str): The file path to the analogies data CSV file (with or without the .csv extension)

    Returns:
        list: A list of lists where each inner list contains the data for a single analogy [item_idx, A, B, C, D, Ans1, Ans2, Ans3, Ans4]
    '''
    if not file_path.endswith('.csv'):
        file_path = f'{file_path}.csv'

    analogies_df = pd.read_csv(f'{DATA_DIR}/{file_path}')
    columns = ['A', 'B', 'C', 'D', 'Ans1', 'Ans2', 'Ans3', 'Ans4']
    analogies_data = analogies_df[columns].reset_index().values.tolist()
    return analogies_data

class AnalogiesDataLoader:
    '''
    A data loader class for loading multiple choice analogy data from a CSV file

    Args:
        file_path (str): The file path to the analogies data CSV file (with or without the .csv extension)
        prompt_type (str): The type of prompt to generate (multiple choice = 'mc', open ended = 'oe')
        batch_size (int): The batch size to use for loading the data
        example_prompt (bool): Whether to use example prompts for the analogies
        n_items (int): The number of items to run
    '''
    def __init__(self, file_path, prompt_type, batch_size=None, example_prompt=True, n_items=None, random_seed=0):
        data = load_analogies_data(file_path)
        if n_items:
            random.seed(random_seed)
            data = random.sample(data, n_items)
        self.data = data
        self.prompt_type = prompt_type
        self.batch_size = batch_size if batch_size else len(self.data)
        self.num_batches = self.calculate_n_batches(self.batch_size)
        self.example_prompt = example_prompt

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise IndexError("Batch index out of range")
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.data))
        batch_data = self.data[start_idx:end_idx]
        
        if self.prompt_type == 'mc':
            return self.get_mc_item(batch_data)
        elif self.prompt_type == 'oe':
            return self.get_open_ended_item(batch_data)
    
    def get_mc_item(self, batch_data):
        batch_prompts = []
        batch_mc_to_option = []
        indices = []
        for item in batch_data:
            prompt, mc_to_option = self.get_item_MC(*item, example_prompt=self.example_prompt)
            batch_prompts.append(prompt)
            batch_mc_to_option.append(mc_to_option)
            indices.append(item[0])
            
        return batch_prompts, batch_mc_to_option, indices
    
    def get_open_ended_item(self, batch_data):
        batch_prompts = []
        Ds = []
        indices = []

        for item in batch_data:
            prompt = self.get_item_open_ended(*item[1:4], example_prompt=self.example_prompt)
            batch_prompts.append(prompt)
            Ds.append(item[4])
            indices.append(item[0])

        return batch_prompts, Ds, indices
    
    def calculate_n_batches(self, batch_size):
        return len(self.data) // batch_size + (0 if len(self.data) % batch_size == 0 else 1)
    
    def set_data(self, data):
        self.data = data
        self.num_batches = len(self.data) // self.batch_size + (0 if len(self.data) % self.batch_size == 0 else 1)
    
    @staticmethod
    def get_item_MC(item_idx, A, B, C, D, Ans1, Ans2, Ans3, Ans4, example_prompt=True) -> Tuple[str, dict]:
        '''
        Generate a multiple choice question prompt for the given analogy item

        Args:
            item_idx (int): The index of the analogy item
            A, B, C: The terms in the analogy 
            D, Ans1, Ans2, Ans3, Ans4: The multiple choice options for the analogy, where D is the correct answer
        
        Returns:
            str: The formatted multiple choice question prompt
            dict: A mapping of the multiple choice letter options to the original response options
        '''
        options = list({'D':D, 'Ans1':Ans1, 'Ans2':Ans2, 'Ans3':Ans3, 'Ans4':Ans4}.items())

        # Shuffle the options based on the seed from item_idx
        random.seed(item_idx)
        random.shuffle(options)
        options = dict(options)
        mc_options = list(options.values())
        
        prompt = f'''
        ### Instruction: {A} is to {B}, as {C} is to
        (a) {mc_options[0]}
        (b) {mc_options[1]}
        (c) {mc_options[2]}
        (d) {mc_options[3]}
        (e) {mc_options[4]}
        ''' 

        example = '''
        ### Instruction: man is to king as woman is to
        (a) girl
        (b) child
        (c) queen
        (d) cat
        (e) crown
        ### Response: (c)
        '''

        if example_prompt:
            prompt = dedent(example) + ' ' + dedent(prompt.lstrip('\n')) + '### Response: ('
        else:
            prompt = dedent(prompt.lstrip('\n')) + '### Response: ('

        # Remove first newline and initial indentations
        prompt = dedent(prompt.lstrip('\n'))

        # Map MC letter options back to their original response options -- {abcde: D Ans1 Ans2 etc}
        mc_to_option = {['a', 'b', 'c', 'd', 'e'][idx]:key for idx, key in enumerate(options.keys())}

        return prompt, mc_to_option
    
    @staticmethod
    def get_item_open_ended(A, B, C, example_prompt=True) -> str:
        '''
        Generate an open-ended question prompt for the given analogy item

        Args:
            A, B, C: The terms in the analogy 
        
        Returns:
            str: The formatted open-ended question prompt
            str: The correct answer for the analogy
        '''
        prompt = f'{A} is to {B}, as {C} is to '

        example = '\n### Instruction: man is to king as woman is to queen'
        if example_prompt:
            prompt = example + '\n### Instruction: ' + prompt
            
        return prompt

        

