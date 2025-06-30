import os
import json
import pickle
import torch
import pandas as pd
from collections import defaultdict
from nnsight import LanguageModel
from .globals import RESULTS_DIR


class ExtendedLanguageModel:
    def __init__(
        self, 
        model_name: str, 
        load_metrics: bool = False,
        metrics_path: str = None,
        remote_run: bool = None
    ):
        self.remote_run = remote_run if remote_run is not None else self.auto_set_remote_run()
        self.name = model_name#.lower()
        self.nickname = model_name.lower().split('/')[1]
        self.lm = self.load_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Set model configuration based on the model type
        self.set_model_config()

        # Construct file paths for head data
        if load_metrics:
            if metrics_path is not None:
                self.metrics_path = metrics_path 
            else:
                self.metrics_path = os.path.join(RESULTS_DIR, 'LUMI', 'RSA', self.nickname, f'rsa.csv')

            cie_mc = pickle.load(open(os.path.join(self.metrics_path, f'cie_{self.nickname}.pkl'), 'rb'))
            cie_mc = torch.stack(cie_mc).to(torch.float32)
            cie_oe = pickle.load(open(os.path.join(self.metrics_path, f'cie_{self.nickname}.pkl'), 'rb'))
            cie_oe = torch.stack(cie_oe).to(torch.float32)
            cie = torch.concat([cie_oe, cie_mc])

            df = pd.read_csv(os.path.join(self.metrics_path, f'rsa.csv'))
            df.rename(columns={'rsa': 'RSA'}, inplace=True)
            df['CIE'] = cie.mean(dim=0).flatten()
            df['CIE_eng'] = cie_oe[::2].mean(dim=0).flatten()
            df['CIE_fr'] = cie_oe[1::2].mean(dim=0).flatten()
            df['CIE_mc'] = cie_mc.mean(dim=0).flatten()
            self.metrics = df

    def __str__(self) -> str:
        return self.name
    
    def auto_set_remote_run(self):
        if any(env in os.environ for env in ['SLURM_JOB_ID', 'SLURM_CLUSTER_NAME']):
            return True
        else:
            return False
    
    def load_model(self):
        if self.remote_run:
            return LanguageModel(self.name)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

            tokenizer = AutoTokenizer.from_pretrained(self.name, padding_side='left')
            if getattr(tokenizer, "pad_token", None) is None:
                # Set pad token to eos token if it is not set
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(self.name, device_map='auto', torch_dtype='auto')
            model.eval()

            config = AutoConfig.from_pretrained(self.name)
            return LanguageModel(model, tokenizer=tokenizer, config=config)
            
    def set_model_config(self):
        if 'llama-3.1' in self.nickname:
            self.config = {
                "n_heads": self.lm.config.num_attention_heads,
                "n_layers": self.lm.config.num_hidden_layers,
                "resid_dim": self.lm.config.hidden_size,
                "hidden_layer": lambda layer: self.lm.model.layers[layer],
                "d_head": self.lm.config.hidden_size // self.lm.config.num_attention_heads,
                "out_proj": lambda layer: self.lm.model.layers[layer].self_attn.o_proj,
                "get_first_token_ids": lambda token_list: [
                    toks[1] for toks in self.lm.tokenizer(token_list)["input_ids"]
                ],
                "get_first_token": lambda string: self.lm.tokenizer.tokenize(string)[0].replace('Ġ', ' ')
            }

    def get_fv_heads(self, n=None):
        if n is None:
            n = self.fv_heads_n

        # Load the CSV file into a DataFrame
        df = pd.read_csv(self.cie_path)

        # Ensure the 'layer' and 'head' columns are of integer type
        df['layer'] = df['Layer'].astype(int)
        df['head'] = df['Head'].astype(int)

        # Create a list of (layer, head) tuples
        fv_heads = list(zip(df['layer'], df['head']))

        # Set the default number of heads if not provided
        if self.fv_heads_n is None:
            if self.name == 'meta-llama/meta-llama-3.1-8b':
                n = 20
            elif self.name == 'meta-llama/meta-llama-3.1-70b':
                n = 100

        # Turn head_list into a dict of {layer: heads we need in this layer}
        head_dict = defaultdict(set)
        for layer, head in fv_heads[:n]:
            head_dict[layer].add(head)
        head_dict = dict(head_dict)

        return head_dict

    def get_rsa_heads(self, task_attribute='relation_verbal', n=None, quantile=None):
        if n is None:
            n = self.rsa_heads_n

        rsa_df = pd.read_csv(self.rsa_path)

        # Sort by the 'relation_verbal' column in descending order
        rsa_df = rsa_df.sort_values(by=task_attribute, ascending=False)

        if quantile is not None:
            n = (rsa_df[task_attribute] > rsa_df[task_attribute].quantile(quantile)).sum()

        # Create a list of (layer, head) tuples
        rsa_heads = list(zip(rsa_df['layer'], rsa_df['head']))

        # Set the default number of heads if not provided
        if self.rsa_heads_n is None:
            if self.name == 'meta-llama/meta-llama-3.1-8b':
                n = 3
            elif self.name == 'meta-llama/meta-llama-3.1-70b':
                n = 3
        
        # Turn head_list into a dict of {layer: heads we need in this layer}
        head_dict = defaultdict(set)
        for layer, head in rsa_heads[:n]:
            head_dict[layer].add(head)
        head_dict = dict(head_dict)

        return head_dict

    def get_heads(self, df: pd.DataFrame, n: int = None, quantile: float = None):
        '''
        Get the heads from a CSV file.
        Args:
            df (pd.DataFrame): The DataFrame to get the heads from.
            The first column should be the layer, the second column should be the head, and the third column should be the value.
            n (int): The number of heads to get.
            quantile (float): The quantile to use to get the heads.
        Returns:
            dict: A dictionary of {layer: heads}
        '''
        df = df.rename(columns={df.columns[0]: 'layer', df.columns[1]: 'head', df.columns[2]: 'value'})

        if quantile is not None and n is None:
            n = (df['value'] > df['value'].quantile(quantile)).sum()
            print(f'Using {n} heads')
        
        if n is not None:
            df = df.sort_values(by='value', ascending=False)
            df = df.head(n)

        head_dict = defaultdict(set)
        for layer, head in zip(df['layer'], df['head']):
            head_dict[layer].add(head)
        head_dict = dict(head_dict)

        return head_dict
