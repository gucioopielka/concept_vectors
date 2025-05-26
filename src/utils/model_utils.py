import os
import json
import pandas as pd
from collections import defaultdict
from nnsight import LanguageModel
from .globals import RESULTS_DIR


class ExtendedLanguageModel:
    def __init__(
        self, model_name: str, 
        rsa_heads_n: int = None,
        fv_heads_n: int = None,
        cie_path: str = None,
        rsa_path: str = None,
        remote_run: bool = False
    ):
        self.remote_run = remote_run
        self.name = model_name#.lower()
        self.nickname = model_name.lower().split('/')[1]
        self.lm = self.load_model()
        self.fv_heads_n = fv_heads_n
        self.rsa_heads_n = rsa_heads_n

        # Set model configuration based on the model type
        self.set_model_config()

        # Construct file paths for head data
        if cie_path is None:
            self.cie_path = os.path.join(RESULTS_DIR, 'CIE_LowLevel', f'{self.nickname}.csv')
        else:
            self.cie_path = cie_path

        if rsa_path is None:
            self.rsa_path = os.path.join(RESULTS_DIR, 'RSA', f'rsa_{self.nickname}.csv')
        else:
            self.rsa_path = rsa_path

        # # Load fv heads if available
        # if os.path.exists(self.cie_path):
        #     self.get_fv_heads()

        # # Load cv heads if available
        # if os.path.exists(self.rsa_path):
        #     self.cv_heads = self.get_rsa_heads(task_attribute='relation_verbal')
        #     self.info_source = self.get_rsa_heads(task_attribute='task_type')
        #     self.response_type = self.get_rsa_heads(task_attribute='response_type')
        #     self.language = self.get_rsa_heads(task_attribute='language')
        #     self.q_type = self.get_rsa_heads(task_attribute='prompt_format')

        # Load best layer to intervene with if available
        self.int_layer_path = os.path.join(RESULTS_DIR, 'intervention', f"{self.nickname}_intervene_layers.json")
        if os.path.exists(self.int_layer_path):
            layers = json.load(open(self.int_layer_path, 'r'))
            self.cv_intervention_layer = int(layers.index(max(layers)))

    def __str__(self) -> str:
        return self.name
    
    def load_model(self):
        if self.remote_run:
            return LanguageModel(self.name)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

            tokenizer = AutoTokenizer.from_pretrained(self.name)
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