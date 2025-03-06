import os
import json
import pandas as pd
from nnsight import LanguageModel
from .globals import RESULTS_DIR


class ExtendedLanguageModel:
    def __init__(
        self, model_name: str, 
        cv_heads_n: int = None,
        fv_heads_n: int = None 
    ):
        self.lm = LanguageModel(model_name)
        self.name = model_name.lower()
        self.nickname = model_name.split('/')[1]
        self.fv_heads_n = fv_heads_n
        self.cv_heads_n = cv_heads_n

        # Set model configuration based on the model type
        self.set_model_config()

        # Construct file paths for head data
        self.cie_path = os.path.join(RESULTS_DIR, 'CIE', f'{self.nickname}.csv')
        self.rsa_path = os.path.join(RESULTS_DIR, 'RSA', f'rsa_{self.nickname}.csv')

        # Load fv heads if available
        if os.path.exists(self.cie_path):
            self.set_fv_heads()
        else:
            self.fv_heads = None

        # Load cv heads if available
        if os.path.exists(self.rsa_path):
            self.set_cv_heads()
        else:
            self.cv_heads = None

        # Load best layer to intervene with if available
        self.int_layer_path = os.path.join(RESULTS_DIR, 'intervention', f"{self.nickname}_intervene_layers.json")
        if os.path.exists(self.int_layer_path):
            layers = json.load(open(self.int_layer_path, 'r'))
            self.cv_intervention_layer = int(layers.index(max(layers)))

    def __str__(self) -> str:
        return self.name

    def set_model_config(self):
        if 'llama-3.1' in self.name:
            self.config = {
                "n_heads": self.lm.config.num_attention_heads,
                "n_layers": self.lm.config.num_hidden_layers,
                "resid_dim": self.lm.config.hidden_size,
                "hidden_layer": lambda layer: self.lm.model.layers[layer],
                "out_proj": lambda layer: self.lm.model.layers[layer].self_attn.o_proj,
                "get_first_token_ids": lambda token_list: [
                    toks[1] for toks in self.lm.tokenizer(token_list)["input_ids"]
                ],
                "get_first_token": lambda string: self.lm.tokenizer.tokenize(string)[0].replace('Ġ', ' ')
            }

    def set_fv_heads(self):
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
                self.fv_heads_n = 20
            elif self.name == 'meta-llama/meta-llama-3.1-70b':
                self.fv_heads_n = 100

        self.fv_heads = fv_heads[:self.fv_heads_n]

    def set_cv_heads(self):
        # Load the CSV file into a DataFrame
        rv_df = pd.read_csv(self.rsa_path)

        # Sort by the 'relation_verbal' column in descending order
        rv_df = rv_df.sort_values(by='relation_verbal', ascending=False)

        # Create a list of (layer, head) tuples
        cv_heads = list(zip(rv_df['layer'], rv_df['head']))

        # Set the default number of heads if not provided
        if self.cv_heads_n is None:
            if self.name == 'meta-llama/meta-llama-3.1-8b':
                self.cv_heads_n = 3
            elif self.name == 'meta-llama/meta-llama-3.1-70b':
                self.cv_heads_n = 3

        self.cv_heads = cv_heads[:self.cv_heads_n]