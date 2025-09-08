import os
import json
import pickle
import torch
import pandas as pd
from collections import defaultdict
from nnsight import LanguageModel
from .globals import RESULTS_DIR

def heads_to_dict(list_of_tuples):
    head_dict = defaultdict(set)
    for layer, head in list_of_tuples:
        head_dict[layer].add(head)
    head_dict = dict(head_dict)
    return head_dict

class ExtendedLanguageModel:
    def __init__(
        self, 
        model_name: str, 
        metrics_path: str = os.path.join(RESULTS_DIR, 'LUMI'),
        remote_run: bool = None
    ):
        self.remote_run = remote_run if remote_run is not None else self.auto_set_remote_run()
        self.name = model_name
        self.nickname = model_name.lower().split('/')[1]
        self.metrics_path = metrics_path
        self.lm = self.load_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Set model configuration based on the model type
        self.set_model_config()

    @property
    def metrics(self):
        if not hasattr(self, '_metrics'):
            self._load_metrics()
        return self._metrics
    
    def _load_metrics(self):
        """Load CIE and RSA metrics"""
        # Load CIE metrics for multiple choice and open-ended tasks
        cie_mc_path = os.path.join(self.metrics_path, 'CIE_mc', f'cie_{self.nickname}.pkl')
        cie_oe_path = os.path.join(self.metrics_path, 'CIE_oe', f'cie_{self.nickname}.pkl')
        
        with open(cie_mc_path, 'rb') as f:
            cie_mc = pickle.load(f)
        cie_mc = torch.stack(cie_mc).to(torch.float32)
        
        with open(cie_oe_path, 'rb') as f:
            cie_oe = pickle.load(f)
        cie_oe = torch.stack(cie_oe).to(torch.float32)
        
        # Concatenate CIE metrics
        cie = torch.concat([cie_oe, cie_mc])

        # Load RSA metrics
        rsa_path = os.path.join(self.metrics_path, 'RSA', self.nickname, 'rsa.csv')
        df = pd.read_csv(rsa_path)
        df.rename(columns={'rsa': 'RSA'}, inplace=True)
        
        # Add CIE metrics to dataframe            
        df['CIE'] = cie.mean(dim=0).flatten()
        df['CIE_eng'] = cie_oe[::2].mean(dim=0).flatten()
        df['CIE_fr'] = cie_oe[1::2].mean(dim=0).flatten()
        df['CIE_mc'] = cie_mc.mean(dim=0).flatten()
        
        self._metrics = df

    def __str__(self) -> str:
        return self.name
    
    def auto_set_remote_run(self):
        if any(env in os.environ for env in ['SLURM_JOB_ID', 'SLURM_CLUSTER_NAME']):
            return False
        else:
            return True
    
    def load_model(self):
        if self.remote_run:
            return LanguageModel(self.name)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
            config = AutoConfig.from_pretrained(self.name)

            tokenizer = AutoTokenizer.from_pretrained(self.name, padding_side='left', config=config)
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(self.name, device_map='auto', torch_dtype='auto')
            model.eval()
            self.hf_model = model
            return LanguageModel(model, tokenizer=tokenizer, config=config)
            
    def set_model_config(self):
        if 'llama-3.1' in self.nickname:
            self.config = {
                "n_heads": self.lm.config.num_attention_heads,
                "n_layers": self.lm.config.num_hidden_layers,
                "resid_dim": self.lm.config.hidden_size,
                "d_head": self.lm.config.hidden_size // self.lm.config.num_attention_heads,
                "hidden_layer": lambda layer: self.lm.model.layers[layer],
                "out_proj": lambda layer: self.lm.model.layers[layer].self_attn.o_proj,
                "get_first_token_ids": lambda token_list: [
                    toks[1] for toks in self.lm.tokenizer(token_list)["input_ids"]
                ],
                "get_first_token": lambda string: self.lm.tokenizer.tokenize(string)[0].replace('Ġ', ' ')
            }
        if 'qwen' in self.nickname:
            self.config = {
                "n_heads": self.lm.config.num_attention_heads,
                "n_layers": self.lm.config.num_hidden_layers,
                "resid_dim": self.lm.config.hidden_size,
                "d_head": self.lm.config.hidden_size // self.lm.config.num_attention_heads,
                "hidden_layer": lambda layer: self.lm.model.layers[layer],
                "out_proj": lambda layer: self.lm.model.layers[layer].self_attn.o_proj,
                "get_first_token_ids": lambda token_list: [
                    toks[0] for toks in self.lm.tokenizer(token_list)["input_ids"]
                ],
                "get_first_token": lambda string: self.lm.tokenizer.tokenize(string)[0].replace('Ġ', ' ')
            }

    def get_top_heads(
            self, 
            metric: str, # 'RSA', 'CIE', 'CIE_eng', 'CIE_fr', 'CIE_mc'
            n: int = 5,
            to_dict: bool = False
        ):
        heads = self.metrics.sort_values(by=metric, ascending=False).head(n)[['layer', 'head']].values.tolist()
        if to_dict:
            return heads_to_dict(heads)
        else:
            return heads