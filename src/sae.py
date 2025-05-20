from sae_lens import SAE
import nnsight
import torch
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from rich.console import Console
import pickle

from utils.model_utils import ExtendedLanguageModel
from utils.ICL_utils import ICLDataset, DatasetConstructor
from utils.query_utils import get_avg_att_output, get_att_output_per_item, compute_similarity_matrix
from utils.eval_utils import SimilarityMatrix, accuracy_completions, InterventionResults
from utils.globals import RESULTS_DIR

device = 'mps'

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="llama_scope_lxr_8x",
    sae_id="l15r_8x",
    device=device,
)

T = -1
n_heads = 5
model_name = 'meta-llama/Meta-Llama-3.1-8B'
low_level_cie_path = os.path.join(RESULTS_DIR, 'CIE_LowLevel', f'Meta-Llama-3.1-8B.csv')
model = ExtendedLanguageModel(model_name, rsa_heads_n=n_heads, fv_heads_n=n_heads, cie_path=low_level_cie_path)
cv_heads = model.get_rsa_heads(task_attribute='relation_verbal')
fv_heads = model.get_fv_heads()


feature_acts = sae.encode(torch.randn(1, model.config['resid_dim']).to(device))
sae_out = sae.decode(feature_acts)


dataset_antonym = DatasetConstructor(
    dataset_ids='antonym_eng', 
    dataset_size=100, 
    n_train=5,
    batch_size=100, 
    seed=42
)
dataset_antonym.prompts

with model.lm.session(remote=True) as sess:
    # Get the FV head outputs
    with model.lm.trace(dataset_antonym.prompts) as t:
        head_outputs = [] 
        for layer, head_list in fv_heads.items():
            out_proj = model.config['out_proj'](layer)
            out_proj_output = get_att_output_per_item(model, out_proj, head_list, token=T)
            head_outputs.append(out_proj_output)
    fvs = torch.stack(head_outputs).sum(dim=0).save()

    # Get the CV head outputs
    with model.lm.trace(dataset_antonym.prompts) as t:
        head_outputs = []
        for layer, head_list in cv_heads.items():
            out_proj = model.config['out_proj'](layer)
            head_outputs.append(get_att_output_per_item(model, out_proj, head_list, token=T))
    cvs = torch.stack(head_outputs).sum(dim=0).save()



sae.encode(fvs.mean(dim=0).to(device)).topk(5)
sae.encode(cvs.mean(dim=0).to(device)).topk(5)
