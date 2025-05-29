from typing import *

import torch

from utils.query_utils import compute_similarity_matrix
from utils.ICL_utils import DatasetConstructor
from utils.model_utils import ExtendedLanguageModel
import nnsight

def get_att_out_proj_input(
    model: ExtendedLanguageModel,
    layer: int,
    token: int = -1,
) -> torch.Tensor:
    out_proj = model.config['out_proj'](layer).inputs[0][0][:, token]
    return out_proj.view(out_proj.shape[0], model.config['n_heads'], model.config['d_head'])

model_id = 'meta-llama/Meta-Llama-3.1-8B'
layer = 13

# Load the model
model = ExtendedLanguageModel(model_id, remote_run=False)
n_heads = model.config['n_heads']
n_layers = model.config['n_layers']
  
# Create the dataset
dataset_constructor = DatasetConstructor(
    dataset_ids=['antonym_eng', 'categorical_eng'],
    dataset_size=10, 
    n_train=3,
    seed=42
)
prompts = dataset_constructor.prompts

with model.lm.session(remote=model.remote_run) as sess:

    sess.log(f"Tracing model ...")
    att_hidden = nnsight.list().save()
    # Collect the hidden states for each head
    with model.lm.trace(prompts) as t:
        att_out = get_att_out_proj_input(model, layer, token=-1)
        for head in range(n_heads):
            act = att_out[:, head]
            act = act if model.remote_run else act.cpu()
            att_hidden.append(act)

    sess.log(f"Computing similarity matrices ...")
    simmats = nnsight.list().save()
    for act in att_hidden:
        v = act.to(model.device)
        simmats.append(compute_similarity_matrix(v).cpu())

torch.save(att_hidden, f'att_hidden_{model.nickname}_{layer}.pt')
torch.save(simmats, f'simmats_{model.nickname}_{layer}.pt')