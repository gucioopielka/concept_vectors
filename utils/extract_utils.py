from typing import List
import torch
from utils.model_utils import ExtendedLanguageModel

def get_relation_vecs(model: ExtendedLanguageModel, prompts: List[str]) -> torch.Tensor:
    '''
    Get the relation vector per item by summing the output of the most influential attention heads for the 
    model's output tokens given a prompt or list of prompts 

    Args:
        model (ExtendedLanguageModel): The model object
        prompts (List[str]): The prompt or list of prompts to run the model on

    Returns:
        torch.Tensor: The relation vectors for the model's output tokens (shape: [batch_size, resid_dim])
    '''
    D_MODEL = model.config['resid_dim']
    N_HEADS = model.config['n_heads']
    D_HEAD = D_MODEL // N_HEADS # dimension of each head
    B = len(prompts)
    T = -1  # values taken from last token
    head_dict = model.top_heads # get the top heads for each layer

    relation_vec_list = []
    with model.trace(remote=True) as runner:
        with runner.invoke(prompts) as invoker:
            for layer, head_list in head_dict.items():
        
                # Get the projection layer output
                out_proj = model.config['out_proj'](layer)

                # Output projection input
                z = out_proj.input[0][0][:, T]

                # Reshape output projection input into heads
                z_ablated = z.view(B, N_HEADS, D_HEAD).clone()
                
                # Zero-ablate all heads which aren't in our list of infuential heads
                heads_to_ablate = list(set(range(N_HEADS)) - head_list)
                z_ablated[:, heads_to_ablate, :] = 0.0

                # Concatanate the heads back into the residual dimension
                z_ablated = z_ablated.view(B, -1) 

                # Get the projection (if more than one head is in the list of heads to keep, 
                # the output will be the sum of those heads)
                out_proj_output = out_proj(z_ablated)

                relation_vec_list.append(out_proj_output.save())

    # Sum all the attention heads per item
    relation_vecs = torch.sum(torch.stack(relation_vec_list),dim=0)
    return relation_vecs.numpy()


