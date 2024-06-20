import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

def get_RDM(mat):
    mat_flattened = np.reshape(mat, (mat.shape[0], -1))
    return pairwise_distances(mat_flattened, metric='cosine')

def get_unique_indices(s):
    s.reset_index(drop=True, inplace=True)

    result_dict = {}
    for value in s.unique():
        # Find the first and last index of the current value
        first_idx = s[s == value].index[0]
        last_idx = s[s == value].index[-1]
        
        # Store the result in the dictionary
        result_dict[value] = (first_idx, last_idx)
    
    return result_dict

def get_rule_sim_diagonal(rdm, rel_idx, sort=True):
    rule_similarity = []
    for relation, indices in rel_idx.items():
        rule_similarity.append(np.mean(rdm[indices[0]:indices[1], indices[0]:indices[1]]))
    
    sim_df = pd.DataFrame({'Concept' : rel_idx.keys(), 'Similarity' : rule_similarity})
    if sort:
        return sim_df.sort_values(by='Similarity', ascending=False)
    else:
        return sim_df
