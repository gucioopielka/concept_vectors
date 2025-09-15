from typing import *
import copy

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import label
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from transformers import AutoTokenizer
import nnsight
from nnsight.intervention.graph.proxy import InterventionProxy

from .model_utils import ExtendedLanguageModel

class InterventionResults:
    def __init__(self):
        self.completion_ids = nnsight.list().save()
        self.completion_probs = nnsight.list().save()
        self.y_probs_1 = nnsight.list().save()
        self.y_probs_2 = nnsight.list().save()
        self.top_delta_ids = nnsight.list().save()
        self.top_delta_probs = nnsight.list().save()
        self.bottom_delta_ids = nnsight.list().save()
        self.bottom_delta_probs = nnsight.list().save()
        self.y_probs_1_fr = nnsight.list().save()
        self.y_probs_1_es = nnsight.list().save()

    def save(self):
        for key, value in self.__dict__.items():
            if isinstance(value, InterventionProxy):
                self.__dict__[key] = value.value


def spearman_rho_torch(x, y):
    """
    Compute Spearman's rank correlation coefficient between two 1D tensors.
    
    Args:
        x (torch.Tensor): A 1D tensor of values.
        y (torch.Tensor): A 1D tensor of values.
        
    Returns:
        torch.Tensor: A scalar tensor containing the Spearman correlation coefficient.
    """
    # Compute ranks using the double argsort method.
    # The first argsort returns indices that would sort the tensor,
    # the second argsort gives the rank of each element.
    x_rank = torch.argsort(torch.argsort(x)).float()
    y_rank = torch.argsort(torch.argsort(y)).float()

    # Compute the mean of the ranks
    x_mean = x_rank.mean()
    y_mean = y_rank.mean()

    # Compute covariance between the ranks
    cov = ((x_rank - x_mean) * (y_rank - y_mean)).mean()
    
    # Compute the standard deviations (using population std, unbiased=False)
    std_x = torch.sqrt(((x_rank - x_mean) ** 2).mean())
    std_y = torch.sqrt(((y_rank - y_mean) ** 2).mean())

    # Spearman correlation is the Pearson correlation of the ranks
    return (cov / (std_x * std_y)).item()

def create_design_matrix(info_list: List[Any]):
    n_items = len(info_list)
    m = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(i+1, n_items):
            m[i, j] = int(info_list[i] == info_list[j])
            m[j, i] = int(info_list[i] == info_list[j])
    np.fill_diagonal(m, 1)
    return m

def rsa(X: np.ndarray, Y: np.ndarray) -> float:
    X = squareform(X, checks=False) if X.ndim == 2 else X
    Y = squareform(Y, checks=False) if Y.ndim == 2 else Y
    return spearmanr(X, Y).correlation

def get_unique_indices(s: pd.Series) -> Dict:
    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    s.reset_index(drop=True, inplace=True)

    result_dict = {}
    for value in s.unique():
        # Find the first and last index of the current value
        first_idx = s[s == value].index[0]
        last_idx = s[s == value].index[-1]
        
        # Store the result in the dictionary
        result_dict[value] = (first_idx, last_idx)
    
    return result_dict
    
def within_task_similarity(rdm: np.ndarray, rel_idx: Dict[str, Tuple[int, int]], metric: str = 'mean') -> Dict[str, float]:
    within_task_similarities = {}
    for task, (start_idx, end_idx) in rel_idx.items():
        task_rdm = rdm[start_idx:end_idx+1, start_idx:end_idx+1]
        lower_triangle = np.tril(task_rdm, -1)  # Only take elements below the diagonal
        lower_triangle_nonzero = lower_triangle[lower_triangle != 0]
        within_task_similarities[task] = eval(f'np.{metric}(lower_triangle_nonzero)')  # Ignore zeros which represent the upper triangle

    return within_task_similarities

def between_task_similarity(rdm: np.ndarray, rel_idx: Dict[str, Tuple[int, int]], metric: str = 'mean') -> Dict[Tuple[str, str], float]:
    between_task_similarities = {}
    for i, (task1, (start1, end1)) in enumerate(rel_idx.items()):
        for j, (task2, (start2, end2)) in enumerate(rel_idx.items()):
            if i > j:  # This ensures that we only consider each pair once
                between_rdm = rdm[start1:end1+1, start2:end2+1]
                between_rdm_nonzero = between_rdm[between_rdm != 0]
                between_task_similarities[(task1, task2)] = eval(f'np.{metric}(between_rdm_nonzero)')

    return between_task_similarities

def accuracy_completions(
    model: ExtendedLanguageModel,
    completions: List[str],
    Ys: List[str],
    return_correct: bool = False
) -> Union[float, Tuple[float, List[bool]]]:
    correct = []
    for completion, y in zip(completions, Ys):
        y_first_token = model.config['get_first_token'](y)
        correct.append(completion == y_first_token)
    accuracy = np.mean(correct).item()
    return (accuracy, correct) if return_correct else accuracy
    
def get_bounds(design_matrix) -> List[Tuple[int, int]]:
    labeled, num_features = label(design_matrix)
    # For each connected component, compute its bounding box.
    # The bounding box is given by the top-left and bottom-right indices.
    rectangles = []
    for region in range(1, num_features + 1):
        # Get the indices (row, col) where the region is located.
        coords = np.argwhere(labeled == region)
        # The top-left corner is the minimum row and col;
        # the bottom-right corner is the maximum row and col.
        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)
        rectangles.append((tuple(top_left), tuple(bottom_right)))
    return rectangles

class SimilarityMatrix:
    def __init__(self, 
        sim_mat: np.ndarray = None,
        tasks: List[str]|np.ndarray = None,
        attribute_list: List[str] = None
    ):
        '''
        Args:
            sim_mat (np.ndarray): The similarity matrix
            tasks (List[str]): The tasks (must be equal to the number of rows in the similarity matrix or divisible by the number of rows)
            attribute_list (List[str]): The attributes of the tasks (e.g. the concepts that represent the tasks). The number of attributes must be equal to the number of rows in the similarity matrix, or the number of rows must be divisible by the number of attributes.
        '''
        self.matrix = self.vec_to_simmat(sim_mat) if sim_mat.ndim == 1 else sim_mat
        self.sim_mode = 'Similarity'
        self.design_matrix = None
        
        if tasks is not None:
            # Repeat the task names if the data is divisible by the number of tasks
            if self.matrix.shape[0] % len(tasks) == 0:
                self.tasks_list = np.repeat(tasks, self.matrix.shape[0] // len(tasks))
            elif self.matrix.shape[0] == len(tasks):
                self.tasks_list = np.array(tasks)
            else:
                raise ValueError('The number of tasks must be equal to the number of rows in the similarity matrix, or the number of rows must be divisible by the number of tasks.')
            
            # Create a dictionary of unique indices for each task
            self.tasks_idx = get_unique_indices(self.tasks_list)
    
        if attribute_list is not None:
            # Repeat the task names if the data is divisible by the number of tasks
            if self.matrix.shape[0] % len(attribute_list) == 0:
                self.attribute_list = np.repeat(attribute_list, self.matrix.shape[0] // len(attribute_list))
            elif self.matrix.shape[0] == len(attribute_list):
                self.attribute_list = np.array(attribute_list)
            else:
                raise ValueError('The number of attributes must be equal to the number of rows in the similarity matrix, or the number of rows must be divisible by the number of attributes.')
            
            self.attribute_idx = get_unique_indices(self.attribute_list)            
            self.design_matrix = create_design_matrix(self.attribute_list)

    def get_within_task_sims(self, metric: str = 'mean'): return within_task_similarity(self.matrix, self.tasks_idx, metric)
    def get_between_task_sims(self, metric: str = 'mean'): return between_task_similarity(self.matrix, self.tasks_idx, metric)

    def get_within_attribute_sims(self, metric: str = 'mean'): return within_task_similarity(self.matrix, self.attribute_idx, metric)
    def get_between_attribute_sims(self, metric: str = 'mean'): return between_task_similarity(self.matrix, self.attribute_idx, metric)

    @staticmethod
    def vec_to_simmat(matrix: np.ndarray | torch.Tensor) -> np.ndarray:
        matrix = squareform(matrix, checks=False)
        np.fill_diagonal(matrix, 1)  # Set diagonal to 1
        return matrix

    def toggle_similarity(self):
        '''
        Toggle between the similarity matrix and the dissimilarity matrix
        '''
        self.matrix = 1 - self.matrix
        self.sim_mode = 'Similarity' if self.sim_mode == 'Dissimilarity' else 'Dissimilarity'

    def create_design_matrix(self):
        '''
        Create a design matrix for the RDM with 0s for the same task and 1s for different tasks
        '''
        design_matrix = np.ones((self.matrix.shape[0], self.matrix.shape[1]))
        for task, (start, end) in self.tasks_idx.items():
            design_matrix[start:end+1, start:end+1] = 0
        np.fill_diagonal(design_matrix, 1)
        return design_matrix
    
    def relocate_tasks(self, tasks: List[str]):
        '''
        Relocate tasks according to the provided order
        '''

        if not all(task in self.tasks_idx for task in tasks):
            raise ValueError('All tasks must be present in the similarity matrix')
        
        new_indices = np.concatenate([np.arange(self.tasks_idx[task][0], self.tasks_idx[task][1]+1) for task in tasks])
        self.tasks_list = np.repeat(tasks, self.matrix.shape[0] // len(tasks))
        self.tasks_idx = get_unique_indices(self.tasks_list)
        self.matrix = self.matrix[np.ix_(new_indices, new_indices)]

        if self.design_matrix is not None:
            self.design_matrix = self.design_matrix[np.ix_(new_indices, new_indices)]
            self.attribute_list = self.attribute_list[new_indices]
            self.attribute_idx = get_unique_indices(self.attribute_list)
        
    def filter_tasks(self, tasks: List[str]):
        '''
        Delete a task from the similarity matrix

        Args:
            tasks (List[str]): The tasks to delete
        '''
        tasks = tasks if isinstance(tasks, list) else [tasks]

        tasks_to_delete = [task for task in self.tasks_idx if task not in tasks]
        indices_to_delete = []
        for task in tasks_to_delete:
            if task not in self.tasks_idx:
                raise ValueError(f'Task {task} not found in the similarity matrix.')
            
            start, end = self.tasks_idx[task]
            self.matrix = np.delete(self.matrix, np.s_[start:end+1], axis=0)
            self.matrix = np.delete(self.matrix, np.s_[start:end+1], axis=1)

            if self.design_matrix is not None:
                self.design_matrix = np.delete(self.design_matrix, np.s_[start:end+1], axis=0)
                self.design_matrix = np.delete(self.design_matrix, np.s_[start:end+1], axis=1)
                self.attribute_list = np.delete(self.attribute_list, np.s_[start:end+1])
            
            self.tasks_list = np.delete(self.tasks_list, np.s_[start:end+1])
            self.tasks_idx = get_unique_indices(self.tasks_list)
            self.attribute_idx = get_unique_indices(self.attribute_list)

            indices_to_delete.append((start, end))

    def copy(self):
        return copy.deepcopy(self)
    
    def triangularize_matrix(self):
        self.matrix = np.tril(self.matrix)        
    
    def full_matrix(self):
        self.matrix = self.matrix + self.matrix .T - np.diag(np.diag(self.matrix ))

    def plot(
            self,
            title: str = None,
            norm: Tuple[int, int] = (0, 1),
            rel_ticks: bool = True, 
            axis: plt.Axes = None, 
            labels: List[str] = None,
            plot_means: bool = False,
            save_path: str = None,
            dpi: int = 500,
            cmap: str = 'coolwarm',
            label_colors: bool = False,
            show_labels: bool = True,
            plot_lower_diag: bool = False,
            grey_upper_diag: bool = False,
            grey_upper_diag_alpha: float = 0.3,
            upper_diag_color: str = 'grey',
            fade_upper_diag: bool = False,
            upper_diag_fade_alpha: float = 0.15,
            bounding_boxes: bool = False,
            bounding_box_color: str = 'black',
            bounding_box_width: int = 2,
            fontsize: int = 12
        ):
            
            # Create a new figure and axis if none is provided
            if axis is None:
                fig, ax = plt.subplots()
            else:
                ax = axis
            
            if plot_lower_diag:
                # Mask the upper diagonal
                mask = np.triu(np.ones_like(self.matrix, dtype=bool), k=0)  
                matrix_to_plot = np.ma.array(self.matrix, mask=mask) 
            else:
                matrix_to_plot = self.matrix
            # Optionally prepare a per-pixel alpha mask to fade the upper triangle (including the diagonal)
            alpha_array = None
            if fade_upper_diag:
                alpha_array = np.ones_like(self.matrix, dtype=float)
                upper_mask = np.triu(np.ones_like(self.matrix, dtype=bool), k=0)
                fade_val = float(np.clip(upper_diag_fade_alpha, 0.0, 1.0))
                alpha_array[upper_mask] = fade_val

            # Plot the RDM on the specified (or new) axes object
            cax = ax.imshow(
                matrix_to_plot,
                cmap=cmap,
                interpolation='nearest',
                norm=plt.Normalize(*norm) if norm else None,
                alpha=alpha_array if fade_upper_diag else None
            )
            if title:
                ax.set_title(title, fontsize=fontsize+4)

            # Optionally shade the upper triangle with a grey overlay (including the main diagonal)
            # Skipped if fading is enabled
            if grey_upper_diag and not fade_upper_diag:
                upper_mask = np.triu(np.ones_like(self.matrix, dtype=bool), k=0)
                overlay_value = 0.5
                overlay = np.ma.masked_where(~upper_mask, np.full_like(self.matrix, overlay_value, dtype=float))
                ax.imshow(
                    overlay,
                    cmap=upper_diag_color,
                    vmin=0.0,
                    vmax=1.0,
                    interpolation='nearest',
                    alpha=grey_upper_diag_alpha
                )

            if self.tasks_idx:            
                if rel_ticks:
                    midpoints = [(start + end) / 2 for start, end in self.tasks_idx.values()]
                    ax.set_xticks(midpoints)
                    ax.set_yticks(midpoints)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                if show_labels:
                    if labels:
                        assert len(labels) == len(self.tasks_idx), 'The number of labels must be equal to the number of tasks'
                    else:
                        labels = self.tasks_idx.keys()
                    ax.set_xticklabels(list(labels), rotation=90, fontsize=fontsize)
                    ax.set_yticklabels(list(labels), fontsize=fontsize)
                    
                    if label_colors:
                        # Set individual colors for tick labels
                        for i, (tick, color) in enumerate(zip(ax.get_xticklabels(), label_colors)):
                            tick.set_color(color)
                        for i, (tick, color) in enumerate(zip(ax.get_yticklabels(), label_colors)):
                            tick.set_color(color)

            if plot_means:    
                if self.tasks_idx and plot_means:
                    for i, (task1, (start1, end1)) in enumerate(self.tasks_idx.items()):
                        midpoint1 = (start1 + end1) / 2
                        for j, (task2, (start2, end2)) in enumerate(self.tasks_idx.items()):
                            midpoint2 = (start2 + end2) / 2
                            if i == j:  # Within-task mean value
                                mean_value = self.within_task_sims[task1]
                            else:
                                mean_value = self.between_task_sims.get((task1, task2), self.between_task_sims.get((task2, task1)))

                            if i > j and plot_lower_diag:  # Between-task mean value (upper diagonal)
                                continue
                            
                            ax.text(midpoint1, midpoint2, f'{mean_value:.2f}', ha='center', va='center', color='black')
            
            if bounding_boxes: 
                assert self.design_matrix is not None, 'Design matrix must be provided to plot bounding boxes'
                bounding_boxes = get_bounds(self.design_matrix)

                for (tl, br) in bounding_boxes:
                    rect = Rectangle(
                        (tl[1] - 0.5, tl[0] - 0.5), 
                        br[1] - tl[1] + 1, br[0] - tl[0] + 1, 
                        linewidth=bounding_box_width, edgecolor=bounding_box_color, facecolor='none', linestyle='--'
                    )
                    ax.add_patch(rect)
                
            # Add colorbar for the axes
            if axis is None:
                cbar = plt.colorbar(cax, ax=ax)
                cbar.ax.set_ylabel(f'Cosine {self.sim_mode}', fontsize=14)
                cbar.ax.tick_params(labelsize=12)

            if plot_lower_diag:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

            if save_path:
                plt.tight_layout()
                plt.savefig(save_path, dpi=dpi)

            if axis is None:
                plt.show()
                plt.close()

class AccSimMat(SimilarityMatrix):
    def __init__(
        self,
        sim_mat: SimilarityMatrix,
        acc_bool: List[bool]
    ):
        self.full_mat = sim_mat.matrix
        self.full_design_matrix_list = sim_mat.attribute_list
        self.acc_bool = np.array(acc_bool)
        self.full_tasks_list = np.array(sim_mat.tasks_list)
        self.acc_mode = 'Correct'
        self.create_matrix(acc_bool)

    def create_matrix(self, acc_bool):
        sim_mat = self.full_mat[np.ix_(acc_bool, acc_bool)]
        tasks_list = self.full_tasks_list[acc_bool]
        design_matrix_list = self.full_design_matrix_list[acc_bool]
        super().__init__(sim_mat=sim_mat, tasks=tasks_list, attribute_list=design_matrix_list)
    
    def toggle_accuracy(self):
        self.acc_mode = 'Correct' if self.acc_mode == 'Incorrect' else 'Incorrect'
        acc_bool = self.acc_bool if self.acc_mode == 'Correct' else ~self.acc_bool
        self.create_matrix(acc_bool)