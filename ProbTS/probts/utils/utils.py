# ---------------------------------------------------------------------------------
# Portions of this file are derived from PyTorch-TS
# - Source: https://github.com/zalandoresearch/pytorch-ts
# - License: MIT, Apache-2.0 license

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import re
import os
import torch
from typing import Optional
import importlib

def repeat(tensor: torch.Tensor, n: int, dim: int = 0):
    return tensor.repeat_interleave(repeats=n, dim=dim)


    
def weighted_average(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: int = None,
    reduce: str = 'mean',
):
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        x: Input tensor, of which the average must be computed.
        weights: Weights tensor, of the same shape as `x`.
        dim: The dim along which to average `x`

    Returns:
        Tensor: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        if reduce != 'mean':
            return weighted_tensor
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim) if dim else x
    
    
def convert_to_list(s):
    '''
    Convert prediction length strings into list
    e.g., '96-192-336-720' will be convert into [96,192,336,720]
    Input: str, list, int
    Returns: list
    '''
    if (type(s).__name__=='int'):
        return [s]
    elif (type(s).__name__=='list'):
        return s
    elif (type(s).__name__=='str'):
        elements = re.split(r'\D+', s)
        return list(map(int, elements))
    else:
        return None
    
def find_best_epoch(ckpt_folder):
    """
    Find the checkpoint with the lowest val_CRPS in the given folder.
    :param ckpt_folder: dir where the checkpoints are being saved.
    :return: (best_epoch_index, best_checkpoint_filename)
    """
    crps_pattern = r"val_CRPS=([0-9]*\.[0-9]+)"
    epoch_pattern = r"epoch=(\d+)"
    matched = []  # list of (filename, crps_float)
    for filename in os.listdir(ckpt_folder):
        match = re.search(crps_pattern, filename)
        if match:
            matched.append((filename, float(match.group(1))))

    if not matched:
        # Fallback to last.ckpt if no epoch checkpoints found
        last = os.path.join(ckpt_folder, 'last.ckpt')
        if os.path.exists(last):
            return 0, 'last.ckpt'
        raise FileNotFoundError(f"No checkpoints found in {ckpt_folder}")

    best_filename, _ = min(matched, key=lambda x: x[1])
    epoch_match = re.search(epoch_pattern, best_filename)
    best_epoch = int(epoch_match.group(1)) if epoch_match else 0
    return best_epoch, best_filename

def ensure_list(input_value, default_value=None):
    """
    Ensures that the input is converted to a list. If the input is None,
    it converts the default value to a list instead.
    """
    result = convert_to_list(input_value)
    if result is None:
        result = convert_to_list(default_value)
    return result


def init_class_helper(class_name):
    """
    Dynamically imports a module and retrieves a class.

    Args:
        class_name (str): The fully qualified name of the class in the format "module_name.ClassName".

    Returns:
        type: The class object retrieved from the specified module.
    """
    module_name, class_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    Class = getattr(module, class_name)
    return Class