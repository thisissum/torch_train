import torch
import numpy as np
import random

def seed_everything(seed=1):
    """
    Add seed to numpy random and torch
    Args:
        seed: int
    Return:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def move_to_device(data, device):
    """
    Flatten all items contained in 'data' and move it to device
    Args:
        data: Iterable[Iterable[..[item]]]]
        device: str or torch.device
    Return:
        output: flattened data contained in a list
    """
    output = []
    for item in data:
        if isinstance(item, (torch.Tensor, torch.cuda.Tensor)):
            output.append(item.to(device))
        elif isinstance(item, (tuple, list)):
            output += move_to_device(item, device)
        else:
            output.append(item)
    return output