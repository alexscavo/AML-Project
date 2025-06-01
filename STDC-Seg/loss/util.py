import numpy as np
import torch

def enet_weighing(label, num_classes, c=1.02, ignore_lb=255):
    """
    Computes ENet class weights:
        w_c = 1 / ln(c + p_c)
    where p_c = freq_c / total_pixels (excluding ignore_lb).
    """
    # move to CPU and flatten
    flat = label.cpu().numpy().flatten()
    # ignore the no-data label
    mask = (flat != ignore_lb)
    flat = flat[mask]
    # now bincount only sees values in [0 .. num_classes-1]
    counts = np.bincount(flat, minlength=num_classes)
    total = flat.size
    # propensity scores & ENet weights
    p = counts / total
    weights = 1.0 / np.log(c + p)
    return torch.from_numpy(weights).float()


def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)
    return output_arr