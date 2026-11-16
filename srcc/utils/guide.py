from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch import Tensor
from torch.nn import functional as F
from einops import reduce
import math
from scipy.optimize import linear_sum_assignment


# add project path to sys to import relative modules
import sys
import os
sys.path.insert(0, '../..') # add parent-parent dir

from srcc.main import params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logger = logging.getLogger(params["running_type"])

def sample_z_from_latents(latents: torch.Tensor, n_samples: int = 64):
    """
    Sample imaginable latents combinations z from infered latents.
    """
    batch_size, n_slots, n_latents = latents.shape

    # flatten the latents tensor into a 2D tensor
    flattened_latents = latents.reshape(batch_size * n_slots, n_latents)

    # sample indices randomly with replacement from the flattened latents tensor
    indices = np.random.choice(
        len(flattened_latents),
        size=n_samples * n_slots,
    )
    # gather the sampled latents from the flattened tensor
    sampled_latents = flattened_latents[indices]
    # reshape the sampled latents tensor back to the original shape
    sampled_z = sampled_latents.reshape(n_samples, n_slots, n_latents)

    return sampled_z.to(latents.device), indices

def linear_scheduler(step, total_steps, initial, final):
    """Linear scheduler"""

    if step >= total_steps:
        return final
    if step <= 0:
        return initial
    if total_steps <= 1:
        return final

    t = step / (total_steps - 1)
    return (1.0 - t) * initial + t * final

def exp_scheduler(step, total_steps, initial, final):
    """Exponential scheduler"""

    if step >= total_steps:
        return final
    if step <= 0:
        return initial
    if total_steps <= 1:
        return final

    t = step / (total_steps - 1)
    return np.exp((step-total_steps)*0.1)*final

def exp_scheduler_neg(step, total_steps, initial, final):
    """Exponential scheduler"""

    if step >= total_steps:
        return final
    if step <= 0:
        return initial
    if total_steps <= 1:
        return final

    t = step / (total_steps - 1)
    return np.exp((1/(step-2020))*20)

def to_int(value: Tensor):
    return int(torch.round(value))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def kl_normal(qm, qv, pm, pv):

    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    if len(element_wise.shape) == 2:
        kl = torch.sum(element_wise, dim=-1)
    else:
        kl = element_wise
    return kl

def permute_dims(z, d):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = torch.tensor([]).to(device)
    perms = []
    for z_j in z.split(d, 1):
        #print(z_j.shape)
        perm = torch.randperm(B).to(z.device)
        perms.append(perm)
        perm_z_j = z_j[perm]
        perm_z = torch.cat((perm_z, perm_z_j), dim=-1)

    return perm_z, perms

def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv

def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"

def linear_std(step):
    if step <= 1000: return 1. - step*0.00095
    else: return 0.05

