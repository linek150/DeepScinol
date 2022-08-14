import math

from torch.nn import Linear, Conv2d, BatchNorm2d
import torch
from .common import get_param_state, M
"""
all functions assume mean reduction for loss function for mini-batch
"""


def __linear(module: Linear, param):
    k = 1/module.out_features
    init_value = math.sqrt(k)
    eta = torch.full_like(param, init_value)
    return eta


def __conv2d(module: Conv2d, param):
    k = 1/(module.out_channels) # it should be multiply be sizes of an output, but it cannot be accessed
    init_value = math.sqrt(k)
    eta = torch.full_like(param, init_value)
    return eta


def __batch_norm2d(module: BatchNorm2d, param):
    return torch.ones_like(param)



eta_init_dict = {Conv2d: __conv2d, Linear: __linear, BatchNorm2d: __batch_norm2d}