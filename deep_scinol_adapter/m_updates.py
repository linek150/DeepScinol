from torch.nn import Linear, Conv2d
import torch
from .common import get_param_state, M
"""
all functions assume mean reduction for loss function for mini-batch
"""


def __linear(module, curr_input):
    assert len(curr_input.shape) == 2, "Input tensor should have dims (N,W)"
    abs_input = torch.abs(curr_input)
    mean_abs_input = torch.mean(abs_input, 0)
    for p_name, p in module.named_parameters():
        state = get_param_state(module, p_name)
        if p_name == 'bias':
            continue
        if p_name == 'weight':
            state[M].set_(torch.maximum(state[M], mean_abs_input))


def __conv2d(module, curr_input):
    assert len(curr_input.shape) == 4, "Input tensor should have dims (N,C,H,W)"
    abs_input = torch.abs(curr_input)
    # two following lines take max_abs from every input tensor in batch, result dims=(N,C)
    max_abs_rows, _ = torch.max(abs_input, -1)
    max_abs_cols, _ = torch.max(max_abs_rows, -1)
    # take mean of abs_max values for every channel over mini-batch, result dims=(N)
    mean_max_abs_channel = torch.mean(max_abs_cols, 0)
    for p_name, p in module.named_parameters():
        state = get_param_state(module, p_name)
        # new M is calculated only for weights, for biases it is constant 1
        if p_name == 'bias':
            continue
        if p_name == 'weight':
            expanded = mean_max_abs_channel.expand(p.shape[0], p.shape[2], p.shape[3], p.shape[1])
            curr_max = expanded.permute(0, 3, 1, 2)
            torch.maximum(curr_max, state[M], out=state[M])


def __batch_norm2d(module, curr_input):
    # curr_inpu channels (N,C,H,W)
    assert len(curr_input.shape) == 4
    abs_input = torch.abs(curr_input)
    # two following lines take max_abs from every input tensor in batch, result dims=(N,C)
    max_abs_rows, _ = torch.max(abs_input, -1)
    max_abs_cols, _ = torch.max(max_abs_rows, -1)
    # take mean of abs_max values for every channel over mini-batch, result dims=(N)
    mean_max_abs_channel = torch.mean(max_abs_cols, 0)
    for p_name, p in module.named_parameters():
        state = get_param_state(module, p_name)
        # new M is calculated only for weights, for biases it is constant 1
        if p_name == 'bias':
            continue
        if p_name == 'weight':
            torch.maximum(mean_max_abs_channel, state[M], out=state[M])


m_update_dict = {Conv2d: __conv2d, Linear: __linear, torch.nn.BatchNorm2d: __batch_norm2d}
