from typing import Iterable, Callable
from config_enums import ModelEnum
import torch
from torch.nn import Linear, Sequential, Module
from scinol_nn import ScinolLinear, ScinolMLP


def get_model(model_enum: ModelEnum, no_inputs: int, no_outputs: int,
              hidden_layers: Iterable, activation: Module):
    if model_enum == ModelEnum.LINEAR:
        return torch.nn.Linear(no_inputs, no_outputs)
    if model_enum == ModelEnum.MLP:
        return _mlp(no_inputs, no_outputs, hidden_layers, activation)


def get_scinol_model(model_enum: ModelEnum, no_inputs: int, no_outputs: int,
                     hidden_layers: Iterable, activation: Module):
    if model_enum == ModelEnum.LINEAR:
        return ScinolLinear(no_inputs, no_outputs, single_layer=True)
    if model_enum == ModelEnum.MLP:
        return ScinolMLP(no_inputs, no_outputs, hidden_layers, activation)


def _mlp(no_inputs: int, no_outputs: int, hidden_layers: Iterable, activation: Module) -> Sequential:
    model = Sequential()
    prev_layer_size = -1
    for idx, layer_size in enumerate(hidden_layers):
        if idx == 0:
            model.add_module('input_layer', Linear(no_inputs, layer_size))
        else:
            model.add_module(f'hidden_layer_{idx}', Linear(prev_layer_size, layer_size))
        prev_layer_size = layer_size
        # activation after every layer except last
        model.add_module(f"relu_{idx}", activation())
    model.add_module('output_layer', Linear(prev_layer_size, no_outputs))
    return model
