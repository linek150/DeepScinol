from typing import Iterable, Callable

import torchvision.models.resnet

from config_enums import ModelEnum, EtaInitEnum
import torch
from torch.nn import Linear, Sequential, Module
from scinol_nn import ScinolLinear, ScinolMLP, MNIST_CNN


def get_model(model_enum: ModelEnum, no_inputs: int, no_outputs: int,
              hidden_layers: Iterable, activation: Module):
    if model_enum == ModelEnum.LINEAR:
        return torch.nn.Linear(no_inputs, no_outputs)
    if model_enum == ModelEnum.MLP:
        return _mlp(no_inputs, no_outputs, hidden_layers, activation)
    if model_enum == ModelEnum.MNIST_CNN:
        return _mnist_cnn(activation)
    if model_enum == ModelEnum.RESNET18_MNIST:
        return _resnet18_mnist()


def get_scinol_model(model_enum: ModelEnum, no_inputs: int, no_outputs: int,
                     hidden_layers: Iterable, activation: Module, eta_init: EtaInitEnum):
    if model_enum == ModelEnum.LINEAR:
        return ScinolLinear(no_inputs, no_outputs, single_layer=True, eta_init=eta_init)
    if model_enum == ModelEnum.MLP:
        return ScinolMLP(no_inputs, no_outputs, hidden_layers, activation, eta_init)
    if model_enum == ModelEnum.MNIST_CNN:
        return MNIST_CNN(eta_init, activation)


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

def _mnist_cnn(activation: Module):
    model = Sequential()
    no_filters=64
    model.add_module('conv1',torch.nn.Conv2d(1, no_filters, 5))
    model.add_module("RELU1", activation())
    model.add_module("Max_pooling_1", torch.nn.MaxPool2d(2))
    model.add_module("CNN_2", torch.nn.Conv2d(no_filters, no_filters, 5))
    model.add_module("RELU12", activation())
    model.add_module("Max_pooling_2", torch.nn.MaxPool2d(2))
    model.add_module("flatten", torch.nn.Flatten())
    model.add_module("lin", Linear(1024, 10))
    return model


def _resnet18_mnist():
    model = torchvision.models.resnet.resnet18()
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7,stride=2, padding=(3,3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = Linear(num_ftrs, 10)
    return model
