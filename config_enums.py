from enum import Enum

import torch.optim.adam
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
from torch.nn import ReLU, Sigmoid, Tanh

import cocob_bp


class DatasetEnum(Enum):
    ONED_REG = '1dReg'
    ND_REG = 'ndReg'
    FLATTEN_MNIST = 'flatten_mnist'
    MNIST = 'mnist'
    DUM_ABS = 'abs'
    CIFAR10 = 'cifar10'
    ADULT = 'adult'
    COVER_TYPE = 'cover_type'


class ModelEnum(Enum):
    LINEAR = 'linear'
    MLP = 'mlp'
    MNIST_CNN = 'mnist_cnn'
    RESNET18_MNIST = 'resnet18_mnist'
    RESNET18_CIFAR10 = 'resnet18_cifar10'
    RESNET34_CIFAR10 = 'resnet34_cifar10'

class LossEnum(Enum):
    DEFAULT = 'dataset_default'
    MSE = MSELoss(reduction='mean')
    CE = CrossEntropyLoss(reduction='mean')
    MAE = L1Loss(reduction='mean')


class ActivationEnum(Enum):
    RELU = ReLU
    SIGMOID = Sigmoid
    TANH = Tanh


class EtaInitEnum(Enum):
    GLOROT = 'glorot'
    ONES = 1
    TENTH = 0.1
    TENS = 10

class OptimizerEnum(Enum):
    ADAM = torch.optim.Adam
    SGD = torch.optim.SGD
    COCOB = cocob_bp.COCOB_Backprop

