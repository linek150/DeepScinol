from enum import Enum
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
from torch.nn import ReLU, Sigmoid, Tanh

class DatasetEnum(Enum):
    ONED_REG = '1dReg'
    ND_REG = 'ndReg'
    FLATTEN_MNIST = 'flatten_mnist'
    MNIST = 'mnist'
    DUM_ABS = 'abs'


class ModelEnum(Enum):
    LINEAR = 'linear'
    MLP = 'mlp'
    MNIST_CNN = 'mnist_cnn'


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
    UNIFORM_GLOROT = 'glorot'
    ONES = 'ones'