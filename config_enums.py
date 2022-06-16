from enum import Enum
from torch.nn.functional import mse_loss, cross_entropy, l1_loss
from torch.nn import ReLU

class DatasetEnum(Enum):
    ONED_REG = '1dReg'
    ND_REG = 'ndReg'
    MNIST = 'mnist'
    DUM_ABS = 'abs'


class ModelEnum(Enum):
    LINEAR = 'linear'
    MLP = 'mlp'


class LossEnum(Enum):
    DEFAULT = 'dataset_default'
    MSE = mse_loss
    CE = cross_entropy
    MAE = l1_loss


class ActivationEnum(Enum):
    RELU = ReLU
