import torch.cuda

from config_enums import DatasetEnum, ModelEnum, LossEnum, ActivationEnum, EtaInitEnum
from typing import Tuple
# to define
#reproducibility
DETERMINISTIC_RES: bool = True
# dataset options
DATASET_NAME: DatasetEnum = DatasetEnum.MNIST
N_DIM: int = 100  # number of feature dimensions for generating ND_REG-ression
DATASET_DIR: str = "./datasets"  # place where datasets will be downloaded
# run options
NO_RUNS: int = 1
NO_EPOCHS: int = 18# number of iterations over all dataset
BATCH_SIZE = 128 # None = whole dataset or int
SCINOL_ONLY: bool = False
VALIDATION: bool = True
# model params
MODEL_TYPE: ModelEnum = ModelEnum.MNIST_CNN
HIDDEN_LAYERS: Tuple = (100,100,100)  # number of neurons in layers in MLP
ACTIVATION: ActivationEnum = ActivationEnum.RELU  # activation after every layer, ignored if MODEL == ModelEnum.LINEAR
# loss definition
LOSS: LossEnum = LossEnum.DEFAULT  # loss function, None default dataset loss
# cuda
CUDA: bool = True
# logging lvl
LOG_GRADS_AND_WEIGHTS: bool = False
LOG_SCINOL_PARAMS: bool = False
# Scinol options
ETA_INIT: EtaInitEnum = EtaInitEnum.UNIFORM_GLOROT

# automatically completed
SCINOL_PREFIX = str(ETA_INIT)+"$"+str(ACTIVATION)+"$"
_model_spec = str(HIDDEN_LAYERS) if MODEL_TYPE == ModelEnum.MLP else ''
WRITER_PREFIX = f'runs/{DATASET_NAME.value}/{MODEL_TYPE.value}{_model_spec}/'
DEVICE = 'cuda' if CUDA and torch.cuda.is_available() else 'cpu'

