import torch.cuda

from config_enums import DatasetEnum, ModelEnum, LossEnum, ActivationEnum, EtaInitEnum, OptimizerEnum
from typing import Tuple
# to define
#reproducibility
DETERMINISTIC_RES: bool = True
# dataset options
DATASET_NAME: DatasetEnum = DatasetEnum.CIFAR10
N_DIM: int = 100  # number of feature dimensions for generating ND_REG-ression
DATASET_DIR: str = "./datasets"  # place where datasets will be downloaded
# run options
NO_RUNS: int = 4
NO_EPOCHS: int = 80# number of iterations over all dataset
BATCH_SIZE = 128 # None = whole dataset or int
RUN_OTHER_ALGORITHMS: bool = False
RUN_SCINOL: bool = True
VALIDATION: bool = True
LR_TUNING: bool = False # Set it to true to run algorithms specified in OTHER_ALGORITHMS with all LR tuple values
OTHER_ALGORITHMS: Tuple[Tuple, ...] = ((OptimizerEnum.ADAM, 0.001), (OptimizerEnum.SGD, 0.1), (OptimizerEnum.COCOB,)) # Entries -> (Optimizer,[lr])
# Learning rates to test
LR: Tuple[float, ...] = (0.1, 0.01, 0.001)
# model params
MODEL_TYPE: ModelEnum = ModelEnum.RESNET18_CIFAR10
HIDDEN_LAYERS: Tuple = (1000, 1000, 1000)  # number of neurons in layers in MLP
ACTIVATION: ActivationEnum = ActivationEnum.RELU  # activation after every layer, ignored if MODEL == ModelEnum.LINEAR
# Scinol options
ETA_INIT: EtaInitEnum = EtaInitEnum.TENS
MOMENTUM: bool = False
# loss definition
LOSS: LossEnum = LossEnum.DEFAULT  # loss function, None default dataset loss
# logging lvl
LOG_GRADS_AND_WEIGHTS: bool = False
LOG_SCINOL_PARAMS: bool = False
LOSS_TO_PICKLE: bool = True
# cuda
CUDA: bool = True
# automatically completed
SCINOL_PREFIX = str(ETA_INIT)+"_"+"momentum."+str(MOMENTUM)+"_"
_model_spec = str(HIDDEN_LAYERS) if MODEL_TYPE == ModelEnum.MLP else ''
WRITER_PREFIX = f'runs/{DATASET_NAME.value}/{MODEL_TYPE.value}{_model_spec}/'
DEVICE = 'cuda' if CUDA and torch.cuda.is_available() else 'cpu'

