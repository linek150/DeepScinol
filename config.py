from config_enums import DatasetEnum, ModelEnum, LossEnum, ActivationEnum
from typing import Tuple
# to define

# dataset options
DATASET_NAME: DatasetEnum = DatasetEnum.ND_REG
N_DIM: int = 2  # number of feature dimensions for generating ND_REG-ression
DATASET_DIR: str = "./datasets"  # place where datasets will be downloaded
# run options
NO_RUNS: int = 1
NO_EPOCHS: int = 100 # number of iterations over all dataset
BATCH_SIZE = None  # None = whole dataset or int
SCINOL_ONLY: bool = False
# model params
MODEL_TYPE: ModelEnum = ModelEnum.MLP
HIDDEN_LAYERS: Tuple = (2,)  # number of neurons in hidden layers, ignored if MODEL == ModelEnum.Linear
ACTIVATION: ActivationEnum = ActivationEnum.RELU  # activation after every layer, ignored if MODEL == ModelEnum.LINEAR
# loss definition
LOSS: LossEnum = LossEnum.DEFAULT  # loss function, None default dataset loss


# automatically completed
_model_spec = str(HIDDEN_LAYERS) if MODEL_TYPE == ModelEnum.MLP else ''
WRITER_PREFIX = f'runs/{DATASET_NAME.value}/{MODEL_TYPE.value}{_model_spec}/'
