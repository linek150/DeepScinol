from common_enums import DatasetEnum

NO_RUNS = 1
NO_EPOCHS = 1
DATASET_NAME = DatasetEnum.ONED_REG
N_DIM = 100
BATCH_SIZE = 1
SCINOL_ONLY = True
LOSS = None
WRITER_PREFIX = f'runs/{DATASET_NAME.name}/linear/'
DATASET_DIR = "./datasets"