import copy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import dataloader
from datasets import CustomDataset
import torch
from test_alg import train, train_scinol
from scinol import Scinol2Dl
# device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
from config import *
from models import get_model, get_scinol_model
from config_enums import LossEnum
from torch.optim import SGD, Adam

dataset_ = CustomDataset(DATASET_NAME, n_dim=N_DIM)
bs=BATCH_SIZE if BATCH_SIZE is not None else len(dataset_)
dataloader_ = torch.utils.data.DataLoader(dataset_, batch_size=bs)

no_inputs = dataset_.input_size
no_outputs = dataset_.output_size
loss = dataset_.default_loss if LOSS == LossEnum.DEFAULT else LOSS

if SCINOL_ONLY:
    optimizers = []  # [Scinol2Dl]
else:
    optimizers = [Adam, SGD, Scinol2Dl]

for run_no in range(NO_RUNS):
    model = get_model(MODEL_TYPE, no_inputs, no_outputs, HIDDEN_LAYERS, ACTIVATION.value)
    for optimizer in optimizers:
        clean_model = copy.deepcopy(model)
        writer = SummaryWriter(log_dir=f'{WRITER_PREFIX}{optimizer.__name__+str(run_no)}')

        train(dataloader_, clean_model, optimizer, writer, no_epochs=NO_EPOCHS, loss=loss)

    scinol_model = get_scinol_model(MODEL_TYPE, no_inputs, no_outputs, HIDDEN_LAYERS, ACTIVATION.value)
    writer = SummaryWriter(log_dir=f'{WRITER_PREFIX}Scinol{str(run_no)}')

    train_scinol(dataloader_, scinol_model, writer, no_epochs=NO_EPOCHS, loss=loss)

    writer.flush()
