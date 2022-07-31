import copy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import dataloader
from datasets import CustomDataset
from test_alg import train
from scinol import Scinol2Dl
from config import *
from models import get_model, get_scinol_model
from config_enums import LossEnum
from torch.optim import SGD, Adam
import torch
from cocob_bp import COCOB_Backprop
from deep_scinol_adapter.deep_scinol_adapter import adapt_to_scinol

if DETERMINISTIC_RES:
    torch.manual_seed(0)

dataset_ = CustomDataset(DATASET_NAME, n_dim=N_DIM)

bs=BATCH_SIZE if BATCH_SIZE is not None else len(dataset_)
valid_dataloader = None
if VALIDATION:
    valid_dataset = CustomDataset(DATASET_NAME, validation=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs)
dataloader_ = torch.utils.data.DataLoader(dataset_, batch_size=bs)

no_inputs = dataset_.input_size
no_outputs = dataset_.output_size
loss = dataset_.default_loss if LOSS == LossEnum.DEFAULT else LOSS

if SCINOL_ONLY:
    optimizers = []  # [Scinol2Dl]
else:
    optimizers = [Adam, COCOB_Backprop]

for run_no in range(NO_RUNS):
    model = get_model(MODEL_TYPE, no_inputs, no_outputs, HIDDEN_LAYERS, ACTIVATION.value)
    for optimizer in optimizers:
        clean_model = copy.deepcopy(model).to(DEVICE)
        writer = SummaryWriter(log_dir=f'{WRITER_PREFIX}{optimizer.__name__+str(run_no)}')
        train(dataloader_, clean_model, writer, optim=optimizer, no_epochs=NO_EPOCHS, log_grads=LOG_GRADS_AND_WEIGHTS,
              loss=loss, val_dataloader = valid_dataloader)
    clean_model = copy.deepcopy(model).to(DEVICE)
    #scinol_model = get_scinol_model(MODEL_TYPE, no_inputs, no_outputs, HIDDEN_LAYERS, ACTIVATION.value, ETA_INIT).to(DEVICE)
    scinol_model = adapt_to_scinol(clean_model)
    writer = SummaryWriter(log_dir=f'{WRITER_PREFIX}{SCINOL_PREFIX}Scinol{str(run_no)}')
    train(dataloader_, scinol_model, writer, no_epochs=NO_EPOCHS, loss=loss, log_grads=LOG_GRADS_AND_WEIGHTS,
          log_scinol_params_=LOG_SCINOL_PARAMS, val_dataloader=valid_dataloader)
    writer.flush()
