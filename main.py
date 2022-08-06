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


def run_training(starting_model: torch.nn.Module, optimizer: OptimizerEnum, run_no:int, dataloader, loss,
                 valid_dataloader, lr):
    clean_model = copy.deepcopy(starting_model).to(DEVICE)
    writer = SummaryWriter(log_dir=f'{WRITER_PREFIX}{optimizer.name}/{lr}/{str(run_no)}')
    train(dataloader, clean_model, writer, optim=optimizer, no_epochs=NO_EPOCHS,
          log_grads=LOG_GRADS_AND_WEIGHTS,
          loss=loss, val_dataloader=valid_dataloader, lr=lr, log_to_pickle=LOSS_TO_PICKLE)



OPTIMIZERS_REQUIRE_LR = (OptimizerEnum.ADAM, OptimizerEnum.SGD)

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
if RUN_OTHER_ALGORITHMS:
    optimizers_with_optional_lr = OTHER_ALGORITHMS
else:
    optimizers_with_optional_lr = []
writer = None
for run_no in range(NO_RUNS):
    model = get_model(MODEL_TYPE, no_inputs, no_outputs, HIDDEN_LAYERS, ACTIVATION.value)
    for optimizer_with_optional_lr in optimizers_with_optional_lr:
        if LR_TUNING:
            assert RUN_SCINOL==False, "I doesnt make sense to run LR_TUNING and SCINOL"
            assert optimizer_with_optional_lr[0] in OPTIMIZERS_REQUIRE_LR, "You try to tune LR for algorithm that doesn't have lr param."
            assert len(optimizer_with_optional_lr) == 1, "lr should be specified only in LR array for all optimizers when LR_TUNING==True"
            for lr in LR:
                run_training(model, optimizer_with_optional_lr[0], run_no, dataloader_, loss, valid_dataloader, lr)
        else:
            lr=None
            if len(optimizer_with_optional_lr) == 1:
                assert optimizer_with_optional_lr[0] not in OPTIMIZERS_REQUIRE_LR, f"You need to specifie lr for this algorithm{optimizer_with_optional_lr}"
            if len(optimizer_with_optional_lr) == 2 :
                assert optimizer_with_optional_lr[0] in OPTIMIZERS_REQUIRE_LR, f"This optimizer {optimizer_with_optional_lr[0].name} doesn't need lr, remove it."
                lr=optimizer_with_optional_lr[1]
            run_training(model, optimizer_with_optional_lr[0], run_no, dataloader_, loss, valid_dataloader, lr)

    if RUN_SCINOL:
        clean_model = copy.deepcopy(model).to(DEVICE)
        #scinol_model = get_scinol_model(MODEL_TYPE, no_inputs, no_outputs, HIDDEN_LAYERS, ACTIVATION.value, ETA_INIT).to(DEVICE)
        scinol_model = adapt_to_scinol(clean_model)
        writer = SummaryWriter(log_dir=f'{WRITER_PREFIX}{SCINOL_PREFIX}Scinol_{str(run_no)}')
        train(dataloader_, scinol_model, writer, no_epochs=NO_EPOCHS, loss=loss, log_grads=LOG_GRADS_AND_WEIGHTS,
              log_scinol_params_=LOG_SCINOL_PARAMS, val_dataloader=valid_dataloader, log_to_pickle=LOSS_TO_PICKLE)
    writer.flush()

