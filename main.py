import copy

from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss

import scinol_nn
from scinol_nn import ScinolLinear
from torch.utils.data import dataloader
from torch.nn import Module
from torch.optim import optimizer
from datasets import CustomDataset
import torch
from test_alg import train, train_scinol
from scinol import Scinol2Dl
from common_enums import DatasetEnum
from tqdm import tqdm
# device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
from config import *


dataset_ = CustomDataset(DATASET_NAME, n_dim=N_DIM)
bs=BATCH_SIZE if BATCH_SIZE is not None else len(dataset_)
dataloader_ = torch.utils.data.DataLoader(dataset_, batch_size=bs)
if LOSS is None:
    loss = dataset_.default_loss
else:
    loss = LOSS

if SCINOL_ONLY:
    optimizers = []#[Scinol2Dl]
else:
    optimizers = [torch.optim.Adam, torch.optim.SGD, Scinol2Dl]

for run_no in range(NO_RUNS):
    model = torch.nn.Sequential(torch.nn.Linear(dataset_.input_size, 100),
                                 torch.nn.ReLU(),
                                 torch.nn.Linear(100,dataset_.output_size))
    for optimizer in optimizers:
        clean_model = copy.deepcopy(model)
        writer = SummaryWriter(log_dir=f'{WRITER_PREFIX}{optimizer.__name__+str(run_no)}')

        train(dataloader_, clean_model, optimizer, writer, no_epochs=NO_EPOCHS, loss=loss)

    model = scinol_nn.ScinolMLP(dataset_.input_size, dataset_.output_size)
    writer = SummaryWriter(log_dir=f'{WRITER_PREFIX}Scinol{str(run_no)}')

    train_scinol(dataloader_, model, writer, no_epochs=NO_EPOCHS, loss=loss)

    writer.flush()
