import torch.nn.functional
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss

import scinol_nn
from scinol_nn import ScinolLinear
from torch.utils.data import dataloader
from torch.nn import Module
from torch.optim import optimizer
from typing import Callable
from tqdm import tqdm


def train(dataloader_: dataloader, model: Module, optimizer_: optimizer, writer: SummaryWriter,
          loss: Callable, no_epochs: int = 100):
    optim = optimizer_(model.parameters(), lr=0.001)
    step=0
    for epoch in tqdm(range(no_epochs), desc=f'{optimizer_.__name__}-Epoch'):
        for x, y in dataloader_:
            y_pred = model(x)
            loss_val = loss(y_pred, y)
            writer.add_scalar("loss", loss_val, step)
            optim.zero_grad()
            loss_val.backward()
            optim.step()
            step+=1

def train_scinol(dataloader_: dataloader, model: scinol_nn.ScinolMLP, writer: SummaryWriter,
                 loss: Callable, no_epochs: int = 100):
    step=0
    for epoch in range(no_epochs):
        for x, y in dataloader_:
            y_pred = model(x)
            loss_val = loss(y_pred,y)
            writer.add_scalar("loss", loss_val, step)
            model.zero_grad()
            loss_val.backward()
            model.step()
            step+=1
