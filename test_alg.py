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
            step += 1


def train_scinol(dataloader_: dataloader, model: scinol_nn.ScinolMLP, writer: SummaryWriter,
                 loss: Callable, no_epochs: int = 100):
    step=0
    for epoch in range(no_epochs):
        for x, y in dataloader_:
            y_pred = model(x)
            loss_val = loss(y_pred,y)
            writer.add_scalar("loss", loss_val, step)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, step)
                if param.grad is not None:
                    writer.add_histogram(name+"_grad",param.grad,step)

            for name, module in model.named_children():
                for p_name, p in module.named_parameters():
                    optim_state = module.optim_state[p]
                    full_name=name+"."+p_name
                    writer.add_histogram(full_name+"_G", optim_state['G'], step)
                    writer.add_histogram(full_name+"_S2", optim_state['S2'], step)
                    writer.add_histogram(full_name + "_eta", optim_state['eta'], step)
                    writer.add_histogram(full_name+"_M", optim_state['M'], step)
            model.zero_grad()
            loss_val.backward()
            model.step()
            step += 1
