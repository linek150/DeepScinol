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


def train(dataloader_: dataloader, model: Module, writer: SummaryWriter,
          loss: Callable, optim: optimizer = None, no_epochs: int = 100,
          log_grads: bool = False, log_scinol_params_: bool = False):

    use_optim= True if optim is not None else False
    opt_name=optim.__name__ if use_optim else "Scinol"
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    step = 0
    optim = optim(model.parameters(), lr=0.01) if use_optim else None
    for epoch in tqdm(range(no_epochs), desc=f'{opt_name}-Epoch'):
        for x, y in dataloader_:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            if not use_optim and log_scinol_params_: log_scinol_params(model, writer, step)
            loss_val = loss(y_pred, y)
            writer.add_scalar("loss", loss_val, step)
            if log_grads:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param, step)
                    if param.grad is not None:
                        writer.add_histogram(name+"_grad", param.grad, step)
                        writer.add_scalar(name+"_no_non_zero_gradients",
                                          torch.count_nonzero(torch.torch.ne(param.grad, torch.tensor(0.))), step)
                        writer.add_scalar(name+"_no_non_zero_weights",
                                          torch.count_nonzero(torch.torch.ne(param, torch.tensor(0.))), step)
                        writer.add_scalar(name+"_no_weights_grater_then_0",
                                          torch.count_nonzero(torch.torch.gt(param.grad, torch.tensor(0.))), step)
            if use_optim:
                optim.zero_grad()
                loss_val.backward()
                optim.step()
            else:
                model.zero_grad()
                loss_val.backward()
                model.step()
            step += 1


def log_scinol_params(model, writer, step):
    for name, module in model.named_children():
        for p_name, p in module.named_parameters():
            optim_state = module.optim_state[p]
            full_name=name+"."+p_name
            writer.add_histogram(full_name+"_G", optim_state['G'], step)
            writer.add_histogram(full_name+"_S2", optim_state['S2'], step)
            writer.add_histogram(full_name + "_eta", optim_state['eta'], step)
            writer.add_histogram(full_name+"_M", optim_state['M'], step)
