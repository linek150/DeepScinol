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
          loss: Callable, log_to_pickle, optim: optimizer = None, no_epochs: int = 100,
          log_grads: bool = False, log_scinol_params_: bool = False,
          val_dataloader: dataloader = None, lr=None):
    use_optim= True if optim is not None else False
    opt_name = optim.name if use_optim else "Scinol"
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    step = 0
    optim = optim.value(model.parameters(), lr=lr) if use_optim else None
    train_loss = []
    train_acc = []
    validation_loss = []
    for epoch in tqdm(range(no_epochs), desc=f'{opt_name}-Epoch'):
        for x, y in dataloader_:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss_val = loss(y_pred, y)
            writer.add_scalar("loss", loss_val, step)
            train_loss.append((step, float(loss_val)))
            if log_grads:
                __log_grads(writer, model, step)

            if use_optim:
                optim.zero_grad()
                loss_val.backward()
                optim.step()
            else:
                model.zero_grad()
                loss_val.backward()
                model.step()
            # test accuracy
            with torch.no_grad():
                labels = torch.argmax(y_pred, 1)
                train_accuracy = (labels == y).sum() / x.shape[0]
            writer.add_scalar("top 1 test accuracy", train_accuracy, step)
            train_acc.append((step,float(train_accuracy)))


            step += 1
            # validation accuracy
        if val_dataloader is not None:

            with torch.no_grad():
                no_correct = 0
                no_all = 0
                no_batches=0
                sum_validation_loss=0
                for x, y in val_dataloader:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                    validation_batch_loss = loss(y_pred, y)
                    labels = torch.argmax(y_pred, 1)
                    no_correct += (labels == y).sum()
                    no_all += y.shape[0]
                    no_batches += 1
                    sum_validation_loss+=validation_batch_loss
                val_acc = no_correct / no_all
                mean_validation_loss = sum_validation_loss/no_batches
            writer.add_scalar("top 1 val accuracy", val_acc, epoch)
            writer.add_scalar("val_loss",mean_validation_loss, epoch)
            validation_loss.append((step, float(mean_validation_loss)))
    if log_to_pickle:
        import pickle
        import os
        path = "./pickles/"+writer.log_dir+"/"
        os.makedirs(path)
        with open(path+"train_loss", "wb+") as train_loss_file:
            pickle.dump(train_loss, train_loss_file)
        with open(path+"train_acc", "wb+") as train_acc_file:
            pickle.dump(train_acc, train_acc_file)
        with open(path+"val_loss", "wb+") as val_loss_file:
            pickle.dump(validation_loss, val_loss_file)



def log_scinol_params(model, writer, step):
    for name, module in model.named_children():
        for p_name, p in module.named_parameters():
            optim_state = module.optim_state[p]
            full_name=name+"."+p_name
            writer.add_histogram(full_name+"_G", optim_state['G'], step)
            writer.add_histogram(full_name+"_S2", optim_state['S2'], step)
            writer.add_histogram(full_name + "_eta", optim_state['eta'], step)
            writer.add_histogram(full_name+"_M", optim_state['M'], step)


def __log_grads(writer, model, step):
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, step)
        if param.grad is not None:
            writer.add_histogram(name + "_grad", param.grad, step)
            writer.add_scalar(name + "_no_non_zero_gradients",
                              torch.count_nonzero(torch.torch.ne(param.grad, torch.tensor(0.))), step)
            writer.add_scalar(name + "_no_non_zero_weights",
                              torch.count_nonzero(torch.torch.ne(param, torch.tensor(0.))), step)
            writer.add_scalar(name + "_no_weights_grater_then_0",
                              torch.count_nonzero(torch.torch.gt(param.grad, torch.tensor(0.))), step)