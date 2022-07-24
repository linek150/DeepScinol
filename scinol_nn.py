import typing

import torch
import torch.nn
from torch.nn import ReLU, Module
from collections import defaultdict
from typing import Iterable
from config_enums import EtaInitEnum, MaxInitEnum

class ScinolModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def step(self):
        for p in self.children():
            p.step()


class ScinolLinear(torch.nn.Linear):  # Scinol Linear

    def __init__(self, in_features, out_features, single_layer: bool, eta_init: EtaInitEnum,
                 **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.optim_state = defaultdict(dict)
        self.single_layer = False
        self.next_size = out_features
        self.eta_init_bound = 1/(self.next_size**(1/2))
        self.eta_init = eta_init

    def forward(self, curr_input):
        # update max input
        self._update_max_input(curr_input)
        # update parameters
        self._update_all_parameters()
        # forward pass
        return super().forward(curr_input)

    def step(self):
        with torch.no_grad():
            """update cumulative gradients, cumulative variance and learning rate """
            for p in self.parameters():
                state = self.optim_state[p]
                # state update
                state['G'].add_(-p.grad)
                state['S2'].add_(p.grad ** 2)
                # Limit eta by assumption that function is a-liphnitz
                new_eta = torch.maximum(state['eta']-(p.grad*p), state['eta']*0.5)
                state['eta'].set_(new_eta)


    def _update_all_parameters(self):
        with torch.no_grad():
            for p in self.parameters():
                self._update_parameter(p)

    def _update_parameter(self, p):
        state = self.optim_state[p]
        # calculate new weights
        denominator = torch.sqrt(state['S2'] + torch.square(state['M']))
        theta = state['G']/denominator
        clipped_theta = torch.clamp(theta, min=-1, max=1)
        p_update = (clipped_theta / (2 * denominator)) * state['eta']
        if not self.single_layer:
            # weight is updated only if accumulated gradient is different from 0
            non_zero_gradient_sum = torch.ne(state['S2'], torch.tensor(0.))
            # if G is zero weight is not updated
            p_update = torch.where(non_zero_gradient_sum, p_update, torch.zeros_like(p_update))
        # weights update
        #if "weight" in name:
        #    print(torch.count_nonzero(non_zero_gradient_sum),"/",torch.numel(p_new),"Niezerowe elementy")
        ## TODO: SPRAWDZIĆ TO
        p_new = state['p0'] + p_update
        p.set_(p_new)

    def _init_param_state(self, p, p_name):
        with torch.no_grad():
            state = self.optim_state[p]
            state['p0']=p.detach().clone()
            if p_name == 'bias':
                state['M'] = torch.ones_like(p, memory_format=torch.preserve_format, requires_grad=False)
            if p_name == 'weight':
                state['M'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
            state['S2'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
            state['G'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
            if self.eta_init == EtaInitEnum.UNIFORM_GLOROT:
                state['eta'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
                torch.nn.init.constant_(state['eta'], self.eta_init_bound)
            if self.eta_init == EtaInitEnum.ONES:
                state['eta'] = torch.ones_like(p, memory_format=torch.preserve_format, requires_grad=False)

    def _update_max_input(self, curr_input):
        with torch.no_grad():
            ## TODO: SPRAWDZIĆ TO
            ## DO ZROBIENIA poprawne obliczanie maksa z średniej wartości maksymalnej z batcha
            abs_input = torch.abs(curr_input)
            # mean reduction for loss function for mini-batch
            mean_abs_input = torch.mean(abs_input, 0)
            #max_abs_input, _ = torch.max(abs_input, 0)
            for p_name, p in self.named_parameters():
                state = self.optim_state[p]
                # first run, initialize state for p in state
                if len(state) == 0:
                    self._init_param_state(p, p_name)
                # new M is calculated only for weights, for biases it is constant 1
                if p_name == 'bias':
                    continue
                if p_name == 'weight':
                    state['M'] = torch.maximum(state['M'], mean_abs_input)


class ScinolMLP(ScinolModule):
    def __init__(self, no_inputs: int, no_outputs: int, hidden_layer_sizes: Iterable, activation: Module = ReLU,
                 eta_init: EtaInitEnum = EtaInitEnum.UNIFORM_GLOROT):
        super().__init__()
        self.activation = (activation(),)  # to prevent from listing as module
        single_layer = False
        prev_layer_size = -1
        for idx, layer_size in enumerate(hidden_layer_sizes):
            if idx == 0:
                self.add_module('input_layer', ScinolLinear(no_inputs, layer_size, single_layer, eta_init))
            else:
                self.add_module(f'{idx}_hidden_layer', ScinolLinear(prev_layer_size, layer_size, single_layer, eta_init))
            prev_layer_size = layer_size
        self.add_module('output_layer', ScinolLinear(prev_layer_size, no_outputs, single_layer, eta_init))

    def forward(self, x):
        layers = list(self.children())
        for idx, layer in enumerate(layers):
            x = layer(x)
            if idx != len(layers)-1:
                x = self.activation[0](x)
        return x


class MNIST_CNN(ScinolModule):
    def __init__(self, eta_init: EtaInitEnum, activation: Module = ReLU):
        super().__init__()
        no_filters=64
        self.optim_prefix = "Scinol"
        self.add_module(f"{self.optim_prefix}CNN_1", Conv2d(1, no_filters, 5))
        self.add_module(f"{activation.__name__}1", activation())
        self.add_module("Max_pooling_1", torch.nn.MaxPool2d(2))
        self.add_module(f"{self.optim_prefix}CNN_2", Conv2d(no_filters, no_filters, 5))
        self.add_module(f"{activation.__name__}2", activation())
        self.add_module("Max_pooling_2", torch.nn.MaxPool2d(2))
        self.add_module("Flatten",torch.nn.Flatten())
        self.add_module(f"{self.optim_prefix}Linear", ScinolLinear(1024, 10, False, eta_init))

    def forward(self, x):
        layers = list(self.children())
        for idx, layer in enumerate(layers):
            x = layer(x)
        return x

    def step(self):
        for name,p in self.named_children():
            if self.optim_prefix in name:
                p.step()


class Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', device=None, dtype=None):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                     dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
                                     device=dtype, dtype=dtype)
        self.optim_state = defaultdict(dict)

    def forward(self, curr_input):
        # update max input
        self._update_max_input(curr_input)
        # update parameters
        self._update_all_parameters()
        # forward pass
        return super(Conv2d, self).forward(curr_input)

    def _update_max_input(self, curr_input):
        """
        :param curr_input: (N,C,H,W)
        :return:
        """
        assert len(curr_input.shape) == 4, "Input tensor should have dims (N,C,H,W)"
        with torch.no_grad():
            abs_input = torch.abs(curr_input)
            # two following lines take max_abs from every input tensor in batch, result dims=(N,C)
            max_abs_rows, _ = torch.max(abs_input, -1)
            max_abs_cols, _ = torch.max(max_abs_rows, -1)
            # take mean of abs_max values for every channel over mini-batch, result dims=(N)
            mean_max_abs_channel = torch.mean(max_abs_cols, 0)
            for p_name, p in self.named_parameters():
                state = self.optim_state[p]
                # first run, initialize state for p
                if len(state) == 0:
                    self._init_param_state(p, p_name)
                # new M is calculated only for weights, for biases it is constant 1
                if p_name == 'bias':
                    continue
                if p_name == 'weight':
                    state['M'] = torch.maximum(mean_max_abs_channel, state['M'])

    def _init_param_state(self, p, p_name):
        with torch.no_grad():
            state = self.optim_state[p]
            state['p0']=p.detach().clone()
            if p_name == 'bias':
                state['M'] = torch.ones(1, device=p.device, requires_grad=False)
            if p_name == 'weight':
                # shape of M is number of input channels
                state['M'] = torch.zeros(p.shape[1], device=p.device, requires_grad=False)
            state['S2'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
            state['G'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
            # TODO ETA INIT FOR CONV
            state['eta'] = torch.ones_like(p, memory_format=torch.preserve_format, requires_grad=False)

    def step(self):
        with torch.no_grad():
            """update cumulative gradients, cumulative variance and learning rate """
            for p in self.parameters():
                state = self.optim_state[p]
                # state update
                state['G'].add_(-p.grad)
                state['S2'].add_(p.grad ** 2)
                new_eta= torch.maximum(state['eta']-(p.grad*p), state['eta']*0.5)
                state['eta'].set_(new_eta)

    def _update_all_parameters(self):
        with torch.no_grad():
            for p_name,p in self.named_parameters():
                self._update_parameter(p,p_name)

    def _update_parameter(self, p, p_name):
        state = self.optim_state[p]
        # calculate new weights
        # TODO SPRWDZIĆ TO
        if p_name == 'weight':
            #dims(p)=(C_input,C_output,kernelW,kernelH)
            #dims(state['M'])=(C_input)
            expanded = state['M'].expand(p.shape[0], p.shape[2], p.shape[3], p.shape[1])
            M = expanded.permute(0, 3, 1, 2)
        else:
            M = state['M']
        denominator = torch.sqrt(state['S2'] + torch.square(M))
        theta = state['G']/denominator
        clipped_theta = torch.clamp(theta, min=-1, max=1)
        p_update = (clipped_theta / (2 * denominator)) * state['eta']
        # weight is updated only if accumulated gradient is different from 0, to prevent disabling parts of net
        non_zero_gradient_sum = torch.ne(state['G'], torch.tensor(0.))
        # if G is zero weight is not updated
        p_update = torch.where(non_zero_gradient_sum, p_update, torch.zeros_like(p_update))
        # weights update
        p_new = state['p0'] + p_update
        p.set_(p_new)

