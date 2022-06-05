from torch.optim import Optimizer as _Optimizer
import torch
from torch.optim import _functional as F
import numpy as np


class Scinol2Dl(_Optimizer):
    def __init__(self, params,
                 epsilon=1.0,
                 s0=1e-7,
                 max_start=1.0,
                 epsilon_scaled=False,
                 clip_grad=True,
                 clip_eta=True,
                 use_updated_max=False,
                 epsilon_is_max=False, **kwargs):
        self.epsilon_is_max = epsilon_is_max
        self.epsilon_start = epsilon
        self.s0 = s0
        self.max_start = max_start
        self.epsilon_scaled = epsilon_scaled
        self.clip_grad = clip_grad
        self.clip_eta = clip_eta
        self.use_updated_max = use_updated_max
        # TODO lol wtf are there groups, this whole design looks like somebody wrote it while drunk
        defaults = dict()
        super(Scinol2Dl, self).__init__(params, defaults)

    # TODO _setstate__ is overriden in different implementations, check wtf it is
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'Scinol2Dl does not support sparse gradients.')

                    state = self.state[p]
                    # Lazy state initialization
                    # TODO check if it makes any sense that init is lazy
                    if len(state) == 0:
                        # TODO step can be a group probably?
                        # sparować maxa z epsilonem?
                        state['step'] = 0
                        state['grads_sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['squared_grads_sum'] = torch.full_like(p, self.s0, memory_format=torch.preserve_format)

                        with torch.no_grad():
                            state['initial_value'] = torch.clone(p, memory_format=torch.preserve_format)

                        if self.epsilon_scaled and len(p.shape) > 1:
                            with torch.no_grad():
                                state['epsilon'] = torch.clone(p.abs(), memory_format=torch.preserve_format)
                                # state['epsilon'] = torch.clone(p**2, memory_format=torch.preserve_format)
                                # state['epsilon'] = torch.full_like(p, 1/np.product(p.shape), memory_format=torch.preserve_format)
                                # state['epsilon'] = torch.full_like(p, 1/(fin + fout), memory_format=torch.preserve_format)
                                # state['epsilon'] = torch.full_like(p, (2*2/(fin+fout))**0.5, memory_format=torch.preserve_format)
                                # state['epsilon'] = torch.full_like(p, (2*2/(fin+fout)), memory_format=torch.preserve_format)
                                # state['epsilon'] = torch.full_like(p, (2 * 6 / (fin + fout)) ** 0.5,
                                #                                    memory_format=torch.preserve_format)
                                # state['epsilon'] = torch.full_like(p, (2*6/(fin+fout)), memory_format=torch.preserve_format)
                        else:
                            state['epsilon'] = torch.full_like(p, self.epsilon_start,
                                                               memory_format=torch.preserve_format)
                        if self.epsilon_is_max:
                            state['max'] = torch.clone(state['epsilon'], memory_format=torch.preserve_format)
                        else:
                            state['max'] = torch.full_like(p, self.max_start, memory_format=torch.preserve_format)
                        state['eta'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['step'] += 1

                    g = p.grad
                    eta = state["eta"]
                    epsilon = state["epsilon"]
                    G = state["grads_sum"]
                    S2 = state["squared_grads_sum"]
                    M = state["max"]
                    var0 = state["initial_value"]

                    old_g = g
                    if self.clip_grad:
                        g = torch.maximum(-M, torch.minimum(g, M))

                    if self.use_updated_max:
                        torch.maximum(M, old_g.abs(), out=state["max"])
                    if self.epsilon_is_max:
                        state["epsilon"].set_(M)

                    theta = G / (S2 + M ** 2) ** 0.5
                    var_delta = torch.sign(theta) * torch.clamp(theta.abs(), max=1.0) / (
                            2 * (S2 + M ** 2) ** 0.5) * (eta + epsilon)
                    # TODO keep epsilon separate and dependant on max

                    state["grads_sum"].add_(-g)
                    state["squared_grads_sum"].add_(g ** 2)
                    if not self.use_updated_max:
                        torch.maximum(M, old_g.abs(), out=state["max"])
                    new_eta = state["eta"] - g * var_delta
                    if self.clip_eta:
                        new_eta = torch.maximum(0.5 * (state["eta"] + epsilon), new_eta + epsilon) - epsilon

                    state["eta"].set_(new_eta)

                    p.set_(var0 + var_delta)
            # for default implementations this function is somewhere else for some reason
            # and multiple lists are stored and used there ... for some convection reasons apparently
            # so.... TODO implement it as they do and check if it changes anything

        return loss
