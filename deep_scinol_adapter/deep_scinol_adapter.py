import torch
from deep_scinol_adapter.m_updates import m_update_dict
from deep_scinol_adapter.eta_init import eta_init_dict
from .common import *

def adapt_to_scinol(module, eta_init_value, momentum:bool):
    """
      Find all modules without submodules. Add max_buffer, forward_pre_hook,
      and fit method.
    """
    # TODO: find better way to add step method to modules
    if not hasattr(module, 'step'):
        step_method = get_scinol_step(momentum)
        torch.nn.Module.step = step_method
    # TODO: consider situation when module that has submodules also has parameters
    # module has no submodules, end of recursion, adapt this module
    if sum(1 for _ in module.children()) == 0:
        # module doesn't have hook yet and has parameters to optimize
        if not has_scinol_hook(module) and sum(1 for _ in module.parameters()) > 0:
            add_scinol_buffers(module, eta_init_value)
            module.register_forward_pre_hook(scinol_pre_hook)
        return module
    else:  # module has submodules
        # check children modules
        for child_module in module.children():
            adapt_to_scinol(child_module, eta_init_value, momentum)

    return module

def get_scinol_step(momentum:bool):
    def _scinol_step(self):
        step(self,momentum)
    return _scinol_step


@torch.no_grad()
def scinol_pre_hook(module, curr_input):
    update_m(module, curr_input)
    update_module_params(module)


def update_m(module, curr_input):
    # in general curr_input is tuple with one entry, check for other possibilities
    assert type(curr_input) == tuple and len(curr_input) == 1
    m_update = m_update_dict[type(module)]
    m_update(module, curr_input[0])


@torch.no_grad()
def update_module_params(module):
    for p_name, p in module.named_parameters():
        state = get_param_state(module, p_name)
        denominator = torch.sqrt(state[S2] + torch.square(state[M]))
        theta = state[G] / denominator
        clipped_theta = torch.clamp(theta, min=-1, max=1)
        p_update = (clipped_theta / (2 * denominator)) * state[ETA]
        # parameter is updated only if accumulated uncentered variance is different from 0
        non_zero_gradient_sum = torch.ne(state[S2], torch.tensor(0.))
        # if G is zero weight is not updated
        p_update = torch.where(non_zero_gradient_sum, p_update, torch.zeros_like(p_update))
        # weights update
        p_new = state[P0] + p_update
        p.set_(p_new)


def has_scinol_hook(module):
    for k, v in module._forward_pre_hooks.items():
        if v == scinol_pre_hook:
            return True
    return False


def add_scinol_buffers(module, eta_init_value):
    module.register_buffer(T, torch.tensor(1))
    for p_name, p in module.named_parameters():
        add_max_buffer(module, p_name, p)

        s2=torch.zeros_like(p)
        module.register_buffer(p_name + S2, s2)

        g=torch.zeros_like(p)
        module.register_buffer(p_name + G, g)
        #ETA INIT
        if str(eta_init_value) == "glorot":
            eta=get_eta_init(module, p, p_name)
        elif isinstance(eta_init_value, float) or isinstance(eta_init_value, int):
            eta=torch.full_like(p, eta_init_value)
        else:
            raise NotImplementedError("Pass float or int or 'glorot' as eta_init_method param.")
        module.register_buffer(p_name + ETA, eta)

        p0=p.detach().clone()
        module.register_buffer(p_name + P0, p0)

# TODO: make this function diferent for every module type to limit model size,
#  it will require to specifie update_param function for every module type
def add_max_buffer(module,p_name,p):
    if "weight" in p_name:
        buffer = torch.zeros_like(p, requires_grad=False)
        module.register_buffer(p_name+M, buffer)
    elif "bias" in p_name:
        buffer = torch.ones_like(p, requires_grad=False)
        module.register_buffer(p_name+M, buffer)
    else:
        raise NotImplementedError("Not supported param type")

B1=0.9
B1_complement=1-B1
B2=0.999
B2_complement=1-B2
@torch.no_grad()
def step(self, momentum):
    if sum(1 for _ in self.children()) == 0 and sum(1 for _ in self.parameters()) != 0:
        # update cumulative gradients, cumulative variance and learning rate
        state=None
        for p_name, p in self.named_parameters():
            state = get_param_state(self, p_name)
            # state update
            if momentum:
                first_moment=B1*state[G]+B1_complement*(-p.grad)
                corrected_first_moment=first_moment/(1-B1**state[T])
                second_moment=B2*state[S2]+B2_complement*(torch.square(p.grad))
                corrected_second_moment=second_moment/(1-B2**state[T])
                state[G].set_(corrected_first_moment)
                state[S2].set_(corrected_second_moment)
            else:
                state[G].add_(-p.grad)
                state[S2].add_(p.grad ** 2)
            delta_p = p-state[P0]
            new_eta = torch.maximum(state[ETA]-(p.grad*delta_p), state[ETA]*0.5)
            state[ETA].set_(new_eta)
        state[T].add_(torch.tensor(1))
        return
    for child in self.children():
        # TODO napisa lepszy warunek na test sytuacji że moduł ma parametry i submoduly
        child.step()



def get_eta_init(module, param, param_name):
    eta_init_fun = eta_init_dict[type(module)]
    return eta_init_fun(module, param)
