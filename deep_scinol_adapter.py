
import torch
import types
M = "scinol_M"
S2 = "scinol_S2"
G = "scinol_G"
ETA = "scinol_eta"
P0 = "scinol_p0"

def adapt_to_scinol(module):
    """
      Find all modules without submodules. Add max_buffer, forward_pre_hook,
      and fit method.
    """
    # module has no submodules, end of recursion, adapt this module
    if sum(1 for _ in module.children()) == 0:
        # module doesn't have hook yet and has parameters to optimize
        if not has_scinol_hook(module) and sum(1 for _ in module.parameters()) > 0:
            add_scinol_buffers(module)
            module.register_forward_pre_hook(scinol_pre_hook)
        return
    # check children modules
    for child_module in module.children():
        adapt_to_scinol(child_module)
    # TODO jakoś sensowniej dodać tę funkcję do Module
    if not hasattr(module, 'step'):
        torch.nn.Module.step = step
    return module


@torch.no_grad()
def scinol_pre_hook(module, curr_input):
    update_m(module, curr_input)
    update_module_params(module)

def update_m(module, curr_input):
    assert type(curr_input) == tuple
    # TODO WERJSA TYLKO DLA LINIOWEGO MODULU LINIOWEGO, DODAC INNE
    abs_input = torch.abs(curr_input[0])
    # mean reduction for loss function for mini-batch
    mean_abs_input = torch.mean(abs_input, 0)
    for p_name, p in module.named_parameters():
        state = get_param_state(module, p_name)
        if p_name == 'bias':
            continue
        if p_name == 'weight':
            state[M].set_(torch.maximum(state[M], mean_abs_input))

@torch.no_grad()
def update_module_params(module):
    for p_name, p in module.named_parameters():
        state = get_param_state(module, p_name)
        denominator = torch.sqrt(state[S2] + torch.square(state[M]))
        theta = state[G] / denominator
        clipped_theta = torch.clamp(theta, min=-1, max=1)
        p_update = (clipped_theta / (2 * denominator)) * state[ETA]
        # weight is updated only if accumulated gradient is different from 0, to prevent disabling parts of net
        non_zero_gradient_sum = torch.ne(state[G], torch.tensor(0.))
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


def add_scinol_buffers(module):
    for p_name, p in module.named_parameters():
        add_max_buffer(module, p_name, p)

        s2=torch.zeros_like(p)
        module.register_buffer(p_name + S2, s2)

        g=torch.zeros_like(p)
        module.register_buffer(p_name + G, g)

        eta=torch.ones_like(p)
        module.register_buffer(p_name + ETA, eta)

        p0=p.detach().clone()
        module.register_buffer(p_name + P0, p0)


def add_max_buffer(module,p_name,p):
    if "weight" in p_name:
        buffer = torch.zeros_like(p, requires_grad=False)
        module.register_buffer(p_name+M, buffer)
    elif "bias" in p_name:
        buffer = torch.ones_like(p, requires_grad=False)
        module.register_buffer(p_name+M, buffer)
    else:
        raise NotImplementedError("Not supported param type")


def get_param_state(module, p_name):
    p_state = dict()
    p_state[G] = module.get_buffer(p_name + G)
    p_state[S2] = module.get_buffer(p_name + S2)
    p_state[M] = module.get_buffer(p_name + M)
    p_state[ETA] = module.get_buffer(p_name + ETA)
    p_state[P0] = module.get_buffer(p_name + P0)
    return p_state

@torch.no_grad()
def step(self):
    if sum(1 for _ in self.children()) == 0:
        # update cumulative gradients, cumulative variance and learning rate
        for p_name,p in self.named_parameters():
            state = get_param_state(self, p_name)
            # state update
            state[G].add_(-p.grad)
            state[S2].add_(p.grad ** 2)
            new_eta = torch.maximum(state[ETA]-(p.grad*p), state[ETA]*0.5)
            state[ETA].set_(new_eta)
        return
    for child in self.children():
        # TODO napisa lepszy warunek na test sytuacji że moduł ma parametry i submoduly
        child.step()
