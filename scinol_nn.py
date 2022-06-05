import torch
from collections import defaultdict
from torch.nn import functional




class ScinolModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def step(self):
        for p in self.children():
            p.step()


class ScinolLinear(torch.nn.Linear):  # Scinol Linear

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.optim_state = defaultdict(dict)

    def forward(self, curr_input):
        # update max input
        self._update_max_input(curr_input)
        # update parameters
        self._update_all_parameters()
        # forward pass
        return super().forward(curr_input)

    def step(self):
        """update cumulative gradients, cumulative variance and learning rate """
        for p in self.parameters():
            state = self.optim_state[p]
            # state update
            state['G'].add_(-p.grad)
            state['S2'].add_(p.grad ** 2)
            state['eta'].add_(-p.grad * p)

    def _update_all_parameters(self):
        for p in self.parameters():
            self._update_parameter(p)

    def _update_parameter(self, p):
        state = self.optim_state[p]
        # calculate new weights
        denominator = torch.sqrt(state['S2'] + torch.square(state['M']))
        non_zero_denom = torch.ne(denominator, torch.tensor(0.))
        theta = state['G']/denominator
        clipped_theta = torch.clamp(theta, min=-1, max=1)
        p_new = (clipped_theta / (2 * denominator)) * state['eta']
        # new weight is equal to 0 if denominator is 0
        p_new = torch.where(non_zero_denom, p_new, torch.tensor([0.]))

        # weights update
        with torch.no_grad():
            p.set_(p_new)

    def _init_param_state(self, p, p_name):
        state = self.optim_state[p]
        if p_name == 'bias':
            state['M'] = torch.ones_like(p, memory_format=torch.preserve_format, requires_grad=False)
        if p_name == 'weight':
            state['M'] = torch.zeros(p.shape[1], requires_grad=False)
        state['S2'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
        state['G'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
        state['eta'] = torch.ones_like(p, memory_format=torch.preserve_format, requires_grad=False)

    def _update_max_input(self, curr_input):
        with torch.no_grad():
            abs_input = torch.abs(curr_input)
            if len(abs_input.shape) == 1:
                abs_input.unsqueeze_(0)
            max_abs_input, _ = torch.max(abs_input, 0)
            for p_name, p in self.named_parameters():
                state = self.optim_state[p]
                # first run, initialize M in state
                if len(state) == 0:
                    self._init_param_state(p, p_name)
                # new M is calculated only for weights, for biases it is constant 1
                if p_name == 'bias':
                    continue
                if p_name == 'weight':
                    state['M'] = torch.where(max_abs_input > state['M'], max_abs_input, state['M'])


class ScinolMLP(ScinolModule):
    def __init__(self, no_inputs, no_outputs):
        super().__init__()
        hiden=1
        self.l1 = ScinolLinear(no_inputs, hiden)
        self.l2 = ScinolLinear(hiden, no_outputs)

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)
        return x