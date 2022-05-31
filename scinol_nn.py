import torch
from collections import defaultdict


class ScinolLinear(torch.nn.Linear):  # Scinol Linear

    def __init__(self, in_features, out_features, **kwargs):
        #torch.manual_seed(0)
        super().__init__(in_features, out_features, **kwargs)
        self.optim_state = defaultdict(dict)

    def forward(self, curr_input):
        # update max input
        self._update_max_input(curr_input)
        # forward pass
        return super().forward(curr_input)

    def step(self):
        for name, p in self.named_parameters():
            state = self.optim_state[p]
            # first run, key 'M' created during forward pass
            if len(state) == 1:
                state['S2'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
                state['G'] = torch.zeros_like(p, memory_format=torch.preserve_format, requires_grad=False)
                state['eta'] = torch.ones_like(p, memory_format=torch.preserve_format, requires_grad=False)
            # state update
            state['G'].add_(-p.grad)
            state['S2'].add_(p.grad ** 2)
            state['eta'].add_(-p.grad * p)
            # calculate new weight
            denominator = torch.sqrt(state['S2'] + torch.square(state['M']))
            theta = state['G'] / denominator
            clipped_theta = torch.clamp(theta, min=-1, max=1)
            p_new = (clipped_theta / (2 * denominator)) * state['eta']
            # print("P_new d type:",p_new.type() )
            # weight update
            with torch.no_grad():
                p.set_(p_new)

    def _update_max_input(self, curr_input):
        with torch.no_grad():
            abs_input = torch.abs(curr_input)
            if len(abs_input.shape) == 1:
                abs_input.unsqueeze_(0)
            max_abs_input, _ = torch.max(abs_input, 0)
            for name, p in self.named_parameters():
                state = self.optim_state[p]
                # first run, initialize M in state
                if len(state) == 0:
                    if name == 'bias':
                        state['M'] = torch.ones_like(p, memory_format=torch.preserve_format, requires_grad=False)
                    if name == 'weight':
                        state['M'] = torch.zeros(p.shape[1], requires_grad=False)
                # new M is calculated only for weights, for biases it is constant 1
                if name == 'bias':
                    continue
                if name == 'weight':
                    state['M'] = torch.where(max_abs_input > state['M'], max_abs_input, state['M'])