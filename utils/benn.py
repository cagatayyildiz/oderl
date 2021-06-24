import torch
import numpy as np

import torch.nn as nn
from torch.nn.parameter import Parameter
from .utils import get_act


class BENN(nn.Module):
    def __init__(self, n_ens: int, n_in: int, n_out: int, n_hid_layers: int=2, n_hidden: int=250, \
                        act: str='relu', requires_grad=True, bias=True, layer_norm=False, skip_con=False):
        super().__init__()
        layers_dim = [n_in] + n_hid_layers*[n_hidden] + [n_out]
        self.n_ens = n_ens
        self.skip_con = skip_con
        self.act = act 
        self.bias = bias
        self.acts  = []
        self.weights,self.biases = nn.ParameterList([]),nn.ParameterList([])
        self.rs,self.ss = nn.ParameterList([]),nn.ParameterList([])
        for i,(n_in,n_out) in enumerate(zip(layers_dim[:-1],layers_dim[1:])):
            self.weights.append(Parameter(torch.Tensor(n_in, n_out),requires_grad=requires_grad))
            self.biases.append(None if not bias else Parameter(torch.Tensor(1,n_out),requires_grad=requires_grad))
            self.acts.append(get_act(act) if i<n_hid_layers else get_act('linear')) # no act. in final layer
            self.rs.append(Parameter(torch.Tensor(n_ens,1,n_in),requires_grad=requires_grad)) # Nens,1,n
            self.ss.append(Parameter(torch.Tensor(n_ens,1,n_out),requires_grad=requires_grad)) # Nens,1,n
        self.reset_parameters()
    
    def shuffle(self):
        rand_idx = torch.randperm(self.n_ens)
        for r,s in zip(self.rs,self.ss):
            r.data = r.data[rand_idx]
            s.data = s.data[rand_idx]

    @property
    def device(self):
        return self.weights[0].device

    def __transform_sig(self,sig):
        # return F.softplus(sig)
        return sig.exp() + 1e-6

    def reset_parameters(self,gain=1.0):
        for i,(weight,bias) in enumerate(zip(self.weights,self.biases)):
            nn.init.xavier_uniform_(weight,gain)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / np.sqrt(fan_in)
            if self.bias:
                nn.init.uniform_(bias, -bound, bound)
        for r,s in zip(self.rs,self.ss):
            nn.init.normal_(r, 1.0, 0.25)
            nn.init.normal_(s, 1.0, 0.25)

    def draw_noise(self, **kwargs):
        return None

    def draw_f(self, L=1, noise_vec=None):
        """ Draws L//n_ens samples from each ensemble component
            Assigns each x[i] to a different sample in a different component
            x  -     [L,N,n]
            output - [L,N,n]
        """
        def f(x):
            for (r,s,weight,bias,act) in zip(self.rs,self.ss,self.weights,self.biases,self.acts):
                x_ = (x*r)@weight + bias
                x_ = x_+x if x.shape==x_.shape and self.skip_con else x_
                x  = act(x_*s)
            return x
        return f

    def forward(self, x, L=1):
        return self.draw_f()(x)

    def kl(self):
        return torch.zeros(1).to(self.device)

    def __repr__(self):
        str_ = f'BENN - {self.n_ens} members\n'
        for i,(weight,act) in enumerate(zip(self.weights,self.acts)):
            str_ += 'Layer-{:d}: '.format(i+1) + ''.join(str([*weight.shape][::-1])) \
                + '\t' + str(act) + '\n'
        return str_