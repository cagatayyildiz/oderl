import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import MultivariateNormal, Normal, Bernoulli, kl_divergence as kl

from .bnn import get_act

class DropoutBNN(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hid_layers: int=2, act: str='relu', dropout_rate=0.0, \
                        n_hidden: int=100, bias=True, requires_grad=True, layer_norm=False):
        super().__init__()
        self.layers_dim = [n_in] + n_hid_layers*[n_hidden] + [n_out]
        self.weights = nn.ParameterList([])
        self.biases  = nn.ParameterList([])
        self.layer_norms = nn.ModuleList([])
        self.dropout_rate = dropout_rate
        self.acts    = []
        self.bias = bias
        self.act = act 
        for i,(n_in,n_out) in enumerate(zip(self.layers_dim[:-1],self.layers_dim[1:])):
            self.weights.append(Parameter(torch.Tensor(n_in, n_out),requires_grad=requires_grad))
            self.biases.append(None if not bias else Parameter(torch.Tensor(n_out),requires_grad=requires_grad))
            self.acts.append(get_act(act) if i<n_hid_layers else get_act('linear')) # no act. in final layer
            self.layer_norms.append(nn.LayerNorm(n_out) if layer_norm and i<n_hid_layers else nn.Identity())
        self.reset_parameters()

    def reset_parameters(self,gain=1.0):
        for i,(weight,bias) in enumerate(zip(self.weights,self.biases)):
            nn.init.xavier_uniform_(weight,gain)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
        for norm in self.layer_norms[:-1]:
            if isinstance(norm,nn.LayerNorm):
                norm.reset_parameters()

    def sample_weights(self):
        pass

    @property
    def device(self):
        return self.weights[0].device

    def draw_noise(self, L=1):
        dropout_masks = []
        dropout_rate = self.dropout_rate
        b = Bernoulli(1-dropout_rate)
        for h in self.layers_dim[1:-1]:
            dropout_masks.append(b.sample([L,1,h]).to(self.device))
        dropout_masks.append(torch.ones([L,1,self.layers_dim[-1]],device=self.device))
        return dropout_masks

    def draw_f(self, L=1, noise_vec=None):
        dropout_masks = self.draw_noise(L) if noise_vec is None else noise_vec # list of [L,1,h]
        def f(x):
            x2d = x.ndim==2
            if x2d:
                x = torch.stack([x]*L) # [L,N,n]
            for (weight,bias,dropout_mask,act,norm) in \
                zip(self.weights,self.biases,dropout_masks,self.acts,self.layer_norms):
                x = act(norm(dropout_mask*(x@weight + bias)))
            return x.squeeze(0) if x2d and L==1 else x
        return f

    def forward(self, x, L=1):
        return self.draw_f(L,None)(x)

    def __repr__(self):
        str_ = 'DBBB\\dropout rate = {:.2f}\n'.format(self.dropout_rate)
        for i,(weight,act) in enumerate(zip(self.weights,self.acts)):
            str_ += 'Layer-{:d}: '.format(i+1) + ''.join(str([*weight.shape][::-1])) \
                + '\t' + str(act) + '\n'
        return str_
