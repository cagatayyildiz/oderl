import torch
import numpy as np
from abc import ABCMeta, abstractmethod

import torch.nn as nn
from torch.nn.parameter import Parameter
from .utils import get_act


class ENN_BASE(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, n_ens, layers_ins, layers_outs, n_hid_layers=2, act='celu', dropout=0.0, skip_con=False,\
                        n_hidden=100, requires_grad=True, logsig0=-3, layer_norm=False):
        super().__init__()
        self.n_ens = n_ens
        self.weights  = nn.ParameterList([])
        self.biases    = nn.ParameterList([])
        self.layer_norms = nn.ModuleList([])
        self.dropout_rate = dropout
        self.skip_con = skip_con
        self.dropout = nn.Dropout(dropout)
        self.acts    = []
        for i,(n_in,n_out) in enumerate(zip(layers_ins,layers_outs)):
            self.weights.append(Parameter(torch.Tensor(n_ens, n_in, n_out), requires_grad=requires_grad))
            self.biases.append(Parameter(torch.Tensor(n_ens, 1, n_out), requires_grad=requires_grad))
            self.acts.append(get_act(act) if i<n_hid_layers else get_act('linear')) # no act. in final layer
            self.layer_norms.append(nn.LayerNorm(n_out, elementwise_affine=False) \
                if layer_norm and i<n_hid_layers else nn.Identity())
        self.reset_parameters()

    @property
    def device(self):
        return self.weights[0].device

    def reset_parameters(self,gain=1.0):
        for i,(weight,bias) in enumerate(zip(self.weights,self.biases)):
            for w,b in zip(weight,bias):
                nn.init.xavier_uniform_(w,gain)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(b, -bound, bound)
        for norm in self.layer_norms[:-1]:
            if isinstance(norm,nn.LayerNorm):
                norm.reset_parameters()

    def kl(self):
        return torch.zeros(1).to(self.device)

    def set_num_particles(self,N):
        self.n_ens = N
        weights_new = [Parameter(weight[:N]) for weight in self.weights]
        biases_new   = [Parameter(bias[:N])   for bias in self.biases]
        del self.weights
        del self.biases
        self.weights = nn.ParameterList(weights_new)
        self.biases = nn.ParameterList(biases_new)

    def shuffle(self):
        rand_idx = torch.randperm(self.n_ens)
        for w,b in zip(self.weights,self.biases):
            w.data = w.data[rand_idx] 
            b.data = b.data[rand_idx]
                
    def name(self):
        str_ = ''
        for i,(weight,act) in enumerate(zip(self.weights,self.acts)):
            str_ += 'Layer-{:d}: '.format(i+1) + ''.join(str([*weight.shape][::-1])) \
                + '\t' + str(act) + '\n'
        return str_
    
    def draw_noise(self, **kwargs):
        return None
        
    def forward(self, x):
        return self.draw_f()(x)
    
    @abstractmethod
    def draw_f(self):
        raise NotImplementedError


class ENN(ENN_BASE):
    def __init__(self, n_ens, n_in, n_out, n_hid_layers=2, act='relu', dropout=0.0, skip_con=False, \
                        n_hidden=100, requires_grad=True, logsig0=-3, layer_norm=False):
        layers_ins  = [n_in] + n_hid_layers*[n_hidden]
        layers_outs = n_hid_layers*[n_hidden] + [n_out]
        super().__init__(n_ens, layers_ins, layers_outs, n_hid_layers=n_hid_layers, \
                          skip_con=skip_con, act=act, dropout=dropout, n_hidden=n_hidden, \
                          requires_grad=requires_grad, logsig0=logsig0, layer_norm=layer_norm)

    def draw_f(self, **kwargs):
        ''' Returns 2D if input is 2D '''
        def f(x): # input/output is [Nens,N,nin] or [N,nin]
            x2d = x.ndim==2
            x = torch.stack([x]*self.n_ens) if x2d else x
            for (W,b,act,norm) in zip(self.weights,self.biases,self.acts,self.layer_norms):
                x_ = self.dropout(torch.baddbmm(b, x, W))
                x_ = x_+x if x.shape==x_.shape and self.skip_con else x_
                x  = norm(act(x_)) # Nens,1,nout & Nens,N,nin & Nens,nin,nout
            return x.mean(0) if x2d else x
        return f

    def __repr__(self):
        super_name = super().name()
        return f'ENN - {self.n_ens} members\n' + super_name


class EPNN(ENN):
    def __init__(self, n_ens, n_in, n_out, n_hid_layers=2, act='relu', dropout=0.0, skip_con=False, \
                        n_hidden=100, requires_grad=True, logsig0=-3, layer_norm=False):
        super().__init__(n_ens, n_in, 2*n_out, n_hid_layers=n_hid_layers, act=act, dropout=dropout, \
                         skip_con=skip_con, n_hidden=n_hidden, requires_grad=requires_grad, \
                         logsig0=logsig0,layer_norm=layer_norm)
        self.n_out = n_out
        self.sp = nn.Softplus()
        self.max_logsig = nn.Parameter(torch.ones([n_out]), requires_grad=requires_grad)
        self.min_logsig = nn.Parameter(-2*torch.ones([n_out]), requires_grad=requires_grad)

    def get_probs(self,x):
        x2d = x.ndim==2
        x = torch.stack([x]*self.n_ens) if x2d else x
        for (W,b,act,norm) in zip(self.weights,self.biases,self.acts,self.layer_norms):
            x_ = self.dropout(torch.baddbmm(b, x, W))
            x_ = x_+x if x.shape==x_.shape and self.skip_con else x_
            x  = norm(act(x_)) # Nens,1,2nout & Nens,N,nin & Nens,nin,2nout
        x = x.mean(0) if x2d else x # ...,2nout
        mean,logvar = x[...,:self.n_out],x[...,self.n_out:]
        logvar = self.max_logsig - self.sp(self.max_logsig - logvar)
        logvar = self.min_logsig + self.sp(logvar - self.min_logsig)
        return mean, logvar.exp()

    def draw_f(self, **kwargs):
        ''' Returns 2D if input is 2D '''
        def f(x): # input/output is [Nens,N,nin] or [N,nin]
            mean,sig = self.get_probs(x)
            return mean + torch.randn_like(sig)*sig
        return f
    
    def __repr__(self):
        super_name = super().name()
        return f'EPNN - {self.n_ens} members\n' + super_name


