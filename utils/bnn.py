import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import MultivariateNormal, Normal, Bernoulli, kl_divergence as kl
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from .utils import get_act

class BNN(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hid_layers: int=2, n_hidden: int=100, act: str='relu', \
                        dropout=0.0, requires_grad=True, logsig0=-3, bnn=True, layer_norm=False,\
                        batch_norm=False, bias=True, var_apr='mf'):
        super().__init__()
        layers_dim = [n_in] + n_hid_layers*[n_hidden] + [n_out]
        assert not (layer_norm and batch_norm), 'Either layer_norm or batch_norm should be True'
        self.weight_mus  = nn.ParameterList([])
        self.bias_mus    = nn.ParameterList([])
        self.norms = nn.ModuleList([])
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.acts    = []
        self.act = act 
        self.bnn = bnn
        self.bias = bias
        self.var_apr = var_apr
        for i,(n_in,n_out) in enumerate(zip(layers_dim[:-1],layers_dim[1:])):
            self.weight_mus.append(Parameter(torch.Tensor(n_in, n_out),requires_grad=requires_grad))
            self.bias_mus.append(None if not bias else Parameter(torch.Tensor(1,n_out),requires_grad=requires_grad))
            self.acts.append(get_act(act) if i<n_hid_layers else get_act('linear')) # no act. in final layer
            norm = nn.Identity()
            if i < n_hid_layers:
                if layer_norm:
                    norm = nn.LayerNorm(n_out)
                elif batch_norm:
                    norm = nn.BatchNorm1d(n_out)
            self.norms.append(norm)
        if bnn:
            self.weight_logsigs = nn.ParameterList([])
            self.bias_logsigs   = nn.ParameterList([])
            self.logsig0 = logsig0
            for i,(n_in,n_out) in enumerate(zip(layers_dim[:-1],layers_dim[1:])):
                self.weight_logsigs.append(Parameter(torch.Tensor(n_in, n_out),requires_grad=requires_grad))
                self.bias_logsigs.append(None if not bias else Parameter(torch.Tensor(1,n_out),requires_grad=requires_grad))
        self.reset_parameters()

    @property
    def device(self):
        return self.weight_mus[0].device

    def __transform_sig(self,sig):
        return torch.log(1 + torch.exp(sig))
        # return sig.exp()

    def reset_parameters(self,gain=1.0):
        for i,(weight,bias) in enumerate(zip(self.weight_mus,self.bias_mus)):
            nn.init.xavier_uniform_(weight,gain)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / np.sqrt(fan_in)
            if self.bias:
                nn.init.uniform_(bias, -bound, bound)
        for norm in self.norms[:-1]:
            if isinstance(norm,nn.LayerNorm):
                norm.reset_parameters()
        if self.bnn:
            for w,b in zip(self.weight_logsigs,self.bias_logsigs):
                nn.init.uniform_(w,self.logsig0-1,self.logsig0+1)
                if self.bias:
                    nn.init.uniform_(b,self.logsig0-1,self.logsig0+1)

    def draw_noise(self, L):
        P = parameters_to_vector(self.parameters()).numel() // 2 # single noise term needed per (mean,var) pair
        noise = torch.randn([L,P],device=self.weight_mus[0].device)
        if self.var_apr == 'mf':
            return noise
        elif self.var_apr == 'radial':
            noise /= noise.norm(dim=1,keepdim=True)
            r = torch.randn([L,1], device=self.weight_mus[0].device)
            return noise * r

    def __sample_weights(self, L, noise_vec=None):
        if self.bnn:
            if noise_vec is None:
                noise_vec = self.draw_noise(L) # L,P
            weights = []
            i = 0
            for weight_mu,weight_sig in zip(self.weight_mus,self.weight_logsigs):
                p = weight_mu.numel()
                weights.append( weight_mu + noise_vec[:,i:i+p].view(L,*weight_mu.shape)*self.__transform_sig(weight_sig) )
                i += p
            if self.bias:
                biases = []
                for bias_mu,bias_sig in zip(self.bias_mus,self.bias_logsigs):
                    p = bias_mu.numel()
                    biases.append( bias_mu + noise_vec[:,i:i+p].view(L,*bias_mu.shape)*self.__transform_sig(bias_sig) )
                    i += p
            else:
                biases = [torch.zeros([L,1,weight_mu.shape[1]],device=weight_mu.device)*1.0 \
                    for weight_mu,bias_mu in zip(self.weight_mus,self.bias_mus)] # list of zeros
        else:
            raise ValueError('This is a NN, not a BNN!')
        return weights,biases

    def draw_f(self, L=1, noise_vec=None):
        """ 
            x=[N,n] & bnn=False ---> out=[N,n]
            x=[N,n] & L=1 ---> out=[N,n]
            x=[N,n] & L>1 ---> out=[L,N,n]
            x=[L,N,n] -------> out=[L,N,n]
        """
        if not self.bnn:
            def f(x):
                for (weight,bias,act,norm) in zip(self.weight_mus,self.bias_mus,self.acts,self.norms):
                    x = act(norm(self.dropout(F.linear(x,weight.T,bias))))
                return x
            return f
        else:
            weights,biases = self.__sample_weights(L, noise_vec)
            def f(x):
                x2d = x.ndim==2
                if x2d:
                    x = torch.stack([x]*L) # [L,N,n]
                for (weight,bias,act,norm) in zip(weights,biases,self.acts,self.norms):
                    x = act(norm(self.dropout(torch.baddbmm(bias, x, weight))))
                return x.squeeze(0) if x2d and L==1 else x
            return f

    def forward(self, x, L=1):
        return self.draw_f(L)(x)

    def kl(self,L=100):
        if not self.bnn:
            return torch.zeros([1],device=self.device)*1.0
        if self.var_apr == 'mf':
            mus = [weight_mu.view([-1]) for weight_mu in self.weight_mus]
            logsigs = [weight_logsig.view([-1]) for weight_logsig in self.weight_logsigs]
            if self.bias:
                mus += [bias_mu.view([-1]) for bias_mu in self.bias_mus]
                logsigs += [bias_logsigs.view([-1]) for bias_logsigs in self.bias_logsigs]
            mus = torch.cat(mus)
            sigs = self.__transform_sig(torch.cat(logsigs))
            q = Normal(mus,sigs)
            N = Normal(torch.zeros_like(mus),torch.ones_like(mus))
            return kl(q,N)
        elif self.var_apr == 'radial':
            weights,biases = self.__sample_weights(L)
            weights = torch.cat([w.view(L,-1) for w in weights],1)
            sigs = torch.cat([weight_sig.view([-1]) for weight_sig in self.weight_logsigs])
            if self.bias:
                biases = torch.cat([b.view(L,-1) for b in biases],1)
                weights = torch.cat([weights,biases],1)
                bias_sigs = torch.cat([bias_sig.view([-1]) for bias_sig in self.bias_logsigs])
                sigs = torch.cat([sigs,bias_sigs])
            cross_entr = -(weights**2).mean(0)/2 - np.log(2*np.pi)
            entr = -self.__transform_sig(sigs).log()
            return entr - cross_entr

    def __repr__(self):
        str_ = 'BNN\n' if self.bnn else 'NN\n'
        for i,(weight,act) in enumerate(zip(self.weight_mus,self.acts)):
            str_ += 'Layer-{:d}: '.format(i+1) + ''.join(str([*weight.shape][::-1])) \
                + '\t' + str(act) + '\n'
        return str_


