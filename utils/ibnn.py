import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import MultivariateNormal, Normal, Bernoulli, kl_divergence as kl
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

from .utils import get_act


class IBNN(nn.Module):
    def __init__(self, n_ens: int, n_in: int, n_out: int, n_hid_layers: int=2, n_hidden: int=250, \
                        act: str='relu', requires_grad=True, bias=True, layer_norm=False, \
                        dropout=0.0, bnn=True, skip_con=False):
        super().__init__()
        print('IBNN: layer_norm, dropout, bnn parameters are discarded')
        layers_dim = [n_in] + n_hid_layers*[n_hidden] + [n_out]
        self.weights  = nn.ParameterList([])
        self.biases  = nn.ParameterList([])
        self.acts  = []
        self.n_ens = n_ens
        self.skip_con = skip_con
        self.act = act 
        self.bias = bias
        for i,(n_in,n_out) in enumerate(zip(layers_dim[:-1],layers_dim[1:])):
            self.weights.append(Parameter(torch.Tensor(n_in, n_out),requires_grad=requires_grad))
            self.biases.append(None if not bias else Parameter(torch.Tensor(1,n_out),requires_grad=requires_grad))
            self.acts.append(get_act(act) if i<n_hid_layers else get_act('linear')) # no act. in final layer
        self.z_mus = nn.ParameterList([])
        self.z_logsigs = nn.ParameterList([])
        for i,n_node in enumerate(layers_dim[:-1]):
            self.z_mus.append(
                Parameter(torch.Tensor(n_ens,1,n_node),requires_grad=requires_grad) # Nens,1,n
            )
            self.z_logsigs.append(
                Parameter(torch.Tensor(n_ens,1,n_node),requires_grad=requires_grad)
            )
        self.reset_parameters()
    
    def shuffle(self):
        rand_idx = torch.randperm(self.n_ens)
        for mu,logsig in zip(self.z_mus,self.z_logsigs):
            mu.data = mu.data[rand_idx] 
            logsig.data = logsig.data[rand_idx] 

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
        for z_mu,z_logsig in zip(self.z_mus,self.z_logsigs):
            nn.init.normal_(z_mu, 1.0, 0.25)
            # nn.init.normal_(z_logsig, 0.05, 0.02)
            nn.init.normal_(z_logsig, -2, 0.01)

    def draw_noise(self, L):
        assert L//self.n_ens, f'L={L} must be a multiple of n_ens={self.n_ens}'
        return [torch.randn([L,1,z_mu.shape[-1]],device=self.device) for z_mu in self.z_mus] # L,1,N

    def __draw_multiplicative_factors(self, noise_vec):
        zs = []
        for i,noise in enumerate(noise_vec): # for each layer
            noise = noise.view([-1,*self.z_mus[i].shape]) # L/Nens,Nens,1,n
            sig = self.__transform_sig(self.z_logsigs[i]) 
            z = self.z_mus[i] + noise*sig # L/Nens,Nens,1,n
            zs.append(z.reshape(-1,1,self.z_mus[i].shape[-1])) # L,1,n
        return zs # list of L,1,n


    def draw_f(self, L=1, noise_vec=None):
        """ Draws L//n_ens samples from each ensemble component
            Assigns each x[i] to a different sample in a different component
            x  -     [N,n] or [L,N,n]
            output - the same shape as input
        """
        # assert L//self.n_ens, f'L={L} must be a multiple of n_ens={self.n_ens}'
        noise_vec = noise_vec if noise_vec is not None else self.draw_noise(L)
        zs = self.__draw_multiplicative_factors(noise_vec) # list of [L,1,n_hidden]
        def f(x):
            x2d = x.ndim==2
            x = torch.stack([x]*L) if x2d else x # L,N,n
            for (z,weight,bias,act) in zip(zs,self.weights,self.biases,self.acts):
                x_ = (x*z)@weight + bias
                x_ = x_+x if x.shape==x_.shape and self.skip_con else x_
                x  = act(x_)
            return x.mean(0) if x2d else x
        return f

    def forward(self, x, L=1):
        return self.draw_f(L)(x)

    def kl(self):
        kls = []
        for mu,logsig in zip(self.z_mus,self.z_logsigs):
            mu_ = mu.mean([0])[0] # n
            sig_ = self.__transform_sig(logsig).pow(2).mean(0)[0].pow(0.5) # n
            qhat = Normal(mu_, sig_)
            p = Normal(torch.ones_like(mu_),torch.ones_like(sig_))
            kl_ = kl(qhat, p).sum()
            kls.append(kl_)
        return torch.stack(kls).sum()

    def __repr__(self):
        str_ = f'iBNN - {self.n_ens} components\n'
        for i,(weight,act) in enumerate(zip(self.weights,self.acts)):
            str_ += 'Layer-{:d}: '.format(i+1) + ''.join(str([*weight.shape][::-1])) \
                + '\t' + str(act) + '\n'
        return str_