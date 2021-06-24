
import torch
import torch.nn as nn
from utils import BNN

tanh_ = torch.nn.Tanh()
def final_activation(env,a):
    return tanh_(a) * env.act_rng

class Policy(nn.Module):
    def __init__(self, env, nl=2, nn=100, act='relu'):
        super().__init__()
        self.env   = env
        self.act   = act
        self._g = BNN(env.n, env.m, n_hid_layers=nl, act=act, n_hidden=nn, dropout=0.0, bnn=False)
        self.reset_parameters()
    
    def reset_parameters(self,w=0.1):
        self._g.reset_parameters(w)
    
    def forward(self,s,t):
        a = self._g(s)
        return final_activation(self.env, a)