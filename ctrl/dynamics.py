import numpy as np
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from utils.utils import odesolve
from utils import ENN, DropoutBNN, IBNN, BENN, EPNN

class Dynamics(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, env, dynamics, L, nl=2, nn=100, act='relu', dropout=0.0, bnn=False):
        super().__init__()
        n,m = env.n, env.m
        self.qin, self.qout = n+m, n
        self.qin = self.qin
        self.env = env
        self.dynamics = dynamics
        self.L = L
        self.ens_method = False
        
        if self.dynamics == 'ibnode':
            self._f = IBNN(L, self.qin, self.qout, n_hid_layers=nl, n_hidden=nn, act=act)

        elif self.dynamics == 'benode':
            self._f = BENN(L, self.qin, self.qout, n_hid_layers=nl, n_hidden=nn, act=act)
            self.ens_method = True

        elif self.dynamics == 'enode':
            self._f = ENN(L, self.qin, self.qout, n_hid_layers=nl, n_hidden=nn, act=act)
            self.ens_method = True

        elif self.dynamics == 'epnn':
            self._f = EPNN(L, self.qin, self.qout, n_hid_layers=nl, n_hidden=nn, act=act)
            self.ens_method = True

        elif self.dynamics == 'dbnn':
            self._f = DropoutBNN(self.qin, self.qout, n_hid_layers=nl, n_hidden=nn, act=act, \
                dropout_rate=dropout)

        self.reset_parameters()
    
    @property 
    def device(self):
        return self._f.device

    def reset_parameters(self,w=0.1):
        self._f.reset_parameters(w)
    
    def kl(self):
        try:
            return self._f.kl()
        except:
            return torch.Tensor(np.zeros(1)*1.0).to(self.device)
        
    def ds_dt(self,f,s,a):
        return f(torch.cat([s,a],-1))
    
    def dv_dt(self,t,s,a,v,tau,compute_rew):
        r = self.env.diff_reward(s,a).unsqueeze(2) if compute_rew else v # L,N,1
        if tau is not None:
            t = t.item() if isinstance(t,torch.Tensor) else t
            r *= np.exp(-t/tau)
        return r
    
    def forward_simulate(self, solver, H, s0, f, g, L, tau=None, compute_rew=True):
        # starting from t=0
        T = int(H/self.env.dt)
        ts  = self.env.dt*torch.arange(T+1,dtype=torch.float32,device=self._f.device)
        t0s = torch.zeros(s0.shape[0],dtype=torch.float32,device=self._f.device)
        st, rt, at = self._forward_simulate(solver, ts, t0s, s0, f, g, L, tau=tau, compute_rew=compute_rew)
        return st, rt, at, torch.stack([ts[:-1]]*s0.shape[0])
    
    def forward_simulate_nonuniform_ts(self, solver, ts, s0, f, g, L, tau=None, compute_rew=True):
        # all t0s are set to be zero
        [N,T] = ts.shape
        ts_norm = ts - ts[:,0:1] # all starting from 0 [[.0 .1 .4],[.1 .3 .4]] --> [[.0 .1 .4],[.0 .2 .3]]
        ts_ode = ts_norm[:,1:].reshape(-1).unique() # [.0 .1 .4 .2 .3]
        ts_ode = torch.cat([torch.zeros(1,device=ts_ode.device),ts_ode]) # handle numerical issues
        ts_ode_sorted,_ = ts_ode.sort() # [.0 .1 .2 .3 .4]
        ts_ode_sorted = torch.cat([ts_ode_sorted,ts_ode_sorted[-1:]+1e-3]) # handle numerical issues
        Tidx = [[torch.where(ts_norm[n,t]==ts_ode_sorted)[0].item() for t in range(T)] for n in range(N)]
        st, rt, at = self._forward_simulate(solver, ts_ode_sorted, ts[:,0], s0, f, g, L, \
                                            tau=tau, compute_rew=compute_rew)
        sts = [st[:,i,Tidx[i]] for i in range(N)]
        rts = [rt[:,i,Tidx[i]] for i in range(N)]
        return torch.stack(sts,1), torch.stack(rts,1), at, ts

    @abstractmethod
    def _forward_simulate(self, solver, ts, t0s, s0, f, g, L, tau, compute_rew):
        ''' Performs forward simulation for L different vector fields
                ts  - [T+1], starting from 0
                t0s - [N]
            Output
                st - [L,N,T,n]
                rt - [L,N,T,n]
                at - [L,N,T-1,m]
                t  - [T]
        '''
        raise NotImplementedError


class NODE(Dynamics):
    def __init__(self, env, dynamics, L, nl=2, nn=100, act='relu'):
        super().__init__(env, dynamics, L, nl=nl, nn=nn, act=act, dropout=0.0)
   
    def odef(self, t, sv, f, g, t0s=None, tau=None, compute_rew=True):
        ''' Input
                t - current time (add t0s to get inputs to the policy!)
                sv - state&value - [Nens,N,n+1]
                f - time differential 
                g - action function
        '''
        t = t if isinstance(t,torch.Tensor) else torch.tensor(t).to(self.device)
        s,v = sv[:,:,:-1],sv[:,:,-1:]
        a  = g(s,t+t0s) # L,N,m
        self.at[t+t0s] = a
        ds = self.ds_dt(f,s,a)
        dv = self.dv_dt(t,s,a,v,tau,compute_rew)
        return torch.cat([ds,dv],-1) # Nens,N,n+1
     
    def _forward_simulate(self, solver, ts, t0s, s0, f, g, L, tau=None, compute_rew=True):
        self.at = {}
        T = len(ts)-1
        [N,n] = s0.shape # N,n
        s0 = torch.stack([s0]*L) # Nens,N,n
        r0 = torch.zeros(s0.shape[:-1],device=s0.device).unsqueeze(2) # Nens,N,1
        s0r0 = torch.cat([s0,r0],-1) # Nens,N,n+1
        odef = lambda t,s: self.odef(t, s, f, g, t0s, tau=tau, compute_rew=compute_rew)
        strt = odesolve(odef, s0r0, ts, solver['step_size'], solver['method'], solver['rtol'], solver['atol'])
        st,rt = strt[:T,...,:n].permute([1,2,0,3]),strt[:T,...,-1].permute([1,2,0]) # L,N,T,n & L,N,T
        return st, rt, self.at


class PETS(Dynamics):
    def __init__(self, env, dynamics, L, nl=2, nn=100, act='relu'):
        super().__init__(env, 'epnn', L, nl=nl, nn=nn, act=act, dropout=0.0)
        self.P = 20
            
    def _forward_simulate(self, solver, ts, t0s, s0, f, g, L, tau=None, compute_rew=True):
        H = len(ts) - 1
        [N,n] = s0.shape # N,n
        s0 = torch.cat([s0]*self.P) # PN,n
        s0 = torch.stack([s0]*L) # Nens,PN,n
        V0 = torch.zeros([*s0.shape[:-1],1],device=s0.device) # Nens,PN,1
        st,Vt,at = [s0],[V0],{}
        delta_t = ts[1:] - ts[:-1]
        for t_,delta_t_ in zip(ts,delta_t): # 0 & 0.1
            a = g(st[-1].reshape(L,self.P,N,n),t_+t0s).reshape(L,self.P*N,-1) # Nens,PN,m
            at[t_+t0s] = a
            dV = self.dv_dt(t_, st[-1], a, Vt[-1], tau, compute_rew)
            V = Vt[-1] + delta_t_*dV
            Vt.append(V)
            ds = self.ds_dt(f,st[-1],a)
            s = st[-1] + delta_t_*ds
            st.append(s)
            self._f.shuffle()
        st,Vt = torch.stack(st)[:H].permute(1,2,0,3),torch.stack(Vt)[:H].permute(1,2,0,3).squeeze(-1)
        # Nens,PN,T,n & Nens,PN,T & Nens,PN,T-1,m
        st = st.reshape([L,self.P,N,H,n]).view(L*self.P,N,H,n)
        Vt = Vt.reshape([L,self.P,N,H]).view(L*self.P,N,H)
        return st, Vt, at
    
    
class DeepPILCO(Dynamics):
    def __init__(self, env, dynamics, L, nl=2,  nn=100, act='relu', dropout=0.0):
        super().__init__(env, 'dbnn', L, nl=nl, nn=nn, act=act, dropout=dropout, bnn=True)
            
    def _forward_simulate(self, solver, ts, t0s, s0, f, g, L, tau=None, compute_rew=True):
        H = len(ts) - 1
        [N,n] = s0.shape # N,n
        s0 = torch.stack([s0]*L) # Nens,N,n
        V0 = torch.zeros([*s0.shape[:-1],1],device=s0.device) # Nens,N,1
        st,Vt,at = [s0],[V0],{}
        delta_t = ts[1:] - ts[:-1]
        for t_,delta_t_ in zip(ts,delta_t): # 0 & 0.1
            a = g(st[-1],t_+t0s) # L,N,m
            at[t_+t0s] = a
            dV = self.dv_dt(t_, st[-1], a, Vt[-1], tau, compute_rew)
            V = Vt[-1] + delta_t_*dV
            Vt.append(V)
            ds = self.ds_dt(f,st[-1],a)
            s = st[-1] + delta_t_*ds
            mu,sig = s.mean(0),s.std(0)
            s_new = torch.randn_like(s)*sig + mu
            st.append(s_new)
        st,Vt = torch.stack(st)[:H].permute(1,2,0,3),torch.stack(Vt)[:H].permute(1,2,0,3) # Nens,N,T,n & Nens,N,T,1
        Vt = Vt.squeeze(-1) # Nens,N,T
        return st, Vt, at
