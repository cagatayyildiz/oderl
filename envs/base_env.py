from abc import ABCMeta, abstractmethod

import gym, torch
import numpy as np
from gym.utils import seeding
from gym import spaces
from utils.utils import numpy_to_torch
from torchdiffeq import odeint

class BaseEnv(gym.Env, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, dt, n, m, act_rng, obs_trans, name, state_actions_names, \
                 device, solver, obs_noise, ts_grid, ac_rew_const=0.01, vel_rew_const=0.01):
        self.dt = dt
        self.n = n
        self.m = m
        self.act_rng = act_rng
        self.obs_trans = obs_trans
        self.name = name
        self.reward_range = [-ac_rew_const*act_rng**2,1.0]
        self.state_actions_names = state_actions_names
        self.ac_rew_const = ac_rew_const
        self.vel_rew_const = vel_rew_const
        self.obs_noise = obs_noise
        self.ts_grid = ts_grid
        # derived
        self.viewer = None
        self.action_space = spaces.Box(low=-self.act_rng,high=self.act_rng,shape=(self.m,))
        self.seed()
        self.ac_lb = numpy_to_torch(self.action_space.low, device=device)
        self.ac_ub = numpy_to_torch(self.action_space.high, device=device)
        self.set_solver(method=solver)

    def set_solver(self, method='euler', rtol=1e-6, atol=1e-9, num_bins=None):
        if num_bins is None:
            if method=='euler':
                num_bins = 1000
            elif method=='rk4':
                num_bins = 50
            else:
                num_bins = 1
        self.solver = {'method':method, 'rtol':rtol, 'atol':atol, 'step_size':self.dt/num_bins} 

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    @property
    def device(self):
        return self.ac_lb.device

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_obs(self):
        if self.obs_trans:
            torch_state = torch.tensor(self.state).unsqueeze(0)
            return list(self.torch_transform_states(torch_state)[0].numpy())
        else:
            return self.state

    def reward(self,obs,a):
        return self.np_obs_reward_fn(obs) + self.np_ac_reward_fn(a)
        
    def diff_reward(self,s,a):
        if not isinstance(s,torch.Tensor) or not isinstance(a,torch.Tensor):
            raise NotImplementedError('Differentiable reward only accepts torch.Tensor inputs\n')
        return self.diff_obs_reward_(s) + self.diff_ac_reward_(a)

    def build_time_grid(self,T):
        if self.ts_grid=='fixed':
            ts = torch.arange(T,device=self.device) * self.dt  
        elif self.ts_grid=='uniform' or self.ts_grid=='random':
            ts = (torch.rand(T,device=self.device)*2*self.dt).cumsum(0)
        elif self.ts_grid=='exp':
            ts = torch.distributions.exponential.Exponential(1/self.dt).sample([T])
            ts = ts.cumsum(0).to(self.device)
        else:
            raise ValueError('Time grid parameter is wrong!')
        return ts

    def integrate_system(self, T, g, s0=None, N=1, return_states=False):
        ''' Returns torch tensors
                states  - [N,T,n] where s0=[N,n]
                actions - [N,T,m]
                rewards - [N,T]
                ts      - [N,T]
        '''
        with torch.no_grad():
            s0 = torch.stack([numpy_to_torch(self.reset()) for _ in range(N)]).to(self.device) \
                if s0 is None else numpy_to_torch(s0)
            s0 = self.obs2state(s0)
            ts = self.build_time_grid(T)
            def odefnc(t, s):
                a = g(self.torch_transform_states(s),t) # 1,m
                return self.torch_rhs(s,a)
            st = odeint(odefnc, s0, ts, rtol=self.solver['rtol'], atol=self.solver['atol'], \
                        method=self.solver['method'])
            at = torch.stack([g(self.torch_transform_states(s_),t_) for s_,t_ in zip(st,ts)])
            rt = self.diff_reward(st,at) # T,N
            st,at,rt = st.permute(1,0,2),at.permute(1,0,2),rt.T
            st_obs = self.torch_transform_states(st)
            st_obs += torch.randn_like(st_obs) * self.obs_noise
            returns = [st_obs,at,rt,torch.stack([ts]*st_obs.shape[0])]
            if return_states:
                returns.append(st)
            return returns
    
    def torch_transform_states(self,state):
        if self.obs_trans:
            raise NotImplementedError
        else:
            return state
        
    def obs2state(self,state):
        if self.obs_trans:
            raise NotImplementedError
        else:
            return state

    def np_terminating_reward(self,state): # [...,n]
        return np.zeros(state.shape[:-1]) * 0.0
        
    def trigonometric2angle(self,costheta,sintheta):
        C = (costheta**2 + sintheta**2).detach()
        costheta,sintheta = costheta/C, sintheta/C
        theta = torch.atan2(sintheta/C,costheta/C)
        return theta
        
    @abstractmethod
    def reset(self):
        raise NotImplementedError
    @abstractmethod
    def torch_rhs(self, state, action):
        raise NotImplementedError
    @abstractmethod
    def diff_obs_reward_(self,s):
        raise NotImplementedError
    @abstractmethod
    def diff_ac_reward_(self,a):
        raise NotImplementedError
    @abstractmethod
    def render(self, mode, **kwargs):
        raise NotImplementedError