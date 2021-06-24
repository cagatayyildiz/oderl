from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from os import path
import copy, torch, numpy as np
from .base_env import BaseEnv

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi) # [-3, -1, 0, 1, 2, 3] --> [-1, -1, 0, -1, 0]

    
class CTPendulum(BaseEnv):
    """ The precise equation for reward:
        -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
        Theta is normalized between -pi and pi. Therefore, the lowest reward is -(pi^2 + 0.1*8^2 + 0.001*2^2) = -16.2736044, 
        and the highest reward is 0. In essence, the goal is to remain at zero angle (vertical), 
        with the least rotational velocity, and the least effort.
    """
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }
    def __init__(self, dt=0.1, device='cpu', obs_trans=True, obs_noise=0.0, ts_grid='fixed', solver='dopri8'):
        name = 'pendulum'
        if obs_trans:
            state_action_names = ['cos_theta','sin_theta','velocity','action']
            name += '-trig'
        else:
            state_action_names = ['angle','velocity','action']
        # self.reward_range = [-17.0, 0.0] # for visualization
        super().__init__(dt, 2+obs_trans, 1, 2.0, obs_trans, name, \
            state_action_names, device, solver, obs_noise, ts_grid)
        self.N0 = 3
        self.Nexpseq = 0
        self.g = 10.0
        self.mass = 1.
        self.l = 1.
        self.reset()

    #################### environment specific ##################
    def extract_velocity(self,state):
        return state[...,-1:]

    def extract_position(self,state):
        return state[...,:-1]

    def merge_velocity_acceleration(self,ds,dv):
        return torch.cat([ds,dv],-1)
    
    def torch_transform_states(self,state):
        ''' Input - [N,n] or [L,N,n]
        '''
        if self.obs_trans:
            theta, theta_dot = state[...,0:1],state[...,1:2]
            return torch.cat([theta.cos(), theta.sin(), theta_dot],-1)
        else:
            return state
    
    def set_state_(self,state):
        assert state.shape[-1]==2, 'Trigonometrically transformed states cannot be set!\n'
        self.state = copy.deepcopy(state)
        return self.get_obs()
        
    def df_du(self,state):
        theta, theta_dot = state[...,0],state[...,1]
        m,l = self.mass, self.l
        return torch.stack([theta*0.0, torch.ones_like(theta_dot)*3./(m*l**2)],-1)

    #################### override ##################
    def reset(self):
        low, high = np.array([-np.pi, -3]), np.array([np.pi, 3])
        # low, high = np.array([-0.75*np.pi, -1]), np.array([-0.5*np.pi, 1])
        self.state = self.np_random.uniform(low=low, high=high)
        return self.get_obs()
            
    def obs2state(self,obs):
        if obs.shape[-1] == 2:
            return obs
        cos_th, sin_th, vel = obs[...,0],obs[...,1],obs[...,2]
        theta = self.trigonometric2angle(cos_th,sin_th)
        return torch.stack([theta,vel],-1)
    
    def torch_rhs(self, state, action):
        ''' Input
                state  [N,n] 
                action [N,m] 
        '''
        # assert state.shape[-1]==2, 'Trigonometrically transformed states do not define ODE rhs!\n'
        g,m,l = self.g, self.mass, self.l
        if state.shape[-1]==2:
            th, thdot = state[...,0],state[...,1]
            return torch.stack([thdot, (-3*g/(2*l) * torch.sin(th+np.pi) + 3./(m*l**2)*action[...,0])],-1)
        elif state.shape[-1]==3:
            costh,sinth,thdot = state[...,0],state[...,1],state[...,2]
            th = self.obs2state(state)[...,0]
            return torch.stack([-sinth*thdot, costh*thdot, \
                (-3*g/(2*l) * torch.sin(th+np.pi) + 3./(m*l**2)*action[...,0])],-1)

    def diff_obs_reward_(self,state):
        if state.shape[-1] == 2:
            th, thdot = state[...,0], state[...,1]
            cos_th, sin_th = th.cos(),th.sin()
        else:
            cos_th, sin_th, thdot = state[...,0],state[...,1],state[...,2]
        state_reward = -self.l**2 * ((1-cos_th)**2+sin_th**2)
        velocity_reward = -thdot**2
        # return state_reward.exp() + self.vel_rew_const*velocity_reward
        return (state_reward + self.vel_rew_const*velocity_reward).exp() # works superb
    
    def diff_ac_reward_(self,action):
        return -self.ac_rew_const * torch.sum(action**2, -1)

    def render(self, mode='human', **kwargs):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        try:
            last_act = kwargs['last_act']
            self.imgtrans.scale = (-last_act/2, np.abs(last_act)/2)
        except:
            pass
        return self.viewer.render(return_rgb_array=mode=='rgb_array')
