"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R

This version is developed on top of Ian Danforth's version
"""

import math, copy, torch
import numpy as np
from .base_env import BaseEnv

class CTCartpole(BaseEnv):
    """
    x, x_dot, theta, theta_dot = self.state
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, dt=0.1, device='cpu', swing_up=True, obs_trans=False, obs_noise=0.0, \
                 ts_grid='fixed', solver='dopri8'):
        name = 'cartpole'
        if obs_trans:
            state_action_names = ['Cart position','Cart velocity','Pole cos', 'Pole sin', 'Pole angular velocity','Action']
            name += '-trig'
        else:
            state_action_names = ['Cart position','Cart velocity','Pole angle','Pole angular velocity','Action']
        theta_threshold_radians = 12*math.pi/180
        x_threshold = 2.4
        super().__init__(dt, 4+obs_trans, 1, 3.0, obs_trans, name, \
            state_action_names, device, solver, obs_noise, ts_grid)
        # Angle at which to fail the episode
        self.theta_threshold_radians = theta_threshold_radians
        self.x_threshold = x_threshold
        self.N0 = 5
        self.Nexpseq = 2
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 1.0  # actually half the pole's length
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 3.0
        self.swing_up = swing_up
        self.reset()

    #################### environment specific ##################
    def extract_velocity(self,state):
        return state[...,[1,4]] if self.obs_trans else state[...,[1,3]]

    def extract_position(self,state):
        return state[...,[0,2,3]] if self.obs_trans else state[...,[0,2]]

    def merge_velocity_acceleration(self,ds,dv):
        if self.obs_trans:
            return torch.cat([ds[...,0:1],dv[...,0:1],ds[...,1:3],dv[...,1:2]],-1)
        return torch.cat([ds[...,0:1],dv[...,0:1],ds[...,1:2],dv[...,1:2]],-1)

    def torch_transform_states(self,state):
        ''' Input - [N,n] or [L,N,n]
        '''
        if self.obs_trans:
            state_ = state.detach().clone()
            x, x_dot, theta, theta_dot = state_[...,0:1],state_[...,1:2],state_[...,2:3],state_[...,3:4]
            state_ = torch.cat([x, x_dot, self.length*theta.cos(), self.length*theta.sin(), theta_dot],-1)
            return state_
        else:
            return state
    
    def set_state_(self,state):
        assert state.shape[-1]==4, 'Trigonometrically transformed states cannot be set!\n'
        self.state = copy.deepcopy(state)
        return self.get_obs()

    def df_du(self,state):
        if state.shape[-1]==4:
            x, _, theta, _ = state[...,0],state[...,1],state[...,2],state[...,3]
            costheta,sintheta = torch.cos(theta),torch.sin(theta)
        elif state.shape[-1]==5:
            x, _, costheta, sintheta, _ = \
                state[...,0], state[...,1], state[...,2], state[...,3], state[...,4]
            C = (costheta**2 + sintheta**2).detach()
            costheta = costheta / C
            sintheta = sintheta / C
        den = (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        d2 = - self.force_mag * costheta / self.total_mass / den
        d1 = (self.force_mag - d2*self.polemass_length*costheta) / self.total_mass
        if state.shape[-1]==4:
            return torch.stack([x*0.0,d1,x*0.0,d2],-1)
        elif state.shape[-1]==5:
            return torch.stack([x*0.0,d1,x*0.0,x*0.0,d2],-1)

    #################### override ##################
    def reset(self):
        rand_state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        if self.swing_up:
            rand_state[2] += np.pi
        self.state = rand_state
        return np.array(self.get_obs())

    def obs2state(self,obs):
        if obs.shape[-1] == 4:
            return obs
        x, x_dot, costheta, sintheta, theta_dot = obs[...,0], obs[...,1], obs[...,2], obs[...,3], obs[...,4]
        theta = self.trigonometric2angle(costheta,sintheta)
        return torch.stack([x,x_dot,theta,theta_dot],-1)
        
    def torch_rhs(self, state, action):
        ''' Input
                state  [N,n] or [b,N,n] 
                action [N,m] or [b,N,n] 
        '''
        # assert state.shape[-1]==4, 'Trigonometrically transformed states do not define ODE rhs!\n'
        fiveD = state.shape[-1]==5
        if fiveD:
            _, x_dot, costheta, sintheta, theta_dot = \
                state[...,0], state[...,1], state[...,2], state[...,3], state[...,4]
            C = (costheta**2 + sintheta**2).detach()
        else:
            _, x_dot, theta, theta_dot = state[...,0], state[...,1], state[...,2], state[...,3]
            costheta = torch.cos(theta)
            sintheta = torch.sin(theta)
        force = action[...,0]*self.force_mag
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if  fiveD:
            return torch.stack([x_dot,xacc,-sintheta*theta_dot/C,costheta*theta_dot/C,thetaacc],-1)
        else:
            return torch.stack([x_dot,xacc,theta_dot,thetaacc],-1)
                    
    def diff_obs_reward_(self,s,penalize_vel=False):
        if s.shape[-1]==4:
            x, xdot, theta, thetadot = s[..., :1], s[..., 1:2], s[..., 2:3], s[..., 3:]
            cos_th, sin_th = torch.cos(theta), torch.sin(theta)
            cos_th_len, sin_th_len = self.length*cos_th, self.length*sin_th
        else:
            x, xdot, cos_th_len, sin_th_len,thetadot = \
                s[..., :1], s[..., 1:2], s[..., 2:3], s[..., 3:4], s[..., 4:]
            cos_th, sin_th = cos_th_len/self.length, sin_th_len/self.length
        if self.swing_up:
            ee_pos = torch.cat([x+sin_th_len, cos_th_len], -1)
            err = ee_pos - torch.Tensor([0.0, self.length]).to(s.device)
            state_reward = -torch.sum(err**2,-1)
            velocity_reward = -torch.sum(xdot**2,-1) - torch.sum(thetadot**2,-1)
            return (state_reward + self.vel_rew_const*velocity_reward).exp() # works superb
        else:  
            return ((cos_th>np.cos(self.theta_threshold_radians)) *
                    (x<self.x_threshold) * (x>-self.x_threshold)).float()
        
    def diff_ac_reward_(self,a):
        return -self.ac_rew_const*torch.sum(a**2, -1) if self.swing_up else 0.0

    def render(self, mode='human', *args, **kwargs):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 3
        scale = screen_width / world_width
        carty = 200  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 5.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            axle = rendering.make_circle(10)
            axle.v = [(screen_width/2-5, polelen+carty-5), (screen_width/2-5, polelen+carty+5), \
                (screen_width/2+5, polelen+carty+5), (screen_width/2+5, polelen+carty-5)]
            axle.set_color(1,0,0)
            self.viewer.add_geom(axle)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))