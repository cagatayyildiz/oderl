from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy, torch, numpy as np
from numpy import pi, cos, sin
from .base_env import BaseEnv

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi) # [-3, -1, 0, 1, 2, 3] --> [-1, -1, 0, -1, 0]

    
class CTAcrobot(BaseEnv):
    """
    Code modified from https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py
    Acrobot is a 2-link pendulum with only the second joint actuated.
    Initially, both links point downwards. The goal is to swing the
    end-effector at a height at least the length of one link above the base.
    Both links can swing freely and can pass by each other, i.e., they don't
    collide when they have the same angle.
    **STATE:**
    The state consists of the sin() and cos() of the two rotational joint
    angles and the joint angular velocities :
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2].
    For the first link, an angle of 0 corresponds to the link pointing downwards.
    The angle of the second link is relative to the angle of the first link.
    An angle of 0 corresponds to having the same angle between the two links.
    A state of [1, 0, 1, 0, ..., ...] means that both links point downwards.
    **ACTIONS:**
    Restricted to a range [-4,4]
    .. note::
        The dynamics equations were missing some terms in the NIPS paper which
        are present in the book. R. Sutton confirmed in personal correspondence
        that the experimental results shown in the paper and the book were
        generated with the equations shown in the book.
        However, there is the option to run the domain with the paper equations
        by setting book_or_nips = 'nips'
    **REFERENCE:**
    .. seealso::
        R. Sutton: Generalization in Reinforcement Learning:
        Successful Examples Using Sparse Coarse Coding (NIPS 1996)
    .. seealso::
        R. Sutton and A. G. Barto:
        Reinforcement learning: An introduction.
        Cambridge: MIT press, 1998.
    .. warning::
        This version of the domain uses the Runge-Kutta method for integrating
        the system dynamics and is more realistic, but also considerably harder
        than the original version which employs Euler integration,
        see the AcrobotLegacy class.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 15
    }
    
    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    def __init__(self, dt=0.1, device='cpu', obs_trans=True, obs_noise=0.0, ts_grid='fixed',\
                 solver='dopri8', fully_act=True):
        self.fully_act = fully_act
        self.N0 = 7
        self.Nexpseq = 3
        name = 'acrobot'
        if obs_trans:
            state_action_names = ['cos_theta1','sin_theta1','cos_theta2','sin_theta2','velocity1','velocity2']
            name += '-trig'
        else:
            state_action_names = ['theta1','theta2','velocity1','velocity2']
        if fully_act:
            state_action_names += ['action1', 'action2']
            print('Running fully actuated Acrobot')
        else:
            state_action_names += ['action']
        super().__init__(dt, 4+2*obs_trans, 1+fully_act, 5.0, obs_trans, name, \
            state_action_names, device, solver, obs_noise, ts_grid, 1e-4)
        self.reset()

    #################### environment specific ##################
    def extract_velocity(self,state):
        return state[...,-2:]

    def extract_position(self,state):
        return state[...,:-2]

    def merge_velocity_acceleration(self,ds,dv):
        return torch.cat(ds,dv,-1)
    
    def torch_transform_states(self,state):
        ''' Input - [N,n] or [L,N,n]
        '''
        if self.obs_trans:
            state_ = state.detach().clone()
            theta1, theta2, vel1, vel2 = state_[...,0:1],state_[...,1:2],state_[...,2:3],state_[...,3:4]
            return torch.cat([theta1.cos(), theta1.sin(), theta2.cos(), theta2.sin(), vel1, vel2],-1)
        else:
            return state
    
    def set_state_(self,state):
        assert state.shape[-1]==4, 'Trigonometrically transformed states cannot be set!\n'
        self.state = copy.deepcopy(state)
        return self.get_obs()
        
    def df_du(self,state):
        raise NotImplementedError()

    #################### override ##################
    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        return self.get_obs()
    
    def obs2state(self,obs):
        if obs.shape[-1]==4:
            return obs
        cos_th1,sin_th1,cos_th2,sin_th2,vel1,vel2 = obs[...,0],obs[...,1],obs[...,2],obs[...,3],obs[...,4],obs[...,5]
        theta1 = self.trigonometric2angle(cos_th1,sin_th1)
        theta2 = self.trigonometric2angle(cos_th2,sin_th2)
        return torch.stack([theta1,theta2,vel1,vel2],-1)
    
    def torch_rhs(self, state, action):
        ''' Input
                state  [N,n] 
                action [N,m] 
        '''
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        theta1,theta2,dtheta1,dtheta2 = state[...,0],state[...,1],state[...,2],state[...,3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * torch.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * torch.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * torch.cos(theta1 + theta2 - pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * torch.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * torch.sin(theta2)  \
            + (m1 * lc1 + m2 * l1) * g * torch.cos(theta1 - pi / 2) + phi2
        ddtheta2 = (action[...,0] + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * torch.sin(theta2) - phi2) \
            / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        if self.fully_act:
            ddtheta1 = -(action[...,1] + d2 * ddtheta2 + phi1) / d1
        else:
            ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return torch.stack([dtheta1, dtheta2, ddtheta1, ddtheta2], -1)
    

    def diff_obs_reward_(self,state):
        if state.shape[-1]==6:
            state = self.obs2state(state)
        th1,th2,vel1,vel2 = state[...,0],state[...,1],state[...,2],state[...,3]
        velocity_reward = -vel1**2 - vel2**2
        p1 = [-self.LINK_LENGTH_1 * torch.cos(th1), self.LINK_LENGTH_1 * torch.sin(th1)]
        p2 = [p1[0] - self.LINK_LENGTH_2 * torch.cos(th1 + th2),
              p1[1] + self.LINK_LENGTH_2 * torch.sin(th1 + th2)]
        state_reward = -(p2[0]-self.LINK_LENGTH_1-self.LINK_LENGTH_2)**2 - (p2[1])**2
        return (state_reward + self.vel_rew_const*velocity_reward).exp()
        
    def diff_ac_reward_(self,action):
        return -self.ac_rew_const * torch.sum(action**2, -1)

    def render(self, mode='human', *args, **kwargs):
        from gym.envs.classic_control import rendering
        s = self.state
        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)
        if s is None: 
            return None
        p1 = [-self.LINK_LENGTH_1 *cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]
        p2 = [p1[0] - self.LINK_LENGTH_2 * cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])]
        # print(p1+p2)
        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]- pi/2, s[0]+s[1]-pi/2]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]
        self.viewer.draw_line((-2.2, 1), (2.2, 1))
        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            link.set_color(0,.8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
