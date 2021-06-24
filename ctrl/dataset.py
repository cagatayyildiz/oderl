import torch, copy, numpy as np

class Dataset:
    def __init__(self, env, D, ts):
        D = Dataset.__compute_rewards_if_needed(env,D) # N,T,n+m+1
        self.env = env
        self.n   = env.n
        self.m   = env.m
        self.D   = D
        self.ts  = ts

    @property
    def device(self):
        return self.env.device

    @property
    def shape(self):
        return self.D.shape

    @property
    def dt(self):
        return self.env.dt

    @property
    def N(self):
        return self.shape[0]

    @property
    def T(self):
        return self.shape[1]
    
    @property
    def s(self):
        return self.D[:,:,:self.n]

    @property
    def a(self):
        return self.D[:,:,self.n:self.n+self.m]

    @property
    def sa(self):
        return self.D[:,:,:self.n+self.m]

    @property
    def r(self):
        return self.D[:,:,-1:]

    def clone(self):
        return copy.deepcopy(self)

    def add_experience(self, Dnew, ts):
        assert (len(Dnew.shape)==3), 'New experience must be a 3D torch tensor' # N,T,nm
        Dnew = self.__compute_rewards_if_needed(self.env,Dnew) # N,T,n+m+1
        Dnew,ts = Dnew.to(self.device),ts.to(self.device)
        self.D  = torch.cat([self.D,Dnew])
        self.ts = torch.cat([self.ts,ts])
            
    def crop_last(self,N=1):
        self.D  = self.D[:-N]
        self.ts = self.ts[:-N]

    @staticmethod
    def __compute_rewards_if_needed(env,D):
        ''' returns (s,a,r) '''
        assert (len(D.shape)==3), 'Dataset must be a 3D torch tensor' # N,T,nm
        if D.shape[-1]==env.n+env.m:
            [N,T,nm] = D.shape
            with torch.no_grad():
                s_ = D[:,:,:env.n].view([-1,env.n])
                a_ = D[:,:,env.n:].view([-1,env.m])
                rewards = env.diff_reward(s_,a_).view([N,T,1])
                D = torch.cat([D,rewards],2) # N,T,n+m+1
        return D
    
    def to(self,device):
        self.D = self.D.to(device)
        self.ts = self.ts.to(device)
        return self
    
    def extract_data(self, H, cont, nrep=1, idx=None):
        ''' extracts sequences randomly subsequenced from the dataset
                H  - in second
                cont - boolean denoting whether the system is continuous
            returns
                g  - policy or None
                st - [N,T,n]
                at - [N,T,m]
                rt - [N,T,1]
        '''
        idx = list(np.arange(0,self.N)) if idx is None else list(idx)
        T = int(H/self.dt) # convert sec to # data points
        idx = [item for sublist in nrep*[idx] for item in sublist]
        t0s = torch.tensor(np.random.randint(0,1+self.T-T,len(idx)),dtype=torch.int32).to(self.device)
        st_at_rt = torch.stack([self.D[seq_idx_,t0:t0+T] for t0,seq_idx_ in zip(t0s,idx)])
        st,at,rt = st_at_rt[:,:,:self.n], st_at_rt[:,:,self.n:self.n+self.m], st_at_rt[:,:,-1:]
        ts = torch.stack([self.ts[seq_idx_,t0:t0+T] for t0,seq_idx_ in zip(t0s,idx)])
        g = self.__extract_policy(idx, at=at, cont=cont, ts=ts, T=T)
        return g, st, at, rt, ts 
    
    def __extract_policy(self, idx, at, cont, ts, T):
        if cont:
            return KernelInterpolatePolicy(at,ts)
        else:
            return DiscreteActions(at, ts)

class DiscreteActions:
    def __init__(self, at, ts):
        if len(at.shape) != 3:
            raise ValueError('Actions must be 3D!\n')
        self.at = at.to(at.device) # N,T,m
        self.ts = ts.to(at.device) # N,T
        self.N  = self.ts.shape[0]
        self.max_idx  = self.at.shape[1]-1
    def __call__(self,s,t):
        # t = t.item() if isinstance(t,torch.Tensor) else t
        if t[0].item()>self.ts[0,-1].item(): # actions outside the defined range
            actions = self.at[:,-1]
        else:
            before_idx = [(t[i]+1e-5>self.ts[i]).sum().item()-1 for i in range(self.N)]
            before_idx = [min(item,self.max_idx) for item in before_idx]
            actions = self.at[np.arange(self.N),before_idx]
        if actions.isnan().sum()>0:
            raise ValueError('Action interpolation is wrong!')
        if s.ndim==2:
            return actions
        elif s.ndim==3:
            return torch.stack([actions]*s.shape[0])
        elif s.ndim==4:
            tmp = torch.stack([actions]*s.shape[1])
            return torch.stack([tmp]*s.shape[0])

from utils.utils import KernelInterpolation
class KernelInterpolatePolicy:
    def __init__(self, at, ts):
        [N,T,m] = at.shape
        sfs  = 1.0 * torch.ones([N,1,1],device=at.device, dtype=torch.float32)
        ells = 0.5 * torch.ones([N,1,1],device=at.device, dtype=torch.float32)
        self.kernel_int = KernelInterpolation(sfs, ells, ts.unsqueeze(-1), at, eps=1e-5)
    
    def __call__(self,s,t):
        actions = self.kernel_int(t.unsqueeze(-1).unsqueeze(-1)) # N,1,n_out
        actions = actions.permute(1,0,2) # 1,N,n_out
        if s.ndim==2:
            return actions
        elif s.ndim==3:
            return torch.cat([actions]*s.shape[0])
        elif s.ndim==4:
            tmp = torch.stack([actions]*s.shape[1])
            return torch.stack([tmp]*s.shape[0])