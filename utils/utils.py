import torch
import torch.nn as nn
import numpy as np 
import math
from TorchDiffEqPack.odesolver import odesolve as torchdiffeqpack_odesolve

def get_act(act="relu"):
    if act=="relu":         return nn.ReLU()
    elif act=="elu":        return nn.ELU()
    elif act=="celu":       return nn.CELU()
    elif act=="leaky_relu": return nn.LeakyReLU()
    elif act=="sigmoid":    return nn.Sigmoid()
    elif act=="tanh":       return nn.Tanh()
    elif act=="sin":        return torch.sin
    elif act=="linear":     return nn.Identity()
    elif act=='softplus':   return nn.modules.activation.Softplus()
    elif act=='swish':      return lambda x: x*torch.sigmoid(x)
    else:                   return None

def sq_dist(X1, X2, ell=1.0):
    X1  = X1 / ell
    X1s = torch.sum(X1**2, dim=-1).view([-1,1]) 
    X2  = X2 / ell
    X2s = torch.sum(X2**2, dim=-1).view([1,-1])
    sq_dist = -2*torch.mm(X1,X2.t()) + X1s + X2s
    return sq_dist

def sq_dist3(X1, X2, ell=1.0):
    N = X1.shape[0]
    X1  = X1 / ell
    X1s = torch.sum(X1**2, dim=-1).view([N,-1,1]) 
    X2  = X2 / ell
    X2s = torch.sum(X2**2, dim=-1).view([N,1,-1])
    sq_dist = -2*X1@X2.transpose(-1,-2) + X1s + X2s
    return sq_dist

def batch_sq_dist(x, y, ell=1.0):
    '''                                                                                              
    Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3         
    Input: x is a bxNxd matrix y is an optional bxMxd matirx                                                             
    Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
    i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2                                                         
    '''
    assert x.ndim==3, 'Input1 must be 3D, not {x.shape}'
    y = y if y.ndim==3 else torch.stack([y]*x.shape[0])
    assert y.ndim==3, 'Input2 must be 3D, not {y.shape}'
    x,y = x/ell, y/ell                                                                                              
    x_norm = (x**2).sum(2).view(x.shape[0],x.shape[1],1)
    y_t = y.permute(0,2,1).contiguous()
    y_norm = (y**2).sum(2).view(y.shape[0],1,y.shape[1])
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    dist[dist != dist] = 0 # replace nan values with 0
    return torch.clamp(dist, 0.0, np.inf)

def K(X1, X2, ell=1.0, sf=1.0, eps=1e-5):
    dnorm2 = sq_dist(X1,X2,ell) if X1.ndim==2 else sq_dist3(X1,X2,ell)
    K_ = sf**2 * torch.exp(-0.5*dnorm2)
    if X1.shape[-2]==X2.shape[-2]:
        return K_ + torch.eye(X1.shape[-2],device=X1.device)*eps
    return K_

def torch_to_numpy(a):
    if isinstance(a,torch.Tensor):
        return a.cpu().detach().numpy()
    else:
        return a

def numpy_to_torch(a, device='cpu', dtype=torch.float32):
    if isinstance(a,np.ndarray) or isinstance(a,list):
        return torch.tensor(a,dtype=dtype).to(device)
    else:
        return a

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, torch.tensor):
            return m + torch.log(sum_exp)
        else:
            return m + math.log(sum_exp)

def flatten_(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat)>0 else torch.tensor([])

def smooth(x,w=7):
    x = np.array(x)
    y = np.zeros_like(x)
    for i in range(len(y)):
        y[i] = x[max(0,i-w):min(i+w,len(y))].mean()
    return y

def odesolve(f, z0, ts, step_size, method, rtol, atol):
    options = {}
    method = 'midpoint' if method=='RK2' else method
    options.update({'method': method})
    options.update({'step_size': step_size})
    options.update({'t0': ts[0].item()})
    options.update({'t1': ts[-1].item()})
    options.update({'rtol': rtol})
    options.update({'atol': atol})
    options.update({'t_eval': ts.tolist()})
    return torchdiffeqpack_odesolve(f, z0, options)
    

def Klinear(X1, X2, ell=1.0, sf=1.0, eps=1e-5):
    dnorm2 = sq_dist(X1,X2,ell) if X1.ndim==2 else sq_dist3(X1,X2,ell)
    K_ = sf**2 * torch.exp(-0.5*dnorm2)
    if X1.shape[-2]==X2.shape[-2]:
        return K_ + torch.eye(X1.shape[-2],device=X1.device)*eps
    return K_

class KernelInterpolation:
    def __init__(self, sf, ell, X, y, eps=1e-5, kernel='exp'):
        self.sf = sf
        self.ell = ell
        self.X = X
        self.y = y
        self.eps = eps
        self.K = K if kernel=='exp' else Klinear
        self.KXX_inv_y = torch.linalg.solve(self.K(X,X,ell,sf,eps), y)[0]
    
    def __call__(self,x):
        x = x if isinstance(x,torch.Tensor) else torch.tensor(x)
        kxX = self.K(x,self.X,self.ell,self.sf,self.eps)
        out = kxX @ self.KXX_inv_y # 1,nout
        return out
