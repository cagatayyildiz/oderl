import numpy as np
import copy, math, os, collections
import matplotlib.pyplot as plt
plt.switch_backend('agg')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch

from utils.utils import K, KernelInterpolation, numpy_to_torch, flatten_
from ctrl.dataset import Dataset
# from gpytorch.utils.cholesky import psd_safe_cholesky

##########################################################################################
######################################## PLOTTING ########################################
##########################################################################################
def plot_model(ctrl, D, rep_buf=10, H=None, L=10, fname=None, verbose=False, savefig=True):
    with torch.no_grad():
        if fname is None:
            fname = '{:s}-train.png'.format(ctrl.name)
        if verbose: 
            print('fname is {:s}'.format(fname))
        H = D.H if H is None else H
        rep_buf = min(rep_buf,D.N)
        idxs = -1*torch.arange(rep_buf,0,-1)
        g,st,at,rt,tobs = D.extract_data(H, ctrl.is_cont, idx=idxs)
        st_hat, rt_hat, at_hat, t = \
            ctrl.forward_simulate(tobs, st[:,0,:], g, L=L, compute_rew=False)
        if verbose: 
            print(st_hat.shape)
        plot_sequences(ctrl, fname, tobs, st, at, rt, t, st_hat, at_hat, rt_hat, verbose=verbose, savefig=savefig)

def plot_test(ctrl, D, N=None, H=None, L=10, fname=None, verbose=False, savefig=True):
    with torch.no_grad():
        if fname is None:
            fname = '{:s}-test.png'.format(ctrl.name)
        if verbose: 
            print('fname is {:s}'.format(fname))
        H = D.H if H is None else H
        N = max(D.N,10) if N is None else N
        D = collect_test_sequences(ctrl, D, N=N, reset=False, explore=True)
        g,st,at,rt,tobs = D.extract_data(H, ctrl.is_cont)
        st_hat, rt_hat, at_hat, t = \
            ctrl.forward_simulate(H, st[:,0,:], g, L=L, compute_rew=False)
        if verbose: 
            print(st_hat.shape)
        plot_sequences(ctrl, fname, tobs, st, at, rt, t, st_hat, at_hat, rt_hat, verbose=verbose, savefig=savefig)

def plot_sequences(ctrl, fname, tobs, st, at, rt, t, st_hat, at_hat, rt_hat, verbose=False, savefig=True):
    plt.close()
    [L,N,T,_] = st_hat.shape
    V = ctrl.V(st_hat).squeeze(-1).cpu().detach().numpy() # L,N,Td
    # rt_hat = ctrl.env.diff_reward(st_hat[:,:,:-1],at_hat)
    t      = t.cpu().detach().numpy() # T
    st_hat = st_hat.cpu().detach().numpy() # L,N,T,n
    rt_hat = rt_hat.cpu().detach().numpy() # L,N,T,m
    err = (st_hat-st.cpu().numpy())**2 # L,N,T,n
    err = err.mean(0).mean(0).mean(1)
    t_acts = torch.stack(list(at_hat.keys())).T
    act_hats = torch.stack(list(at_hat.values())).permute(1,2,0,3)
    t_acts   = t_acts.cpu().detach().numpy() # T
    tobs     = tobs.cpu().detach().numpy() # T
    act_hats = act_hats.cpu().detach().numpy() # T
    if verbose: 
        print(f'average error is {err}')
    w = ctrl.env.n + ctrl.env.m + 2
    plt.figure(1,((ctrl.env.n+ctrl.env.m)*5,N*3))
    for j in range(N):
        for i in range(ctrl.env.n):
            plt.subplot(N,w,j*w+i+1)
            plt.plot(t[j], st_hat[:,j,:,i].T, '-b',linewidth=.75)
            if i==0:
                plt.ylabel('Seq. {:d}'.format(j+1),fontsize=20)
            plt.plot(tobs[j], st[j,:,i].cpu().numpy(), '.r',linewidth=2,markersize=10)
            if j==0:
                plt.title(ctrl.env.state_actions_names[i],fontsize=25)    
            # rang = (st[j,:,i].max() - st[j,:,i].min()).item()
            # plt.ylim([st[j,:,i].min().item()-rang/5, st[j,:,i].max().item()+rang/5])
        for i in range(ctrl.env.m):
            plt.subplot(N,w,j*w+ctrl.env.n+i+1)
            plt.plot(tobs[j], at[j,:,i].cpu().numpy(), '.r',linewidth=1.0)
            plt.plot(t_acts[j], act_hats[:,j,:,i].T,'-b',linewidth=.75)
            if j==0:
                plt.title(ctrl.env.state_actions_names[ctrl.env.n+i],fontsize=25)
            plt.ylim([ctrl.env.ac_lb[i].item()-0.1,ctrl.env.ac_ub[i].item()+0.1])
        # plot reward
        plt.subplot(N,w,j*w+ctrl.env.n+ctrl.env.m+1) 
        # plt.plot(t[j,:-1], rt_hat[:,j].T,'-b')
        plt.plot(tobs[j], rt[j].cpu().numpy(),'or',markersize=4)
        if j==0:
            plt.title('rewards',fontsize=25)
        plt.ylim([ctrl.env.reward_range[0]-0.2,ctrl.env.reward_range[1]+0.2])
        # plot value
        plt.subplot(N,w,j*w+ctrl.env.n+ctrl.env.m+2) 
        # min,max = V[:,j].min().item(),V[:,j].max().item()
        # scaled_rew = (max-min)*rt[j]+min
        # plt.plot(t, scaled_rew.cpu().numpy(),'-or',markersize=3,linewidth=0.5)
        plt.plot(t[j], V[:,j].T,'-b')
        if j==0:
            plt.title('Values',fontsize=25)
        plt.grid(True)
    plt.tight_layout()
    if savefig:
        plt.savefig(fname)
    plt.close()        
    

##########################################################################################
######################################## TRAIN #########################################
##########################################################################################

def get_train_functions(dynamics):
    if 'ode' in dynamics:
        return train_ode
    elif 'pilco' in dynamics:
        return train_deep_pilco
    elif 'pets' in dynamics:
        return train_pets

def train_loop(ctrl, D, fname, Nround, **kwargs):
    H, L  = kwargs['H'], kwargs['L']
    verbose, print_times, N_pol_iter, Nexpseq = True, 10, 250, ctrl.env.Nexpseq
    dyn_tr_fnc = get_train_functions(ctrl.dynamics)
    print('fname is {:s}'.format(fname))
    # plot_model(ctrl, D, H=H, L=L, rep_buf=10, fname=fname+'-train.png')
    r0 = int( (D.N-ctrl.env.N0) / (1+ctrl.env.Nexpseq) )
    for round in range(r0,r0+Nround):
        print(f'Round {round}/{r0+Nround} starting')
        # dynamics training
        dyn_tr_fnc(ctrl, D, fname, verbose, print_times, **kwargs)
        # policy learning
        train_policy(ctrl, D, H=H, V_const=min(round/5.0,1), verbose=verbose, \
            Niter=N_pol_iter, L=L, save_fname=fname, print_times=print_times)
        # test the policy
        Htest,Ntest,Tup = 30,10,int(3.0/ctrl.env.dt)
        s0 = torch.stack([numpy_to_torch(ctrl.env.reset()) for _ in range(Ntest)]).to(ctrl.device) 
        _,_,test_rewards,_ = ctrl.env.integrate_system(T=int(Htest//ctrl.env.dt), s0=s0, g=ctrl._g)
        # get the rewards after 3 seconds to understand the pole never falls (reward must be >.9)
        true_test_rewards = test_rewards[...,Tup:].mean().item()
        st_hat, rt_hat, at_hat, t = ctrl.forward_simulate(Htest, s0, ctrl._g, compute_rew=True)
        rt_hat = ctrl.env.diff_obs_reward_(st_hat[:,:,:-1])
        imagined_test_rewards = rt_hat[...,Tup:].mean().item()
        print('Model tested. True/imagined reward sum is {:.2f}/{:.2f}'.\
              format(true_test_rewards,imagined_test_rewards))
        print(f'Minimum test reward is {test_rewards[...,Tup:].min().item()}')
        if (test_rewards[...,Tup:]<.8).sum()==0:
            print(f'All {Ntest} tests are solved in {round} rounds!')
            ctrl.save(D=D,fname=fname+'-solved')
            break
        print('Collecting experience...\n')
        mean_act = torch.stack(list(at_hat.values())).abs().mean()
        print('Imagined mean action is {:.2f}'.format(mean_act))
        if mean_act<0.25: # if the policy does not explore, collect data with random policy
            D = collect_data(ctrl.env, H=D.dt*D.T, N=Nexpseq+1, D=D)
        else:
            D = collect_experience(ctrl, D=D, N=Nexpseq, H=D.dt*D.T, reset=False, explore=True) 
            D = collect_experience(ctrl, D=D, N=1, H=D.dt*D.T, reset=True)
        plot_model(ctrl, D, H=H, L=10, rep_buf=10, fname=fname+'-train.png', verbose=False)
        ctrl.save(D=D, fname=fname)

def train_policy(ctrl, D, H=2.0, Niter=250, verbose=True, tau=5.0, N=100, L=10, V_const=1.0, save_every=50, 
            eta=1e-3, save_fname=None, rep_buf=5, opt='adam', print_times=10):
    if rep_buf<0:
        rep_buf = D.N
    if verbose: 
        print('policy training started')
    if save_fname is None:
        save_fname = ctrl.name
    N = D.N if N==-1 else N
    opt_cls = get_opt(opt)
    policy_optimizer = opt_cls(ctrl._g.parameters(),lr=eta)
    opt_cls = get_opt(opt)
    value_optimizer = opt_cls(ctrl.V.parameters(),lr=eta)
    L = ctrl.get_L(L)
    rewards,opt_objs = [],[]
    for itr in range(Niter):
        # update the critic copy
        if itr%100==0:
            Vtarget = copy.deepcopy(ctrl.V)
        s0 = get_ivs(ctrl.env,D,N,rep_buf) # N,n
        noise_vec = ctrl.draw_noise(L)
        fs = ctrl.draw_f(L, noise_vec=noise_vec)
        policy_optimizer.zero_grad()
        st, rt, at, ts = ctrl.forward_simulate(H, s0, ctrl._g, f=fs, L=L, tau=tau, compute_rew=True)
        rew_int  = rt[:,:,-1].mean(0) # N
        if rt.isnan().any():
            print('Reward is nan. Breaking.')
            break
        ts = ts[0]
        st = torch.cat([st]*5) if st.shape[0]==1 else st
        [L,N_,Hdense,n] = st.shape
        gammas = (-ts/tau).exp() # H
        V_st_gam = ctrl.V(st.contiguous())[:,:,1:,0] * gammas[1:] # L,N,H-1
        n_step_returns = rt[:,:,1:] + V_const*V_st_gam # ---> n_step_returns[:,:,k] is the sum in (5)
        opimized_returns = n_step_returns.mean(-1) # L,N
        mean_cost = -opimized_returns.mean()
        mean_cost.backward()
        grad_norm = torch.norm(flatten_([p.grad for p in ctrl._g.parameters()])).item()
        policy_optimizer.step()
        rewards.append(rew_int.mean().item()/H)
        opt_objs.append(mean_cost.mean().item())
        print_log = 'Iter:{:4d}/{:<4d},  opt. target:{:.3f}  mean reward:{:.3f}  '\
            .format(itr, Niter, np.mean(opt_objs), np.mean(rewards)) + \
            'H={:.2f},  grad_norm={:.3f},  '.format(H,grad_norm) 
        # minimize TD error
        with torch.no_grad():
            # regress all intermediate values
            last_states = st.detach().contiguous()[:,:,1:,:] # L,N,T-1,n
            last_values = Vtarget(last_states).squeeze(-1) # L,N,T-1
            Vtargets = rt[:,:,1:] + (-ts[1:]/tau).exp()*last_values # L,N,T-1
            Vtargets = Vtargets.mean(0).mean(-1) # N
        mean_val_err = 0
        for inner_iter in range(10):
            value_optimizer.zero_grad()
            td_error = ctrl.V(s0).squeeze(-1) - Vtargets # L,N
            td_error = torch.mean(td_error**2)
            td_error.backward()
            mean_val_err += td_error.item() / 10
            if inner_iter==0:
                first_val_err = td_error.item()
            value_optimizer.step()
        print_log += 'first/final value error:{:.3f}/{:.3f}  kl:{:.3f}'\
            .format(first_val_err,td_error.item(),ctrl.V.kl().item())
        if verbose and itr%(Niter//print_times)==0:
            print(print_log)
        if (itr+1)%save_every == 0:
            ctrl.save(D, fname=save_fname)
    
    
def dynamics_loss(ctrl, st, ts, at, g, L=1):
    f = ctrl.draw_f(L)
    outputs = ctrl.forward_simulate(ts, st[:,0,:], g, f=f, L=L, compute_rew=False)
    st_hat,at_hat = outputs[0], outputs[2]
    [N,T,n] = st.shape 
    [L,N,T,_] = st_hat.shape
    sq_err = (torch.stack([st]*L)-st_hat)**2  # L,N,T,n
    sq_err = sq_err.view([-1,ctrl.env.n]) / ctrl.sn[:n]**2 / 2
    lhood = -sq_err - torch.mean(ctrl.logsn[:n]) - 0.5*np.log(2*np.pi)
    lhood = lhood.sum() / L
    mse = sq_err.mean().item()
    return mse, lhood, st_hat, at_hat  # N,T,n
def train_dynamics(ctrl, D, Niter=1000, verbose=True, H=10, N=-1, L=1, eta=1e-3, eta_final=2e-4, \
        save_every=50, save_fname=None, func_KL=False, lr_sch=False, kl_w=1, rep_buf=-1, temp_opt=True, \
        num_plots=0, opt='adam', print_times=10, rnode=False, stop_mse=1e-3, nrep=3):
    if save_fname is None:
        save_fname = ctrl.name
    opt_cls = get_opt(opt, temp=temp_opt)
    losses, mses, lhoods, kls, grad_norms = [], [], [], [], []
    opt_pars = ctrl.dynamics_parameters
    optimizer = opt_cls(opt_pars,lr=eta)
    if verbose: 
        print('dynamics training started')
    if verbose: 
        print(f'Dataset size = {list(D.shape)}')
    num_below_thresholds = 0
    for k in range(Niter):
        idx_ = np.arange(D.N)[-rep_buf:] if rep_buf>0 else  np.arange(D.N)
        g,st,at,rt,tobs = D.extract_data(H=H, idx=idx_, nrep=nrep, cont=ctrl.is_cont)
        optimizer.zero_grad()
        mse, sum_lhood, st_hat, at_hat = dynamics_loss(ctrl, st, tobs, at, g, L=L) # lhood = N,T,n
        loss   = -sum_lhood * D.T / (H/ctrl.env.dt) / nrep
        lhood_ = sum_lhood.item()
        kl_ = 0.0
        if kl_w > 0:
            kl_w_ = kl_w*min(1,(2*k/Niter))
            kl = kl_w_ * ctrl._f.kl().sum()
            loss += kl
            kl_ = kl.item()
        loss.backward()
        if math.isnan(loss.item()):
            print('Dynamics loss is nan, no gradient computation.')
            break
        grad_norm_ = torch.norm(flatten_([p.grad for p in opt_pars if p is not None and p.grad is not None])).item()
        optimizer.step()
        losses.append(loss.item())
        mses.append(mse)
        lhoods.append(lhood_)
        kls.append(kl_)
        grad_norms.append(grad_norm_)
        if verbose and k%(Niter//print_times)==0:
            print('Iter:{:4d}/{:<4d} loss:{:<.3f} lhood:{:<.3f} KL:{:<.3f} mse:{:<.3f}  grad norm:{:.4f} T:{:d}'.\
                format(k, Niter, np.mean(losses), np.mean(lhoods), np.mean(kls), np.mean(mses), np.mean(grad_norms), \
                st.shape[1]))
        if (k+1)%save_every == 0:
            ctrl.save(D, fname=save_fname)
        if num_plots>0 and (k+1)%(Niter//num_plots) == 0:
            plot_model(ctrl, D, L=10, rep_buf=10, fname=save_fname+'-train.png', verbose=False)
            plot_test(ctrl,  D, L=10, N=10, fname=save_fname+'-test.png', verbose=False)
        if mse/(H/ctrl.env.dt) < stop_mse:
            num_below_thresholds += 1
            if num_below_thresholds > 10:
                print(f'Optimization converged at iter {k}. Breaking...')
                ctrl.save(D, fname=save_fname)
                break
    if num_plots>0:
        plot_model(ctrl, D, L=10, rep_buf=10, fname=save_fname+'-train.png')
        plot_test(ctrl,  D, L=10, N=10, fname=save_fname+'-test.png')
    return losses,optimizer

def compute_full_batch_loss(ctrl, D, L=10):
    g,st,at,rt,tobs = D.extract_data(H=D.T*D.dt, cont=ctrl.is_cont)
    mse, sum_lhood, st_hat, at_hat = dynamics_loss(ctrl, st, tobs, at, g, L=L)
    return mse

def train_ode(ctrl, D, save_fname, verbose, print_times, **kwargs):
    if D.N==ctrl.env.N0: # if the training has just started
        print('Drift is being initialized with gradient matching.')
        ctrl0 = copy.deepcopy(ctrl)
        loss0 = compute_full_batch_loss(ctrl0, D)
        gradient_match(ctrl, D, Niter=500, L=kwargs['L'], print_times=print_times)
        loss1 = compute_full_batch_loss(ctrl, D)
        ctrl = copy.deepcopy(ctrl0) if loss1>loss0 else ctrl
        print('Drift initialized.')
    for H_ in [ctrl.env.dt*5]:
        train_dynamics(ctrl, D, Niter=1250, L=kwargs['L'], H=H_, eta=1e-3, save_fname=save_fname, \
            verbose=verbose, print_times=print_times, rep_buf=50, nrep=3)

def train_deep_pilco(ctrl, D, save_fname, verbose, print_times, **kwargs):
    Niter = 5000
    L = 100
    rep_buf = -1
    rep_buf = D.N if rep_buf<1 else rep_buf
    ds = (D.s[-rep_buf:,1:]-D.s[-rep_buf:,:-1]).reshape(-1,ctrl.env.n)
    dt = (D.ts[-rep_buf:,1:] - D.ts[-rep_buf:,:-1]).reshape(-1,1)
    ds_dt_opt = ds / dt # _,n
    s_opt = D.s[-rep_buf:,:-1].reshape(-1,ctrl.env.n) # _,n 
    a_opt = D.a[-rep_buf:,:-1].reshape(-1,ctrl.env.m) # _,m
    errors = []
    L = ctrl.get_L(L=L)
    s_opt,a_opt = torch.stack([s_opt]*L), torch.stack([a_opt]*L)
    optimizer = torch.optim.Adam(ctrl._f.parameters(),lr=1e-3)
    for i in range(Niter):
        optimizer.zero_grad()
        f = ctrl.draw_f(L=L)
        # optimize buffer
        ds_dt_opt_hat = ctrl._f.ds_dt(f,s_opt,a_opt) # N*(T-1),n
        opt_error = torch.sum((ds_dt_opt_hat-ds_dt_opt)**2)
        error = opt_error + ctrl._f.kl()
        errors.append(error.item())
        error.backward()
        optimizer.step()
        if verbose and i%(Niter/print_times)==0:
            print('Iter={:4d}/{:<4d}   opt_error={:.3f}'.format(i, Niter, np.mean(errors)))
        if (i+1)%(Niter//10) == 0:
            ctrl.save(D, fname=save_fname)

def train_pets(ctrl, D, save_fname, verbose, print_times, **kwargs):
    Niter = 5000
    C = 0.01
    rep_buf = -1
    rep_buf = D.N if rep_buf<1 else rep_buf
    ds = (D.s[-rep_buf:,1:]-D.s[-rep_buf:,:-1]).reshape(-1,ctrl.env.n)
    dt = (D.ts[-rep_buf:,1:] - D.ts[-rep_buf:,:-1]).reshape(-1,1)
    ds_dt_opt = ds / dt # _,n
    ds_dt_opt_L = torch.stack([ds_dt_opt]*ctrl.n_ens)
    s_opt = D.s[-rep_buf:,:-1].reshape(-1,ctrl.env.n) # _,n 
    a_opt = D.a[-rep_buf:,:-1].reshape(-1,ctrl.env.m) # _,m
    errors = []
    L = ctrl.get_L()
    s_opt,a_opt = torch.stack([s_opt]*L), torch.stack([a_opt]*L)
    optimizer = torch.optim.Adam(ctrl._f.parameters(),lr=1e-3)
    for i in range(Niter):
        optimizer.zero_grad()
        means,sig = ctrl._f._f.get_probs(torch.cat([s_opt,a_opt],-1))
        lhood = torch.distributions.Normal(means,sig).log_prob(ds_dt_opt_L).sum() / L
        error = -lhood + C*(ctrl._f._f.max_logsig-ctrl._f._f.min_logsig).sum()
        errors.append(error.item())
        error.backward()
        optimizer.step()
        if verbose and i%(Niter/print_times)==0:
            print('Iter={:4d}/{:<4d}   opt_error={:.3f}'.format(i, Niter, np.mean(errors)))
        if (i+1)%(Niter//10) == 0:
            ctrl.save(D, fname=save_fname)

##########################################################################################
################################# DATA COLLECTION ########################################
##########################################################################################
def _clip_actions(env,U):
    if env.ac_lb is not None:
        ac_lb = env.ac_lb.repeat([*(U.shape[:-1]),1])
        U[U<ac_lb] = ac_lb[U<ac_lb]
    if env.ac_ub is not None:
        ac_ub = env.ac_ub.repeat([*(U.shape[:-1]),1])
        U[U>ac_ub] = ac_ub[U>ac_ub]
    return U

def draw_from_gp(inputs, sf, ell, L=1, N=1, n_out=1, eps=1e-5):
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(1) 
    T = inputs.shape[0]
    cov  = K(inputs,inputs,ell,sf,eps=eps) # T,T
    L_ = torch.cholesky(cov) # psd_safe_cholesky(cov)
    # L,N,T,n_out or N,T,n_out or T,n_out
    return L_ @ torch.randn([L,N,T,n_out],device=inputs.device).squeeze(0).squeeze(0)

def obtain_smooth_test_acts(env, T, sf=1.0, ell=0.5, eps=1e-5):
    with torch.no_grad():
        t = torch.arange(T,device=env.device) * env.dt
        acs = draw_from_gp(t, sf, ell, n_out=env.m, eps=eps).to(torch.float32)
        acs = acs * (env.ac_ub-env.ac_lb)/2 # T,m
        return _clip_actions(env,acs) # T,m

def get_kernel_interpolation(env, T, N=1, ell=0.25, sf=0.5):
    with torch.no_grad():
        ell = numpy_to_torch([ell], env.device)
        sf  = numpy_to_torch([sf],  env.device)
        ts = env.dt * torch.arange(T,device=env.device) # T
        smooth_noise = torch.stack([obtain_smooth_test_acts(env,T,sf=sf,ell=ell) for _ in range(N)]) # N,T,1
        ells = torch.stack([ell]*N).unsqueeze(-1) # N,1,1
        sfs  = torch.stack([sf]*N).unsqueeze(-1) # N,1,1
        tss  = torch.stack([ts]*N).unsqueeze(-1) # N,T,1
        kernel_int = KernelInterpolation(sfs, ells, tss, smooth_noise)
        def g(s,t):
            return kernel_int(torch.stack([t.view([1,1])]*N)).squeeze(1)
        return g
    
def build_policy(env, T, g_pol=None, sf=0.1, ell=0.5):
    g_exp = get_kernel_interpolation(env, T, ell=ell, sf=sf)
    tanh_ = torch.nn.Tanh()
    def g(s,t):
        a_pol = g_pol(s,t) if g_pol is not None else 0.0
        a_exp = g_exp(s,t)
        return tanh_(a_pol+a_exp) * (env.ac_ub-env.ac_lb) / 2.0
    return g

def collect_data(env, H, N=1, sf=0.5, ell=0.5, D=None):
    ''' H in seconds '''
    with torch.no_grad():
        if N<1:
            print('Since N<1, data not collected!')
            return D
        T = int(H/env.dt)
        s0 = torch.stack([numpy_to_torch(env.reset(), env.device) for _ in range(N)]) # N,n
        for i in range(N):
            g = build_policy(env,T)
            st,at,rt,ts = env.integrate_system(T, g, s0[i:i+1])
            st_at = torch.cat([st,at],-1) # N,T,n+m
            if D is None:
                D = Dataset(env, st_at, ts)
            else:
                D.add_experience(st_at, ts)
        return D

def collect_test_sequences(ctrl, D, N=1, reset=True, explore=False, sf=0.1):
    with torch.no_grad():
        env = ctrl.env
        Dnew = None
        T = D.T
        if reset:
            s0 = torch.stack([torch.tensor(env.reset(),dtype=torch.float32)\
                   for _ in range(N)]).to(ctrl.device)
        else:
            s0 = get_high_f_uncertainty_iv(ctrl, D, N=N)
        for i in range(N):
            g = build_policy(env, T, sf=sf) if explore else ctrl._g
            st_obs,at,rt,ts = env.integrate_system(T, g, s0[i:i+1])
            st_at = torch.cat([st_obs,at],-1) # T,n+m
            if Dnew is None:
                Dnew = Dataset(env, st_at, ts)
            else:
                Dnew.add_experience(st_at, ts)
        return Dnew


def collect_experience(ctrl, D, N=1, H=None, reset=False, explore=False, sf=0.1):
    with torch.no_grad():
        Dnew = collect_test_sequences(ctrl, D, N=N, reset=reset, explore=explore, sf=sf)
        D.add_experience(Dnew.sa, Dnew.ts)
        return D


##########################################################################################
######################################## UTILS ########################################
##########################################################################################
class TemperedOpt:
    def __init__(self, OPT_CLS, params, lr):
        self.opt = OPT_CLS(params,lr=lr)
        my_lambda = lambda ep: min(10.0,10**(ep/100)) / 10
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=my_lambda)
    def zero_grad(self):
        self.opt.zero_grad()
    def step(self):
        self.opt.step()
        self.scheduler.step()
class OptWrapper:
    def __init__(self, opt_name):
        self.OPT_CLS = get_opt(opt_name)
    def __call__(self,params,lr=1e-3):
        return TemperedOpt(self.OPT_CLS, params, lr)

def get_opt(opt, temp=False):
    if opt=='adam':
        CLS = torch.optim.Adam
    elif opt=='sgd':
        CLS = torch.optim.SGD
    elif opt=='sgld':
        from utils.sgld import SGLD
        CLS = SGLD
    elif opt=='rmsprop':
        CLS = torch.optim.RMSprop
    elif opt=='radam':
        from utils.radam import RAdam
        CLS = RAdam
    else:
        raise ValueError('optimizer parameter is wrong\n')
    if temp:
        return OptWrapper(opt)
    else:
        return CLS


def gradient_match(ctrl, D, Niter=5000, eta=5e-3, verbose=False, L=10, print_times=100, \
                   rep_buf=-1, opt='adam'):
    rep_buf = D.N if rep_buf<1 else rep_buf
    ds = (D.s[-rep_buf:,1:]-D.s[-rep_buf:,:-1]).reshape(-1,ctrl.env.n)
    dt = (D.ts[-rep_buf:,1:] - D.ts[-rep_buf:,:-1]).reshape(-1,1)
    ds_dt_opt = ds / dt # _,n
    s_opt = D.s[-rep_buf:,:-1].reshape(-1,ctrl.env.n) # _,n 
    a_opt = D.a[-rep_buf:,:-1].reshape(-1,ctrl.env.m) # _,m
    errors = []
    L = ctrl.get_L(L=L)
    s_opt,a_opt = torch.stack([s_opt]*L), torch.stack([a_opt]*L)
    OPT_CLS = get_opt(opt)
    optimizer = OPT_CLS(ctrl._f.parameters(),lr=eta)
    for i in range(Niter):
        optimizer.zero_grad()
        f = ctrl.draw_f(L=L)
        # optimize buffer
        ds_dt_opt_hat = ctrl._f.ds_dt(f,s_opt,a_opt) # N*(T-1),n
        opt_error = torch.sum((ds_dt_opt_hat-ds_dt_opt)**2)
        error = opt_error + ctrl._f.kl()
        errors.append(error.item())
        error.backward()
        optimizer.step()
        if verbose and i%(Niter/print_times)==0:
            print('Iter={:4d}/{:<4d}   opt_error={:.3f}'.format(i, Niter, np.mean(errors)))

def get_ivs(env, D, N, rep_buf=None):
    if rep_buf is None and D is not None:
        rep_buf = D.N
    seq_idx = np.arange(D.N)[-rep_buf:]
    s0  = D.s[seq_idx].view([-1,env.n])
    idx = torch.randint(s0.shape[0],[N])
    s0  = torch.stack([s0[idx_.item()] for idx_ in idx])
    return s0.to(env.device)

def get_high_f_uncertainty_iv(ctrl, D, N=1, rep_buf=5, nrep=5, L=10):
    st = get_ivs(ctrl.env, D, 5*N, rep_buf=rep_buf)
    at = ctrl._g(st,None) # nrep,N,m
    rt = ctrl.env.diff_reward(st,at) # nrep,N
    st,at,rt = st.view([-1,D.n]), at.view([-1,D.m]), rt.view([-1])
    L = ctrl.get_L(L=L)
    f = ctrl.draw_f(L=L)
    stL,atL = torch.stack([st]*L),torch.stack([at]*L)
    fL = ctrl._f.ds_dt(f,stL,atL)
    var_ = fL.var(0).mean(-1)
    rt = rt / (rt.max()-rt.min())
    var_ = var_ / (var_.max()-var_.min())
    scores = rt + var_
    winner_idx = scores.argsort().flip(0)
    return st[winner_idx[:N]]






























