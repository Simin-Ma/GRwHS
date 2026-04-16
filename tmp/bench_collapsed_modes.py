import time
import numpy as np
import yaml
from grrhs.models.grrhs_gibbs import GRRHS_Gibbs
from grrhs.diagnostics.convergence import summarize_convergence
from grrhs.experiments.runner import _convergence_config, _check_convergence


def dataset(seed=20260409):
    rng=np.random.default_rng(seed)
    n,p,G=140,48,6
    gs=p//G
    groups=[list(range(g*gs,(g+1)*gs)) for g in range(G)]
    X=rng.normal(size=(n,p)); X[:,1:]+=0.25*X[:,:-1]
    b=np.zeros(p)
    for g,idx in enumerate(groups):
        if g==0: b[np.array(idx[:3])] = [1.4,-1.0,0.8]
        elif g==1: b[np.array(idx[:2])] = [0.7,-0.6]
        elif g==2: b[np.array(idx[:1])] = [0.45]
    y=X@b+rng.normal(scale=1.0,size=n)
    return X,y,groups


def run(tag, **kw):
    with open('configs/base.yaml','r',encoding='utf-8') as f:
        conv_cfg = _convergence_config(yaml.safe_load(f))
    X,y,groups = dataset()
    m=GRRHS_Gibbs(c=1.0,tau0=0.1,eta=0.5,alpha_c=2.0,beta_c=2.0,s0=1.0,
                  iters=4400,burnin=2200,thin=1,num_chains=4,seed=321,
                  adapt_proposals=True,adapt_only_during_burnin=True,
                  tau2_refresh_steps=6,use_tau_slice_refresh=True,
                  global_block_sd_u=0.2,global_block_sd_alpha=0.25,global_comp_sd=0.16,adapt_step_size=0.08,
                  **kw)
    t0=time.perf_counter(); m.fit(X,y,groups=groups); dt=time.perf_counter()-t0
    arr={'beta':np.asarray(m.coef_samples_),'tau':np.asarray(m.tau_samples_),'a':np.asarray(m.a_samples_),'c2':np.asarray(m.c2_samples_),'lambda':np.asarray(m.lambda_samples_)}
    s=summarize_convergence(arr,min_chains_for_rhat=int(conv_cfg.get('min_chains_for_rhat',4)))
    ok, fails=_check_convergence(s,conv_cfg,model_name='grrhs_gibbs',sampler_diagnostics=m.sampler_diagnostics_)
    print(f"{tag}: {dt:.2f}s ok={ok} beta_ess={s['beta']['ess_min']:.1f} tau_ess={s['tau']['ess_min']:.1f} beta_rhat={s['beta']['rhat_max']:.4f}")

if __name__=='__main__':
    run('collapsed_true', use_collapsed_scale_updates=True)
    run('collapsed_false', use_collapsed_scale_updates=False)
