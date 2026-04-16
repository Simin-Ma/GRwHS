import numpy as np
from grrhs.experiments.runner import _fit_model_with_retry, _convergence_config
from data.preprocess import StandardizationConfig

cfg = {
  'model': {
    'name':'grrhs_gibbs','c':1.0,'eta':0.5,'alpha_c':2.0,'beta_c':2.0,'s0':1.0,
    'iters':4400,'use_pcabs_lite':True,'use_collapsed_scale_updates':False,
    'tau': {'mode':'fixed','value':0.1}
  },
  'inference': {
    'gibbs': {
      'burn_in':2200,'thin':1,'num_chains':4,'seed':321,
      'adapt_proposals':True,'adapt_only_during_burnin':True,
      'tau2_refresh_steps':6,'use_tau_slice_refresh':True,
    }
  },
  'experiments': {
    'bayesian_fairness': {'enabled':True,'disable_budget_retry':False,'enforce_shared_sampling_budget':False},
    'convergence': {
      'enabled': True, 'max_rhat':1.02, 'min_ess':50,
      'min_ess_by_block': {'beta':120,'tau':200,'a':120,'c2':120,'lambda':80},
      'min_chains_for_rhat':4, 'max_retries':2, 'retry_scale':2.0,
      'auto_stop': {'enabled': True,'initial_scale': 0.5,'growth': 1.6}
    }
  }
}

conv = _convergence_config(cfg)
rng=np.random.default_rng(20260409)
n,p,G=140,48,6
gs=p//G
groups=[list(range(g*gs,(g+1)*gs)) for g in range(G)]
X=rng.normal(size=(n,p)); X[:,1:]+=0.25*X[:,:-1]
b=np.zeros(p); b[:3]=[1.2,-0.9,0.7]
y=X@b+rng.normal(size=n)

_, _, _, attempts, _, _ = _fit_model_with_retry(
    cfg, groups, p, X.astype(np.float32), y.astype(np.float32), None,
    'regression', StandardizationConfig(X='unit_variance', y_center=True), conv,
)
print('attempt_count=', len(attempts))
print('scales=', [round(float(a.get('budget_scale') or 0.0), 3) for a in attempts])
print('iters=', [int(a.get('iters') or 0) for a in attempts])
