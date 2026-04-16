import time
import yaml
import numpy as np

from grrhs.models.grrhs_gibbs import GRRHS_Gibbs
from grrhs.diagnostics.convergence import summarize_convergence
from grrhs.experiments.runner import _convergence_config, _check_convergence

with open('configs/base.yaml', 'r', encoding='utf-8') as f:
    base_cfg = yaml.safe_load(f)
conv_cfg = _convergence_config(base_cfg)

rng = np.random.default_rng(20260409)
n, p, G = 140, 48, 6
gs = p // G
groups = [list(range(g * gs, (g + 1) * gs)) for g in range(G)]
X = rng.normal(size=(n, p))
X[:, 1:] += 0.25 * X[:, :-1]
beta = np.zeros(p)
for g, idx in enumerate(groups):
    if g == 0:
        beta[np.array(idx[:3])] = [1.4, -1.0, 0.8]
    elif g == 1:
        beta[np.array(idx[:2])] = [0.7, -0.6]
    elif g == 2:
        beta[np.array(idx[:1])] = [0.45]
y = X @ beta + rng.normal(scale=1.0, size=n)

plans = [
    (1600, 1600),
    (2200, 2200),
]
print('=== high-budget scan ===')
for burnin, kept in plans:
    iters = burnin + kept
    model = GRRHS_Gibbs(
        c=1.0,
        tau0=0.1,
        eta=0.5,
        alpha_c=2.0,
        beta_c=2.0,
        s0=1.0,
        iters=iters,
        burnin=burnin,
        thin=1,
        num_chains=4,
        seed=321,
        use_pcabs_lite=True,
        use_collapsed_scale_updates=True,
        adapt_proposals=True,
        adapt_interval=50,
        adapt_until_frac=0.8,
        adapt_only_during_burnin=True,
    )
    t0 = time.perf_counter()
    model.fit(X, y, groups=groups)
    elapsed = time.perf_counter() - t0

    arrays = {
        'beta': np.asarray(model.coef_samples_),
        'tau': np.asarray(model.tau_samples_),
        'a': np.asarray(model.a_samples_),
        'c2': np.asarray(model.c2_samples_),
        'lambda': np.asarray(model.lambda_samples_),
    }
    summary = summarize_convergence(arrays, min_chains_for_rhat=int(conv_cfg.get('min_chains_for_rhat', 4)))
    ok, failures = _check_convergence(summary, conv_cfg, model_name='grrhs_gibbs', sampler_diagnostics=model.sampler_diagnostics_)

    print(f"burnin={burnin}, kept={kept}, iters={iters}, time={elapsed:.2f}s, converged={ok}")
    if not ok:
        print('  top failures:', '; '.join(failures[:6]))
        print(f"  beta_rhat={summary['beta'].get('rhat_max'):.4f}, beta_ess={summary['beta'].get('ess_min'):.1f}, tau_ess={summary['tau'].get('ess_min'):.1f}")
    else:
        print(f"  beta_rhat={summary['beta'].get('rhat_max'):.4f}, beta_ess={summary['beta'].get('ess_min'):.1f}, tau_ess={summary['tau'].get('ess_min'):.1f}")
