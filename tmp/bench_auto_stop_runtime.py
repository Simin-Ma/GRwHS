import yaml
import numpy as np
from copy import deepcopy
from pathlib import Path

from grrhs.experiments.runner import _fit_model_with_retry, _convergence_config
from data.preprocess import StandardizationConfig


def deep_update(base, upd):
    out = deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def main():
    with open('configs/base.yaml', 'r', encoding='utf-8') as f:
        base = yaml.safe_load(f)
    with open('configs/methods/grrhs_regression.yaml', 'r', encoding='utf-8') as f:
        method = yaml.safe_load(f)

    cfg = deep_update(base, method)

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

    conv = _convergence_config(cfg)
    _, _, summary, attempts, _, _ = _fit_model_with_retry(
        cfg,
        groups,
        p,
        X.astype(np.float32),
        y.astype(np.float32),
        None,
        'regression',
        StandardizationConfig(X='unit_variance', y_center=True),
        conv,
    )

    total_sec = sum(float(a.get('elapsed_sec') or 0.0) for a in attempts)
    print('attempt_count=', len(attempts))
    print('total_elapsed_sec=', round(total_sec, 2))
    for a in attempts:
        print({
            'attempt': int(a.get('attempt') or 0),
            'scale': round(float(a.get('budget_scale') or 0.0), 3),
            'iters': int(a.get('iters') or 0),
            'burn_in': int(a.get('burn_in') or 0),
            'elapsed_sec': round(float(a.get('elapsed_sec') or 0.0), 2),
            'converged': bool(a.get('converged')),
            'failures_top3': list((a.get('failures') or [])[:3]),
        })
    if summary and isinstance(summary, dict):
        b = summary.get('beta', {})
        t = summary.get('tau', {})
        print('final_beta_rhat=', round(float(b.get('rhat_max', float('nan'))), 4))
        print('final_beta_ess=', round(float(b.get('ess_min', float('nan'))), 1))
        print('final_tau_rhat=', round(float(t.get('rhat_max', float('nan'))), 4))
        print('final_tau_ess=', round(float(t.get('ess_min', float('nan'))), 1))


if __name__ == '__main__':
    main()
