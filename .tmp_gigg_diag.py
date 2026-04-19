from simulation_project.src.run_experiment import _fit_all_methods, _build_benchmark_beta, experiment_seed, _gigg_config_for_profile
from simulation_project.src.utils import SamplerConfig, canonical_groups, sample_correlated_design
from simulation_project.src.dgp_grouped_linear import sigma2_for_target_snr
import numpy as np

settings=[]
signals=['concentrated','distributed','boundary']
rhos=[0.3,0.8]
snrs=[0.5,2.0]
sid=0
for sig in signals:
    for rho in rhos:
        for snr in snrs:
            sid+=1
            settings.append((sid,sig,rho,snr))

for sid,sig,rho,snr in settings:
    r=1
    s=experiment_seed(3,sid,r,master_seed=20260415)
    gsz=[5,5,5,5,5]
    groups=canonical_groups(gsz)
    X,cov_x=sample_correlated_design(n=100,group_sizes=gsz,rho_within=rho,rho_between=0.1,seed=s)
    beta0=_build_benchmark_beta(sig,gsz)
    if sig=='boundary':
        sigma2=1.0
    else:
        sigma2=sigma2_for_target_snr(beta=beta0,cov_x=cov_x,target_snr=snr)
    y=X@beta0 + np.random.default_rng(s+17).normal(0.0,float(np.sqrt(sigma2)),100)
    p0=int(np.sum(np.abs(beta0)>1e-12))
    p0g=int(np.sum([int(np.any(np.abs(beta0[np.asarray(g,dtype=int)])>1e-12)) for g in groups]))

    fits=_fit_all_methods(
        X,y,groups,task='gaussian',seed=s,p0=p0,p0_groups=p0g,
        sampler=SamplerConfig(),grrhs_kwargs={'backend':'nuts','tau_target':'groups'},
        methods=['GIGG_MMLE'],gigg_config=_gigg_config_for_profile('full'),
        enforce_bayes_convergence=True,max_convergence_retries=1,
    )
    res=fits['GIGG_MMLE']
    cd=(res.diagnostics or {}).get('convergence_detail',{})
    beta=cd.get('beta',{}) if isinstance(cd,dict) else {}
    gamma=cd.get('gamma2',{}) if isinstance(cd,dict) else {}
    print(f"sid={sid:02d} {sig:12s} rho={rho} snr={snr} status={res.status:5s} conv={res.converged} attempts={(res.diagnostics or {}).get('convergence_retry',{}).get('attempts_used')} rhat={res.rhat_max:.4f} ess={res.bulk_ess_min:.1f} beta_rhat={beta.get('rhat_max')} beta_ess={beta.get('ess_min')} gamma_rhat={gamma.get('rhat_max')} gamma_ess={gamma.get('ess_min')}")
