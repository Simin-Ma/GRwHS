from simulation_project.src.run_experiment import run_exp4_variant_ablation

paths = run_exp4_variant_ablation(
    save_dir="simulation_project",
    profile="full",
    repeats=20,
    p0_list=[5, 30],
    include_oracle=True,
    max_convergence_retries=0,
    sampler_backend="collapsed",
    n_jobs=6,
)
print(paths)
