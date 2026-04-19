from simulation_project.src.run_experiment import run_exp3_linear_benchmark

if __name__ == "__main__":
    out = run_exp3_linear_benchmark(
        save_dir="ab_runs/exp3_gigg_mmle_n8_fast_20260419",
        profile="full",
        repeats=1,
        methods=["GIGG_MMLE"],
        group_configs=[{"name":"G5x5","group_sizes":[5,5,5,5,5],"active_groups":[0,1]}],
        n_jobs=8,
        max_convergence_retries=0,
        sampler_backend="nuts",
    )
    print(out)
