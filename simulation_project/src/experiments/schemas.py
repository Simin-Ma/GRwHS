from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunCommonConfig:
    n_jobs: int
    seed: int
    save_dir: str
    profile: str
    enforce_bayes_convergence: bool
    max_convergence_retries: int | None
    until_bayes_converged: bool
    sampler_backend: str

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "n_jobs": int(self.n_jobs),
            "seed": int(self.seed),
            "save_dir": str(self.save_dir),
            "profile": str(self.profile),
            "enforce_bayes_convergence": bool(self.enforce_bayes_convergence),
            "max_convergence_retries": self.max_convergence_retries,
            "until_bayes_converged": bool(self.until_bayes_converged),
            "sampler_backend": str(self.sampler_backend),
        }


@dataclass(frozen=True)
class RunManifest:
    exp_key: str
    timestamp: str
    run_dir: str
    result_paths: dict[str, Any]
    run_summary_table: str
    run_summary_md: str
    run_analysis_json: str
    archived_artifacts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "exp_key": str(self.exp_key),
            "timestamp": str(self.timestamp),
            "run_dir": str(self.run_dir),
            "result_paths": dict(self.result_paths),
            "run_summary_table": str(self.run_summary_table),
            "run_summary_md": str(self.run_summary_md),
            "run_analysis_json": str(self.run_analysis_json),
            "archived_artifacts": list(self.archived_artifacts),
        }
