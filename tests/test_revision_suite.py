from __future__ import annotations

import json

import numpy as np

from grrhs.simulations.revision_suite import profile_kappa_posterior_summary, run_revision_suite


def test_profile_kappa_summary_is_well_formed():
    y = np.zeros(20, dtype=float)
    summary = profile_kappa_posterior_summary(y, tau=0.1, sigma=1.0, alpha_kappa=1.0, beta_kappa=1.0)
    assert 0.0 < summary["mean"] < 1.0
    assert 0.0 < summary["median"] < 1.0
    assert 0.0 < summary["mode"] < 1.0


def test_revision_suite_quick_theory_and_hyper_runs(tmp_path):
    result = run_revision_suite(
        output_dir=tmp_path,
        sections=["theory", "hyper"],
        quick=True,
        seed=123,
        methods=["grrhs", "rhs", "hs"],
    )
    summary_path = tmp_path / "revision_suite_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "theory" in payload
    assert "hyper" in payload
    assert (tmp_path / "theory" / "theory_null_contraction.csv").exists()
    assert (tmp_path / "hyper" / "hyper_tau_meff.csv").exists()
    assert result["summary_path"].endswith("revision_suite_summary.json")
