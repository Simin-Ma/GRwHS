from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simulation_project.src.fit_rhs as fit_rhs_module
from grrhs.models.baselines import (
    HorseshoeRegression,
    RegularizedHorseshoeRegression,
)
from simulation_project.src.utils import SamplerConfig, logistic_pseudo_sigma, rhs_style_tau0


def _synthetic_regression(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = 40, 6
    X = rng.normal(size=(n, p)).astype(np.float32)
    beta = np.array([2.0, 0.0, 0.0, -1.5, 0.0, 0.75], dtype=np.float32)
    noise = rng.normal(scale=0.1, size=n).astype(np.float32)
    y = (X @ beta + noise).astype(np.float32)
    return X, y, beta



def test_horseshoe_regression_prefers_signal_features():
    X, y, _ = _synthetic_regression(seed=123)
    model = HorseshoeRegression(
        num_warmup=150,
        num_samples=150,
        num_chains=1,
        target_accept_prob=0.9,
        progress_bar=False,
        seed=2024,
    )
    fitted = model.fit(X, y)

    assert fitted.coef_samples_ is not None
    assert fitted.coef_samples_.shape == (150, X.shape[1])
    assert fitted.tau_samples_ is not None

    preds = fitted.predict(X[:5])
    assert preds.shape == (5,)

    support = np.array([0, 3, 5])
    inactive = np.array([1, 2, 4])
    active_mean = float(np.mean(np.abs(fitted.coef_[support])))
    inactive_mean = float(np.mean(np.abs(fitted.coef_[inactive])))
    assert active_mean > inactive_mean


def test_regularized_horseshoe_reports_posterior_summaries():
    X, y, _ = _synthetic_regression(seed=321)
    model = RegularizedHorseshoeRegression(
        slab_scale=0.5,
        slab_df=4.0,
        num_warmup=120,
        num_samples=120,
        num_chains=1,
        target_accept_prob=0.9,
        progress_bar=False,
        seed=2025,
    )
    fitted = model.fit(X, y)

    assert fitted.lambda_samples_ is not None
    assert fitted.c_samples_ is not None
    summaries = fitted.get_posterior_summaries()
    assert "tau_mean" in summaries
    assert "lambda_mean" in summaries
    assert "c_mean" in summaries
    assert summaries["lambda_mean"].shape == (X.shape[1],)
    assert summaries["c_mean"] > 0.0

    preds = fitted.predict(X)
    mse = float(np.mean((preds - y) ** 2))
    baseline = float(np.var(y))
    assert mse < baseline

    sampler_diag = getattr(fitted, "sampler_diagnostics_", {})
    assert isinstance(sampler_diag, dict)
    hmc_diag = sampler_diag.get("hmc")
    assert isinstance(hmc_diag, dict)
    assert "divergences" in hmc_diag
    assert "ebfmi_min" in hmc_diag
    assert "treedepth_hits" in hmc_diag


def test_logistic_pseudo_sigma_balanced_binary_is_two():
    y = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=float)
    assert np.isclose(logistic_pseudo_sigma(y), 2.0)


def test_fit_rhs_logistic_scales_tau0_with_pseudo_sigma(monkeypatch):
    captured: dict[str, float] = {}

    class _DummyRHS:
        def __init__(self, **kwargs):
            captured["scale_global"] = float(kwargs["scale_global"])
            captured["backend"] = str(kwargs.get("backend"))
            self.backend = captured["backend"]
            self.coef_samples_ = None
            self.coef_ = None
            self.sampler_diagnostics_ = {}

        def fit(self, X, y, groups=None):
            p = int(np.asarray(X).shape[1])
            self.coef_samples_ = np.zeros((2, p), dtype=float)
            self.coef_ = np.zeros((p,), dtype=float)
            self.sampler_diagnostics_ = {"hmc": {"divergences": 0}}
            return self

    monkeypatch.setattr(fit_rhs_module, "RegularizedHorseshoeRegression", _DummyRHS)
    monkeypatch.setattr(
        fit_rhs_module,
        "diagnostics_summary_for_method",
        lambda **kwargs: (1.0, 100.0, 0.0, True, {}),
    )

    X = np.zeros((4, 3), dtype=float)
    y = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=float)
    groups = [[0], [1], [2]]
    sampler = SamplerConfig(chains=1, warmup=2, post_warmup_draws=2)
    out = fit_rhs_module.fit_rhs(X, y, groups, task="logistic", seed=7, p0=1, sampler=sampler)

    expected_tau0 = rhs_style_tau0(n=4, p=3, p0=1) * 2.0
    assert np.isclose(captured["scale_global"], expected_tau0)
    assert captured["backend"].lower() == "numpyro"
    assert out.status == "ok"


def test_regularized_horseshoe_defaults_to_diagonal_mass_matrix():
    model = RegularizedHorseshoeRegression()
    assert model.dense_mass is False


def test_stan_rhs_omits_explicit_logsigma_prior():
    stan_file = ROOT / "grrhs" / "models" / "baselines" / "stan" / "rhs_gaussian_regression.stan"
    text = stan_file.read_text(encoding="utf-8")
    assert "logsigma ~" not in text

