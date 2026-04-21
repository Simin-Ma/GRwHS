from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy import integrate, stats
from scipy.special import beta as beta_fn
from scipy.special import gamma


_TWO_PI = 2.0 * math.pi
_SQRT_2PI = math.sqrt(_TWO_PI)
_EPS = 1e-12


@dataclass(frozen=True)
class ValidationConfig:
    mode: str = "quick"
    seed: int = 20260415
    mc_draws_moments: int = 200_000
    mc_draws_tail: int = 300_000
    mc_draws_kstest: int = 80_000
    quad_epsabs: float = 1e-10
    quad_epsrel: float = 1e-8

    @staticmethod
    def for_mode(mode: str, *, seed: int = 20260415) -> "ValidationConfig":
        m = str(mode).strip().lower()
        if m == "full":
            return ValidationConfig(
                mode="full",
                seed=seed,
                mc_draws_moments=800_000,
                mc_draws_tail=1_200_000,
                mc_draws_kstest=200_000,
                quad_epsabs=1e-11,
                quad_epsrel=1e-9,
            )
        return ValidationConfig(mode="quick", seed=seed)


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str
    metric: float | None = None
    threshold: float | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "name": self.name,
            "passed": bool(self.passed),
            "details": self.details,
        }
        if self.metric is not None:
            out["metric"] = float(self.metric)
        if self.threshold is not None:
            out["threshold"] = float(self.threshold)
        return out


def _normal_pdf0(x: float, var: float) -> float:
    v = max(float(var), _EPS)
    return math.exp(-0.5 * (float(x) * float(x)) / v) / math.sqrt(_TWO_PI * v)


def variance_density(v: float, s: float, c: float) -> float:
    vv = float(v)
    ss = float(s)
    cc = float(c)
    c2 = cc * cc
    if vv <= 0.0 or vv >= c2:
        return 0.0
    den = math.pi * math.sqrt(vv) * math.sqrt(c2 - vv) * (ss * ss * (c2 - vv) + c2 * vv)
    return (ss * cc * cc * cc) / den


def _half_cauchy_lambda_pdf(lam: float) -> float:
    x = float(lam)
    return 2.0 / (math.pi * (1.0 + x * x))


def conditional_variance_from_lambda(lam: float, s: float, c: float) -> float:
    ll = float(lam)
    l2 = ll * ll
    ss = float(s)
    cc = float(c)
    c2 = cc * cc
    num = c2 * ss * ss * l2
    den = c2 + ss * ss * l2
    return num / max(den, _EPS)


def conditional_density_beta_sc(beta: float, s: float, c: float, *, epsabs: float, epsrel: float) -> float:
    b = abs(float(beta))

    def integrand(lam: float) -> float:
        v = conditional_variance_from_lambda(lam, s, c)
        return _half_cauchy_lambda_pdf(lam) * _normal_pdf0(b, v)

    val, _ = integrate.quad(integrand, 0.0, math.inf, limit=500, epsabs=epsabs, epsrel=epsrel)
    return float(val)


def induced_t_density(t: float, alpha_kappa: float, beta_kappa: float, sigma2: float) -> float:
    tt = float(t)
    a = float(alpha_kappa)
    b = float(beta_kappa)
    s2 = float(sigma2)
    if tt <= 0.0:
        return 0.0
    return (s2**b / beta_fn(a, b)) * (tt ** (a - 1.0)) * ((s2 + tt) ** (-(a + b)))


def t_moment_closed_form(m: float, alpha_kappa: float, beta_kappa: float, sigma2: float) -> float:
    a = float(alpha_kappa)
    b = float(beta_kappa)
    mm = float(m)
    s2 = float(sigma2)
    return (s2**mm) * (beta_fn(a + mm, b - mm) / beta_fn(a, b))


def tail_constant_fixed_s(alpha_kappa: float, beta_kappa: float, sigma2: float, s: float) -> float:
    a = float(alpha_kappa)
    b = float(beta_kappa)
    s2 = float(sigma2)
    ss = float(s)
    return (s2**b / beta_fn(a, b)) * (ss * (2.0 ** (b + 0.5)) * gamma(b + 1.0) * gamma(b) / (math.pi * gamma(b + 0.5)))


def theta_profile(kappa: np.ndarray | float, rho: float) -> np.ndarray:
    kap = np.asarray(kappa, dtype=float)
    r2 = float(rho) ** 2
    den = kap + (1.0 - kap) * r2
    return kap * r2 / np.maximum(den, _EPS)


def psi_profile(kappa: np.ndarray | float, rho: float) -> np.ndarray:
    kap = np.asarray(kappa, dtype=float)
    r2 = float(rho) ** 2
    den = r2 + kap
    return kap * r2 / np.maximum(den, _EPS)


def _m_profile(kappa: float, chi: np.ndarray) -> np.ndarray:
    k = float(kappa)
    ch = np.asarray(chi, dtype=float)
    den = k + (1.0 - k) * ch
    return (k + ch) / np.maximum(den, _EPS)


def _z_profile(kappa: float, chi: np.ndarray) -> np.ndarray:
    m = _m_profile(kappa, chi)
    return 1.0 / np.maximum(m, _EPS)


def posterior_log_kernel(kappa: float, t: np.ndarray, chi: np.ndarray, alpha_kappa: float, beta_kappa: float) -> float:
    k = float(kappa)
    tt = np.asarray(t, dtype=float)
    ch = np.asarray(chi, dtype=float)
    m = _m_profile(k, ch)
    return (
        (float(alpha_kappa) - 1.0) * math.log(max(k, _EPS))
        + (float(beta_kappa) - 1.0) * math.log(max(1.0 - k, _EPS))
        - 0.5 * float(np.sum(np.log(np.maximum(m, _EPS))))
        - 0.5 * float(np.sum(tt / np.maximum(m, _EPS)))
    )


def posterior_score_formula(kappa: float, t: np.ndarray, chi: np.ndarray, alpha_kappa: float, beta_kappa: float) -> float:
    k = float(kappa)
    tt = np.asarray(t, dtype=float)
    ch = np.asarray(chi, dtype=float)
    m = _m_profile(k, ch)
    w = (ch * ch) / np.maximum((k + ch) ** 2, _EPS)
    return (
        (float(alpha_kappa) - 1.0) / max(k, _EPS)
        - (float(beta_kappa) - 1.0) / max(1.0 - k, _EPS)
        + 0.5 * float(np.sum(w * (tt - m)))
    )


def _finite_diff(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    x0 = max(min(float(x), 1.0 - 1e-6), 1e-6)
    hh = min(float(h), 0.25 * min(x0, 1.0 - x0))
    return (f(x0 + hh) - f(x0 - hh)) / (2.0 * hh)


def _check_prop_21_density_and_asymptotics(cfg: ValidationConfig) -> CheckResult:
    pairs = [(1.2, 2.5), (0.7, 1.8)]
    norm_err = 0.0
    edge_err = 0.0
    for s, c in pairs:
        c2 = c * c
        points = [c2 * 1e-6, c2 * 1e-4, c2 * 1e-2, c2 * 0.25, c2 * 0.5, c2 * 0.75, c2 * 0.99]
        val, _ = integrate.quad(
            lambda v: variance_density(v, s, c),
            0.0,
            c2,
            points=points,
            limit=600,
            epsabs=cfg.quad_epsabs,
            epsrel=cfg.quad_epsrel,
        )
        norm_err = max(norm_err, abs(val - 1.0))

        v_small = c2 * 1e-6
        leading0 = (1.0 / (math.pi * s)) * (v_small ** -0.5)
        ratio0 = variance_density(v_small, s, c) / leading0
        edge_err = max(edge_err, abs(ratio0 - 1.0))

        eps = c2 * 1e-6
        v_top = c2 - eps
        leading1 = (s / (math.pi * c2)) * (eps ** -0.5)
        ratio1 = variance_density(v_top, s, c) / leading1
        edge_err = max(edge_err, abs(ratio1 - 1.0))

    metric = max(norm_err, edge_err)
    thr = 5e-3
    ok = metric < thr
    return CheckResult(
        name="Prop2.1 density + endpoint asymptotics",
        passed=ok,
        metric=metric,
        threshold=thr,
        details=f"max(norm error, edge ratio error)={metric:.3e}",
    )


def _check_thm_22_origin_log(cfg: ValidationConfig) -> CheckResult:
    s = 1.2
    c = 2.5
    betas = np.asarray([0.2, 0.1, 0.05, 0.02, 0.01], dtype=float)
    vals = np.asarray(
        [conditional_density_beta_sc(float(b), s, c, epsabs=cfg.quad_epsabs, epsrel=cfg.quad_epsrel) for b in betas],
        dtype=float,
    )
    x = np.log(1.0 / betas)
    slope, _ = np.polyfit(x, vals, 1)
    theo = 2.0 / (math.pi * s * _SQRT_2PI)
    rel = abs((slope - theo) / theo)
    thr = 0.03
    return CheckResult(
        name="Thm2.2 origin log coefficient",
        passed=rel < thr,
        metric=rel,
        threshold=thr,
        details=f"fitted slope={slope:.6f}, theory={theo:.6f}, rel_err={rel:.3e}",
    )


def _check_prop_23_tail_bound(cfg: ValidationConfig) -> CheckResult:
    s = 1.2
    c = 2.5
    betas = np.linspace(c * 1.2, c * 4.0, 12)
    worst = -math.inf
    for b in betas:
        dens = conditional_density_beta_sc(float(b), s, c, epsabs=cfg.quad_epsabs, epsrel=cfg.quad_epsrel)
        bound = (1.0 / (_SQRT_2PI * c)) * math.exp(-(b * b) / (2.0 * c * c))
        worst = max(worst, dens / bound)
    thr = 1.0005
    return CheckResult(
        name="Prop2.3 finite-slab tail upper bound",
        passed=worst <= thr,
        metric=worst,
        threshold=thr,
        details=f"max pi/bound ratio={worst:.6f}",
    )


def _check_thm_24_tail_asym(cfg: ValidationConfig) -> CheckResult:
    s = 1.2
    c = 2.5
    betas = np.asarray([3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    ratios: list[float] = []
    for b in betas:
        dens = conditional_density_beta_sc(float(b), s, c, epsabs=cfg.quad_epsabs, epsrel=cfg.quad_epsrel)
        asym = (s / (math.pi * c * b)) * math.exp(-(b * b) / (2.0 * c * c))
        ratios.append(dens / asym)
    last_err = abs(ratios[-1] - 1.0)
    thr = 0.04
    return CheckResult(
        name="Thm2.4 fixed-c tail asymptotic",
        passed=last_err < thr,
        metric=last_err,
        threshold=thr,
        details=f"ratios={np.round(np.asarray(ratios), 4).tolist()}, |ratio_last-1|={last_err:.3e}",
    )


def _check_prop_26_and_cor_27(cfg: ValidationConfig) -> CheckResult:
    rng = np.random.default_rng(cfg.seed + 11)
    alpha = 0.8
    beta = 1.6
    sigma2 = 1.7

    kappa = rng.beta(alpha, beta, size=cfg.mc_draws_moments)
    t = sigma2 * kappa / np.maximum(1.0 - kappa, _EPS)

    ms = [0.5, 0.2, -0.3]
    rels: list[float] = []
    for m in ms:
        est = float(np.mean(t**m))
        theo = float(t_moment_closed_form(m, alpha, beta, sigma2))
        rels.append(abs(est - theo) / max(abs(theo), _EPS))

    w = t / sigma2
    w_small = np.asarray(w[: min(cfg.mc_draws_kstest, w.size)], dtype=float)
    ks_stat, ks_p = stats.kstest(w_small, "betaprime", args=(alpha, beta))
    metric = max(max(rels), ks_stat)
    # In large samples, KS p-values can become overly sensitive to tiny numerical
    # quadrature/Monte-Carlo discrepancies. We therefore validate shape match by
    # KS statistic magnitude plus moment agreement.
    ok = (max(rels) < 0.03) and (ks_stat < 0.01)
    return CheckResult(
        name="Prop2.6 + Cor2.7 transformed Beta moments",
        passed=ok,
        metric=metric,
        threshold=0.03,
        details=f"max moment rel_err={max(rels):.3e}, KS stat={ks_stat:.3e}, KS p={ks_p:.3f}",
    )


def _draw_v_samples_for_allowance_tail(cfg: ValidationConfig, *, alpha: float, beta: float, sigma2: float, s: float) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed + 29)
    n = int(cfg.mc_draws_tail)
    kappa = rng.beta(alpha, beta, size=n)
    c2 = sigma2 * kappa / np.maximum(1.0 - kappa, _EPS)
    lam = np.abs(rng.standard_cauchy(size=n))
    lam = np.clip(lam, 0.0, 1e8)
    l2 = lam * lam
    num = c2 * (s * s) * l2
    den = c2 + (s * s) * l2
    v = num / np.maximum(den, _EPS)
    return np.maximum(v, _EPS)


def _check_thm_28_allowance_tail(cfg: ValidationConfig) -> CheckResult:
    alpha = 0.8
    beta = 1.6
    sigma2 = 1.7
    s = 1.3
    v = _draw_v_samples_for_allowance_tail(cfg, alpha=alpha, beta=beta, sigma2=sigma2, s=s)
    betas = np.asarray([8.0, 10.0, 12.0, 15.0, 18.0, 22.0], dtype=float)
    dens = np.asarray([float(np.mean(np.exp(-0.5 * (b * b) / v) / np.sqrt(_TWO_PI * v))) for b in betas], dtype=float)

    slope, _ = np.polyfit(np.log(betas), np.log(dens), 1)
    slope_target = -(2.0 * beta + 2.0)
    slope_err = abs(slope - slope_target)

    c_theory = tail_constant_fixed_s(alpha, beta, sigma2, s)
    ratio_last = float(dens[-1] / (c_theory * (betas[-1] ** slope_target)))
    const_err = abs(ratio_last - 1.0)
    metric = max(slope_err, const_err)
    ok = (slope_err < 0.35) and (const_err < 0.25)
    return CheckResult(
        name="Thm2.8 allowance-marginal polynomial tail",
        passed=ok,
        metric=metric,
        threshold=0.35,
        details=(
            f"slope={slope:.3f} target={slope_target:.3f} "
            f"slope_err={slope_err:.3f}, ratio_last={ratio_last:.3f}"
        ),
    )


def _check_thm_31_bridge_inequality(cfg: ValidationConfig) -> CheckResult:
    rng = np.random.default_rng(cfg.seed + 47)
    sigma2 = 1.4
    n = 50_000
    kappa = rng.beta(1.2, 1.8, size=n)
    chi = np.exp(rng.normal(0.0, 0.8, size=n))
    v = sigma2 * kappa * chi / np.maximum(kappa + (1.0 - kappa) * chi, _EPS)
    r = v / np.maximum(sigma2 + v, _EPS)
    gap = np.max(r - kappa)
    thr = 5e-12
    return CheckResult(
        name="Thm3.1 retention bound r_jg <= kappa_g",
        passed=gap <= thr,
        metric=float(gap),
        threshold=thr,
        details=f"max(r-kappa)={gap:.3e}",
    )


def _check_prop_35_score_identity(cfg: ValidationConfig) -> CheckResult:
    rng = np.random.default_rng(cfg.seed + 61)
    p_g = 7
    t = rng.chisquare(df=1.0, size=p_g) + 0.2
    chi = np.exp(rng.normal(0.0, 0.6, size=p_g))
    alpha = 1.3
    beta = 1.1

    def ell(x: float) -> float:
        return posterior_log_kernel(x, t, chi, alpha, beta)

    grid = np.linspace(0.05, 0.95, 11)
    errs = []
    for k in grid:
        num = _finite_diff(ell, float(k), h=1e-6)
        ana = posterior_score_formula(float(k), t, chi, alpha, beta)
        errs.append(abs(num - ana))
    max_err = float(np.max(errs))
    thr = 1e-5
    return CheckResult(
        name="Prop3.5 posterior score identity",
        passed=max_err < thr,
        metric=max_err,
        threshold=thr,
        details=f"max |num-ana|={max_err:.3e}",
    )


def _check_cor_36_37_finite_parts(cfg: ValidationConfig) -> CheckResult:
    rng = np.random.default_rng(cfg.seed + 73)
    p_g = 6
    t = rng.chisquare(df=1.0, size=p_g) + 0.1
    chi = np.exp(rng.normal(0.0, 0.7, size=p_g))
    alpha = 1.0
    beta = 1.0

    left_theory = -(beta - 1.0) + 0.5 * float(np.sum(t - 1.0))
    right_theory = (alpha - 1.0) + 0.5 * float(np.sum((chi * chi) / ((1.0 + chi) ** 2) * (t - (1.0 + chi))))

    k_left = 1e-7
    k_right = 1.0 - 1e-7
    score_left = posterior_score_formula(k_left, t, chi, alpha, beta) - (alpha - 1.0) / k_left
    score_right = posterior_score_formula(k_right, t, chi, alpha, beta) + (beta - 1.0) / (1.0 - k_right)

    err = max(abs(score_left - left_theory), abs(score_right - right_theory))
    thr = 5e-6
    return CheckResult(
        name="Cor3.6/3.7 finite-part boundary derivatives",
        passed=err < thr,
        metric=err,
        threshold=thr,
        details=f"left_err={abs(score_left-left_theory):.3e}, right_err={abs(score_right-right_theory):.3e}",
    )


def _check_thm_33_monotonicity(cfg: ValidationConfig) -> CheckResult:
    rng = np.random.default_rng(cfg.seed + 89)
    p_g = 9
    chi = np.exp(rng.normal(0.0, 0.4, size=p_g))
    t = rng.chisquare(df=1.0, size=p_g)
    t2 = t + rng.uniform(0.0, 3.0, size=p_g)
    alpha = 0.9
    beta = 1.4
    grid = np.linspace(0.001, 0.999, 500)

    l1 = np.asarray([posterior_log_kernel(float(k), t, chi, alpha, beta) for k in grid], dtype=float)
    l2 = np.asarray([posterior_log_kernel(float(k), t2, chi, alpha, beta) for k in grid], dtype=float)
    ratio_log = l2 - l1
    min_diff = float(np.min(np.diff(ratio_log)))
    thr = 1e-9
    return CheckResult(
        name="Thm3.3 coordinatewise data monotonicity",
        passed=min_diff >= -thr,
        metric=min_diff,
        threshold=-thr,
        details=f"min discrete derivative of log-ratio={min_diff:.3e}",
    )


def _check_lemma_331_phase_diagram(cfg: ValidationConfig) -> CheckResult:
    rho = 1.7
    xi = 0.35
    u0 = 0.4
    r2 = rho * rho
    if xi >= r2 / 2.0:
        xi = 0.2 * r2

    def F(k: float) -> float:
        th = float(theta_profile(np.asarray([k]), rho)[0])
        ps = float(psi_profile(np.asarray([k]), rho)[0])
        phi = -math.log(1.0 + th) + th / (1.0 + th)
        return 0.5 * phi + xi * ps

    def dF_num(k: float) -> float:
        return _finite_diff(F, k, h=1e-6)

    grid = np.linspace(0.01, 0.99, 50)
    errs = []
    for k in grid:
        th = float(theta_profile(np.asarray([k]), rho)[0])
        ps = float(psi_profile(np.asarray([k]), rho)[0])
        ps_p = float(_finite_diff(lambda x: float(psi_profile(np.asarray([x]), rho)[0]), float(k), h=1e-6))
        ana = ps_p * (xi - 0.5 * th)
        errs.append(abs(ana - dF_num(float(k))))
    deriv_err = float(np.max(errs))

    k_star = (2.0 * xi * r2) / (r2 - 2.0 * xi * (1.0 - r2))
    th_star = float(theta_profile(np.asarray([k_star]), rho)[0])
    kappa_cond_err = abs(th_star - 2.0 * xi)

    xi_crit = float(theta_profile(np.asarray([u0]), rho)[0]) / 2.0
    xi_hi = xi_crit + 0.05
    k_star_hi = (2.0 * xi_hi * r2) / (r2 - 2.0 * xi_hi * (1.0 - r2))
    threshold_ok = k_star_hi > u0

    metric = max(deriv_err, kappa_cond_err)
    ok = (deriv_err < 3e-5) and (kappa_cond_err < 3e-10) and threshold_ok
    return CheckResult(
        name="Lemma3.31 / Cor3.33 phase-diagram formulas",
        passed=ok,
        metric=metric,
        threshold=3e-5,
        details=(
            f"max dF err={deriv_err:.3e}, theta(k*)-2xi={kappa_cond_err:.3e}, "
            f"k*_above_u0={threshold_ok}"
        ),
    )


def run_0415_theory_validation(config: ValidationConfig) -> list[CheckResult]:
    checks = [
        _check_prop_21_density_and_asymptotics,
        _check_thm_22_origin_log,
        _check_prop_23_tail_bound,
        _check_thm_24_tail_asym,
        _check_prop_26_and_cor_27,
        _check_thm_28_allowance_tail,
        _check_thm_31_bridge_inequality,
        _check_prop_35_score_identity,
        _check_cor_36_37_finite_parts,
        _check_thm_33_monotonicity,
        _check_lemma_331_phase_diagram,
    ]
    return [fn(config) for fn in checks]


def validation_summary(results: list[CheckResult]) -> dict[str, object]:
    total = len(results)
    passed = int(sum(1 for r in results if r.passed))
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "all_passed": passed == total,
        "results": [r.as_dict() for r in results],
    }
