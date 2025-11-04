from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


@dataclass
class MethodResult:
    name: str
    threshold: float
    metrics: Dict[str, float]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _compute_threshold(probs: np.ndarray, labels: np.ndarray, *, grid: Iterable[float]) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    best_acc = -1.0
    for t in grid:
        preds = (probs >= t).astype(int)
        if len(np.unique(preds)) < 2:
            f1 = 0.0
        else:
            f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        if f1 > best_f1 + 1e-9 or (abs(f1 - best_f1) < 1e-9 and acc > best_acc):
            best_thr = float(t)
            best_f1 = f1
            best_acc = acc
    return best_thr


def _evaluate(prob: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    prob = np.clip(prob, 1e-9, 1 - 1e-9)
    preds = (prob >= threshold).astype(int)
    return {
        "ClassAccuracy": float(accuracy_score(labels, preds)),
        "ClassF1": float(f1_score(labels, preds)),
        "ClassAUROC": float(roc_auc_score(labels, prob)),
        "ClassAveragePrecision": float(average_precision_score(labels, prob)),
        "ClassLogLoss": float(log_loss(labels, prob)),
        "ClassBrier": float(brier_score_loss(labels, prob)),
    }


def calibrate_repeat(run_dir: Path) -> Dict[str, Dict[str, float]]:
    dataset = np.load(run_dir / "dataset.npz")
    post = np.load(run_dir / "posterior_samples.npz")

    X_val = dataset["X_val"]
    y_val = dataset["y_val"]
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]

    coef = post["beta"].mean(axis=0)
    logits_val = X_val @ coef
    logits_test = X_test @ coef
    prob_val = _sigmoid(logits_val)
    prob_test = _sigmoid(logits_test)

    grid = np.linspace(0.01, 0.99, 199)

    base_thr = _compute_threshold(prob_val, y_val, grid=grid)
    base_metrics = _evaluate(prob_test, y_test, base_thr)

    # Platt scaling (logistic regression on logits)
    X_platt = logits_val.reshape(-1, 1)
    platt = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs")
    platt.fit(X_platt, y_val)
    prob_val_platt = platt.predict_proba(X_platt)[:, 1]
    prob_test_platt = platt.predict_proba(logits_test.reshape(-1, 1))[:, 1]
    platt_thr = _compute_threshold(prob_val_platt, y_val, grid=grid)
    platt_metrics = _evaluate(prob_test_platt, y_test, platt_thr)

    # Isotonic regression on probabilities
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_val, y_val)
    prob_val_iso = iso.predict(prob_val)
    prob_test_iso = iso.predict(prob_test)
    iso_thr = _compute_threshold(prob_val_iso, y_val, grid=grid)
    iso_metrics = _evaluate(prob_test_iso, y_test, iso_thr)

    return {
        "baseline": {"threshold": base_thr, **base_metrics},
        "platt": {"threshold": platt_thr, **platt_metrics},
        "isotonic": {"threshold": iso_thr, **iso_metrics},
    }


def aggregate(results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    summary: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for method, metrics_list in results.items():
        summary[method] = {}
        keys = metrics_list[0].keys()
        for key in keys:
            vals = np.array([m[key] for m in metrics_list], dtype=float)
            summary[method][key] = (float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
    return summary


def main() -> None:
    run_root = Path("outputs/runs/scenarioB_cls-20251019-205003")
    repeats = sorted(run_root.glob("repeat_*"))
    per_method_results: Dict[str, list] = {"baseline": [], "platt": [], "isotonic": []}
    per_repeat_output: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rep in repeats:
        metrics = calibrate_repeat(rep)
        per_repeat_output[rep.name] = metrics
        for method, vals in metrics.items():
            per_method_results[method].append(vals)

    summary = aggregate(per_method_results)

    payload = {
        "repeats": per_repeat_output,
        "summary": summary,
    }

    out_path = run_root / "calibration_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote calibration summary to {out_path}")


if __name__ == "__main__":
    main()
