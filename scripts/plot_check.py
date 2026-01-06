"""Generate standard and posterior plots for a GRRHS run."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grrhs.viz import plots
from grrhs.models.baselines import Ridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sanity and posterior plots for a GRRHS run")
    parser.add_argument("run_dir", nargs="?", default=None, help="Path to run directory")
    parser.add_argument("--out", type=Path, default=None, help="Destination directory for plots")
    return parser.parse_args()


def resolve_run(run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg).expanduser().resolve()
        if not run_dir.exists():
            raise SystemExit(f"Run directory {run_dir} not found")
        return run_dir
    runs_root = Path("outputs/runs")
    if not runs_root.exists():
        raise SystemExit("outputs/runs does not exist; provide run_dir explicitly")
    candidates = sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit("No run directories available")
    return candidates[0]


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_plots(run_dir: Path, out_dir: Path) -> None:
    data = np.load(run_dir / "dataset.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    out_dir.mkdir(parents=True, exist_ok=True)

    save_fig(plots.scatter_truth_vs_pred(y_test, y_pred, title="Pred vs Truth").figure, out_dir / "scatter_pred_vs_truth.png")
    save_fig(plots.residual_histogram(y_test, y_pred, title="Residuals").figure, out_dir / "residual_hist.png")
    save_fig(plots.prediction_over_index(y_test, y_pred, title="Prediction Trajectory").figure, out_dir / "prediction_over_index.png")
    save_fig(plots.coefficient_bar(ridge.coef_, sort=True, title="Coefficient Magnitudes").figure, out_dir / "coefficients_sorted.png")
    save_fig(plots.plot_placeholder(), out_dir / "placeholder.png")

    posterior_path = run_dir / "posterior_samples.npz"
    if posterior_path.exists():
        samples = np.load(posterior_path)
        if "beta" in samples:
            beta = samples["beta"].reshape(-1, samples["beta"].shape[-1])
            idx = 0
            trace_fig, trace_ax = plt.subplots()
            trace_ax.plot(beta[:, idx])
            trace_ax.set_title(f"Beta[{idx}] trace")
            trace_ax.set_xlabel("Draw")
            trace_ax.set_ylabel("Value")
            save_fig(trace_fig, out_dir / "posterior_trace_beta0.png")

            hist_fig, hist_ax = plt.subplots()
            hist_ax.hist(beta[:, idx], bins=30, alpha=0.8)
            hist_ax.set_title(f"Beta[{idx}] posterior")
            hist_ax.set_xlabel("Value")
            hist_ax.set_ylabel("Frequency")
            save_fig(hist_fig, out_dir / "posterior_hist_beta0.png")

        for key in ["tau", "sigma", "sigma2"]:
            if key in samples:
                arr = samples[key].reshape(-1)
                fig, ax = plt.subplots()
                ax.plot(arr)
                ax.set_title(f"{key} trace")
                ax.set_xlabel("Draw")
                ax.set_ylabel(key)
                save_fig(fig, out_dir / f"posterior_trace_{key}.png")

                fig, ax = plt.subplots()
                ax.hist(arr, bins=30, alpha=0.8)
                ax.set_title(f"{key} posterior")
                ax.set_xlabel(key)
                ax.set_ylabel("Frequency")
                save_fig(fig, out_dir / f"posterior_hist_{key}.png")

    print(f"[OK] Plots written to {out_dir}")


def main() -> None:
    args = parse_args()
    run_dir = resolve_run(args.run_dir)
    out_dir = args.out or (run_dir / "plots_check")
    generate_plots(run_dir, out_dir)


if __name__ == "__main__":
    main()
