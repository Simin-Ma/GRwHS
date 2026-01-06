"""Plotting utilities for GRRHS experiments."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _get_fig_ax(ax: Optional[plt.Axes]) -> Tuple[plt.Figure, plt.Axes]:
    """Return a figure/axes pair, creating one if needed."""
    if ax is None:
        fig, new_ax = plt.subplots()
        return fig, new_ax
    return ax.figure, ax


def scatter_truth_vs_pred(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Scatter plot comparing predictions with ground truth."""
    fig, ax = _get_fig_ax(ax)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    ax.scatter(y_true, y_pred, alpha=0.7)

    if y_true.size:
        lims = np.array([y_true.min(), y_true.max(), y_pred.min(), y_pred.max()])
        low, high = lims.min(), lims.max()
        ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)

    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")
    ax.set_title(title or "Predictions vs truth")
    fig.tight_layout()
    return ax


def residual_histogram(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Histogram of prediction residuals."""
    fig, ax = _get_fig_ax(ax)
    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    ax.hist(residuals, bins=bins, alpha=0.8)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title(title or "Residual distribution")
    fig.tight_layout()
    return ax


def coefficient_bar(
    coefficients: Sequence[float],
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    sort: bool = False,
) -> plt.Axes:
    """Bar plot of coefficient magnitudes."""
    fig, ax = _get_fig_ax(ax)
    coef = np.asarray(coefficients).ravel()
    idx = np.arange(coef.size)
    if sort:
        order = np.argsort(np.abs(coef))[::-1]
        coef = coef[order]
        idx = np.arange(coef.size)
    ax.bar(idx, coef)
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Value")
    ax.set_title(title or "Coefficient values")
    fig.tight_layout()
    return ax


def prediction_over_index(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """Line plot showing predictions and truth across sample index."""
    fig, ax = _get_fig_ax(ax)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    x = np.arange(y_true.size)
    ax.plot(x, y_true, label="Truth")
    ax.plot(x, y_pred, label="Prediction")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Value")
    ax.set_title(title or "Prediction trajectory")
    ax.legend()
    fig.tight_layout()
    return ax


def plot_placeholder() -> plt.Figure:
    """Placeholder plot generator."""
    fig, ax = plt.subplots()
    ax.set_title("GRRHS Plot Placeholder")
    fig.tight_layout()
    return fig
