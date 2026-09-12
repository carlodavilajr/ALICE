"""Chart rendering. Each function returns a base64-encoded PNG string."""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#0F6E77", "#B5762C", "#5B4B8A", "#3A7D44", "#A23E48", "#4A6C8C"]
INK = "#1B2431"
GRID = "#D9DEE4"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.dpi": 120,
})


def _encode(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _x_values(df: pd.DataFrame, time_col: str | None):
    if time_col is None:
        return np.arange(len(df))
    x = df[time_col]
    if x.dtype == object:
        try:
            return pd.to_datetime(x)
        except (ValueError, TypeError):
            return np.arange(len(df))
    return x


def line_chart(df: pd.DataFrame, series: dict[str, pd.Series], time_col: str | None, title: str, ylabel: str = "") -> str:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = _x_values(df, time_col)
    for i, (name, s) in enumerate(series.items()):
        ax.plot(x, s.to_numpy(), label=name, color=PALETTE[i % len(PALETTE)], linewidth=2)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12)
    ax.set_xlabel(time_col or "observation")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color=GRID, linewidth=0.8) if any((s < 0).any() for s in series.values()) else None
    if len(series) > 1:
        ax.legend(frameon=False)
    return _encode(fig)


def bar_chart(df: pd.DataFrame, series: dict[str, pd.Series], time_col: str | None, title: str, ylabel: str = "") -> str:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(df))
    labels = df[time_col].astype(str).to_numpy() if time_col else x
    k = len(series)
    width = 0.8 / max(k, 1)
    for i, (name, s) in enumerate(series.items()):
        ax.bar(x + i * width - 0.4 + width / 2, s.to_numpy(), width=width, label=name, color=PALETTE[i % len(PALETTE)])
    step = max(1, len(x) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=45, ha="right")
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    if k > 1:
        ax.legend(frameon=False)
    return _encode(fig)


def scatter_regression(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, fit: dict | None, labels: pd.Series | None = None) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(x, y, color=PALETTE[0], alpha=0.85, s=38, edgecolor="white", linewidth=0.6)
    if labels is not None and len(x) <= 60:
        for xi, yi, lab in zip(x, y, labels):
            if pd.notna(xi) and pd.notna(yi):
                ax.annotate(str(lab), (xi, yi), fontsize=7, color="#5C6773", xytext=(3, 3), textcoords="offset points")
    if fit:
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, fit["slope"] * xs + fit["intercept"], color=PALETTE[1], linewidth=2,
                label=f"OLS fit, R² = {fit['r_squared']:.3f}")
        ax.legend(frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} against {xlabel}", loc="left", fontsize=13, color=INK, pad=12)
    return _encode(fig)


def histogram(s: pd.Series, name: str, bins: int = 20) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.hist(s.dropna(), bins=bins, color=PALETTE[0], edgecolor="white")
    ax.axvline(s.mean(), color=PALETTE[1], linewidth=2, label=f"mean = {s.mean():.3g}")
    ax.set_title(f"Distribution of {name}", loc="left", fontsize=13, color=INK, pad=12)
    ax.set_xlabel(name)
    ax.set_ylabel("count")
    ax.legend(frameon=False)
    return _encode(fig)


def heatmap(columns: list[str], matrix: list[list[float]]) -> str:
    m = np.array([[np.nan if v is None else v for v in row] for row in matrix], dtype=float)
    n = len(columns)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2.5, 1.0 * n + 1.5))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            v = m[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.55 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation matrix", loc="left", fontsize=13, color=INK, pad=12)
    return _encode(fig)
