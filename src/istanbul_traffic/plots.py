"""Report figures with one consistent, colour-blind-checked style."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

if "ipykernel" not in sys.modules:  # headless backend for CLI/CI; keep inline plots in notebooks
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
MODEL_COLORS = {
    "Seasonal naive (same hour yesterday)": SERIES[3],
    "Seasonal naive (same hour last week)": SERIES[1],
    "LightGBM without holiday/school features": SERIES[2],
    "LightGBM (full)": SERIES[0],
}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        }
    )


def save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def cell_map(catalogue: pd.DataFrame, selected: list[str], path: str | Path) -> Path:
    """Morning-peak congestion of every well-covered grid cell; modelled cells are outlined."""
    apply_style()
    cat = catalogue[(catalogue["coverage"] >= 0.5) & catalogue["congestion_index"].notna()]
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sc = ax.scatter(
        cat["longitude"],
        cat["latitude"],
        c=cat["congestion_index"],
        cmap=cmap,
        s=9,
        vmin=0,
        vmax=0.6,
        marker="s",
        linewidths=0,
    )
    sel = cat.loc[cat.index.intersection(selected)]
    ax.scatter(
        sel["longitude"],
        sel["latitude"],
        s=28,
        facecolors="none",
        edgecolors=SERIES[1],
        linewidths=1.1,
        label=f"{len(sel)} modelled cells (busiest, ≥97 % coverage)",
    )
    ax.set_aspect(1 / np.cos(np.deg2rad(41.0)))
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, format=mticker.PercentFormatter(1))
    cb.set_label("Speed loss at morning peak vs. night (free-flow)")
    ax.set_title("Istanbul morning-peak congestion by geohash cell (school days 07–10 h, 2023–24)")
    return save(fig, path)


def hour_dow_heatmap(df: pd.DataFrame, path: str | Path) -> Path:
    apply_style()
    piv = df.pivot_table(index="dow", columns="hour", values="average_speed", aggfunc="mean")
    cmap = LinearSegmentedColormap.from_list("blue_r", BLUE_RAMP[::-1])
    fig, ax = plt.subplots(figsize=(12, 3.8))
    im = ax.imshow(piv.to_numpy(), cmap=cmap, aspect="auto")
    ax.set_yticks(range(7), DOW)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.grid(False)
    ax.set(xlabel="Hour of day", title="Average speed by weekday and hour (darker = slower)")
    fig.colorbar(im, ax=ax, fraction=0.02, label="km/h")
    return save(fig, path)


def daily_profile(daily: pd.DataFrame, cal_days: pd.DataFrame, path: str | Path) -> Path:
    """Daily vehicles and speed through 2024 with religious holidays and school breaks marked."""
    apply_style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    a1.plot(daily.index, daily["vehicles"] / 1e3, color=SERIES[0], lw=1.4)
    a2.plot(daily.index, daily["speed"], color=SERIES[0], lw=1.4)
    for ax in (a1, a2):
        for d in cal_days.index[cal_days["is_religious_holiday"] == 1]:
            ax.axvspan(d, d + pd.Timedelta(days=1), color=SERIES[1], alpha=0.25, lw=0)
        brk = cal_days["school_break"]
        starts = brk.index[brk & ~brk.shift(1, fill_value=False)]
        ends = brk.index[brk & ~brk.shift(-1, fill_value=False)]
        for s, e in zip(starts, ends, strict=True):
            ax.axvspan(s, e + pd.Timedelta(days=1), color=MUTED, alpha=0.12, lw=0)
    a1.set(title="Daily traffic in the modelled cells, 2024", ylabel="Vehicles (thousands)")
    a2.set(ylabel="Mean speed (km/h)")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    a1.text(
        0.995,
        0.97,
        "orange = Ramazan / Kurban Bayramı · grey = school breaks · gaps = missing source data",
        transform=a1.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=INK_2,
    )
    return save(fig, path)


def fold_dotplot(per_fold: pd.DataFrame, path: str | Path) -> Path:
    """MAE per fold (rows) and model (dots)."""
    apply_style()
    folds = list(dict.fromkeys(per_fold["fold"]))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(folds) + 1.5))
    for model, color in MODEL_COLORS.items():
        sub = per_fold[per_fold["model"] == model].set_index("fold").reindex(folds)
        ax.scatter(
            sub["mae"],
            range(len(folds)),
            color=color,
            s=46,
            label=model,
            zorder=3,
            edgecolors=SURFACE,
            linewidths=1.5,
        )
    ax.set_yticks(range(len(folds)), folds)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="MAE (km/h) ↓", title="Backtest error by test week")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    return save(fig, path)


def forecast_example(pred: pd.DataFrame, path: str | Path, title: str) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pred["date_time"], pred["y"], color=INK, lw=1.6, label="actual")
    ax.plot(
        pred["date_time"], pred["naive_week"], color=SERIES[1], lw=1.4, label="same hour last week"
    )
    ax.plot(pred["date_time"], pred["lgbm"], color=SERIES[0], lw=1.8, label="LightGBM (full)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.set(title=title, ylabel="Average speed (km/h)")
    ax.legend(loc="lower left", ncol=3)
    return save(fig, path)


def error_by_hour(pred: pd.DataFrame, models: dict[str, str], path: str | Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    for col, name in models.items():
        mae = (pred["y"] - pred[col]).abs().groupby(pred["hour"]).mean()
        ax.plot(mae.index, mae.to_numpy(), color=MODEL_COLORS[name], label=name, marker="o", ms=3)
    ax.set_xticks(range(0, 24, 2))
    ax.set(title="Error by hour of day (all test weeks)", xlabel="Hour", ylabel="MAE (km/h)")
    ax.legend(loc="upper left")
    return save(fig, path)


def importance(mean_abs: pd.Series, path: str | Path) -> Path:
    apply_style()
    s = mean_abs.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(s.index, s.to_numpy(), color=SERIES[0], height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set(title="LightGBM drivers (mean |SHAP|, km/h)", xlabel="Mean |SHAP value|")
    return save(fig, path)
