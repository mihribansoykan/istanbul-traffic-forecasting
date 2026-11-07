"""Rolling-origin backtesting: retrain before every test week, forecast day-ahead, score.

Each fold trains only on data strictly before its test window, exactly as the model would be
used in production (weekly retraining, daily day-ahead forecasts).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from istanbul_traffic.features import CALENDAR_FEATURES, FEATURES, TARGET


@dataclass(frozen=True)
class Fold:
    name: str
    start: str
    end: str  # exclusive
    kind: str  # "holiday" | "regular"

    @property
    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end)


HOLIDAY_FOLDS = [
    Fold("Ramazan Bayramı (Apr 8-14)", "2024-04-08", "2024-04-15", "holiday"),
    Fold("Kurban Bayramı (Jun 14-20)", "2024-06-14", "2024-06-21", "holiday"),
]
REGULAR_FOLDS = [
    Fold(f"Week of {d:%b %d}", f"{d:%Y-%m-%d}", f"{d + pd.Timedelta(days=7):%Y-%m-%d}", "regular")
    for d in pd.date_range("2024-10-21", "2024-12-09", freq="7D")
]
FOLDS = HOLIDAY_FOLDS + REGULAR_FOLDS

LGBM_PARAMS = {
    "n_estimators": 700,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbosity": -1,
}


def fit_lgbm(train: pd.DataFrame, features: list[str], seed: int = 42) -> LGBMRegressor:
    train = train.dropna(subset=[TARGET])
    model = LGBMRegressor(random_state=seed, **LGBM_PARAMS)
    return model.fit(train[features], train[TARGET], categorical_feature=["cell"])


def run_fold(df: pd.DataFrame, fold: Fold, seed: int = 42) -> tuple[pd.DataFrame, LGBMRegressor]:
    train = df[df["date_time"] < fold.start_ts]
    test = df[(df["date_time"] >= fold.start_ts) & (df["date_time"] < fold.end_ts)]
    test = test.dropna(subset=[TARGET])

    no_cal = [f for f in FEATURES if f not in CALENDAR_FEATURES]
    full = fit_lgbm(train, FEATURES, seed)
    ablation = fit_lgbm(train, no_cal, seed)

    out = test[["geohash", "date_time", TARGET, "day_type", "hour"]].rename(columns={TARGET: "y"})
    out["naive_week"] = test["lag_168"].to_numpy()
    out["naive_day"] = test["lag_24"].to_numpy()
    out["lgbm_no_calendar"] = ablation.predict(test[no_cal])
    out["lgbm"] = full.predict(test[FEATURES])
    out["fold"] = fold.name
    out["fold_kind"] = fold.kind
    return out, full


MODELS = {
    "naive_day": "Seasonal naive (same hour yesterday)",
    "naive_week": "Seasonal naive (same hour last week)",
    "lgbm_no_calendar": "LightGBM without holiday/school features",
    "lgbm": "LightGBM (full)",
}


def score(pred: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """MAE, RMSE, WAPE and skill vs. the weekly seasonal naive, computed on common rows."""
    models = models or list(MODELS)
    p = pred.dropna(subset=models)
    naive_mae = np.mean(np.abs(p["y"] - p["naive_week"]))
    rows = {}
    for m in models:
        err = p["y"] - p[m]
        mae = float(np.mean(np.abs(err)))
        rows[MODELS.get(m, m)] = {
            "mae": mae,
            "rmse": float(np.sqrt(np.mean(err**2))),
            "wape": float(np.sum(np.abs(err)) / np.sum(np.abs(p["y"]))),
            "skill_vs_naive_week": float(1 - mae / naive_mae),
            "n": len(p),
        }
    return pd.DataFrame(rows).T


def congestion_metrics(
    pred: pd.DataFrame, free_flow: pd.Series, model: str = "lgbm", ratio: float = 0.6
) -> dict[str, float]:
    """Alert quality when a cell-hour is flagged as congested (speed < ratio x free-flow speed)."""
    ff = pred["geohash"].map(free_flow)
    actual = pred["y"] < ratio * ff
    alert = pred[model] < ratio * ff
    tp = float((actual & alert).sum())
    precision = tp / max(alert.sum(), 1)
    recall = tp / max(actual.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "congested_share": actual.mean()}
