"""Leakage-safe features for day-ahead forecasting.

The forecast for every hour of day D is issued at midnight, so every feature built from the
target series uses information that is at least 24 hours old (all lags >= 24 h).
"""

from __future__ import annotations

import pandas as pd

from istanbul_traffic.calendar import calendar_features

TARGET = "average_speed"
LAGS = [24, 48, 72, 168, 336]
MIN_LAG = 24


def add_lag_features(panel: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Per-cell lags and rolling statistics. ``panel`` must be a complete cell x hour grid."""
    df = panel.sort_values(["geohash", "date_time"]).reset_index(drop=True)
    g = df.groupby("geohash", sort=False)[target]
    for k in LAGS:
        df[f"lag_{k}"] = g.shift(k)
    same_hour = [g.shift(168 * w) for w in range(1, 5)]
    df["same_hour_4w_mean"] = pd.concat(same_hour, axis=1).mean(axis=1)
    base = g.shift(MIN_LAG)
    by_cell = base.groupby(df["geohash"], sort=False)
    df["roll24_mean"] = by_cell.transform(lambda s: s.rolling(24, min_periods=12).mean())
    df["roll168_mean"] = by_cell.transform(lambda s: s.rolling(168, min_periods=84).mean())
    df["roll168_std"] = by_cell.transform(lambda s: s.rolling(168, min_periods=84).std())
    veh = df.groupby("geohash", sort=False)["number_of_vehicles"]
    df["veh_lag24"] = veh.shift(24)
    df["veh_lag168"] = veh.shift(168)
    return df


def build_features(panel: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Join lag, calendar and static cell features. ``cells`` is indexed by geohash."""
    df = add_lag_features(panel)
    cal = calendar_features(pd.DatetimeIndex(df["date_time"].unique()))
    df = df.join(cal, on="date_time")

    # Was the lagged hour itself a holiday? Lets the model discount "last week was Bayram".
    hol = cal["is_holiday"]
    for k in (24, 168):
        df[f"lag{k}_was_holiday"] = (
            hol.reindex(df["date_time"] - pd.Timedelta(hours=k)).fillna(0).to_numpy().astype(int)
        )

    df = df.join(cells[["latitude", "longitude"]], on="geohash")
    df["cell"] = pd.Categorical(df["geohash"], categories=sorted(cells.index.astype(str)))
    return df


FEATURES = [
    "cell",
    "latitude",
    "longitude",
    "hour",
    "dow",
    "is_weekend",
    "is_holiday",
    "is_religious_holiday",
    "is_eve_afternoon",
    "days_to_holiday",
    "days_since_holiday",
    "school_day",
    *[f"lag_{k}" for k in LAGS],
    "same_hour_4w_mean",
    "roll24_mean",
    "roll168_mean",
    "roll168_std",
    "veh_lag24",
    "veh_lag168",
    "lag24_was_holiday",
    "lag168_was_holiday",
]
CALENDAR_FEATURES = [
    "is_holiday",
    "is_religious_holiday",
    "is_eve_afternoon",
    "days_to_holiday",
    "days_since_holiday",
    "school_day",
    "lag24_was_holiday",
    "lag168_was_holiday",
]
