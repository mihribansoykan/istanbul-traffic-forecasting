"""Synthetic hourly traffic for a few cells with realistic rush-hour, school and holiday effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from istanbul_traffic.calendar import calendar_features

CELLS = {"sxk9jr": (41.00, 29.06), "sxkc42": (40.96, 29.28), "sxk3xy": (41.04, 28.98)}


def make_raw(start="2024-01-01", end="2024-12-31 23:00", seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, end, freq="h")
    cal = calendar_features(hours)
    rows = []
    for i, (gh, (lat, lon)) in enumerate(CELLS.items()):
        base = 70 - 8 * i
        rush = cal["hour"].isin([7, 8, 9, 17, 18, 19]).to_numpy() & (cal["is_weekend"] == 0)
        speed = (
            base
            - 22 * rush * cal["school_day"].to_numpy()
            - 10 * rush * (1 - cal["school_day"].to_numpy())
            + 12 * cal["is_holiday"].to_numpy()
            + rng.normal(0, 3, len(hours))
        )
        vehicles = (
            300 + 400 * rush - 150 * cal["is_holiday"].to_numpy() + rng.normal(0, 20, len(hours))
        )
        rows.append(
            pd.DataFrame(
                {
                    "date_time": hours,
                    "latitude": np.float32(lat),
                    "longitude": np.float32(lon),
                    "geohash": gh,
                    "minimum_speed": (speed - 20).astype("float32"),
                    "maximum_speed": (speed + 40).astype("float32"),
                    "average_speed": speed.astype("float32"),
                    "number_of_vehicles": vehicles.clip(1).astype("float32"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return make_raw()
