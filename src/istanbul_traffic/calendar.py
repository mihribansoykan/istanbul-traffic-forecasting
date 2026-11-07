"""Turkish calendar features: public & religious holidays, half-day eves (arife) and school terms.

Everything here is known in advance, so it can be used for day-ahead forecasting without leakage.
"""

from __future__ import annotations

from datetime import date

import holidays
import numpy as np
import pandas as pd

# Turkish school breaks (Ministry of National Education calendar), weekends included.
SCHOOL_BREAKS: list[tuple[str, str]] = [
    ("2023-01-21", "2023-02-19"),  # semester break, extended after the 6 Feb 2023 earthquakes
    ("2023-04-15", "2023-04-23"),  # 2nd mid-term break + Ramazan Bayrami
    ("2023-06-17", "2023-09-10"),  # summer 2023
    ("2023-11-11", "2023-11-19"),  # 1st mid-term break 2023-24
    ("2024-01-20", "2024-02-04"),  # semester break
    ("2024-04-06", "2024-04-14"),  # 2nd mid-term break (overlaps Ramazan Bayrami)
    ("2024-06-15", "2024-09-08"),  # summer 2024
    ("2024-11-09", "2024-11-17"),  # 1st mid-term break 2024-25
    ("2025-01-18", "2025-02-02"),  # semester break
]
RELIGIOUS = ("Ramazan", "Kurban")


def _holiday_tables(years: list[int]) -> tuple[dict[date, str], dict[date, str]]:
    full = holidays.Turkey(years=years)
    half = holidays.Turkey(years=years, categories=("half_day",))
    return dict(full.items()), dict(half.items())


def school_break_days(index: pd.DatetimeIndex) -> np.ndarray:
    days = index.normalize()
    flag = np.zeros(len(index), dtype=bool)
    for start, end in SCHOOL_BREAKS:
        flag |= (days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))
    return flag


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour-level calendar features for every timestamp in ``index``."""
    index = pd.DatetimeIndex(index)
    years = sorted(set(index.year) | {int(index.year.max()) + 1})
    full, half = _holiday_tables(years)
    day = index.normalize()
    day_dates = [d.date() for d in day]

    is_holiday = np.array([d in full for d in day_dates])
    is_religious = np.array([d in full and full[d].startswith(RELIGIOUS) for d in day_dates])
    is_eve = np.array([d in half for d in day_dates])
    weekend = index.dayofweek >= 5

    # Distance (in days) to the next / from the previous holiday, capped at 7: captures the
    # pre-holiday exodus and the return wave around long religious holidays.
    hol_days = pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in full))
    pos = hol_days.searchsorted(day)
    nxt = hol_days[np.minimum(pos, len(hol_days) - 1)]
    prv = hol_days[np.maximum(pos - 1, 0)]
    days_to = np.clip((nxt - day).days, 0, 7)
    days_since = np.clip((day - prv).days, 0, 7)
    days_since = np.where(pos == 0, 7, days_since)

    out = pd.DataFrame(
        {
            "hour": index.hour,
            "dow": index.dayofweek,
            "is_weekend": weekend.astype(int),
            "is_holiday": is_holiday.astype(int),
            "is_religious_holiday": is_religious.astype(int),
            "is_eve_afternoon": (is_eve & (index.hour >= 13)).astype(int),
            "days_to_holiday": days_to,
            "days_since_holiday": days_since,
            "school_day": (~weekend & ~is_holiday & ~school_break_days(index)).astype(int),
        },
        index=index,
    )
    out["day_type"] = np.select(
        [out["is_holiday"] == 1, out["is_weekend"] == 1, out["school_day"] == 0],
        ["holiday", "weekend", "weekday (school break)"],
        default="weekday (school day)",
    )
    return out
