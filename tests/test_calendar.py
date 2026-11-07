import pandas as pd

from istanbul_traffic.calendar import calendar_features


def _row(ts: str) -> pd.Series:
    return calendar_features(pd.DatetimeIndex([pd.Timestamp(ts)])).iloc[0]


def test_religious_holiday():
    r = _row("2024-04-10 08:00")  # Ramazan Bayrami day 1
    assert r["is_holiday"] == 1 and r["is_religious_holiday"] == 1
    assert r["school_day"] == 0 and r["day_type"] == "holiday"


def test_national_holiday_is_not_religious():
    r = _row("2024-10-29 08:00")  # Republic Day
    assert r["is_holiday"] == 1 and r["is_religious_holiday"] == 0


def test_eve_is_half_day_from_13h():
    assert _row("2024-06-15 10:00")["is_eve_afternoon"] == 0  # Kurban Bayrami arife
    assert _row("2024-06-15 14:00")["is_eve_afternoon"] == 1


def test_school_calendar():
    assert _row("2024-03-05 08:00")["school_day"] == 1  # ordinary Tuesday in term
    assert _row("2024-07-10 08:00")["school_day"] == 0  # summer holiday
    assert _row("2024-11-13 08:00")["school_day"] == 0  # November mid-term break
    assert _row("2024-03-09 08:00")["school_day"] == 0  # Saturday
    assert _row("2024-07-10 08:00")["day_type"] == "weekday (school break)"


def test_earthquake_extended_semester_break_2023():
    assert _row("2023-02-14 08:00")["school_day"] == 0  # schools closed after 6 Feb 2023
    assert _row("2023-02-20 08:00")["school_day"] == 1  # reopened in Istanbul


def test_days_to_and_since_holiday():
    r = _row("2024-04-08 12:00")
    assert r["days_to_holiday"] == 2  # Apr 10
    assert _row("2024-04-13 12:00")["days_since_holiday"] == 1  # Apr 12
    assert _row("2024-03-01 12:00")["days_to_holiday"] == 7  # capped
