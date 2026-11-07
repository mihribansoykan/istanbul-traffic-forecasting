import numpy as np
import pandas as pd
import pytest

from istanbul_traffic import data
from istanbul_traffic.features import FEATURES, build_features


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory, raw):
    d = tmp_path_factory.mktemp("raw")
    for month, part in raw.groupby(raw["date_time"].dt.strftime("%Y%m")):
        part.to_parquet(data.month_path(month, d), index=False)
    return d


def test_catalogue_and_selection(raw_dir):
    months = [f"2024{m:02d}" for m in range(1, 13)]
    cat = data.build_catalogue(months, raw_dir)
    assert len(cat) == 3
    assert cat["coverage"].between(0.99, 1.0).all()
    # night free-flow speed must exceed the school-day morning-peak speed
    assert (cat["free_flow_speed"] > cat["rush_speed"]).all()
    assert cat["congestion_index"].between(0, 1).all()
    assert data.select_cells(cat, 2) == cat.index[:2].tolist()


def test_select_cells_respects_coverage():
    cat = pd.DataFrame({"coverage": [0.5, 0.99, 0.98]}, index=["a", "b", "c"])
    assert data.select_cells(cat, 5) == ["b", "c"]


def test_to_panel_is_complete_grid(raw):
    holed = raw.drop(index=raw.index[:10])
    cells = sorted(raw["geohash"].unique())
    panel = data.to_panel(holed, cells)
    assert len(panel) == len(cells) * 366 * 24
    assert panel["average_speed"].isna().sum() == 10


@pytest.fixture(scope="module")
def feats(raw):
    cells = sorted(raw["geohash"].unique())
    cat = raw.groupby("geohash")[["latitude", "longitude"]].first()
    return build_features(data.to_panel(raw, cells), cat)


def test_feature_columns_present(feats):
    assert set(FEATURES) <= set(feats.columns)


def test_lag_values_are_exact(feats):
    one = feats[feats["geohash"] == "sxk9jr"].set_index("date_time")
    t = pd.Timestamp("2024-05-15 08:00")
    assert one.loc[t, "lag_24"] == pytest.approx(
        one.loc[t - pd.Timedelta(hours=24), "average_speed"]
    )
    assert one.loc[t, "lag_168"] == pytest.approx(
        one.loc[t - pd.Timedelta(days=7), "average_speed"]
    )


def test_no_leakage_from_last_24_hours(raw):
    """Day-ahead guard: changing the target in (t-24h, t] must not change features at t."""
    cells = sorted(raw["geohash"].unique())
    cat = raw.groupby("geohash")[["latitude", "longitude"]].first()
    t = pd.Timestamp("2024-05-15 08:00")
    window = (raw["date_time"] > t - pd.Timedelta(hours=24)) & (raw["date_time"] <= t)
    perturbed = raw.copy()
    perturbed.loc[window, "average_speed"] += 50
    perturbed.loc[window, "number_of_vehicles"] += 999
    cols = [c for c in FEATURES if c != "cell"]
    a = build_features(data.to_panel(raw, cells), cat)
    b = build_features(data.to_panel(perturbed, cells), cat)
    sel_a = a.loc[a["date_time"] == t, cols].reset_index(drop=True)
    sel_b = b.loc[b["date_time"] == t, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(sel_a, sel_b)


def test_lag_was_holiday_flag(feats):
    one = feats[feats["geohash"] == "sxk9jr"].set_index("date_time")
    assert one.loc[pd.Timestamp("2024-04-17 09:00"), "lag168_was_holiday"] == 1  # Apr 10
    assert one.loc[pd.Timestamp("2024-04-24 09:00"), "lag168_was_holiday"] == 0  # Apr 17
    assert one.loc[pd.Timestamp("2024-04-30 09:00"), "lag168_was_holiday"] == 1  # Apr 23
    assert np.isin(one["lag24_was_holiday"].unique(), [0, 1]).all()
