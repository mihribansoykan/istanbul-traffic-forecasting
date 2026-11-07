import json

import numpy as np
import pandas as pd
import pytest

from istanbul_traffic import backtest, data, pipeline
from istanbul_traffic.features import build_features


@pytest.fixture(autouse=True)
def fast_lgbm(monkeypatch):
    monkeypatch.setitem(backtest.LGBM_PARAMS, "n_estimators", 80)
    monkeypatch.setitem(backtest.LGBM_PARAMS, "num_leaves", 31)


@pytest.fixture(scope="module")
def feats(raw):
    cells = sorted(raw["geohash"].unique())
    cat = raw.groupby("geohash")[["latitude", "longitude"]].first()
    return build_features(data.to_panel(raw, cells), cat)


def test_folds_are_well_formed():
    for f in backtest.FOLDS:
        assert (f.end_ts - f.start_ts) == pd.Timedelta(days=7)
    assert len(backtest.REGULAR_FOLDS) == 8
    assert {f.kind for f in backtest.FOLDS} == {"holiday", "regular"}


def test_run_fold_trains_only_on_the_past(feats):
    fold = backtest.HOLIDAY_FOLDS[0]
    pred, model = backtest.run_fold(feats, fold)
    assert pred["date_time"].between(fold.start_ts, fold.end_ts, inclusive="left").all()
    assert set(backtest.MODELS) <= set(pred.columns)
    # Holiday-aware model must beat "same hour last week" on Bayram days of the synthetic data.
    hol = pred[pred["day_type"] == "holiday"]
    assert (hol["y"] - hol["lgbm"]).abs().mean() < (hol["y"] - hol["naive_week"]).abs().mean()


def test_score_and_skill():
    p = pd.DataFrame(
        {
            "y": [10.0, 20.0, 30.0],
            "naive_day": [12.0, 18.0, 33.0],
            "naive_week": [14.0, 24.0, 26.0],
            "lgbm_no_calendar": [11.0, 21.0, 29.0],
            "lgbm": [10.0, 20.0, 30.0],
        }
    )
    s = backtest.score(p)
    assert s.loc["LightGBM (full)", "mae"] == 0
    assert s.loc["LightGBM (full)", "skill_vs_naive_week"] == pytest.approx(1.0)
    assert s.loc["Seasonal naive (same hour last week)", "skill_vs_naive_week"] == pytest.approx(0)
    assert s.loc["Seasonal naive (same hour last week)", "wape"] == pytest.approx(12 / 60)


def test_congestion_metrics():
    p = pd.DataFrame({"geohash": ["a"] * 4, "y": [20.0, 50, 20, 50], "lgbm": [20.0, 50, 50, 20]})
    m = backtest.congestion_metrics(p, pd.Series({"a": 60.0}), model="lgbm")
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["congested_share"] == pytest.approx(0.5)


def test_pipeline_end_to_end(raw, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for month, part in raw.groupby(raw["date_time"].dt.strftime("%Y%m")):
        data.month_path(month).parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(data.month_path(month), index=False)

    metrics = pipeline.run(months=[f"2024{m:02d}" for m in range(1, 13)], n_cells=3)

    saved = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert saved["data"]["cells_modelled"] == 3
    scores = metrics["scores"]["all_folds"]
    assert scores["LightGBM (full)"]["mae"] < scores["Seasonal naive (same hour last week)"]["mae"]
    assert len(list((tmp_path / "reports" / "figures").glob("*.png"))) == 7
    assert np.isfinite(pd.read_csv(tmp_path / "reports" / "backtest_by_fold.csv")["mae"]).all()
