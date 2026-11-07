"""End-to-end pipeline: download -> select cells -> features -> rolling-origin backtest -> reports.

uv run traffic-backtest
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from istanbul_traffic import backtest, data, plots
from istanbul_traffic.calendar import calendar_features, school_break_days
from istanbul_traffic.features import FEATURES, build_features

log = logging.getLogger("istanbul_traffic")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
PROCESSED = Path("data/processed")
N_CELLS = 150


def _per_fold(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, part in pred.groupby("fold", sort=False):
        s = backtest.score(part)
        s = s.assign(fold=fold, kind=part["fold_kind"].iloc[0]).reset_index(names="model")
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


def run(months: list[str] = data.DEFAULT_MONTHS, n_cells: int = N_CELLS) -> dict:
    FIGS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1 ── data -------------------------------------------------------------------------
    data.download(months)
    catalogue = data.build_catalogue(months)
    cells = data.select_cells(catalogue, n_cells)
    catalogue.round(5).to_csv(REPORTS / "cell_catalogue.csv")
    panel = data.to_panel(data.load_cells(cells, months), cells)
    panel.to_parquet(PROCESSED / "panel.parquet", index=False)
    log.info(
        "cells: %d of %d | panel rows: %d | missing target: %.2f%%",
        len(cells),
        len(catalogue),
        len(panel),
        100 * panel["average_speed"].isna().mean(),
    )

    # 2 ── features + backtest ----------------------------------------------------------
    df = build_features(panel, catalogue.loc[cells])
    preds, last_model = [], None
    for fold in backtest.FOLDS:
        p, last_model = backtest.run_fold(df, fold)
        preds.append(p)
        log.info(
            "%-28s LightGBM MAE %.2f | naive-week MAE %.2f",
            fold.name,
            (p["y"] - p["lgbm"]).abs().mean(),
            (p["y"] - p["naive_week"]).abs().mean(),
        )
    pred = pd.concat(preds, ignore_index=True)
    pred.to_parquet(PROCESSED / "backtest_predictions.parquet", index=False)

    # 3 ── scores -----------------------------------------------------------------------
    overall = backtest.score(pred)
    regular = backtest.score(pred[pred["fold_kind"] == "regular"])
    holiday_weeks = backtest.score(pred[pred["fold_kind"] == "holiday"])
    holiday_days = backtest.score(pred[pred["day_type"] == "holiday"])
    by_day_type = {
        t: backtest.score(part).round(4).to_dict(orient="index")
        for t, part in pred.groupby("day_type")
    }
    per_fold = _per_fold(pred)
    per_fold.round(4).to_csv(REPORTS / "backtest_by_fold.csv", index=False)
    overall.round(4).to_csv(REPORTS / "backtest_overall.csv")

    free_flow = catalogue.loc[cells, "free_flow_speed"]
    reg = pred[pred["fold_kind"] == "regular"].dropna(subset=["naive_week"])
    congestion = {
        m: backtest.congestion_metrics(reg, free_flow, model=m)
        for m in ("naive_week", "lgbm_no_calendar", "lgbm")
    }

    test_last = df[df["date_time"] >= pd.Timestamp(backtest.FOLDS[-1].start)].dropna(
        subset=["average_speed"]
    )
    sample = test_last.sample(min(20_000, len(test_last)), random_state=0)
    contrib = last_model.booster_.predict(sample[FEATURES], pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=FEATURES).sort_values(ascending=False)

    # 4 ── figures ----------------------------------------------------------------------
    plots.cell_map(catalogue, cells, FIGS / "congestion_map.png")
    plots.hour_dow_heatmap(df[df["is_holiday"] == 0], FIGS / "speed_hour_weekday.png")
    by_day = panel[panel["date_time"].dt.year == 2024].set_index("date_time").resample("D")
    daily = pd.DataFrame(
        {
            "vehicles": by_day["number_of_vehicles"].sum(),
            "speed": by_day["average_speed"].mean(),
            "coverage": by_day["average_speed"].count() / (24 * len(cells)),
        }
    )
    daily.loc[daily["coverage"] < 0.8, ["vehicles", "speed"]] = np.nan  # gaps in the source
    cal_days = calendar_features(daily.index)
    cal_days["school_break"] = school_break_days(daily.index)
    plots.daily_profile(daily, cal_days, FIGS / "daily_profile_2024.png")
    plots.fold_dotplot(per_fold, FIGS / "backtest_by_fold.png")
    plots.error_by_hour(
        pred.dropna(subset=["naive_week"]),
        {k: backtest.MODELS[k] for k in ("naive_week", "lgbm_no_calendar", "lgbm")},
        FIGS / "error_by_hour.png",
    )
    busiest = cells[0]
    kurban = pred[(pred["fold"] == backtest.HOLIDAY_FOLDS[1].name) & (pred["geohash"] == busiest)]
    plots.forecast_example(
        kurban.sort_values("date_time"),
        FIGS / "forecast_kurban_bayrami.png",
        f"Day-ahead forecasts through Kurban Bayramı 2024 (cell {busiest})",
    )
    plots.importance(imp, FIGS / "feature_importance.png")

    metrics = {
        "data": {
            "months": [months[0], months[-1]],
            "cells_total": len(catalogue),
            "cells_modelled": len(cells),
            "panel_rows": len(panel),
            "missing_target_share": round(float(panel["average_speed"].isna().mean()), 4),
            "hours_with_data_by_month": (
                panel.dropna(subset=["average_speed"])
                .groupby(panel["date_time"].dt.strftime("%Y-%m"))["date_time"]
                .nunique()
                .div(panel.groupby(panel["date_time"].dt.strftime("%Y-%m"))["date_time"].nunique())
                .round(3)
                .to_dict()
            ),
            "busiest_cell": busiest,
        },
        "folds": [f.__dict__ for f in backtest.FOLDS],
        "scores": {
            "all_folds": overall.round(4).to_dict(orient="index"),
            "regular_weeks": regular.round(4).to_dict(orient="index"),
            "holiday_weeks": holiday_weeks.round(4).to_dict(orient="index"),
            "holiday_days_only": holiday_days.round(4).to_dict(orient="index"),
            "by_day_type": by_day_type,
        },
        "congestion_alerts_regular_weeks": {
            backtest.MODELS[k]: {kk: round(float(vv), 4) for kk, vv in v.items()}
            for k, v in congestion.items()
        },
        "top_features": imp.head(10).round(3).to_dict(),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), "utf-8")
    log.info("Overall:\n%s", overall.round(3).to_string())
    return metrics


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore", message="Usage of np.ndarray subset")
    parser = argparse.ArgumentParser(description="Backtest day-ahead traffic speed forecasts")
    parser.add_argument("--cells", type=int, default=N_CELLS)
    parser.add_argument(
        "--months",
        nargs="+",
        default=data.DEFAULT_MONTHS,
        help="YYYYMM months to use, e.g. only 2024 to reproduce the 'no prior Bayram' experiment",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(months=args.months, n_cells=args.cells)


if __name__ == "__main__":
    main()
