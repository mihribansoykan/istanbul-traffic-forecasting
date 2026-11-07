"""Download and prepare the IBB "Hourly Traffic Density" open dataset.

Source: Istanbul Metropolitan Municipality (IBB) Open Data Portal, dataset
``hourly-traffic-density-data-set`` - one CSV per month with, for every geohash-6 cell
(~1.2 km x 0.6 km) and hour: min / max / average speed (km/h) and number of vehicles.
Licence: IBB Open Data Licence.

    uv run traffic-download                 # 2023-01 .. 2024-12 (default)
    uv run traffic-download --months 202403 202404
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

CKAN_URL = "https://data.ibb.gov.tr/api/3/action/package_show"
DATASET_ID = "hourly-traffic-density-data-set"
RAW_DIR = Path("data/raw")
DEFAULT_MONTHS = [f"{y}{m:02d}" for y in (2023, 2024) for m in range(1, 13)]
HEADERS = {"User-Agent": "istanbul-traffic-forecasting/1.0 (open-data research project)"}

DTYPES = {
    "LATITUDE": "float32",
    "LONGITUDE": "float32",
    "GEOHASH": "string",
    "MINIMUM_SPEED": "float32",
    "MAXIMUM_SPEED": "float32",
    "AVERAGE_SPEED": "float32",
    "NUMBER_OF_VEHICLES": "float32",
}


def resource_urls() -> dict[str, str]:
    """Map 'YYYYMM' -> download URL, read from the portal's CKAN API."""
    resp = requests.get(CKAN_URL, params={"id": DATASET_ID}, timeout=60)
    resp.raise_for_status()
    urls = {}
    for res in resp.json()["result"]["resources"]:
        url = res.get("url", "")
        stem = Path(url).stem  # e.g. traffic_density_202401
        if stem.startswith("traffic_density_"):
            urls[stem.rsplit("_", 1)[-1]] = url
    return urls


def month_path(month: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"traffic_{month}.parquet"


def download_month(month: str, url: str, raw_dir: Path = RAW_DIR, retries: int = 5) -> Path:
    """Stream one monthly CSV to disk, convert it to typed Parquet and delete the CSV."""
    out = month_path(month, raw_dir)
    if out.exists():
        return out
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv = raw_dir / f"traffic_{month}.csv"
    for attempt in range(1, retries + 1):
        try:
            log.info("Downloading %s (attempt %d)", month, attempt)
            with requests.get(url, stream=True, timeout=600, headers=HEADERS) as r:
                r.raise_for_status()
                with open(csv, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            break
        except requests.RequestException as exc:  # the portal occasionally drops connections
            if attempt == retries:
                raise
            wait = 15 * attempt
            log.warning("%s failed (%s) - retrying in %ds", month, exc, wait)
            time.sleep(wait)
    df = pd.read_csv(csv, dtype=DTYPES, parse_dates=["DATE_TIME"])
    df.columns = [c.lower() for c in df.columns]
    df.to_parquet(out, index=False)
    csv.unlink()
    log.info("%s: %d rows -> %s", month, len(df), out)
    return out


def download(months: list[str] = DEFAULT_MONTHS, raw_dir: Path = RAW_DIR) -> list[Path]:
    missing = [m for m in months if not month_path(m, raw_dir).exists()]
    urls = resource_urls() if missing else {}
    return [download_month(m, urls.get(m, ""), raw_dir) for m in months]


def read_month(month: str, raw_dir: Path = RAW_DIR, cells: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(month_path(month, raw_dir))
    df["geohash"] = df["geohash"].astype(str)
    if cells is not None:
        df = df[df["geohash"].isin(cells)]
    return df


def month_cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Additive per-cell statistics for one month (so a year never has to sit in memory at once).

    * night speed (01:00-04:59) approximates the free-flow speed of a cell
    * rush speed = school-day 07:00-09:59 (morning peak)
    """
    from istanbul_traffic.calendar import calendar_features

    cal = calendar_features(pd.DatetimeIndex(df["date_time"].unique()))
    hour = df["date_time"].dt.hour
    school = df["date_time"].map(cal["school_day"]).to_numpy() == 1
    night = hour.between(1, 4).to_numpy()
    rush = school & hour.between(7, 9).to_numpy()
    s = df["average_speed"].to_numpy()
    tmp = pd.DataFrame(
        {
            "geohash": df["geohash"].to_numpy(),
            "latitude": df["latitude"].to_numpy(),
            "longitude": df["longitude"].to_numpy(),
            "vehicles": df["number_of_vehicles"].to_numpy(),
            "speed_sum": s,
            "night_sum": np.where(night, s, 0.0),
            "night_n": night.astype(int),
            "rush_sum": np.where(rush, s, 0.0),
            "rush_n": rush.astype(int),
            "hours": 1,
        }
    )
    return tmp.groupby("geohash").agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        hours_observed=("hours", "sum"),
        vehicles=("vehicles", "sum"),
        speed_sum=("speed_sum", "sum"),
        night_sum=("night_sum", "sum"),
        night_n=("night_n", "sum"),
        rush_sum=("rush_sum", "sum"),
        rush_n=("rush_n", "sum"),
    )


def build_catalogue(months: list[str] = DEFAULT_MONTHS, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per geohash cell: location, coverage, volume, free-flow and rush-hour speed."""
    stats, total_hours = [], 0
    for m in months:
        df = read_month(m, raw_dir)
        total_hours += df["date_time"].nunique()
        stats.append(month_cell_stats(df))
    allm = pd.concat(stats)
    g = allm.groupby(level=0)
    cat = g[
        ["hours_observed", "vehicles", "speed_sum", "night_sum", "night_n", "rush_sum", "rush_n"]
    ].sum()
    cat[["latitude", "longitude"]] = g[["latitude", "longitude"]].first()
    cat["coverage"] = cat["hours_observed"] / total_hours
    cat["mean_speed"] = cat["speed_sum"] / cat["hours_observed"]
    cat["free_flow_speed"] = cat["night_sum"] / cat["night_n"].replace(0, np.nan)
    cat["rush_speed"] = cat["rush_sum"] / cat["rush_n"].replace(0, np.nan)
    cat["congestion_index"] = (1 - cat["rush_speed"] / cat["free_flow_speed"]).clip(lower=0)
    cols = [
        "latitude",
        "longitude",
        "coverage",
        "hours_observed",
        "vehicles",
        "mean_speed",
        "free_flow_speed",
        "rush_speed",
        "congestion_index",
    ]
    return cat[cols].sort_values("vehicles", ascending=False)


def load_cells(
    cells: list[str], months: list[str] = DEFAULT_MONTHS, raw_dir: Path = RAW_DIR
) -> pd.DataFrame:
    return pd.concat([read_month(m, raw_dir, cells) for m in months], ignore_index=True)


def select_cells(catalogue: pd.DataFrame, n: int, min_coverage: float = 0.97) -> list[str]:
    """The ``n`` busiest cells among those observed in at least ``min_coverage`` of all hours."""
    eligible = catalogue[catalogue["coverage"] >= min_coverage]
    return eligible.head(n).index.astype(str).tolist()


def to_panel(
    raw: pd.DataFrame, cells: list[str], start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Complete cell x hour grid (missing hours become NaN rather than silently disappearing).

    The grid spans whole calendar months from the first to the last month present in ``raw``,
    so days missing at the start or end of the source files also show up as gaps.
    """
    sub = raw[raw["geohash"].isin(cells)]
    first, last = raw["date_time"].min(), raw["date_time"].max()
    start = start or first.to_period("M").start_time
    end = end or last.to_period("M").end_time.floor("h")
    hours = pd.date_range(start, end, freq="h")
    grid = pd.MultiIndex.from_product([cells, hours], names=["geohash", "date_time"])
    panel = (
        sub.groupby(["geohash", "date_time"], observed=True)[
            ["average_speed", "number_of_vehicles", "minimum_speed", "maximum_speed"]
        ]
        .mean()
        .reindex(grid)
        .reset_index()
    )
    panel["geohash"] = panel["geohash"].astype(str)
    return panel


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download IBB hourly traffic data")
    parser.add_argument("--months", nargs="+", default=DEFAULT_MONTHS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download(args.months)


if __name__ == "__main__":
    main()
