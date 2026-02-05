from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessConfig:
    gap_s: int = 30 * 60
    alt_min_m: Optional[float] = 8000
    alt_max_m: Optional[float] = 13000
    trim_start_min: int = 20
    trim_end_min: int = 20
    downsample_minute: bool = True
    max_segment_gap_s: int = 5 * 60


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_raw_parquets(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {raw_dir}")
    dfs = [pd.read_parquet(p) for p in files]
    return pd.concat(dfs, ignore_index=True)


def build_flights(points: pd.DataFrame, gap_s: int) -> pd.DataFrame:
    points = points.sort_values(["icao24", "callsign", "time"], kind="stable").reset_index(drop=True)
    dt = points.groupby(["icao24", "callsign"])["time"].diff()
    new_flight = dt.isna() | (dt > gap_s)
    flight_idx = new_flight.groupby([points["icao24"], points["callsign"]]).cumsum()
    points = points.copy()
    points["flight_id"] = (
        points["icao24"].astype(str)
        + "_"
        + points["callsign"].fillna("NA").astype(str)
        + "_"
        + flight_idx.astype(str)
    )
    return points


def build_segments(
    points: pd.DataFrame,
    trim_start_min: int = 0,
    trim_end_min: int = 0,
    max_segment_gap_s: int = 5 * 60,
) -> pd.DataFrame:
    points = points.sort_values(["flight_id", "time"], kind="stable").reset_index(drop=True)
    g = points.groupby("flight_id", sort=False)
    next_time = g["time"].shift(-1)
    next_lat = g["lat"].shift(-1)
    next_lon = g["lon"].shift(-1)
    next_alt = g["baroaltitude"].shift(-1)

    dt_s = next_time - points["time"]
    dist_km = haversine_km(points["lat"], points["lon"], next_lat, next_lon)

    mid_time = points["time"] + dt_s / 2.0
    mid_lat = (points["lat"] + next_lat) / 2.0
    mid_lon = (points["lon"] + next_lon) / 2.0
    mid_alt_m = (points["baroaltitude"] + next_alt) / 2.0

    seg = pd.DataFrame(
        {
            "flight_id": points["flight_id"].astype(str),
            "icao24": points["icao24"],
            "callsign": points["callsign"],
            "mid_time": mid_time,
            "mid_lat": mid_lat,
            "mid_lon": mid_lon,
            "mid_alt_m": mid_alt_m,
            "dist_km": dist_km,
            "dt_s": dt_s,
        }
    )
    seg = seg.dropna(subset=["dt_s", "dist_km", "mid_lat", "mid_lon", "mid_alt_m"])
    seg = seg[(seg["dt_s"] > 0) & (seg["dt_s"] <= max_segment_gap_s)].copy()
    if trim_start_min > 0 or trim_end_min > 0:
        trim_start_s = trim_start_min * 60
        trim_end_s = trim_end_min * 60
        seg_g = seg.groupby("flight_id", sort=False)
        flight_start = seg_g["mid_time"].transform("min").to_numpy()
        flight_end = seg_g["mid_time"].transform("max").to_numpy()
        keep = (seg["mid_time"].to_numpy() >= (flight_start + trim_start_s)) & (
            seg["mid_time"].to_numpy() <= (flight_end - trim_end_s)
        )
        seg = seg[keep].copy()
    return seg.reset_index(drop=True)


def process_opensky_raw(
    raw_dir: Path,
    out_path: Path,
    cfg: PreprocessConfig,
) -> Path:
    df = load_raw_parquets(raw_dir)
    keep = ["time", "icao24", "callsign", "lat", "lon", "baroaltitude"]
    df = df[keep].copy()
    df["callsign"] = df["callsign"].astype("string").str.strip()
    df.loc[df["callsign"].isna() | (df["callsign"] == ""), "callsign"] = pd.NA
    df = df.dropna(subset=["time", "lat", "lon", "baroaltitude"])
    if cfg.alt_min_m is not None:
        df = df[df["baroaltitude"] >= cfg.alt_min_m]
    if cfg.alt_max_m is not None:
        df = df[df["baroaltitude"] <= cfg.alt_max_m]

    if cfg.downsample_minute:
        df = df.copy()
        df["minute"] = (df["time"] // 60).astype("int64")
        df = df.sort_values(["icao24", "callsign", "minute", "time"], kind="mergesort")
        df = df.drop_duplicates(subset=["icao24", "callsign", "minute"], keep="first")
        df = df.drop(columns=["minute"])

    points = build_flights(df, gap_s=cfg.gap_s)
    segments = build_segments(
        points,
        trim_start_min=cfg.trim_start_min,
        trim_end_min=cfg.trim_end_min,
        max_segment_gap_s=cfg.max_segment_gap_s,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    segments.to_parquet(out_path, index=False)
    return out_path
