from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoringConfig:
    pressure_levels_hpa: List[int]
    night_weight: float = 1.5
    rhi_excess_scale: float = 20.0
    issr_threshold: float = 100.0


def alt_m_to_pressure_pa(h_m: np.ndarray) -> np.ndarray:
    h = np.clip(h_m, 0, 20000)
    p0 = 101325.0
    T0 = 288.15
    L = 0.0065
    g = 9.80665
    R = 287.05
    return p0 * (1 - L * h / T0) ** (g / (R * L))


def e_si_pa(T: np.ndarray) -> np.ndarray:
    return np.exp(9.550426 - (5723.265 / T) + 3.53068 * np.log(T) - 0.00728332 * T)


def vapor_pressure_from_q(p_pa: np.ndarray, q: np.ndarray) -> np.ndarray:
    w = q / np.clip(1 - q, 1e-12, None)
    return (w * p_pa) / (0.622 + w)


def compute_rhi(T: np.ndarray, q: np.ndarray, p_pa: np.ndarray) -> np.ndarray:
    e = vapor_pressure_from_q(p_pa, q)
    e_si = e_si_pa(T)
    return 100.0 * (e / e_si)


def _nearest_level(levels: np.ndarray, p_hpa: np.ndarray) -> np.ndarray:
    levels = levels.reshape(1, -1)
    p = p_hpa.reshape(-1, 1)
    idx = np.argmin(np.abs(p - levels), axis=1)
    return levels[0, idx]


def _nearest_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    idx = np.searchsorted(grid, values, side="left")
    idx = np.clip(idx, 0, len(grid) - 1)
    left = np.clip(idx - 1, 0, len(grid) - 1)
    right = idx
    left_val = grid[left]
    right_val = grid[right]
    choose_right = np.abs(values - right_val) < np.abs(values - left_val)
    return np.where(choose_right, right_val, left_val)


def _add_up_down_levels(df: pd.DataFrame, levels: List[int]) -> pd.DataFrame:
    levels_sorted = np.array(sorted(levels))
    plev = df["plev_hpa"].astype(float).to_numpy()
    # nearest index (robust to any out-of-set values)
    idx = np.abs(plev.reshape(-1, 1) - levels_sorted.reshape(1, -1)).argmin(axis=1)
    idx = np.clip(idx, 0, len(levels_sorted) - 1)
    plev_here = np.take(levels_sorted, idx, mode="clip")
    plev_up = np.take(levels_sorted, idx - 1, mode="clip")
    plev_down = np.take(levels_sorted, idx + 1, mode="clip")
    df = df.copy()
    df["plev_up"] = plev_up
    df["plev_down"] = plev_down
    return df


def attach_era5_and_score(
    segments: pd.DataFrame,
    era5: pd.DataFrame,
    cfg: ScoringConfig,
) -> pd.DataFrame:
    seg = segments.copy()
    seg["dt_utc"] = pd.to_datetime(seg["mid_time"], unit="s", utc=True)
    seg["time_hour"] = seg["dt_utc"].dt.floor("h").dt.tz_convert(None)

    era5 = era5.copy()
    time_col = None
    for candidate in ["time", "valid_time", "datetime"]:
        if candidate in era5.columns:
            time_col = candidate
            break
    if time_col is None:
        raise KeyError("ERA5 dataframe missing time column (expected one of: time, valid_time, datetime)")
    era5["time_hour"] = pd.to_datetime(era5[time_col], utc=True).dt.floor("h").dt.tz_convert(None)

    lat_vals = np.sort(era5["lat"].unique())
    lon_vals = np.sort(era5["lon"].unique())

    seg["lat_g"] = _nearest_grid(seg["mid_lat"].to_numpy(), lat_vals)
    seg["lon_g"] = _nearest_grid(seg["mid_lon"].to_numpy(), lon_vals)

    p_pa = alt_m_to_pressure_pa(seg["mid_alt_m"].to_numpy())
    seg["p_hpa_est"] = p_pa / 100.0
    levels = np.array(cfg.pressure_levels_hpa, dtype=int)
    seg["plev_hpa"] = _nearest_level(levels, seg["p_hpa_est"].to_numpy()).astype(int)
    seg = _add_up_down_levels(seg, cfg.pressure_levels_hpa)

    era5_key = era5.rename(columns={"lat": "lat_g", "lon": "lon_g"})
    if "plev_hpa" not in era5_key.columns:
        for cand in ["level", "pressure_level"]:
            if cand in era5_key.columns:
                era5_key = era5_key.rename(columns={cand: "plev_hpa"})
                break
    if "plev_hpa" not in era5_key.columns:
        raise KeyError("ERA5 dataframe missing pressure level column (expected one of: plev_hpa, level, pressure_level)")
    era5_key = era5_key.drop_duplicates(["time_hour", "plev_hpa", "lat_g", "lon_g"])
    era5_key = era5_key.set_index(["time_hour", "plev_hpa", "lat_g", "lon_g"])[["T_K", "q_kgkg"]]

    idx = pd.MultiIndex.from_arrays(
        [seg["time_hour"], seg["plev_hpa"], seg["lat_g"], seg["lon_g"]],
        names=["time_hour", "plev_hpa", "lat_g", "lon_g"],
    )
    tq = era5_key.reindex(idx).reset_index(drop=True)
    seg["T_K"] = tq["T_K"].to_numpy()
    seg["q_kgkg"] = tq["q_kgkg"].to_numpy()

    def _lookup(level_col: str, prefix: str) -> None:
        idx2 = pd.MultiIndex.from_arrays(
            [seg["time_hour"], seg[level_col], seg["lat_g"], seg["lon_g"]],
            names=["time_hour", "plev_hpa", "lat_g", "lon_g"],
        )
        tq2 = era5_key.reindex(idx2).reset_index(drop=True)
        seg[f"T_K_{prefix}"] = tq2["T_K"].to_numpy()
        seg[f"q_kgkg_{prefix}"] = tq2["q_kgkg"].to_numpy()

    _lookup("plev_up", "up")
    _lookup("plev_down", "down")

    seg["RHi"] = compute_rhi(seg["T_K"].to_numpy(), seg["q_kgkg"].to_numpy(), seg["p_hpa_est"].to_numpy() * 100.0)
    seg["ISSR"] = seg["RHi"] > cfg.issr_threshold

    seg["RHi_up"] = compute_rhi(seg["T_K_up"].to_numpy(), seg["q_kgkg_up"].to_numpy(), seg["p_hpa_est"].to_numpy() * 100.0)
    seg["ISSR_up"] = seg["RHi_up"] > cfg.issr_threshold
    seg["RHi_down"] = compute_rhi(seg["T_K_down"].to_numpy(), seg["q_kgkg_down"].to_numpy(), seg["p_hpa_est"].to_numpy() * 100.0)
    seg["ISSR_down"] = seg["RHi_down"] > cfg.issr_threshold

    local_hour = (seg["dt_utc"].dt.hour + seg["mid_lon"] / 15.0) % 24
    seg["is_night"] = (local_hour >= 18) | (local_hour < 6)

    excess = np.maximum(0.0, (seg["RHi"] - 100.0) / cfg.rhi_excess_scale)
    night_w = np.where(seg["is_night"], cfg.night_weight, 1.0)
    seg["segment_score"] = seg["dist_km"] * excess * night_w * seg["ISSR"].astype(float)

    excess_up = np.maximum(0.0, (seg["RHi_up"] - 100.0) / cfg.rhi_excess_scale)
    seg["segment_score_up"] = seg["dist_km"] * excess_up * night_w * seg["ISSR_up"].astype(float)

    excess_down = np.maximum(0.0, (seg["RHi_down"] - 100.0) / cfg.rhi_excess_scale)
    seg["segment_score_down"] = seg["dist_km"] * excess_down * night_w * seg["ISSR_down"].astype(float)

    return seg
