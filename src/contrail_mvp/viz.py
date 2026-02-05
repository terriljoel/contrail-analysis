from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_altitude_profile(
    hourly: pd.DataFrame,
    policy_hourly: pd.DataFrame,
    flight_id: str,
    use_mid_alt: bool = False,
    title: Optional[str] = None,
) -> plt.Axes:
    df_actual = hourly[hourly["flight_id"] == flight_id].copy()
    df_policy = policy_hourly[policy_hourly["flight_id"] == flight_id].copy()
    if df_actual.empty or df_policy.empty:
        raise ValueError(f"flight_id {flight_id} not found in hourly/policy data")

    cols = ["flight_id", "time_hour", "action"]
    if "plev_reco" in df_policy.columns:
        cols.append("plev_reco")
    df = df_actual.merge(
        df_policy[cols],
        on=["flight_id", "time_hour"],
        how="left",
    ).sort_values("time_hour", kind="stable")

    x = pd.to_datetime(df["time_hour"])
    if use_mid_alt:
        if "alt_reco_m" not in df.columns:
            raise ValueError("alt_reco_m missing; provide recommended altitude in meters")
        y_actual = df["mid_alt_m_mean"]
        y_reco = df["alt_reco_m"]
    else:
        y_actual = df["plev_hpa_mode"]
        y_reco = df["plev_reco"] if "plev_reco" in df.columns else df["plev_hpa_mode"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(x, y_actual, label="Actual", color="#2a5d9f", linewidth=2, where="post", alpha=0.9)
    ax.step(x, y_reco, label="Recommended", color="#c45a1c", linewidth=2, where="post", linestyle="--", alpha=0.9)

    ax.set_xlabel("Time (hour)")
    ax.set_ylabel("Altitude (m)" if use_mid_alt else "Pressure level (hPa)")
    ax.set_title(title or f"Altitude profile: {flight_id}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if not use_mid_alt:
        ax.invert_yaxis()
    return ax


def plot_score_scatter(
    summary_df: pd.DataFrame,
    title: str = "Original vs optimized contrail score",
) -> plt.Axes:
    if not {"no_change_cost", "best_cost"}.issubset(summary_df.columns):
        raise ValueError("summary_df must include no_change_cost and best_cost")

    fig, ax = plt.subplots(figsize=(5, 5))
    x = summary_df["no_change_cost"].to_numpy()
    y = summary_df["best_cost"].to_numpy()
    ax.scatter(x, y, s=12, alpha=0.6, color="#2a5d9f")
    lim = max(np.nanmax(x), np.nanmax(y)) if len(x) else 1.0
    ax.plot([0, lim], [0, lim], color="#888", linestyle="--", linewidth=1)
    ax.set_xlabel("Original score")
    ax.set_ylabel("Optimized score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_improvement_hist(
    summary_df: pd.DataFrame,
    bins: int = 50,
    title: str = "Contrail score improvement",
) -> plt.Axes:
    if "improvement_abs" not in summary_df.columns:
        raise ValueError("summary_df must include improvement_abs")

    fig, ax = plt.subplots(figsize=(6, 4))
    vals = summary_df["improvement_abs"].to_numpy()
    ax.hist(vals, bins=bins, color="#2a5d9f", alpha=0.8)
    ax.set_xlabel("Improvement (original - optimized)")
    ax.set_ylabel("Flights")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax
