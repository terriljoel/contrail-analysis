from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


ACTION_NAMES = {0: "HOLD", 1: "UP", 2: "DOWN"}


def _mode_first(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    mode_vals = series.mode()
    if mode_vals.empty:
        return float("nan")
    return float(mode_vals.iloc[0])


def build_hourly_table(segments: pd.DataFrame, time_bin: str = "1h") -> pd.DataFrame:
    required = {"flight_id", "time_hour", "dist_km"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"segments missing required columns: {sorted(missing)}")

    df = segments.copy()
    df["time_hour"] = pd.to_datetime(df["time_hour"]).dt.tz_localize(None).dt.floor(time_bin)

    agg_kwargs = {}
    if "segment_score" in df.columns:
        agg_kwargs["cost_hold_hour"] = ("segment_score", "sum")
    if "segment_score_up" in df.columns:
        agg_kwargs["cost_up_hour"] = ("segment_score_up", "sum")
    if "segment_score_down" in df.columns:
        agg_kwargs["cost_down_hour"] = ("segment_score_down", "sum")
    agg_kwargs["dist_km"] = ("dist_km", "sum")

    if "RHi" in df.columns:
        agg_kwargs["rhi_mean"] = ("RHi", "mean")
        agg_kwargs["rhi_max"] = ("RHi", "max")
    if "ISSR" in df.columns:
        agg_kwargs["issr_frac"] = ("ISSR", "mean")
    if "is_night" in df.columns:
        agg_kwargs["is_night_mean"] = ("is_night", "mean")
    if "mid_alt_m" in df.columns:
        agg_kwargs["mid_alt_m_mean"] = ("mid_alt_m", "mean")
    if "plev_hpa" in df.columns:
        agg_kwargs["plev_hpa_mode"] = ("plev_hpa", _mode_first)
    if "plev_up" in df.columns:
        agg_kwargs["plev_up_mode"] = ("plev_up", _mode_first)
    if "plev_down" in df.columns:
        agg_kwargs["plev_down_mode"] = ("plev_down", _mode_first)

    hourly = (
        df.groupby(["flight_id", "time_hour"], as_index=False)
        .agg(**agg_kwargs)
        .sort_values(["flight_id", "time_hour"], kind="stable")
        .reset_index(drop=True)
    )

    if "is_night_mean" in hourly.columns:
        hourly["is_night_majority"] = hourly["is_night_mean"] >= 0.5
    if "issr_frac" in hourly.columns:
        hourly["issr_frac"] = hourly["issr_frac"].fillna(0.0)

    hourly["step_idx"] = hourly.groupby("flight_id").cumcount()
    return hourly


def _solve_hourly_sequence(
    df_f: pd.DataFrame,
    max_changes: int,
    fuel_penalty: float,
    scale_by_distance: bool,
    prefer_hold_start: bool,
) -> Tuple[List[int], float]:
    df_f = df_f.sort_values("time_hour", kind="stable").reset_index(drop=True)
    n = len(df_f)
    if n == 0:
        return [], float("nan")

    cost_hold = df_f["cost_hold_hour"].to_numpy(dtype=float)
    cost_up = df_f.get("cost_up_hour", pd.Series([np.nan] * n)).to_numpy(dtype=float)
    cost_down = df_f.get("cost_down_hour", pd.Series([np.nan] * n)).to_numpy(dtype=float)
    cost = np.vstack([cost_hold, cost_up, cost_down]).T
    cost = np.where(np.isfinite(cost), cost, np.inf)

    dist = df_f["dist_km"].to_numpy(dtype=float) if "dist_km" in df_f.columns else np.ones(n)
    penalty = fuel_penalty * (dist if scale_by_distance else np.ones(n))

    K = max_changes
    dp = np.full((n, K + 1, 3), np.inf, dtype=float)
    prev_a = np.full((n, K + 1, 3), -1, dtype=int)
    prev_k = np.full((n, K + 1, 3), -1, dtype=int)

    for a in range(3):
        if prefer_hold_start and a != 0:
            continue
        dp[0, 0, a] = cost[0, a]

    for t in range(1, n):
        for k in range(K + 1):
            for a in range(3):
                best_val = dp[t - 1, k, a]
                best_prev = a
                best_k = k
                if k > 0:
                    for ap in range(3):
                        if ap == a:
                            continue
                        cand = dp[t - 1, k - 1, ap] + penalty[t]
                        if cand < best_val:
                            best_val = cand
                            best_prev = ap
                            best_k = k - 1
                dp[t, k, a] = cost[t, a] + best_val
                prev_a[t, k, a] = best_prev
                prev_k[t, k, a] = best_k

    best_cost = np.inf
    best_a = 0
    best_k = 0
    for k in range(K + 1):
        for a in range(3):
            if dp[n - 1, k, a] < best_cost:
                best_cost = dp[n - 1, k, a]
                best_a = a
                best_k = k

    actions = [best_a]
    k = best_k
    a = best_a
    for t in range(n - 1, 0, -1):
        pa = prev_a[t, k, a]
        pk = prev_k[t, k, a]
        actions.append(pa)
        a = pa
        k = pk
    actions.reverse()

    return actions, float(best_cost)


def solve_hourly_policy(
    hourly: pd.DataFrame,
    max_changes: int = 1,
    fuel_penalty: float = 0.0,
    scale_by_distance: bool = False,
    prefer_hold_start: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "cost_hold_hour" not in hourly.columns:
        raise ValueError("hourly missing cost_hold_hour; run build_hourly_table first")

    records = []
    summary = []
    for flight_id, df_f in hourly.groupby("flight_id", sort=False):
        actions, best_cost = _solve_hourly_sequence(
            df_f,
            max_changes=max_changes,
            fuel_penalty=fuel_penalty,
            scale_by_distance=scale_by_distance,
            prefer_hold_start=prefer_hold_start,
        )
        df_sorted = df_f.sort_values("time_hour", kind="stable").reset_index(drop=True)
        action_names = [ACTION_NAMES[a] for a in actions]
        df_sorted["action"] = action_names
        df_sorted["action_code"] = actions
        records.append(df_sorted)

        no_change_cost = float(df_sorted["cost_hold_hour"].sum())
        changes = int((df_sorted["action_code"].diff().fillna(0) != 0).sum())
        summary.append(
            {
                "flight_id": flight_id,
                "best_cost": best_cost,
                "no_change_cost": no_change_cost,
                "improvement_abs": no_change_cost - best_cost,
                "improvement_pct": (no_change_cost - best_cost) / no_change_cost if no_change_cost > 0 else 0.0,
                "changes_used": changes,
            }
        )

    policy_hourly = pd.concat(records, ignore_index=True)
    summary_df = pd.DataFrame(summary)

    def _choose_cost(row: pd.Series) -> float:
        if row["action"] == "HOLD":
            return row["cost_hold_hour"]
        if row["action"] == "UP":
            return row.get("cost_up_hour", np.nan)
        return row.get("cost_down_hour", np.nan)

    policy_hourly["cost_chosen_hour"] = policy_hourly.apply(_choose_cost, axis=1)

    if "plev_hpa_mode" in policy_hourly.columns:
        def _choose_plev(row: pd.Series) -> float:
            if row["action"] == "HOLD":
                return row["plev_hpa_mode"]
            if row["action"] == "UP":
                val = row.get("plev_up_mode", np.nan)
                return row["plev_hpa_mode"] if pd.isna(val) else val
            val = row.get("plev_down_mode", np.nan)
            return row["plev_hpa_mode"] if pd.isna(val) else val

        policy_hourly["plev_reco"] = policy_hourly.apply(_choose_plev, axis=1)

    return policy_hourly, summary_df
