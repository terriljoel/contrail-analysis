from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ACTION_NAMES = {0: "HOLD", 1: "UP", 2: "DOWN"}


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 3
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.1
    max_changes: int = 2
    fuel_penalty: float = 0.05
    scale_by_distance: bool = False
    seed: int = 7


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return np.array([-np.inf, np.inf])
    edges = np.quantile(clean, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def fit_feature_bins(hourly: pd.DataFrame, features: Iterable[str], n_bins: int = 5) -> Dict[str, np.ndarray]:
    bins: Dict[str, np.ndarray] = {}
    for f in features:
        vals = hourly[f].to_numpy(dtype=float)
        bins[f] = _quantile_bins(vals, n_bins)
    return bins


def _bin_value(x: float, edges: np.ndarray) -> int:
    if len(edges) <= 2:
        return 0
    return int(np.digitize([x], edges[1:-1], right=False)[0])


def _state_from_row(
    row: pd.Series,
    *,
    feature_bins: Dict[str, np.ndarray],
    prev_action: int,
    changes_left: int,
) -> Tuple[int, ...]:
    feats = [_bin_value(float(row[f]), feature_bins[f]) for f in feature_bins]
    feats.append(int(prev_action))
    feats.append(int(changes_left))
    return tuple(feats)


def _action_cost(row: pd.Series, action: int) -> float:
    if action == 0:
        return float(row["cost_hold_hour"])
    if action == 1:
        return float(row.get("cost_up_hour", np.nan))
    return float(row.get("cost_down_hour", np.nan))


def train_q_learning(
    hourly: pd.DataFrame,
    *,
    feature_bins: Dict[str, np.ndarray],
    cfg: QLearningConfig,
) -> Dict[Tuple[int, ...], np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    q: Dict[Tuple[int, ...], np.ndarray] = {}

    for _ in range(cfg.episodes):
        for _, df_f in hourly.groupby("flight_id", sort=False):
            df_f = df_f.sort_values("time_hour", kind="stable").reset_index(drop=True)
            prev_action = 0
            changes_left = cfg.max_changes
            for t in range(len(df_f)):
                row = df_f.iloc[t]
                state = _state_from_row(row, feature_bins=feature_bins, prev_action=prev_action, changes_left=changes_left)
                if state not in q:
                    q[state] = np.zeros(3, dtype=float)

                if rng.random() < cfg.epsilon:
                    action = int(rng.integers(0, 3))
                else:
                    action = int(np.argmin(q[state]))

                if changes_left == 0 and action != prev_action:
                    action = prev_action

                cost = _action_cost(row, action)
                if np.isnan(cost):
                    action = prev_action
                    cost = _action_cost(row, action)

                switch_pen = 0.0
                if action != prev_action:
                    if cfg.scale_by_distance:
                        switch_pen = cfg.fuel_penalty * float(row.get("dist_km", 1.0))
                    else:
                        switch_pen = cfg.fuel_penalty

                reward = -(cost + switch_pen)

                next_prev = action
                next_changes = changes_left - (1 if action != prev_action else 0)
                next_changes = max(0, next_changes)

                if t < len(df_f) - 1:
                    next_row = df_f.iloc[t + 1]
                    next_state = _state_from_row(
                        next_row,
                        feature_bins=feature_bins,
                        prev_action=next_prev,
                        changes_left=next_changes,
                    )
                    if next_state not in q:
                        q[next_state] = np.zeros(3, dtype=float)
                    td_target = reward + cfg.gamma * np.min(q[next_state])
                else:
                    td_target = reward

                q[state][action] += cfg.alpha * (td_target - q[state][action])

                prev_action = next_prev
                changes_left = next_changes

    return q


def rollout_policy(
    hourly: pd.DataFrame,
    *,
    feature_bins: Dict[str, np.ndarray],
    q_table: Dict[Tuple[int, ...], np.ndarray],
    cfg: QLearningConfig,
) -> pd.DataFrame:
    records: List[pd.DataFrame] = []
    for flight_id, df_f in hourly.groupby("flight_id", sort=False):
        df_f = df_f.sort_values("time_hour", kind="stable").reset_index(drop=True)
        prev_action = 0
        changes_left = cfg.max_changes
        actions: List[int] = []
        for t in range(len(df_f)):
            row = df_f.iloc[t]
            state = _state_from_row(row, feature_bins=feature_bins, prev_action=prev_action, changes_left=changes_left)
            if state not in q_table:
                q_table[state] = np.zeros(3, dtype=float)
            action = int(np.argmin(q_table[state]))

            if changes_left == 0 and action != prev_action:
                action = prev_action

            cost = _action_cost(row, action)
            if np.isnan(cost):
                action = prev_action
                cost = _action_cost(row, action)

            if action != prev_action:
                changes_left = max(0, changes_left - 1)
            prev_action = action
            actions.append(action)

        df_f = df_f.copy()
        df_f["action_code"] = actions
        df_f["action"] = [ACTION_NAMES[a] for a in actions]
        records.append(df_f)

    return pd.concat(records, ignore_index=True)


def summarize_policy(policy_hourly: pd.DataFrame) -> pd.DataFrame:
    def _choose_cost(row: pd.Series) -> float:
        if row["action"] == "HOLD":
            return row["cost_hold_hour"]
        if row["action"] == "UP":
            return row.get("cost_up_hour", np.nan)
        return row.get("cost_down_hour", np.nan)

    policy_hourly = policy_hourly.copy()
    policy_hourly["cost_chosen_hour"] = policy_hourly.apply(_choose_cost, axis=1)

    summary = (
        policy_hourly.groupby("flight_id", as_index=False)
        .agg(
            best_cost=("cost_chosen_hour", "sum"),
            no_change_cost=("cost_hold_hour", "sum"),
        )
        .sort_values("flight_id", kind="stable")
    )
    summary["improvement_abs"] = summary["no_change_cost"] - summary["best_cost"]
    summary["improvement_pct"] = np.where(
        summary["no_change_cost"] > 0,
        summary["improvement_abs"] / summary["no_change_cost"],
        0.0,
    )
    return summary
