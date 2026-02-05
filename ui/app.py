from __future__ import annotations

import subprocess
import sys
from datetime import date, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
load_dotenv(ROOT / ".env")

from contrail_mvp.config import get_default_config
from contrail_mvp.data_catalog import (
    filter_scored_with_counterfactuals,
    list_processed_segments,
    list_raw_opensky,
    list_scored_segments,
)
from contrail_mvp.optimize import build_hourly_table, solve_hourly_policy
from contrail_mvp.rl_tabular import (
    QLearningConfig,
    fit_feature_bins,
    rollout_policy,
    summarize_policy,
    train_q_learning,
)
from contrail_mvp.viz import plot_altitude_profile, plot_improvement_hist, plot_score_scatter
from contrail_mvp.trino_opensky import TrinoConfig, check_trino_auth, download_hours_to_parquet
from contrail_mvp.era5 import Era5Request, download_era5_pressure_levels, nc_to_parquet
from contrail_mvp.preprocess import PreprocessConfig, process_opensky_raw
from contrail_mvp.scoring import ScoringConfig, attach_era5_and_score


st.set_page_config(page_title="Contrail Optimization Demo", layout="wide")


cfg = get_default_config()
data_root = Path(cfg.paths.get("data_root", "data"))
ui_cfg = cfg.raw.get("ui", {})
max_download_days = int(ui_cfg.get("max_download_days", 1))
max_poll_minutes = int(ui_cfg.get("max_poll_minutes", 30))
max_flights = int(ui_cfg.get("max_flights", 300))
polite_notice = bool(ui_cfg.get("polite_notice", True))

st.title("Contrail Optimization Demo")

st.sidebar.header("Data")
st.sidebar.caption("Use existing data if available; download only if missing.")

raw_files = list_raw_opensky(data_root)
if raw_files:
    st.sidebar.success(f"Raw OpenSky files: {len(raw_files)}")
else:
    st.sidebar.warning("No raw OpenSky files found.")

st.subheader("Available raw data")
if raw_files:
    st.dataframe(pd.DataFrame({"path": [str(p) for p in raw_files]}), width="stretch")
else:
    st.write("No raw data found in data/raw/opensky.")

st.subheader("Available processed data")
processed_list = list_processed_segments(data_root, include_archive=False)
if processed_list:
    st.dataframe(pd.DataFrame({"path": [str(p) for p in processed_list]}), width="stretch")
else:
    st.write("No processed files found.")

st.subheader("Data availability summary")
def _summarize_paths(paths):
    rows = []
    for p in paths:
        parts = p.parts
        region = None
        day = None
        for i, part in enumerate(parts):
            if part in ("opensky", "opensky_segments", "scores", "weather_ERA5"):
                if i + 1 < len(parts):
                    region = parts[i + 1]
                if i + 2 < len(parts):
                    day = parts[i + 2]
        rows.append({"region": region, "day": day})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summary = (
        df.groupby("region", dropna=False)
        .agg(
            days=("day", lambda x: sorted(set([d for d in x if isinstance(d, str)]))),
            files=("day", "size"),
        )
        .reset_index()
    )
    summary["day_range"] = summary["days"].apply(lambda d: f"{d[0]} -> {d[-1]}" if len(d) else "-")
    summary = summary.drop(columns=["days"])
    return summary

raw_summary = _summarize_paths(raw_files)
proc_summary = _summarize_paths(processed_list)

if not raw_summary.empty:
    st.write("Raw OpenSky")
    st.dataframe(raw_summary, width="stretch")
if not proc_summary.empty:
    st.write("Processed (segments/scores)")
    st.dataframe(proc_summary, width="stretch")

st.subheader("Download OpenSky data (limited)")
if polite_notice:
    st.info(
        "Please keep downloads minimal and respectful. "
        "Use cached data when available and avoid repeated downloads."
    )
region = st.selectbox("Region", options=list(cfg.opensky.get("bbox", {}).keys()) or ["europe", "germany"])
day = st.date_input("Day (YYYY-MM-DD)", value=date(2025, 1, 13))
download_hours = st.slider("Download hours (UTC)", min_value=0, max_value=23, value=(0, 23))
max_rows_per_hour = st.number_input("Max rows per hour (0 = no limit)", min_value=0, max_value=500000, value=0, step=10000)
use_alt_filter = st.checkbox("Apply cruise altitude filter", value=True)
alt_min = st.number_input("Altitude min (m)", min_value=0, max_value=20000, value=8000, step=500)
alt_max = st.number_input("Altitude max (m)", min_value=0, max_value=20000, value=13000, step=500)
confirm = st.checkbox("I will keep downloads minimal and avoid repeated requests.", value=False)
auth_btn = st.button("Check / Authenticate Trino")
download = st.button("Download data", disabled=not confirm)

trino_cfg_raw = cfg.opensky.get("trino", {})
trino_cfg = TrinoConfig(
    host=trino_cfg_raw.get("host", "trino.opensky-network.org"),
    port=int(trino_cfg_raw.get("port", 443)),
    http_scheme=trino_cfg_raw.get("http_scheme", "https"),
    user=str(trino_cfg_raw.get("user", "")),
    catalog=trino_cfg_raw.get("catalog", "minio"),
    schema=trino_cfg_raw.get("schema", "osky"),
    table=trino_cfg_raw.get("table", "state_vectors_data4"),
    max_rows_per_hour=int(max_rows_per_hour or trino_cfg_raw.get("max_rows_per_hour", 0)),
)
if not trino_cfg.user or trino_cfg.user == "your_username":
    st.error("Set opensky.trino.user in config/config.yaml or OPENSKY_TRINO_USER env var.")
else:
    masked = trino_cfg.user[:2] + "***" if len(trino_cfg.user) > 2 else trino_cfg.user
    st.caption(f"Trino user detected: {masked}")
if auth_btn and trino_cfg.user and trino_cfg.user != "your_username":
    with st.spinner("Checking Trino auth..."):
        authed, url = check_trino_auth(trino_cfg)
        st.session_state["trino_authed"] = authed
        if url:
            st.session_state["trino_auth_url"] = url
        if authed:
            st.session_state["trino_auth_msg"] = "Authenticated successfully."
        else:
            st.session_state["trino_auth_msg"] = "Login required. Open the link and then click Check again."
if "trino_auth_url" in st.session_state:
    st.link_button("Open Trino login", st.session_state["trino_auth_url"])
    st.caption("After login, return here and click Check / Authenticate again.")
if "trino_auth_msg" in st.session_state and st.session_state["trino_auth_msg"]:
    if st.session_state.get("trino_authed"):
        st.success(st.session_state["trino_auth_msg"])
    else:
        st.warning(st.session_state["trino_auth_msg"])
if download:
    if max_download_days < 1:
        st.error("Downloads are disabled by config.")
    else:
        out_dir = data_root / "raw" / "opensky" / region / str(day)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not st.session_state.get("trino_authed", False):
            st.warning("Please click Check / Authenticate first and complete login.")
            st.stop()
        bbox_cfg = cfg.opensky.get("bbox", {}).get(region, [])
        if len(bbox_cfg) != 4:
            st.error("Invalid bbox config.")
            st.stop()
        bbox = {
            "lat_min": float(bbox_cfg[0]),
            "lon_min": float(bbox_cfg[1]),
            "lat_max": float(bbox_cfg[2]),
            "lon_max": float(bbox_cfg[3]),
        }
        if not trino_cfg.user or trino_cfg.user == "your_username":
            st.error("Set opensky.trino.user in config/config.yaml before downloading.")
            st.stop()
        start_dt = pd.Timestamp(day).replace(hour=download_hours[0]).tz_localize("UTC").to_pydatetime()
        end_dt = pd.Timestamp(day).replace(hour=download_hours[1]).tz_localize("UTC").to_pydatetime()
        alt_min_val = float(alt_min) if use_alt_filter else None
        alt_max_val = float(alt_max) if use_alt_filter else None
        with st.spinner("Downloading from Trino..."):
            try:
                written = download_hours_to_parquet(
                    out_dir=out_dir,
                    start_utc=start_dt,
                    end_utc=end_dt,
                    bbox=bbox,
                    alt_min=alt_min_val,
                    alt_max=alt_max_val,
                    trino_cfg=trino_cfg,
                    prefix=f"states_{region}",
                )
                if written:
                    total_rows = 0
                    for p in written:
                        try:
                            total_rows += len(pd.read_parquet(p))
                        except Exception:
                            pass
                    st.success(f"Downloaded {len(written)} hourly files to {out_dir}")
                    st.caption(f"Rows downloaded (approx): {total_rows:,}")
                else:
                    st.warning(
                        "Downloaded 0 files. Possible reasons: no rows for the selected time/bbox/alt band, "
                        "or files already exist. Try widening the hour range or disabling altitude filter."
                    )
            except Exception as exc:
                st.error(f"Trino download failed: {exc}")

st.subheader("Download ERA5 weather (pressure levels)")
st.caption("Requires a valid CDS API key in ~/.cdsapirc. Data is cached under data/raw/weather_ERA5.")
era5_region = st.selectbox("ERA5 region", options=list(cfg.opensky.get("bbox", {}).keys()) or ["europe", "germany"], key="era5_region")
era5_day = st.date_input("ERA5 day (YYYY-MM-DD)", value=date(2025, 1, 13), key="era5_day")
era5_hours = st.slider("ERA5 hours (UTC)", min_value=0, max_value=23, value=(0, 23), key="era5_hours")
levels_default = cfg.era5.get("pressure_levels_hpa", [200, 225, 250, 300, 350])
era5_levels = st.multiselect("Pressure levels (hPa)", options=levels_default, default=levels_default)
era5_download = st.button("Download ERA5")
era5_process = st.button("Process ERA5 to parquet")

if era5_download:
    bbox_cfg = cfg.opensky.get("bbox", {}).get(era5_region, [])
    if len(bbox_cfg) != 4:
        st.error("Invalid bbox config.")
        st.stop()
    bbox = [bbox_cfg[2], bbox_cfg[1], bbox_cfg[0], bbox_cfg[3]]  # north, west, south, east
    area_name = f"states_{era5_region}"
    out_nc = data_root / "raw" / "weather_ERA5" / area_name / str(era5_day) / f"era5_pl_{area_name}_T_q.nc"
    req = Era5Request(
        day=era5_day,
        bbox=bbox,
        pressure_levels_hpa=[int(p) for p in era5_levels],
        variables=["temperature", "specific_humidity"],
        hours=list(range(era5_hours[0], era5_hours[1] + 1)),
    )
    with st.spinner("Downloading ERA5 from CDS..."):
        try:
            download_era5_pressure_levels(req, out_nc)
            st.success(f"Downloaded ERA5 to {out_nc}")
        except Exception as exc:
            st.error(f"ERA5 download failed: {exc}")

if era5_process:
    area_name = f"states_{era5_region}"
    in_nc = data_root / "raw" / "weather_ERA5" / area_name / str(era5_day) / f"era5_pl_{area_name}_T_q.nc"
    out_pq = data_root / "processed" / "weather_ERA5" / area_name / str(era5_day) / f"era5_pl_{area_name}_T_q.parquet"
    if not in_nc.exists():
        st.error(f"Missing ERA5 file: {in_nc}")
        st.stop()
    with st.spinner("Processing ERA5 to parquet..."):
        try:
            nc_to_parquet(in_nc, out_pq)
            st.success(f"Saved parquet to {out_pq}")
            try:
                era5_rows = len(pd.read_parquet(out_pq))
                st.caption(f"ERA5 rows: {era5_rows:,}")
            except Exception:
                pass
        except Exception as exc:
            st.error(f"ERA5 processing failed: {exc}")

st.subheader("Process OpenSky + ERA5 to segments with scores")
st.caption("This builds segments and merges ERA5 to compute RHi/ISSR and scores.")
proc_region = st.selectbox("Processing region", options=list(cfg.opensky.get("bbox", {}).keys()) or ["europe", "germany"], key="proc_region")
proc_day_range = st.date_input(
    "Processing day range (YYYY-MM-DD)",
    value=(date(2025, 1, 13), date(2025, 1, 13)),
    key="proc_day_range",
)
proc_alt_filter = st.checkbox("Apply altitude filter", value=True, key="proc_alt_filter")
proc_alt_min = st.number_input("Proc alt min (m)", min_value=0, max_value=20000, value=8000, step=500, key="proc_alt_min")
proc_alt_max = st.number_input("Proc alt max (m)", min_value=0, max_value=20000, value=13000, step=500, key="proc_alt_max")
trim_start = st.number_input("Trim start (minutes)", min_value=0, max_value=120, value=20, step=5, key="trim_start")
trim_end = st.number_input("Trim end (minutes)", min_value=0, max_value=120, value=20, step=5, key="trim_end")
downsample_minute = st.checkbox("Downsample to 1 point per minute", value=True, key="downsample_minute")
max_seg_gap = st.number_input("Max segment gap (seconds)", min_value=60, max_value=1800, value=300, step=60, key="max_seg_gap")
run_processing = st.button("Run processing")

if run_processing:
    if isinstance(proc_day_range, tuple) and len(proc_day_range) == 2:
        proc_start, proc_end = proc_day_range
    else:
        proc_start = proc_end = proc_day_range

    proc_cfg = PreprocessConfig(
        gap_s=30 * 60,
        alt_min_m=float(proc_alt_min) if proc_alt_filter else None,
        alt_max_m=float(proc_alt_max) if proc_alt_filter else None,
        trim_start_min=int(trim_start),
        trim_end_min=int(trim_end),
        downsample_minute=bool(downsample_minute),
        max_segment_gap_s=int(max_seg_gap),
    )
    area_name = f"states_{proc_region}"
    score_cfg = ScoringConfig(pressure_levels_hpa=cfg.era5.get("pressure_levels_hpa", [200, 225, 250, 300, 350]))

    current = pd.Timestamp(proc_start)
    end_ts = pd.Timestamp(proc_end)
    while current <= end_ts:
        day_str = current.date().isoformat()
        raw_dir = data_root / "raw" / "opensky" / proc_region / day_str
        if not raw_dir.exists():
            st.warning(f"Skipping {day_str}: missing OpenSky raw folder {raw_dir}")
            current += pd.Timedelta(days=1)
            continue
        era5_pq = data_root / "processed" / "weather_ERA5" / area_name / day_str / f"era5_pl_{area_name}_T_q.parquet"
        if not era5_pq.exists():
            st.warning(f"Skipping {day_str}: missing ERA5 parquet {era5_pq}")
            current += pd.Timedelta(days=1)
            continue

        out_segments = data_root / "processed" / "opensky_segments" / area_name / day_str / f"opensky_segments_{area_name}_{day_str}.parquet"
        with st.spinner(f"Building segments for {day_str}..."):
            try:
                process_opensky_raw(raw_dir, out_segments, proc_cfg)
                seg_rows = 0
                try:
                    seg_rows = len(pd.read_parquet(out_segments))
                except Exception:
                    pass
                st.success(f"Segments saved to {out_segments}")
                if seg_rows:
                    st.caption(f"Segments rows: {seg_rows:,}")
            except Exception as exc:
                st.error(f"Segment processing failed for {day_str}: {exc}")
                current += pd.Timedelta(days=1)
                continue

        with st.spinner(f"Scoring segments for {day_str}..."):
            try:
                era5 = pd.read_parquet(era5_pq)
                seg = pd.read_parquet(out_segments)
                scored = attach_era5_and_score(seg, era5, score_cfg)
                out_scored = data_root / "processed" / "scores" / area_name / day_str / "segments_with_era5_rhi_issr.parquet"
                out_scored.parent.mkdir(parents=True, exist_ok=True)
                scored.to_parquet(out_scored, index=False)
                st.success(f"Scored segments saved to {out_scored}")
                st.caption(f"Scored rows: {len(scored):,}")
            except Exception as exc:
                st.error(f"Scoring failed for {day_str}: {exc}")
        current += pd.Timedelta(days=1)

st.subheader("Run optimization on scored segments")
only_counterfactuals = st.checkbox("Only show files with segment_score_up/down", value=True)
scored_files = list_scored_segments(data_root, include_archive=False)
if only_counterfactuals:
    scored_files = filter_scored_with_counterfactuals(scored_files)
if not scored_files:
    st.warning("No scored segment files found. Run scoring first or disable the counterfactual filter.")
    st.stop()

seg_path = st.selectbox("Scored segments file (parquet)", options=[str(p) for p in scored_files])
use_filters = st.checkbox("Use filters", value=True)
date_range = st.date_input(
    "Date range",
    value=(date(2025, 1, 13), date(2025, 1, 13)),
)
hour_range = st.slider("Hour range (UTC)", min_value=0, max_value=23, value=(0, 23))
max_flights_ui = st.number_input(
    "Top N flights by baseline contrail score",
    min_value=50,
    max_value=5000,
    value=max_flights,
    step=50,
)
if use_filters:
    st.caption("When enabled, this keeps the highest-impact flights by baseline score.")
min_flight_hours = st.number_input(
    "Min flight duration (hours)",
    min_value=0,
    max_value=24,
    value=2,
    step=1,
)
policy_mode = st.selectbox("Policy mode", options=["Optimizer (constrained)", "RL (tabular)"])
constraint_mode = st.selectbox(
    "Constraint mode",
    options=[
        "One change total",
        "No change within 1 hour",
        "No change within 30 minutes",
    ],
)
episodes = st.slider("RL episodes", min_value=1, max_value=10, value=3, disabled=policy_mode != "RL (tabular)")
epsilon = st.slider("Exploration (epsilon)", min_value=0.0, max_value=0.5, value=0.1, step=0.05, disabled=policy_mode != "RL (tabular)")
max_changes = st.slider(
    "Max altitude changes per flight",
    min_value=0,
    max_value=6,
    value=2,
    disabled=constraint_mode == "One change total",
)
if constraint_mode == "One change total":
    st.caption("Max changes is fixed to 1 by this constraint.")
else:
    st.caption("Higher values allow more frequent altitude changes (less operationally realistic).")
fuel_penalty = st.number_input("Fuel penalty per change", min_value=0.0, max_value=10.0, value=0.05, step=0.05)

run_rl = st.button("Run RL policy")
if run_rl:
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range
    with st.spinner("Loading segments..."):
        segments = pd.read_parquet(seg_path)
    if "segment_score_up" not in segments.columns or "segment_score_down" not in segments.columns:
        st.error("segment_score_up/down missing. Use a segments file with counterfactual scores.")
        st.stop()

    if "flight_id" not in segments.columns:
        st.error("Segments file missing flight_id.")
        st.stop()

    if constraint_mode == "No change within 30 minutes":
        time_bin = "30min"
    else:
        time_bin = "1h"

    with st.spinner("Building time-binned table..."):
        hourly = build_hourly_table(segments, time_bin=time_bin)

    if len(hourly) == 0:
        st.error("No hourly data found.")
        st.stop()

    st.caption(f"Flights available before filtering: {hourly['flight_id'].nunique():,}")

    if use_filters:
        hourly["time_hour"] = pd.to_datetime(hourly["time_hour"])
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        hourly = hourly[(hourly["time_hour"] >= start_dt) & (hourly["time_hour"] <= end_dt)].copy()
        hourly = hourly[
            (hourly["time_hour"].dt.hour >= hour_range[0]) & (hourly["time_hour"].dt.hour <= hour_range[1])
        ].copy()

        if len(hourly) == 0:
            st.error("No hourly data after date/hour filters.")
            st.stop()

        if min_flight_hours > 0:
            sizes = hourly.groupby("flight_id").size()
            keep_ids = sizes[sizes >= int(min_flight_hours)].index
            hourly = hourly[hourly["flight_id"].isin(keep_ids)].copy()
            if len(hourly) == 0:
                st.error("No flights left after min duration filter.")
                st.stop()

        if max_flights_ui > 0:
            top_flights = (
                hourly.groupby("flight_id")["cost_hold_hour"]
                .sum()
                .sort_values(ascending=False)
                .head(max_flights_ui)
                .index
            )
            hourly = hourly[hourly["flight_id"].isin(top_flights)].copy()
            st.caption(f"Flights after top-N filter: {hourly['flight_id'].nunique():,}")

    feature_cols = ["rhi_mean", "rhi_max", "issr_frac", "is_night_mean", "plev_hpa_mode"]
    feature_cols = [c for c in feature_cols if c in hourly.columns]
    if not feature_cols:
        st.error("Missing required features for RL (rhi_mean/rhi_max/issr_frac/is_night_mean/plev_hpa_mode).")
        st.stop()

    if constraint_mode == "One change total":
        max_changes = 1

    if policy_mode == "Optimizer (constrained)":
        with st.spinner("Running constrained optimizer..."):
            policy_hourly, summary = solve_hourly_policy(
                hourly,
                max_changes=max_changes,
                fuel_penalty=fuel_penalty,
                scale_by_distance=False,
                prefer_hold_start=True,
            )
    else:
        cfg_rl = QLearningConfig(
            episodes=episodes,
            epsilon=epsilon,
            max_changes=max_changes,
            fuel_penalty=fuel_penalty,
        )
        with st.spinner("Training tabular Q-learning..."):
            bins = fit_feature_bins(hourly, feature_cols, n_bins=5)
            q = train_q_learning(hourly, feature_bins=bins, cfg=cfg_rl)

        with st.spinner("Rolling out policy..."):
            policy_hourly = rollout_policy(hourly, feature_bins=bins, q_table=q, cfg=cfg_rl)
        summary = summarize_policy(policy_hourly)
    policy_hourly["flight_id"] = policy_hourly["flight_id"].astype(str)
    hourly["flight_id"] = hourly["flight_id"].astype(str)
    summary["flight_id"] = summary["flight_id"].astype(str)

    st.session_state["rl_hourly"] = hourly
    st.session_state["rl_policy_hourly"] = policy_hourly
    st.session_state["rl_summary"] = summary

if "rl_summary" in st.session_state:
    hourly = st.session_state["rl_hourly"]
    policy_hourly = st.session_state["rl_policy_hourly"]
    summary = st.session_state["rl_summary"]

    st.subheader("Example flight")
    action_counts = (
        policy_hourly.groupby(["flight_id", "action"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    count_cols = [c for c in action_counts.columns if c != "flight_id"]
    if "HOLD" in action_counts.columns:
        action_counts["non_hold"] = action_counts[count_cols].sum(axis=1) - action_counts["HOLD"]
    else:
        action_counts["non_hold"] = action_counts[count_cols].sum(axis=1)

    candidates = action_counts[action_counts["non_hold"] > 0]["flight_id"].astype(str).tolist()
    if not candidates:
        candidates = summary.sort_values("improvement_abs", ascending=False)["flight_id"].head(30).tolist()
    if not candidates:
        candidates = policy_hourly["flight_id"].astype(str).unique().tolist()

    example_flight = st.selectbox("Flight (prefers non-HOLD)", options=candidates)

    st.subheader("Score interpretation (selected flight)")
    if not summary.empty:
        sel = summary[summary["flight_id"] == example_flight]
        if not sel.empty:
            orig = float(sel["no_change_cost"].iloc[0])
            opt = float(sel["best_cost"].iloc[0])
            impr_pct = float(sel["improvement_pct"].iloc[0]) * 100.0
            orig_scores = summary["no_change_cost"].to_numpy()
            median = float(np.median(orig_scores))
            pct_rank = float((orig_scores <= orig).mean() * 100.0)
            st.write(
                f"Original score: {orig:.2f} | Optimized: {opt:.2f} | Improvement: {impr_pct:.1f}%"
            )
            st.write(
                f"Percentile rank: {pct_rank:.1f}th | Relative to median: {orig/median:.1f}?"
                if median > 0 else f"Percentile rank: {pct_rank:.1f}th"
            )
    try:
        ax3 = plot_altitude_profile(hourly, policy_hourly, example_flight)
        st.pyplot(ax3.figure)
    except Exception as exc:
        st.error(f"Could not plot flight {example_flight}: {exc}")

    st.subheader("Flight timeline by action (selected flight only)")
    flight_actions = (
        policy_hourly[policy_hourly["flight_id"] == example_flight]
        .sort_values("time_hour", kind="stable")
    )
    if not flight_actions.empty:
        timeline = flight_actions[["time_hour", "action", "plev_reco", "plev_hpa_mode"]].copy()
        # Derive action from recommended altitude change if possible (more intuitive)
        if "plev_reco" in timeline.columns and timeline["plev_reco"].notna().any():
            reco = timeline["plev_reco"].astype(float)
            delta = reco.diff().fillna(0)
            action_from_reco = np.where(delta < 0, "UP", np.where(delta > 0, "DOWN", "HOLD"))
            timeline["action_plot"] = action_from_reco
        else:
            timeline["action_plot"] = timeline["action"]

        timeline["action_code"] = timeline["action_plot"].map({"HOLD": 0, "UP": 1, "DOWN": -1}).fillna(0)
        color_map = {"HOLD": "#4c78a8", "UP": "#54a24b", "DOWN": "#e45756"}
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.step(timeline["time_hour"], timeline["action_code"], where="post", color="#888", linewidth=1)
        for action, grp in timeline.groupby("action_plot"):
            ax.scatter(
                grp["time_hour"],
                grp["action_code"],
                label=action,
                color=color_map.get(action, "#888"),
                s=30,
            )
        ax.set_yticks([-1, 0, 1], labels=["DOWN", "HOLD", "UP"])
        ax.set_xlabel("Time (hour)")
        ax.set_ylabel("Action")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.subheader("Impact summary")
    st.dataframe(summary.describe(), width="stretch")
    total_base = float(summary["no_change_cost"].sum())
    total_opt = float(summary["best_cost"].sum())
    total_impr = total_base - total_opt
    total_impr_pct = (total_impr / total_base * 100.0) if total_base > 0 else 0.0
    st.caption(
        f"Total baseline: {total_base:.2f} | Total optimized: {total_opt:.2f} | "
        f"Total improvement: {total_impr:.2f} ({total_impr_pct:.1f}%)"
    )

    col1, col2 = st.columns(2)
    with col1:
        ax1 = plot_score_scatter(summary)
        st.pyplot(ax1.figure)
    with col2:
        ax2 = plot_improvement_hist(summary)
        st.pyplot(ax2.figure)

    st.subheader("Distribution of baseline scores")
    st.bar_chart(summary["no_change_cost"], width="stretch")

    st.subheader("Per-flight improvement (top 30)")
    top_improve = summary.sort_values("improvement_abs", ascending=False).head(30)
    st.bar_chart(
        top_improve.set_index("flight_id")["improvement_abs"],
        width="stretch",
    )

    st.subheader("Action share (overall)")
    action_share = policy_hourly["action"].value_counts(normalize=True).rename("share")
    st.bar_chart(action_share, width="stretch")
