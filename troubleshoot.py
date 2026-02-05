import pandas as pd
from pathlib import Path
from src.contrail_mvp.optimize import build_hourly_table
from src.contrail_mvp.rl_tabular import QLearningConfig, fit_feature_bins, train_q_learning, rollout_policy, summarize_policy

# pick a scored file
# files = list(Path("data").rglob("segments_with_era5_rhi_issr.parquet"))
# print("scored files:", len(files))
# path = files[0]
# print("using", path)

# seg = pd.read_parquet(path)
# print("segments", seg.shape)
# print("segment_score stats")
# print(seg["segment_score"].describe())
# print("ISSR share", seg["ISSR"].mean())

# # check if counterfactual scores differ
# print("segment_score_up stats")
# print(seg["segment_score_up"].describe())
# print("segment_score_down stats")
# print(seg["segment_score_down"].describe())
# print("mean abs diff up", (seg["segment_score_up"] - seg["segment_score"]).abs().mean())
# print("mean abs diff down", (seg["segment_score_down"] - seg["segment_score"]).abs().mean())

# hourly = build_hourly_table(seg)
# print("hourly rows", len(hourly), "flights", hourly["flight_id"].nunique())
# print("hourly cost stats")
# print(hourly[["cost_hold_hour","cost_up_hour","cost_down_hour"]].describe())

# features = [c for c in ["rhi_mean","rhi_max","issr_frac","is_night_mean","plev_hpa_mode"] if c in hourly.columns]
# bins = fit_feature_bins(hourly, features, n_bins=5)
# q = train_q_learning(hourly, feature_bins=bins, cfg=QLearningConfig(episodes=2, max_changes=2, fuel_penalty=0.05, epsilon=0.1))
# policy = rollout_policy(hourly, feature_bins=bins, q_table=q, cfg=QLearningConfig(episodes=2, max_changes=2, fuel_penalty=0.05, epsilon=0.1))
# summary = summarize_policy(policy)

# print(summary[["no_change_cost","best_cost","improvement_abs","improvement_pct"]].describe())
# print("improvement>0 share", (summary["improvement_abs"]>0).mean())
# print("actions", policy["action"].value_counts())
import pandas as pd
from pathlib import Path

path = Path("data/processed").rglob("segments_with_era5_rhi_issr.parquet").__next__()
seg = pd.read_parquet(path)

print("ISSR share", seg["ISSR"].mean())
print("RHi stats", seg["RHi"].describe())
print("Missing T_K", seg["T_K"].isna().mean())
print("Missing q_kgkg", seg["q_kgkg"].isna().mean())
print("Score nonzero share", (seg["segment_score"] > 0).mean())
