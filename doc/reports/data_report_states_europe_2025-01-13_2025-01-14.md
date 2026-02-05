# Data Report - states_europe (2025-01-13 to 2025-01-14)

Source folder: `data/processed/scores/states_europe/2025-01-13_2025-01-14`

## Overview
This report summarizes three Parquet datasets that describe flight-level scores and segment-level meteorological context for the Europe state window **2025-01-13 to 2025-01-14**.

Files:
- `flight_scores_rhi_issr.parquet`
- `flight_scores_top2pct.parquet`
- `segments_with_era5_rhi_issr.parquet`

## Dataset: `flight_scores_rhi_issr.parquet`
**Shape:** 19,986 rows x 6 columns

**Columns**
- `flight_id` (object): Flight identifier string.
- `flight_score` (float64): Aggregate flight-level score.
- `dist_in_issr_km` (float64): Distance flown in ISSR (km).
- `seg_count` (int64): Total number of segments for the flight.
- `issr_seg_count` (int64): Number of segments in ISSR.
- `mean_RHi` (float64): Mean relative humidity w.r.t. ice.

**Key stats (numeric)**
- `flight_score`: mean 25.77, median 0.00, max 1438.64
- `dist_in_issr_km`: mean 81.10, median 0.00, max 2179.31
- `seg_count`: mean 69.45, max 341
- `issr_seg_count`: mean 5.98, max 149
- `mean_RHi`: mean 47.32, max 122.996

**Missing values**
- `mean_RHi`: 70 missing
- All other columns: no missing values

## Dataset: `flight_scores_top2pct.parquet`
**Shape:** 400 rows x 6 columns

**Description**
Same schema as `flight_scores_rhi_issr.parquet`, containing the top 2% of flights by score.

**Key stats (numeric)**
- `flight_score`: mean 439.63, median 382.14, max 1438.64
- `dist_in_issr_km`: mean 716.48, median 625.88, max 2179.31
- `seg_count`: mean 122.82, max 285
- `issr_seg_count`: mean 51.88, max 149
- `mean_RHi`: mean 88.67, max 111.97

**Missing values**
- None

## Dataset: `segments_with_era5_rhi_issr.parquet`
**Shape:** 1,336,164 rows x 19 columns

**Columns**
- `flight_id` (object)
- `mid_time` (int64): Segment midpoint time as Unix seconds.
- `mid_lat`, `mid_lon` (float64): Segment midpoint latitude/longitude.
- `mid_alt_m` (float64): Segment midpoint altitude (m).
- `dt_s` (int64): Segment duration (seconds).
- `dist_km` (float64): Segment distance (km).
- `dt_utc` (datetime64[ns, UTC]): Segment time (UTC).
- `era5_time` (datetime64[ns, UTC]): ERA5 time stamp (UTC).
- `p_hpa_est` (float64): Estimated pressure (hPa).
- `plev_hpa` (int16): ERA5 pressure level (hPa).
- `lat_g`, `lon_g` (float32): ERA5 grid lat/lon.
- `T_K` (float32): Temperature (K).
- `q_kgkg` (float32): Specific humidity (kg/kg).
- `RHi` (float64): Relative humidity w.r.t. ice (%).
- `ISSR` (bool): Ice supersaturated region flag.
- `is_night` (bool): Night segment flag.
- `segment_score` (float64): Segment-level score.

**Key stats (numeric)**
- `mid_lat`: mean 46.62, range 35.00-71.37
- `mid_lon`: mean 11.44, range -24.65-33.84
- `mid_alt_m`: mean 10,660, max 12,496
- `RHi`: mean 44.10, median 34.14, max 133.99
- `segment_score`: mean 0.388, max 788.40

**Missing values**
- `T_K`, `q_kgkg`, `RHi`, `segment_score`: 8,380 missing each

## Notes / Interpretation
- Many flights have zero `flight_score` and zero `dist_in_issr_km`, suggesting a large share of flights without ISSR exposure in this window.
- The top 2% flights show substantially higher `dist_in_issr_km` and `mean_RHi`, indicating stronger ISSR-related conditions.
- Segment-level missing values likely correspond to gaps in ERA5 interpolation for some segments.

## Reproducibility
Stats were computed with the project's Python venv at:
`C:\Users\HiWi\Desktop\Terril\05_python_venv\Scripts\python.exe`

