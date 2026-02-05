# Contrail Optimization Demo — Project Documentation

This document explains the end‑to‑end pipeline in the app: data ingestion, preprocessing, segmentation, ERA5 merge, RHi/ISSR physics, scoring, and policy optimization.

## Overview

The app evaluates contrail‑aware cruise‑altitude adjustments using:
- OpenSky flight trajectories (raw point data).
- ERA5 pressure‑level weather (temperature + specific humidity).

Only cruise altitude adjustments are considered (UP/DOWN/HOLD), with operational constraints on how often altitude can change.

## Data Layout

```
data/
  raw/
    opensky/{region}/{YYYY-MM-DD}/
    weather_ERA5/{area}/{YYYY-MM-DD}/
  processed/
    opensky_segments/{area}/{YYYY-MM-DD}/
    weather_ERA5/{area}/{YYYY-MM-DD}/
    scores/{area}/{YYYY-MM-DD}/
```

Where:
- `region` is `germany` or `europe`.
- `area` is `states_{region}` (e.g., `states_germany`).

## OpenSky Download (Trino)

OpenSky data are queried from the Trino server using OAuth2. You select:
- Region (bbox).
- Date and hour range.
- Optional altitude band filter.

Downloaded files are saved as hourly parquet files under:
```
data/raw/opensky/{region}/{YYYY-MM-DD}/
```

## ERA5 Download (CDS API)

ERA5 pressure‑level data are downloaded via `cdsapi`:
- Variables: temperature, specific humidity.
- Pressure levels: typically [200, 225, 250, 300, 350] hPa.
- Hour range and bbox from the UI.

Saved to:
```
data/raw/weather_ERA5/{area}/{YYYY-MM-DD}/era5_pl_{area}_T_q.nc
```

Then converted to parquet:
```
data/processed/weather_ERA5/{area}/{YYYY-MM-DD}/era5_pl_{area}_T_q.parquet
```

## Preprocessing (OpenSky Points → Segments)

The preprocessing follows the logic in `notebooks/03_data_preprocessing.ipynb`.

### 1) Downsample to 1 point per minute

Per `(icao24, callsign)`:
- `minute = time // 60`
- Sort by `(icao24, callsign, minute, time)`
- Keep the first record per minute bin

This reduces noise and makes segmentation stable.

### 2) Flight ID

Flights are split by time gaps:
- A new flight starts when time gap > 30 minutes (`gap_s = 30 * 60`).

`flight_id` is:
```
icao24 + "_" + callsign + "_" + flight_idx
```

### 3) Segment creation

Within each flight:
- Shift next point to form segments.
- Compute segment duration `dt_s`.
- Keep only reasonable gaps: `0 < dt_s <= 5 minutes`.
- Compute haversine distance.
- Compute midpoints (time, lat, lon, altitude).

Output:
```
data/processed/opensky_segments/{area}/{YYYY-MM-DD}/opensky_segments_{area}_{date}.parquet
```

### 4) Cruise trimming

Optional trimming removes climb/descend:
- Drop segments in the first N minutes and last N minutes of each flight.

Defaults:
- Trim start = 20 minutes
- Trim end = 20 minutes

## ERA5 Merge + RHi / ISSR Physics

### 1) Time alignment

Segments are aligned to ERA5 hourly bins:
```
time_hour = floor(mid_time to nearest hour)
```

### 2) Pressure level assignment

Estimate pressure from altitude:
```
p_pa = p0 * (1 - L*h/T0)^(g/(R*L))
```
Then assign nearest ERA5 pressure level from:
```
{200, 225, 250, 300, 350} hPa
```

Also compute counterfactual levels:
- `plev_up` (next lower pressure → higher altitude)
- `plev_down` (next higher pressure → lower altitude)

### 3) Grid snapping

Segment midpoints are snapped to the **nearest ERA5 grid** (lat/lon) to ensure merge hits.

### 4) Relative humidity over ice (RHi)

Using Murphy & Koop (2005):
```
e_si(T) = exp(9.550426 - (5723.265/T) + 3.53068*ln(T) - 0.00728332*T)
```
Vapor pressure from specific humidity:
```
w = q / (1 - q)
e = (w * p) / (0.622 + w)
RHi = 100 * e / e_si(T)
```

ISSR condition:
```
ISSR = (RHi > 100)
```

## Contrail Scoring

Segment score:
```
excess = max(0, (RHi - 100) / 20)
night_w = 1.5 if night else 1.0
segment_score = dist_km * excess * night_w * ISSR
```

Night is a simple proxy:
```
local_hour = (UTC hour + lon/15) % 24
night = local_hour >= 18 or < 6
```

Counterfactual scores are computed at `plev_up` and `plev_down` using the same formula:
- `segment_score_up`
- `segment_score_down`

Scored segments output:
```
data/processed/scores/{area}/{YYYY-MM-DD}/segments_with_era5_rhi_issr.parquet
```

## Policy / Optimization

The app can run two modes:

1) **Optimizer (constrained)**
   - Deterministic baseline using dynamic programming.
   - Constraints:
     - One change total, or
     - No change within 1 hour, or
     - No change within 30 minutes
   - Optional fuel penalty per change.

2) **RL (tabular)**
   - Lightweight Q‑learning with discretized state bins.
   - Uses the same constraints as above.

## UI Outputs

Per‑flight:
- Actual vs recommended altitude profile.
- Action timeline (UP/DOWN/HOLD).
- Score interpretation (percentile + improvement).

Overall:
- Original vs optimized scatter.
- Improvement histogram.
- Per‑flight improvement bar chart.
- Action share distribution.
- Baseline score distribution.

## Notes / Limitations

- The score is **relative** (not a direct climate unit).
- Improvements are strongest in high‑impact flights.
- Results depend on ERA5 resolution (hourly, pressure levels).

## Configuration

All defaults are stored in:
```
config/config.yaml
```

Key parameters:
- pressure levels
- bbox
- policy constraints
- fuel penalty

