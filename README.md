 # Contrail MVP

## Motivation
Contrails can contribute significantly to aviation’s near-term climate impact. This project explores a practical, data-driven workflow to estimate contrail formation potential and test cruise-altitude adjustments that reduce contrail risk while keeping operational constraints in view. The goal is a transparent MVP that connects flight trajectories with weather physics and produces interpretable, per-flight recommendations.

## What This Project Does
- Downloads OpenSky flight trajectories (raw points) and ERA5 pressure-level weather.
- Processes trajectories into flight segments, merges weather, and computes RHi/ISSR.
- Scores contrail risk per segment and evaluates counterfactual altitudes (UP/DOWN).
- Runs a constrained optimizer or lightweight RL to recommend altitude changes.
- Provides a Streamlit UI to guide data download, processing, and analysis.

## Quickstart
1. Create a virtual environment and install dependencies.
2. Configure auth and settings (see Configuration and Auth below).
3. Run the Streamlit app:

```bash
python -m venv .venv
# On Windows:
.venv\\Scripts\\activate
# On macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui/app.py
```

## Configuration
Project settings live in `config/config.yaml`. Key items to review:
- `paths`: data locations (raw, processed, outputs).
- `opensky.bbox`: regional bounding boxes.
- `opensky.trino`: Trino host/catalog/schema/table and user.
- `era5.pressure_levels_hpa`: pressure levels used in ERA5 downloads.
- `policy` and `rl`: optimization constraints and RL hyperparameters.

Environment variables are loaded from `.env` (not committed). Create a local `.env` file with:

```
OPENSKY_CLIENT_ID=your_client_id
OPENSKY_CLIENT_SECRET=your_client_secret
OPENSKY_TRINO_USER=your_trino_username
```

## Auth Prerequisites (Do This Before Fetching Data)
### OpenSky (OAuth2 client credentials)
1. Create an OpenSky account.
2. Create an OAuth2 API client in the OpenSky portal.
3. Set `OPENSKY_CLIENT_ID` and `OPENSKY_CLIENT_SECRET` in `.env`.

### OpenSky Trino (historical database access)
Creating an OpenSky account is not sufficient for Trino access. You must request permission for the historical database:
1. Request access at `https://opensky-network.org/data/trino`.
2. Once approved, set `opensky.trino.user` in `config/config.yaml` or `OPENSKY_TRINO_USER` in `.env`.
3. In the UI, click “Check / Authenticate Trino”.
4. Open the login link, sign in, then click “Check / Authenticate” again.

### ERA5 (Copernicus CDS API)
1. Create a Copernicus CDS account.
2. Create `~/.cdsapirc` with your CDS API key (CDS site provides the snippet).
3. Validate by running any ERA5 download in the UI.

## Data Fetching and Processing Flow
1. OpenSky download (UI): Trino (OAuth).
2. ERA5 download (UI): Pressure-level temperature and specific humidity.
3. Process OpenSky to segments.
4. Merge ERA5 and score segments (RHi/ISSR).
5. Run optimizer or RL and inspect results.

## Scripts (Optional CLI)
Smoke test OpenSky auth:

```bash
python scripts/00_smoketest_opensky.py
```

Single snapshot:

```bash
python scripts/01_fetch_snapshot.py
```

Polling time series:

```bash
python scripts/02_poll_states.py
```

## Project Layout
```
config/                 # YAML config
data/                   # Raw, processed, and outputs (ignored by git)
doc/                    # Documentation and reports
notebooks/              # Exploratory notebooks
scripts/                # CLI helpers for OpenSky
src/contrail_mvp/        # Core library code
ui/                     # Streamlit app
```

## Notes
- Full pipeline details are in `doc/APP_DOC.md`.
- Data files are large; `data/` is gitignored.
- `.env` contains secrets and must not be committed.
- This is a research MVP: scores are relative, not direct climate units.

## App Documentation (Full)

Contrail Optimization Demo — Project Documentation

This document explains the end-to-end pipeline in the app: data ingestion, preprocessing, segmentation, ERA5 merge, RHi/ISSR physics, scoring, and policy optimization.

Overview

The app evaluates contrail-aware cruise-altitude adjustments using:
- OpenSky flight trajectories (raw point data).
- ERA5 pressure-level weather (temperature + specific humidity).

Only cruise altitude adjustments are considered (UP/DOWN/HOLD), with operational constraints on how often altitude can change.

Data Layout

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

OpenSky Download (Trino)

OpenSky data are queried from the Trino server using OAuth2. You select:
- Region (bbox).
- Date and hour range.
- Optional altitude band filter.

Downloaded files are saved as hourly parquet files under:
```
data/raw/opensky/{region}/{YYYY-MM-DD}/
```

ERA5 Download (CDS API)

ERA5 pressure-level data are downloaded via `cdsapi`:
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

Preprocessing (OpenSky Points -> Segments)

The preprocessing follows the logic in `notebooks/03_data_preprocessing.ipynb`.

1) Downsample to 1 point per minute

Per `(icao24, callsign)`:
- `minute = time // 60`
- Sort by `(icao24, callsign, minute, time)`
- Keep the first record per minute bin

This reduces noise and makes segmentation stable.

2) Flight ID

Flights are split by time gaps:
- A new flight starts when time gap > 30 minutes (`gap_s = 30 * 60`).

`flight_id` is:
```
icao24 + "_" + callsign + "_" + flight_idx
```

3) Segment creation

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

4) Cruise trimming

Optional trimming removes climb/descend:
- Drop segments in the first N minutes and last N minutes of each flight.

Defaults:
- Trim start = 20 minutes
- Trim end = 20 minutes

ERA5 Merge + RHi / ISSR Physics

1) Time alignment

Segments are aligned to ERA5 hourly bins:
```
time_hour = floor(mid_time to nearest hour)
```

2) Pressure level assignment

Estimate pressure from altitude:
```
p_pa = p0 * (1 - L*h/T0)^(g/(R*L))
```
Then assign nearest ERA5 pressure level from:
```
{200, 225, 250, 300, 350} hPa
```

Also compute counterfactual levels:
- `plev_up` (next lower pressure -> higher altitude)
- `plev_down` (next higher pressure -> lower altitude)

3) Grid snapping

Segment midpoints are snapped to the nearest ERA5 grid (lat/lon) to ensure merge hits.

4) Relative humidity over ice (RHi)

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

Contrail Scoring

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

Policy / Optimization

The app can run two modes:

1) Optimizer (constrained)
   - Deterministic baseline using dynamic programming.
   - Constraints:
     - One change total, or
     - No change within 1 hour, or
     - No change within 30 minutes
   - Optional fuel penalty per change.

2) RL (tabular)
   - Lightweight Q-learning with discretized state bins.
   - Uses the same constraints as above.

UI Outputs

Per-flight:
- Actual vs recommended altitude profile.
- Action timeline (UP/DOWN/HOLD).
- Score interpretation (percentile + improvement).

Overall:
- Original vs optimized scatter.
- Improvement histogram.
- Per-flight improvement bar chart.
- Action share distribution.
- Baseline score distribution.

Notes / Limitations

- The score is relative (not a direct climate unit).
- Improvements are strongest in high-impact flights.
- Results depend on ERA5 resolution (hourly, pressure levels).

Configuration

All defaults are stored in:
```
config/config.yaml
```

Key parameters:
- pressure levels
- bbox
- policy constraints
- fuel penalty
