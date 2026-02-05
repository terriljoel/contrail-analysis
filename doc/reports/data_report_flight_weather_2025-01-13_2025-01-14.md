# Data Report - processed flight and weather data (2025-01-13 to 2025-01-14)

Source folders:
- `data/processed/flight_data`
- `data/processed/weather_ERA5`

This report covers flight and weather data for the Europe state window 2025-01-13 to 2025-01-14.

## Flight data

### File: `flight_data/states_europe/2025-01-13_2025-01-14/opensky_segments_europe_winter_day.parquet`
**Shape:** 1,336,164 rows x 7 columns

**Columns**
- `flight_id` (object)
- `mid_time` (int64): Segment midpoint time as Unix seconds
- `mid_lat`, `mid_lon` (float64): Segment midpoint latitude/longitude
- `mid_alt_m` (float64): Segment midpoint altitude (m)
- `dt_s` (int64): Segment duration (seconds)
- `dist_km` (float64): Segment distance (km)

**Key stats (numeric)**
- `mid_lat`: mean 46.62, range 35.00 to 71.37
- `mid_lon`: mean 11.44, range -24.65 to 33.84
- `mid_alt_m`: mean 10,660, max 12,496
- `dt_s`: mean 60.00, max 300
- `dist_km`: mean 13.56, max 1,619.64

**Missing values**
- None

## Weather data (ERA5)

### File: `weather_ERA5/states_europe/2025-01-13_2025-01-14/era5_pl_europe_20250113_T_q.parquet`
**Shape:** 3,593,880 rows x 6 columns

**Columns**
- `time` (datetime64[ns])
- `plev_hpa` (int16): Pressure level (hPa)
- `lat`, `lon` (float32): Grid latitude/longitude
- `T_K` (float32): Temperature (K)
- `q_kgkg` (float32): Specific humidity (kg/kg)

**Key stats (numeric)**
- `time`: 2025-01-13 00:00:00 to 2025-01-13 23:00:00
- `plev_hpa`: 200 to 350
- `lat`: 35.0 to 72.0
- `lon`: -25.0 to 45.0
- `T_K`: mean 220.88, min 202.95, max 242.32
- `q_kgkg`: mean 5.82e-05, min 1.19e-08, max 5.69e-04

**Missing values**
- None

## Reproducibility
Stats were computed with the project Python venv at:
`C:\Users\HiWi\Desktop\Terril\05_python_venv\Scripts\python.exe`
