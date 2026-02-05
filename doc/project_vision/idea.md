# Contrail-Aware Cruise Altitude Optimization Using Reinforcement Learning

## 1. Objective

The objective of this work is to reduce the climate impact of aviation contrails by allowing **small, operationally realistic cruise altitude adjustments**, while ensuring that **fuel consumption does not increase**. The problem is addressed using physically based contrail metrics combined with a constrained reinforcement learning (RL) formulation.

---

## 2. Data Sources

### 2.1 Flight Trajectory Data
- One day of flight trajectory data over Europe.
- Each flight is segmented into cruise segments.
- Each segment contains:
  - position (`mid_lat`, `mid_lon`)
  - time (`mid_time`, `dt_utc`)
  - altitude (`mid_alt_m`)
  - distance (`dist_km`)
  - flight identifier (`flight_id`)

### 2.2 Meteorological Data (ERA5)
- ERA5 pressure-level reanalysis data.
- Hourly temporal resolution.
- Discrete pressure levels used:
  - **200, 225, 250, 300, 350 hPa**
- Variables used:
  - Temperature (`T_K`)
  - Specific humidity (`q_kgkg`)
- Spatially matched using nearest ERA5 grid point.

---

## 3. Matching Aircraft Altitude to ERA5 Pressure Levels

ERA5 pressure-level data is indexed by **pressure (hPa)**, not geometric altitude.

### 3.1 Altitude-Pressure Conversion
- Aircraft altitude in meters (`mid_alt_m`) is converted to an estimated pressure (`p_hpa_est`).
- Each segment is **snapped to the nearest available ERA5 pressure level**:

\[
plev\_hpa = \arg\min_{L \in \{200,225,250,300,350\}} |p\_hpa\_est - L|
\]

This mapping covers cruise altitudes approximately between **8 km and 13 km**:
- 350 hPa ~= 8-9 km  
- 300 hPa ~= 9-10 km  
- 250 hPa ~= 10-11 km  
- 225 hPa ~= 11-12 km  
- 200 hPa ~= 12-13 km  

A sanity check confirmed a **100% nearest-level match rate**.

---

## 4. Time and Spatial Alignment

### 4.1 Temporal Matching
- ERA5 is hourly.
- Segment timestamps are mapped using:
\[
time\_hour = \lfloor era5\_time \rfloor_{hour}
\]
- One missing hour not covered by ERA5 was removed to ensure complete coverage.

### 4.2 Spatial Matching
- Nearest-neighbor matching to ERA5 latitude-longitude grid.
- Join success rate > 99.9%.

---

## 5. Contrail Physics

### 5.1 Ice Relative Humidity (RHi)
Ice relative humidity is computed using:
- Vapor pressure derived from specific humidity.
- Saturation vapor pressure over ice (Murphy & Koop, 2005).

\[
RHi = 100 \cdot \frac{e}{e_{si}(T)}
\]

### 5.2 Ice-Supersaturated Regions (ISSR)
\[
ISSR = (RHi > 100\%)
\]

---

## 6. Segment-Level Contrail Impact Score

### 6.1 Night-Time Weighting
Local solar time is approximated as:
\[
local\_hour = (UTC\_hour + longitude/15) \bmod 24
\]

Night condition:
- `local_hour >= 18` or `local_hour < 6`
- Night weighting factor: **1.5**
- Day weighting factor: **1.0**

### 6.2 Segment Score Definition
\[
segment\_score =
dist_{km}
\times \max\left(0, \frac{RHi - 100}{20}\right)
\times w_{night}
\times \mathbb{1}_{ISSR}
\]

Flight-level score is the sum of segment scores.

---

## 7. Altitude Action Space

Altitude changes are restricted to **existing ERA5 pressure levels**:

- **HOLD**: current pressure level (`plev_hpa`)
- **UP**: next higher altitude (lower pressure)
- **DOWN**: next lower altitude (higher pressure)

For each segment:
- ERA5 temperature and humidity are retrieved at:
  - `plev_up`
  - `plev_down`
- Counterfactual contrail metrics are recomputed:
  - `RHi_up`, `ISSR_up`, `segment_score_up`
  - `RHi_down`, `ISSR_down`, `segment_score_down`

---

## 8. Operational Constraints

This study discusses multiple operational constraints to balance climate benefit, ATC feasibility, and fuel burn.

### 8.1 Constraint A: At Most One Cruise Altitude Change per Flight
- **Rule:** Each flight may execute **0 or 1** altitude change during cruise.
- **Motivation:** ATC-friendly, interpretable, avoids oscillatory behavior, and provides a strong baseline.

### 8.2 Constraint B: Hourly Decision Grid with Limited Changes
To avoid unrealistic "instant" changes and to align with hourly ERA5 meteorology, an alternative is to make decisions on a coarser temporal grid.
- **Rule:** Altitude changes are only allowed at **hour boundaries** (or fixed decision windows), and the chosen altitude is held for the next window.
- **Additional limit:** No more than **N changes per flight** (e.g., N = 1-2), and/or minimum dwell time between changes (e.g., >= 60 minutes).
- **Motivation:** Practical control frequency, avoids chasing meteorological noise, and matches data resolution.

### 8.3 Fuel Consumption Constraint (No Fuel Increase)
Fuel is incorporated as a constraint in two complementary ways:

#### (i) Soft Constraint (Penalty in Reward)
A fuel penalty is added to discourage changes unless contrail benefit is large:
\[
r_t = -segment\_score_{action} - \lambda \Delta fuel
\]
where \(\Delta fuel\) can be approximated with a proxy (e.g., one-time penalty per change, optionally scaled by altitude step).

#### (ii) Hard Constraint (Budget / Feasibility)
A strict fuel limit is enforced by allowing an action only if:
\[
\sum \Delta fuel \le \text{FuelBudget}
\]
This can represent "fuel must not increase" (budget = 0) or a small allowable increase (e.g., <= 0.1-0.2%) to reflect operational uncertainty.

### 8.4 Baseline and Improvement Methods (summary)
- **Baseline:** existing flight path (no altitude changes).
- **Improvement Method 1:** allow at most one altitude change per flight, subject to a fuel constraint.
- **Improvement Method 2:** allow changes only at hour boundaries (or once per hour), subject to a fuel constraint.
- **Suggested Improvement Method 3:** restrict changes to cruise-only segments and cap total altitude steps (e.g., at most one level change overall), to improve ATC feasibility and passenger comfort.

---

## 9. Baseline Optimization (Exact Solution)

For each flight, the optimal strategy is computed under the **one-change constraint**:

1. No altitude change (all HOLD)
2. Switch once to UP at step *k*
3. Switch once to DOWN at step *k*

The minimum-cost option is selected.

### Observations
- ~40% of flights benefit from one altitude change.
- UP and DOWN are selected in similar proportions.
- Contrail impact is **highly skewed**:
  - majority of flights have zero impact
  - a small subset dominates total contrail impact.

---

## 10. Reinforcement Learning Formulation

### 10.1 Episode Definition
- One flight = one episode.
- Steps = ordered cruise segments (or optionally aggregated to hourly blocks when using the hourly decision constraint).

### 10.2 State
Includes:
- `RHi`, `ISSR`
- `RHi_up/down`, `ISSR_up/down`
- night flag
- normalized flight progress

### 10.3 Actions
Two equivalent action designs are considered:

#### (A) One-change RL (switch-now formulation)
- `HOLD`
- `SWITCH_UP_NOW`
- `SWITCH_DOWN_NOW`  
After a switch occurs, only HOLD is permitted for the remainder of the flight.

#### (B) Hourly RL (multi-change with limit)
- `HOLD`, `UP`, `DOWN` chosen per decision window
- Changes are limited by maximum number of changes per flight and/or minimum dwell time.

### 10.4 Reward
Reward balances contrail reduction and fuel impact:
\[
r_t = -segment\_score_{action} - \lambda \Delta fuel
\]
Optionally, actions that violate the hard fuel budget are disallowed (or penalized heavily).

Rationale for RL vs. single-change optimization:
"The one-change problem is solvable exactly, but to capture operationally realistic multi-step decisions under uncertainty and fuel budgets, we need a policy that adapts to state over time. RL provides a compact decision rule for sequential altitude changes that a static optimizer would need to re-solve at each step."

Note on terminology: the shared per-segment table used in Methods 1 and 2 is referred to as the **cost table** (not an RL-specific table).

---

## 11. Evaluation

Policies are evaluated by comparing:
- no-change baseline
- optimal constrained baseline (e.g., one-change optimum)
- RL policy under the same constraints (including fuel)

Evaluation focuses on **high-impact flights** (top percentiles), since most flights have zero contrail score.

---

## 12. Key Contribution

This work demonstrates that:
- contrail impact is concentrated in a small subset of flights,
- constrained altitude adjustments (single-change or limited hourly changes) can reduce contrail impact substantially,
- physically informed RL can learn when and how to apply such changes while respecting **operational constraints** and **fuel limitations**.

The framework is extensible to:
- finer temporal interpolation of meteorology,
- more detailed fuel models,
- and additional ATC/traffic feasibility constraints.

---

## 13. Real-Time Decision Service (Software Concept)

### 13.1 Goal
Provide a real-time decision service that ingests live flight trajectory data and weather streams, and returns altitude change recommendations (HOLD/UP/DOWN) with expected contrail impact reduction and fuel compliance.

### 13.2 Inputs
- Live or replayed flight segments (position, time, altitude, flight_id)
- ERA5 or equivalent weather data stream (T, q at pressure levels)
- Operational constraints (max changes, hourly boundaries, fuel budget)

### 13.3 Core Components
- **Feature pipeline**: transforms incoming segments into state features (RHi, ISSR, up/down variants).
- **Policy engine**:
  - Exact optimization for Method 1 (one change + fuel).
  - Exact optimization for Method 2 (hourly grid + max changes + fuel).
  - Optional RL policy for richer constraints or uncertainty.
- **Decision API**: REST endpoint that returns action + rationale + expected impact.

### 13.4 Outputs
- Action recommendation: HOLD / UP / DOWN
- Expected contrail impact (delta vs baseline)
- Fuel constraint status (OK / violated)

### 13.5 Demo Setup (Real-Time Feel)
- Replay historical data as a time-ordered stream.
- Call the Decision API per segment (configurable speed).
- Show decisions and savings in a dashboard or log view.

### 13.6 Targeting High-Impact Flights in Real Time
- For each active flight, estimate expected contrail impact over the next decision window (baseline HOLD cost).
- Rank flights by expected impact and prioritize the top 1-5% for action recommendations.
- Recompute rankings at each hour boundary (or decision window) to adapt to changing weather.

### 13.7 Weather Data Sources (Feasibility)
- Short-term demo uses ground-based weather products (forecast/nowcast or ERA5 replay) combined with live trajectories.
- Onboard sensors can provide local T/p (and possibly humidity), but adding new sensors and certified onboard changes are slow.
- Operationally, ground-based integration is the most feasible near-term path; onboard sensing can be a future enhancement.

### 13.8 How RL Fits (Near-Real-Time Policy, Not a Replacement for Baselines)
To keep the approach credible, exact optimization remains the reference baseline, and RL is used as a scalable near-real-time decision policy:
- **Offline (training):** use historical flights and the physics-based reward to learn a policy `pi(state) -> {HOLD, UP, DOWN}` under operational constraints (fuel budget, max changes, hourly boundaries).
- **Online (inference):** at each allowed decision point (e.g., hour boundary), compute the current state and query the policy to recommend an action.
- **Safety/feasibility layer:** mask invalid actions (fuel budget exceeded, change not allowed in this window, max changes already used). If all non-HOLD actions are invalid, fall back to `HOLD`.
- **Why RL here:** instead of repeatedly re-solving an optimization for many flights each decision window, RL provides a compact policy that can scale to streaming deployment and can be trained for robustness under weather uncertainty (forecast error).

### 13.9 Confidence / Abstain Rule (Operational Caution)
For practical deployment, the service can abstain from recommending a change when expected benefit is small or uncertain:
- If predicted savings are below a threshold (or confidence is low), recommend `HOLD` and log the flight for monitoring.
