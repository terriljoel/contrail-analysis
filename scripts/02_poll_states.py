import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
API_URL = "https://opensky-network.org/api/states/all"

# Germany bounding box
BBOX = {"lamin": 47.2, "lomin": 5.9, "lamax": 55.1, "lomax": 15.3}

STATE_COLS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude_m", "on_ground", "velocity_mps",
    "true_track_deg", "vertical_rate_mps", "sensors", "geo_altitude_m",
    "squawk", "spi", "position_source", "category",
]

def get_token(client_id: str, client_secret: str) -> str:
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in token response: {r.text[:300]}")
    return token

def fetch_snapshot(headers: dict) -> pd.DataFrame:
    r = requests.get(API_URL, params={**BBOX, "extended": 1}, headers=headers, timeout=30)
    r.raise_for_status()
    payload = r.json()

    snapshot_time = int(payload.get("time"))
    states = payload.get("states") or []
    df = pd.DataFrame(states, columns=STATE_COLS)
    df["snapshot_time_unix"] = snapshot_time
    df["callsign"] = df["callsign"].astype("string").str.strip()
    return df

def main(poll_seconds: int = 15, duration_minutes: int = 30):
    load_dotenv()
    cid = os.getenv("OPENSKY_CLIENT_ID")
    csec = os.getenv("OPENSKY_CLIENT_SECRET")
    if not cid or not csec:
        raise RuntimeError("Missing OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET in .env")

    token = get_token(cid, csec)
    headers = {"Authorization": f"Bearer {token}"}

    n_iters = int((duration_minutes * 60) / poll_seconds)
    print(f"Polling {n_iters} times every {poll_seconds}s (~{duration_minutes} min)")

    all_dfs = []
    start_utc = datetime.now(timezone.utc)

    for i in range(n_iters):
        try:
            df = fetch_snapshot(headers)
            all_dfs.append(df)
            print(f"[{i+1}/{n_iters}] rows={len(df)} snapshot_time={df['snapshot_time_unix'].iloc[0] if len(df) else 'NA'}")
        except requests.HTTPError as e:
            print(f"[{i+1}/{n_iters}] HTTPError: {e}")
        except Exception as e:
            print(f"[{i+1}/{n_iters}] Error: {e}")

        if i < n_iters - 1:
            time.sleep(poll_seconds)

    end_utc = datetime.now(timezone.utc)

    if not all_dfs:
        raise RuntimeError("No snapshots collected.")

    big = pd.concat(all_dfs, ignore_index=True)

    start_tag = start_utc.strftime("%Y%m%d_%H%M%SZ")
    end_tag = end_utc.strftime("%Y%m%d_%H%M%SZ")
    out_path = f"data/raw/opensky/states_germany_timeseries_{start_tag}_to_{end_tag}.parquet"
    big.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Total rows:", len(big))
    print("Unique aircraft (icao24):", big["icao24"].nunique())

if __name__ == "__main__":
    main(poll_seconds=15, duration_minutes=30)
