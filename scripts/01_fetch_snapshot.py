import os
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

def main():
    load_dotenv()
    cid = os.getenv("OPENSKY_CLIENT_ID")
    csec = os.getenv("OPENSKY_CLIENT_SECRET")
    if not cid or not csec:
        raise RuntimeError("Missing OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET in .env")

    token = get_token(cid, csec)
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(API_URL, params={**BBOX, "extended": 1}, headers=headers, timeout=30)
    r.raise_for_status()
    payload = r.json()

    snapshot_time = int(payload.get("time"))
    states = payload.get("states") or []

    df = pd.DataFrame(states, columns=STATE_COLS)
    df["snapshot_time_unix"] = snapshot_time
    df["callsign"] = df["callsign"].astype("string").str.strip()

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_path = f"data/raw/opensky/states_germany_{ts}.parquet"
    df.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(df))
    print(df.head(8)[["icao24","callsign","latitude","longitude","baro_altitude_m","velocity_mps","vertical_rate_mps","on_ground"]])

if __name__ == "__main__":
    main()