#!/usr/bin/env python3
"""
OpenSky ingestion:
  A) realtime: poll /api/states/all and append to CSV/CSV.GZ

Requires:
  pip install requests pandas python-dotenv

OAuth2 client credentials flow is documented by OpenSky. :contentReference[oaicite:2]{index=2}
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict

import pandas as pd
import requests
from dotenv import load_dotenv


# -----------------------------
# Config / constants
# -----------------------------

OPENSKY_API_ROOT = "https://opensky-network.org/api"
TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)

# Bounding boxes
GERMANY_BBOX = (47.2, 5.9, 55.1, 15.3)     # (lamin, lomin, lamax, lomax)
EUROPE_BBOX = (34.0, -25.0, 72.0, 45.0)    # rough Europe coverage


STATE_COLS_API = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude_m", "on_ground", "velocity_mps",
    "true_track_deg", "vertical_rate_mps", "sensors", "geo_altitude_m",
    "squawk", "spi", "position_source", "category",
]

# -----------------------------
# Helpers
# -----------------------------

@dataclass(frozen=True)
class BBox:
    lamin: float
    lomin: float
    lamax: float
    lomax: float


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def request_with_retries(
    method: str,
    url: str,
    *,
    session: Optional[requests.Session] = None,
    max_tries: int = 5,
    backoff_s: float = 1.5,
    timeout: int = 60,
    **kwargs
) -> requests.Response:
    s = session or requests.Session()
    last_err = None
    for i in range(max_tries):
        try:
            r = s.request(method, url, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            sleep_s = backoff_s * (2 ** i)
            print(f"[retry {i+1}/{max_tries}] {url} -> {e} (sleep {sleep_s:.1f}s)", file=sys.stderr)
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed after {max_tries} tries: {url}\nLast error: {last_err}")


# -----------------------------
# OAuth2 + realtime snapshots
# -----------------------------

def get_oauth_token(client_id: str, client_secret: str, session: Optional[requests.Session] = None) -> str:
    r = request_with_retries(
        "POST",
        TOKEN_URL,
        session=session,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    tok = r.json().get("access_token")
    if not tok:
        raise RuntimeError(f"No access_token in token response: {r.text[:300]}")
    return tok


def fetch_states_snapshot(
    *,
    headers: Dict[str, str],
    bbox: Optional[BBox] = None,
    extended: bool = True,
    t_unix: Optional[int] = None,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    params: Dict[str, object] = {}
    if bbox is not None:
        params.update({"lamin": bbox.lamin, "lomin": bbox.lomin, "lamax": bbox.lamax, "lomax": bbox.lomax})
    if extended:
        params["extended"] = 1
    if t_unix is not None:
        params["time"] = int(t_unix)

    url = f"{OPENSKY_API_ROOT}/states/all"  # documented endpoint :contentReference[oaicite:5]{index=5}
    r = request_with_retries("GET", url, session=session, headers=headers, params=params, timeout=30)
    payload = r.json()
    snapshot_time = int(payload.get("time", int(time.time())))
    states = payload.get("states") or []
    df = pd.DataFrame(states, columns=STATE_COLS_API)
    df["snapshot_time_unix"] = snapshot_time
    if "callsign" in df.columns:
        df["callsign"] = df["callsign"].astype("string").str.strip()
    return df


def realtime_poll_to_csv(
    out_csv: str,
    *,
    bbox: Optional[BBox],
    poll_seconds: int,
    duration_minutes: int,
    client_id: str,
    client_secret: str,
    gzip_out: bool = False,
) -> None:
    ensure_parent_dir(out_csv)
    session = requests.Session()

    token = get_oauth_token(client_id, client_secret, session=session)
    headers = {"Authorization": f"Bearer {token}"}

    n_iters = int((duration_minutes * 60) / poll_seconds)
    print(f"Polling {n_iters} snapshots every {poll_seconds}s into {out_csv}")

    # Stream append to avoid RAM blow-ups
    write_header = True
    opener = gzip.open if gzip_out else open

    with opener(out_csv, "wt", newline="") as f:
        for i in range(n_iters):
            # token expiry is short-lived; if you hit 401, refresh once
            try:
                df = fetch_states_snapshot(headers=headers, bbox=bbox, session=session)
            except requests.HTTPError as e:
                if "401" in str(e):
                    token = get_oauth_token(client_id, client_secret, session=session)
                    headers = {"Authorization": f"Bearer {token}"}
                    df = fetch_states_snapshot(headers=headers, bbox=bbox, session=session)
                else:
                    raise

            if df.empty:
                print(f"[{i+1}/{n_iters}] empty snapshot", file=sys.stderr)
            else:
                df.to_csv(f, index=False, header=write_header)
                write_header = False
                print(f"[{i+1}/{n_iters}] rows={len(df)} snapshot_time={df['snapshot_time_unix'].iloc[0]}")

            if i < n_iters - 1:
                time.sleep(poll_seconds)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenSky realtime downloader")

    sub = p.add_subparsers(dest="cmd", required=True)

    # realtime
    p_rt = sub.add_parser("realtime", help="Poll /api/states/all (OAuth2) and append to CSV")
    p_rt.add_argument("--out", required=True, help="Output CSV path (use .gz to compress if --gzip)")
    p_rt.add_argument("--poll", type=int, default=15, help="Polling interval seconds (default: 15)")
    p_rt.add_argument("--minutes", type=int, default=30, help="Duration minutes (default: 30)")
    p_rt.add_argument("--bbox", choices=["germany", "europe", "none"], default="germany")
    p_rt.add_argument("--gzip", action="store_true", help="Write gzipped CSV")
    p_rt.add_argument("--env", default=".env", help="Path to .env (default: .env)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "realtime":
        load_dotenv(args.env)

        cid = os.getenv("OPENSKY_CLIENT_ID")
        csec = os.getenv("OPENSKY_CLIENT_SECRET")
        if not cid or not csec:
            raise RuntimeError("Missing OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET in your .env")

        bbox = None
        if args.bbox == "germany":
            bbox = BBox(*GERMANY_BBOX)
        elif args.bbox == "europe":
            bbox = BBox(*EUROPE_BBOX)

        realtime_poll_to_csv(
            args.out,
            bbox=bbox,
            poll_seconds=args.poll,
            duration_minutes=args.minutes,
            client_id=cid,
            client_secret=csec,
            gzip_out=args.gzip,
        )

if __name__ == "__main__":
    main()
