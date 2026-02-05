from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import webbrowser

from trino.auth import OAuth2Authentication
from trino.dbapi import connect


@dataclass(frozen=True)
class TrinoConfig:
    host: str
    port: int
    http_scheme: str
    user: str
    catalog: str
    schema: str
    table: str
    max_rows_per_hour: int = 0


def _to_utc(ts: datetime) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def hour_partitions(start_utc: datetime, end_utc: datetime) -> List[Tuple[datetime, int]]:
    start = _to_utc(start_utc).floor("h")
    end = _to_utc(end_utc).floor("h")
    hours = pd.date_range(start, end, freq="h", inclusive="both", tz="UTC")
    return [(h.to_pydatetime(), int(h.timestamp())) for h in hours]


class UILinkRedirectHandler:
    def __init__(self, base_handler=None) -> None:
        self.url: Optional[str] = None
        self.base_handler = base_handler

    def __call__(self, url: str) -> None:
        self.url = url
        if self.base_handler is not None:
            self.base_handler(url)
            return
        try:
            webbrowser.open(url)
        except Exception:
            pass
        print(url, flush=True)


class AuthContext:
    def __init__(self, auth: OAuth2Authentication, handler: UILinkRedirectHandler) -> None:
        self.auth = auth
        self.handler = handler


_AUTH_CACHE: Dict[str, AuthContext] = {}


def _auth_cache_key(cfg: TrinoConfig) -> str:
    return f"{cfg.host}@{cfg.user}"


def get_auth_context(cfg: TrinoConfig) -> AuthContext:
    key = _auth_cache_key(cfg)
    if key in _AUTH_CACHE:
        return _AUTH_CACHE[key]

    base_handler = None
    try:
        from trino.auth import CompositeRedirectHandler, ConsoleRedirectHandler, WebBrowserRedirectHandler

        base_handler = CompositeRedirectHandler([WebBrowserRedirectHandler(), ConsoleRedirectHandler()])
    except Exception:
        base_handler = None

    handler = UILinkRedirectHandler(base_handler=base_handler)
    auth = OAuth2Authentication(redirect_auth_url_handler=handler)
    ctx = AuthContext(auth=auth, handler=handler)
    _AUTH_CACHE[key] = ctx
    return ctx


def connect_trino(cfg: TrinoConfig):
    ctx = get_auth_context(cfg)
    return connect(
        host=cfg.host,
        port=cfg.port,
        http_scheme=cfg.http_scheme,
        user=cfg.user,
        auth=ctx.auth,
        catalog=cfg.catalog,
        schema=cfg.schema,
    )


def check_trino_auth(cfg: TrinoConfig) -> Tuple[bool, Optional[str]]:
    ctx = get_auth_context(cfg)
    try:
        conn = connect_trino(cfg)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchall()
        return True, ctx.handler.url
    except Exception:
        return False, ctx.handler.url


def fetch_states_one_hour(
    conn,
    *,
    hour_unix: int,
    bbox: Dict[str, float],
    alt_min: Optional[float],
    alt_max: Optional[float],
    table: str,
    max_rows: int = 0,
) -> pd.DataFrame:
    where = f"""
      hour = {hour_unix}
      AND onground = false
      AND time - lastcontact <= 15
      AND lat BETWEEN {bbox['lat_min']} AND {bbox['lat_max']}
      AND lon BETWEEN {bbox['lon_min']} AND {bbox['lon_max']}
    """
    if alt_min is not None and alt_max is not None:
        where += f"\n  AND baroaltitude BETWEEN {alt_min} AND {alt_max}"

    limit = f"\nLIMIT {int(max_rows)}" if max_rows and max_rows > 0 else ""

    q = f"""
    SELECT
      hour,
      time, lastcontact,
      icao24, callsign,
      lat, lon,
      baroaltitude, geoaltitude,
      velocity, heading, vertrate
    FROM {table}
    WHERE {where}
    {limit}
    """
    return pd.read_sql(q, conn)


def download_hours_to_parquet(
    *,
    out_dir: Path,
    start_utc: datetime,
    end_utc: datetime,
    bbox: Dict[str, float],
    alt_min: Optional[float],
    alt_max: Optional[float],
    trino_cfg: TrinoConfig,
    prefix: str,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = connect_trino(trino_cfg)
    written: List[Path] = []
    hours = hour_partitions(start_utc, end_utc)
    for hour_dt, hour_unix in hours:
        out_path = out_dir / f"{prefix}_{hour_dt.strftime('%Y%m%d_%H%M%SZ')}.parquet"
        if out_path.exists():
            continue
        df = fetch_states_one_hour(
            conn,
            hour_unix=hour_unix,
            bbox=bbox,
            alt_min=alt_min,
            alt_max=alt_max,
            table=f"{trino_cfg.catalog}.{trino_cfg.schema}.{trino_cfg.table}",
            max_rows=trino_cfg.max_rows_per_hour,
        )
        df.to_parquet(out_path, index=False)
        written.append(out_path)
    return written
