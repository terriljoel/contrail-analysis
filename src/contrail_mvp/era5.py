from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
import xarray as xr
import cdsapi


@dataclass(frozen=True)
class Era5Request:
    day: date
    bbox: List[float]  # [north, west, south, east]
    pressure_levels_hpa: List[int]
    variables: List[str]
    hours: Optional[List[int]] = None


def _hours_list(hours: Optional[List[int]]) -> List[str]:
    if hours is None:
        hours = list(range(0, 24))
    return [f"{h:02d}:00" for h in hours]


def download_era5_pressure_levels(req: Era5Request, out_nc: Path) -> Path:
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "data_format": "netcdf",
            "download_format": "unarchived",
            "variable": req.variables,
            "pressure_level": [str(p) for p in req.pressure_levels_hpa],
            "year": f"{req.day:%Y}",
            "month": f"{req.day:%m}",
            "day": f"{req.day:%d}",
            "time": _hours_list(req.hours),
            "area": [
                req.bbox[0],
                req.bbox[1],
                req.bbox[2],
                req.bbox[3],
            ],
        },
        str(out_nc),
    )
    return out_nc


def nc_to_parquet(nc_path: Path, out_parquet: Path) -> Path:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(nc_path)
    rename = {}
    if "temperature" in ds:
        rename["temperature"] = "T_K"
    if "specific_humidity" in ds:
        rename["specific_humidity"] = "q_kgkg"
    if "t" in ds:
        rename["t"] = "T_K"
    if "q" in ds:
        rename["q"] = "q_kgkg"
    ds = ds.rename(rename)
    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    if "level" in ds.coords:
        ds = ds.rename({"level": "plev_hpa"})

    if "T_K" not in ds or "q_kgkg" not in ds:
        missing = [k for k in ["T_K", "q_kgkg"] if k not in ds]
        raise KeyError(f"Missing variables in ERA5 dataset: {missing}")

    df = ds[["T_K", "q_kgkg"]].to_dataframe().reset_index()
    df.to_parquet(out_parquet, index=False)
    return out_parquet
