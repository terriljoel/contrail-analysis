from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class DataFile:
    category: str
    path: Path


def list_files(root: Path, patterns: Iterable[str]) -> List[DataFile]:
    files: List[DataFile] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                files.append(DataFile(category=pattern, path=path))
    return sorted(files, key=lambda f: str(f.path))


def list_raw_opensky(data_root: Path) -> List[Path]:
    root = data_root / "raw" / "opensky"
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file()])


def list_processed_segments(data_root: Path, include_archive: bool = False) -> List[Path]:
    roots = [data_root / "processed"]
    if include_archive:
        roots.append(data_root / "archive")
    patterns = [
        "scores/**/*.parquet",
        "opensky_segments/**/*.parquet",
    ]
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            out.extend(root.glob(pattern))
    return sorted([p for p in out if p.is_file()])


def list_scored_segments(data_root: Path, include_archive: bool = False) -> List[Path]:
    roots = [data_root / "processed"]
    if include_archive:
        roots.append(data_root / "archive")
    patterns = [
        "scores/**/segments_with_era5_rhi_issr.parquet",
    ]
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            out.extend(root.glob(pattern))
    return sorted([p for p in out if p.is_file()])


def filter_scored_with_counterfactuals(paths: List[Path]) -> List[Path]:
    try:
        import pyarrow.parquet as pq
    except Exception:
        return paths

    keep: List[Path] = []
    for p in paths:
        try:
            schema = pq.read_schema(p)
            names = set(schema.names)
            if {"segment_score_up", "segment_score_down"}.issubset(names):
                keep.append(p)
        except Exception:
            continue
    return keep
