from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import os


DEFAULT_CONFIG_PATH = Path("config") / "config.yaml"


@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]

    @property
    def paths(self) -> Dict[str, Any]:
        return self.raw.get("paths", {})

    @property
    def opensky(self) -> Dict[str, Any]:
        return self.raw.get("opensky", {})

    @property
    def era5(self) -> Dict[str, Any]:
        return self.raw.get("era5", {})

    @property
    def scoring(self) -> Dict[str, Any]:
        return self.raw.get("scoring", {})

    @property
    def policy(self) -> Dict[str, Any]:
        return self.raw.get("policy", {})

    @property
    def rl(self) -> Dict[str, Any]:
        return self.raw.get("rl", {})

    def path(self, key: str) -> Path:
        paths = self.paths
        if key not in paths:
            raise KeyError(f"Path key not found: {key}")
        return Path(paths[key])


def load_config(path: Optional[Path] = None) -> Config:
    cfg_path = path or DEFAULT_CONFIG_PATH
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    trino_user = os.getenv("OPENSKY_TRINO_USER")
    if trino_user:
        data.setdefault("opensky", {}).setdefault("trino", {})["user"] = trino_user
    return Config(raw=data)


def get_default_config() -> Config:
    return load_config(DEFAULT_CONFIG_PATH)
