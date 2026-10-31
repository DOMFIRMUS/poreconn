"""Configuration loading and path helpers.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Sample:
    id: str
    group: str
    label: str
    raw: str
    voxel_um: float
    bin: int = 1
    grain_mm: float | None = None
    solid_mode: str = "dominant"
    xtekct: str | None = None
    legacy: str | None = None

    @property
    def voxel_mm(self) -> float:
        """Voxel size of the analysed VOI (after binning), in mm."""
        return self.voxel_um * self.bin / 1000.0


class Config:
    def __init__(self, path: str | Path | None = None, quick: bool = False):
        self.path = Path(path) if path else ROOT / "config.yaml"
        with open(self.path, encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)
        self.quick = quick
        self.samples = [Sample(**s) for s in self.raw["samples"]]

    def __getitem__(self, key):
        return self.raw[key]

    def p(self, key: str) -> Path:
        """Resolve a path from the `paths` section (relative to the config file).

        In quick mode the VOI folder is redirected to <voi_dir>_quick, so that a smoke test never
        overwrites the deposited 300^3 volumes.
        """
        if key == "raw_root" and os.environ.get("PORECONN_RAW"):
            return Path(os.environ["PORECONN_RAW"])
        v = Path(self.raw["paths"][key])
        v = v if v.is_absolute() else (self.path.parent / v)
        if key == "voi_dir" and self.quick:
            v = v.parent / (v.name + "_quick")
        return v

    @property
    def out(self) -> Path:
        d = self.p("outputs")
        if self.quick:
            d = d.parent / (d.name + "_quick")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def outdir(self, name: str) -> Path:
        d = self.out / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def sample(self, sid: str) -> Sample:
        for s in self.samples:
            if s.id == sid:
                return s
        raise KeyError(sid)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_default)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
