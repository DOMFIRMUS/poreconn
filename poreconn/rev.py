"""Stage `rev`: sub-volume statistics (representative volume and finite-size percolation).

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Every segmented VOI (all replicates) is tiled with non-overlapping cubes of edge e voxels, for each e
in `rev.edges`; when only one such cube fits (e > N/2, e < N), the eight corner-anchored cubes of edge e
are used instead and flagged `overlapping` (they share volume, so the report leaves them, and the single
cube of edge N, out of the spread across cubes). For every cube the porosity, the Crofton surface
density S_V, the Euler characteristic density, the connected fraction and percolation along z, y and x
are recorded. The spread of porosity and chi_V across cubes gives the representative-volume analysis;
the percolation outcomes, pooled over cubes, axes and samples, are the data for the percolation-
probability analysis (stage `predict` and the report).

The analysis is repeated for each air reference in `air_variants` (column `variant`): T50 is the stored
ISO-50 segmentation; T50int re-thresholds the median-filtered VOI with the internal-air reference of
the rocks. The bead packs have no T50int (their air reference is already the pore air), so they appear
under T50 only.

Each sample is written to <id>_rev.json and rev.csv is always rebuilt from the per-sample files present
(in the order of config.yaml), so that a re-run of some samples keeps the table consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, read_json, write_json
from .morph import connectivity, crofton_sv, euler, euler_density, label6
from .segment import load_filtered, load_pore, replicates, seg_paths


def variant_pore(cfg: Config, sid: str, k: int, variant: str, filt=None):
    """Pore mask of VOI k at an air-reference variant (None when the variant is not defined)."""
    if variant == "T50":
        return load_pore(cfg, sid, k)
    thr = read_json(seg_paths(cfg, sid, k)["json"]).get(variant)
    if thr is None or not np.isfinite(thr):
        return None
    if filt is None:
        filt = load_filtered(cfg, sid, k)
    return filt < thr


def run(cfg: Config, only=None, log=print) -> None:
    out = cfg.outdir("rev")
    variants = cfg.raw.get("air_variants", ["T50"])
    for s in cfg.samples:
        if only and s.id not in only:
            continue
        srows = []
        for k in replicates(cfg, s.id):
            for var in variants:
                pore = variant_pore(cfg, s.id, k, var)
                if pore is None:
                    continue
                N = pore.shape[0]
                for e in cfg["rev"]["edges"]:
                    if e > N:
                        continue
                    m = N // e
                    if m >= 2 or e == N:
                        starts = [(i * e, j * e, l * e) for i in range(m) for j in range(m) for l in range(m)]
                    else:
                        starts = [(a, b, c) for a in (0, N - e) for b in (0, N - e) for c in (0, N - e)]
                    for (a0, b0, c0) in starts:
                        c = pore[a0:a0 + e, b0:b0 + e, c0:c0 + e]
                        lab, n = label6(c)
                        r = {"id": s.id, "group": s.group, "replicate": k, "variant": var, "edge": e,
                             "edge_mm": e * s.voxel_mm, "cube": f"{a0}-{b0}-{c0}", "overlapping": m < 2 and e != N,
                             **connectivity(c, lab, n)}
                        del lab
                        r["euler"] = euler(c)
                        r["chi_V"] = euler_density(c, s.voxel_mm)
                        r["S_V"] = crofton_sv(c, s.voxel_mm)["S_V"] if c.any() else 0.0
                        srows.append(r)
                del pore
        log(f"  {s.id}: {len(srows)} sub-volumes")
        write_json(out / f"{s.id}_rev.json", srows)
    rows = []
    for s in cfg.samples:
        f = out / f"{s.id}_rev.json"
        if f.exists():
            rows.extend(read_json(f))
    if rows:
        pd.DataFrame(rows).to_csv(out / "rev.csv", index=False)
