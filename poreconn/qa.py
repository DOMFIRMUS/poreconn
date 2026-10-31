"""Stage `qa`: checks of the assumptions behind the ISO-50 segmentation.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

1. Internal air (rocks). The median grey value of the pore voxels at distances >= 2, 3, 5 and 8 voxels
   from the solid (distance transform of a generous pore phase, f = 0.6 relative to the outside air;
   a shell of the same depth at the VOI faces is excluded) is expressed as a fraction of the outside
   air-solid contrast above the outside air. This documents the internal-air offset used for the
   T50int upper bound; it cannot tell an imaging artefact from partial infill of the pores. A
   provisional phase cut at f = 0.6 cannot return an offset above 0.6 and biases it low when the offset
   approaches that level, so the depth-5 offset is repeated with a cut at f = 0.8 (offset_depth5_prov08).
2. Far-field air (rocks). The shift of the air beyond 100 px from the plug relative to the reference
   annulus (20-100 px), as a fraction of the contrast (air_far_shift_frac): the size of the air halo.
3. Cupping (beam hardening). The median of the solid voxels within +-15 % of the contrast around the
   solid reference is compared between the central and the outer part of the VOI cross-section
   (inside and outside the circle of radius 0.35 x edge); a large difference would make a single
   threshold per VOI inappropriate.

One `<id>_qa.json` is written per sample; `qa.csv` is rebuilt from all of them on every run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from edt import edt

from .config import Config, read_json, write_json
from .segment import load_filtered, load_pore, seg_paths


def deep_offsets(filt: np.ndarray, air: float, contrast: float, f_prov: float, depths, min_n: int = 200) -> dict:
    """Median (grey - air) / contrast of the voxels >= d voxels inside a provisional pore phase cut at
    air + f_prov * contrast, away from a shell of d voxels at the VOI faces."""
    dtp = edt(filt < air + f_prov * contrast)
    out = {}
    for dd in depths:
        mm = dtp >= dd
        mm[:dd] = mm[-dd:] = False          # exclude the shell where the face hides the solid
        mm[:, :dd] = mm[:, -dd:] = False
        mm[:, :, :dd] = mm[:, :, -dd:] = False
        n = int(mm.sum())
        out[dd] = (float((np.median(filt[mm]) - air) / contrast) if n >= min_n else np.nan, n)
    return out


def sample_qa(cfg: Config, s) -> dict:
    seg = read_json(seg_paths(cfg, s.id, 0)["json"])
    filt = load_filtered(cfg, s.id, 0)
    pore = load_pore(cfg, s.id, 0)
    n = filt.shape[1]
    yy, xx = np.indices(filt.shape[1:])
    rr = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2)
    inner, outer = rr < 0.35 * n, rr >= 0.35 * n
    # cupping on one tracked mode: solid voxels within +-15 % of the contrast around I_solid
    c = seg["I_solid"] - seg["I_air"]
    band = (~pore) & (np.abs(filt.astype(np.float64) - seg["I_solid"]) < 0.15 * c)
    sol_in = filt[:, inner][band[:, inner]]
    sol_out = filt[:, outer][band[:, outer]]
    del band, pore
    r = {"id": s.id, "group": s.group, "I_air": seg["I_air"], "I_solid": seg["I_solid"],
         "solid_median_centre": float(np.median(sol_in)), "solid_median_edge": float(np.median(sol_out))}
    r["cupping_frac"] = (r["solid_median_edge"] - r["solid_median_centre"]) / c
    ao = seg.get("air_outside", float("nan"))
    af = seg.get("air_far", float("nan"))
    ao = float("nan") if ao is None else ao
    af = float("nan") if af is None else af
    r["air_far"] = af
    r["air_far_shift_frac"] = (af - ao) / (seg["I_solid"] - ao) if np.isfinite(ao) and np.isfinite(af) else np.nan
    if np.isfinite(ao):
        for dd, (off, nd) in deep_offsets(filt, ao, c, 0.6, (2, 3, 5, 8)).items():
            r[f"offset_depth{dd}"] = off
            r[f"n_depth{dd}"] = nd
        off, nd = deep_offsets(filt, ao, c, 0.8, (5,))[5]
        r["offset_depth5_prov08"] = off
        r["n_depth5_prov08"] = nd
    return r


def run(cfg: Config, only=None, log=print) -> None:
    out = cfg.outdir("qa")
    for s in cfg.samples:
        if only and s.id not in only:
            continue
        r = sample_qa(cfg, s)
        write_json(out / f"{s.id}_qa.json", r)
        log(f"  {s.id}: cupping {r['cupping_frac']:+.3f} of contrast; far-field air {r['air_far_shift_frac']:+.3f}; "
            "internal-air offset by depth "
            + " ".join(f"{dd}:{r.get(f'offset_depth{dd}', float('nan')):.3f}" for dd in (2, 3, 5, 8))
            + f" (depth 5, cut f=0.8: {r.get('offset_depth5_prov08', float('nan')):.3f})")
    # qa.csv from every per-sample file, in config order
    order = {s.id: i for i, s in enumerate(cfg.samples)}
    rows = [read_json(f) for f in out.glob("*_qa.json")]
    rows.sort(key=lambda q: (order.get(q["id"], len(order)), q["id"]))
    pd.DataFrame(rows).to_csv(out / "qa.csv", index=False)
