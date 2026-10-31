"""Stage `topology`: porosity, connectivity, percolation, Euler characteristic, surface and pore size.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

For every VOI the connectivity, the integer Euler characteristic of the window and the edge-corrected
Euler number density chi_V are computed at the ISO-50 threshold, at the bounding thresholds
(f = 0.4, 0.6), at the internal-air upper bound (T50int, rocks only) and at Otsu's threshold. Surface
density and Crofton chord lengths are computed at ISO-50; the dimensionless chi_V l_p^3 uses the mean
pore chord l_p = 4 phi / S_V (morph.crofton_sv), consistent with the reported S_V. For the main VOI of
each sample (replicate 0) the pore-size distribution (local thickness) and, along every percolating
axis, the critical diameter d_c of morphological drainage (the largest sphere diameter that still
invades the VOI from one face and reaches the opposite face; used in the Katz-Thompson estimate) are
computed.

Sphere sizes are taken on an explicit grid of 60 log-spaced radii between 1 voxel and just below the
largest distance-transform value, so that the largest size always invades; d_c and the pore diameters
are reported in voxels as well, and d_c below 3 voxels is flagged as unresolved (an upper bound set by
the voxel, not a measurement). Resolution of d_c: the diameter is read as 2 x EDT, and the Euclidean
distance of a central voxel to the nearest solid voxel centre is (w + 1) / 2 for a throat of odd width
w voxels, so d_c reads one voxel high for odd widths (exact for even widths); it is then rounded down
to the radius grid (relative step (EDT_max)^(1/59) - 1, about 6 % for EDT_max = 30). The two effects
are of opposite sign and of order one voxel, so d_c is resolved to about one voxel.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from edt import edt

from .config import Config, read_json, write_json
from .morph import connectivity, crofton_sv, euler, euler_density, label6
from .segment import load_filtered, replicates, seg_paths

D_C_MIN_VOX = 3.0


def size_grid(pore: np.ndarray, n: int = 60) -> np.ndarray:
    dtmax = float(edt(pore).max())          # float32 EDT (scipy's needs ~5x the memory)
    if dtmax <= 1.0:
        return np.array([1.0])
    return np.logspace(np.log10(dtmax * (1 - 1e-6)), 0.0, n)


def pore_size(pore: np.ndarray, dx: float) -> dict:
    import porespy as ps
    if not pore.any():
        return {k: np.nan for k in ("d10", "d50", "d90", "d_mean", "d50_vox")}
    lt = ps.filters.local_thickness(pore, sizes=size_grid(pore), mode="dt")
    r = lt[pore]
    q = np.percentile(r, [10, 50, 90]) * 2
    return {"d10": float(q[0] * dx), "d50": float(q[1] * dx), "d90": float(q[2] * dx),
            "d_mean": float(2 * dx * r.mean()), "d50_vox": float(q[1])}


def critical_diameter_vox(pore: np.ndarray, axis: int) -> float:
    """Largest invading-sphere diameter (voxels) that spans the VOI along `axis` (access-limited)."""
    import porespy as ps
    inlets = np.zeros_like(pore, dtype=bool)
    sl = [slice(None)] * 3
    sl[axis] = 0
    inlets[tuple(sl)] = True
    inlets &= pore
    if not inlets.any():
        return float("nan")
    sizes = ps.filters.porosimetry(pore, sizes=size_grid(pore), inlets=inlets, access_limited=True, mode="dt")
    sl[axis] = -1
    out = sizes[tuple(sl)]
    return float(2 * out.max()) if out.max() > 0 else float("nan")


def topology_voi(cfg: Config, s, k: int, log=print) -> list[dict]:
    import porespy as ps
    ps.settings.tqdm["disable"] = True
    t0 = time.time()
    seg = read_json(seg_paths(cfg, s.id, k)["json"])
    filt = load_filtered(cfg, s.id, k)
    dx = s.voxel_mm
    rows = []
    variants = {"T50": seg["T50"], "T40": seg["T40"], "T60": seg["T60"], "T50int": seg.get("T50int", np.nan),
                "otsu": seg["T_otsu"]}
    for var, thr in variants.items():
        if thr is None or not np.isfinite(thr):
            continue
        pore = filt < thr
        lab, n = label6(pore)
        r = {"id": s.id, "group": s.group, "replicate": k, "variant": var, "threshold": thr, "voxel_mm": dx,
             "V_mm3": pore.size * dx ** 3, **connectivity(pore, lab, n)}
        del lab
        r["euler"] = euler(pore)
        r["chi_V"] = euler_density(pore, dx)          # edge-corrected, per mm^3
        r["chi_V_closed"] = r["euler"] / r["V_mm3"]   # closed-window count / volume (for reference)
        if var == "T50":
            r["chi_V_26"] = euler_density(pore, dx, conn=26)   # sensitivity to the adjacency choice
            r.update(crofton_sv(pore, dx))
            r["chi_V_l3"] = r["chi_V"] * r["chord_pore_mean"] ** 3 if np.isfinite(r["chord_pore_mean"]) else np.nan
            if k == 0:
                r.update(pore_size(pore, dx))
                for ax, a in enumerate("zyx"):
                    dv = critical_diameter_vox(pore, ax) if r[f"perc_{a}"] else float("nan")
                    r[f"d_c_vox_{a}"] = dv
                    r[f"d_c_{a}"] = dv * dx
                    r[f"d_c_resolved_{a}"] = bool(np.isfinite(dv) and dv >= D_C_MIN_VOX)
        rows.append(r)
    b = rows[0]
    log(f"  {s.id} v{k}: phi {100 * b['phi']:.2f}% conn {100 * b['frac_connected']:.1f}% "
        f"iso {100 * b['frac_isolated']:.1f}% perc zyx {int(b['perc_z'])}{int(b['perc_y'])}{int(b['perc_x'])} "
        f"chi {b['euler']} chiV {b['chi_V']:.3g}/mm3 SV {b['S_V']:.2f}/mm lp {b['chord_pore_mean']:.3f} mm "
        f"d50 {b.get('d50', float('nan')):.3f} mm ({time.time() - t0:.0f} s)")
    return rows


def run(cfg: Config, only=None, log=print) -> None:
    out = cfg.outdir("topology")
    for s in cfg.samples:
        if only and s.id not in only:
            continue
        for k in replicates(cfg, s.id):
            write_json(out / f"{s.id}_v{k}_topology.json", topology_voi(cfg, s, k, log=log))
    # the table is always rebuilt from the per-VOI files, so partial re-runs stay consistent
    rows = []
    for f in sorted(out.glob("*_topology.json")):
        rows.extend(read_json(f))
    pd.DataFrame(rows).to_csv(out / "topology.csv", index=False)
