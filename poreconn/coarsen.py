"""Stage `coarsen`: how resolved connectivity degrades when the voxel grows.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

The median-filtered grey VOI is block-averaged by integer factors b (each coarse voxel is the mean
attenuation of b^3 fine voxels, i.e. a coarser acquisition with an ideal box point-spread function)
and thresholded again at the same absolute threshold. This mimics, to first order, scanning the same
specimen with a b times larger voxel; it cannot reveal what lies below the native resolution, but it
shows how close each pore space is to digital disconnection.

Every VOI (`coarsening.all_replicates`) is coarsened at each air reference in `air_variants` (the bead
packs at T50 only). At the native voxel the critical diameter of each percolating axis of the rocks is
recorded (d_c_vox_<axis>), so that the voxel at which an axis stops percolating can be compared with its
critical throat in coarse voxels.

Each sample is written to <id>_coarsen.json and coarsen.csv is always rebuilt from the per-sample files
present (in the order of config.yaml), so that a re-run of some samples keeps the table consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, read_json, write_json
from .morph import block_mean, connectivity, euler, euler_density, label6
from .segment import load_filtered, replicates, seg_paths
from .topology import critical_diameter_vox


def run(cfg: Config, only=None, log=print) -> None:
    out = cfg.outdir("coarsen")
    variants = cfg.raw.get("air_variants", ["T50"])
    for s in cfg.samples:
        if only and s.id not in only:
            continue
        ks = replicates(cfg, s.id) if cfg["coarsening"].get("all_replicates", False) else [0]
        srows = []
        for k in ks:
            seg = read_json(seg_paths(cfg, s.id, k)["json"])
            filt = load_filtered(cfg, s.id, k)
            for var in variants:
                thr = seg.get(var)
                if thr is None or not np.isfinite(thr):
                    continue
                for b in cfg["coarsening"]["factors"]:
                    g = block_mean(filt, b)
                    pore = g < thr
                    del g
                    lab, n = label6(pore)
                    dx = s.voxel_mm * b
                    r = {"id": s.id, "group": s.group, "replicate": k, "variant": var, "factor": b, "voxel_mm": dx,
                         "V_mm3": pore.size * dx ** 3, **connectivity(pore, lab, n)}
                    del lab
                    r["euler"] = euler(pore)
                    r["chi_V"] = euler_density(pore, dx)
                    if b == 1 and s.group != "granular":    # bead packs: throats of many voxels (topology)
                        for ax, a in enumerate("zyx"):
                            r[f"d_c_vox_{a}"] = critical_diameter_vox(pore, ax) if r[f"perc_{a}"] else np.nan
                    del pore
                    srows.append(r)
                    log(f"  {s.id} v{k} {var} x{b} ({1000 * dx:.0f} um): phi {100 * r['phi']:.2f}% "
                        f"conn {100 * r['frac_connected']:.1f}% perc zyx {int(r['perc_z'])}{int(r['perc_y'])}"
                        f"{int(r['perc_x'])} chiV {r['chi_V']:.3g}")
            del filt
        write_json(out / f"{s.id}_coarsen.json", srows)
    rows = []
    for s in cfg.samples:
        f = out / f"{s.id}_coarsen.json"
        if f.exists():
            rows.extend(read_json(f))
    if rows:
        pd.DataFrame(rows).to_csv(out / "coarsen.csv", index=False)
