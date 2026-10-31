"""Stage `grains`: size and shape of the beads in the glass-bead packs.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

The porosity of the central VOIs of the bead columns (about 30-34 %) is below the random close packing
of equal spheres (about 36 %), which the wall effect cannot explain. Two explanations are tested on
the segmented solid phase of every VOI of a column (main and replicates, which do not overlap): (i)
segmentation bias (beads imaged larger than they are), and (ii) beads that are not equal spheres
(non-spherical or of mixed size pack more densely).

Beads touch through necks a few voxels wide, and some contain gas bubbles. The pore clusters that touch
no VOI face (the bubbles) are filled first; the beads are then separated by a watershed of the distance
map seeded with the cores farther than EROSION_VOX from the pore space, which breaks the necks without
splitting a bead. For every bead that does not touch a VOI face we record the volume-equivalent diameter
and the ratio of the major to the minor axis of the inertia ellipsoid; regions smaller than MIN_VOX
voxels (slivers at the necks) are counted and left out. The partition is repeated with the seeds at
SEED_SENSITIVITY voxels and the summaries are kept as a check of the seed choice. Each column is written
to <id>_grains.json and the summary tables are rebuilt from all of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from edt import edt
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import watershed

from .config import Config, read_json, write_json
from .morph import face_labels, label6
from .segment import load_pore, replicates

EROSION_VOX = 12
SEED_SENSITIVITY = (10, 14, 16)
MIN_VOX = 2000


def solid_dt(pore: np.ndarray):
    """Solid phase with the bubbles (pore clusters touching no face) filled, and its distance map."""
    lab, _ = label6(pore)
    outer = np.unique(np.concatenate([face_labels(lab, a, s) for a in range(3) for s in (0, 1)]))
    solid = ~np.isin(lab, outer)
    del lab
    return solid, edt(solid)


def beads(pore: np.ndarray, seed_vox: float = EROSION_VOX, sd=None) -> np.ndarray:
    """Label image of the beads: bubbles filled, seeded watershed of the distance map of the solid."""
    solid, dt = sd if sd is not None else solid_dt(pore)
    seeds, _ = ndi.label(dt > seed_vox)
    return watershed(-dt, markers=seeds, mask=solid)


def measure(reg: np.ndarray, s) -> tuple[list[dict], int]:
    """Size and shape of every bead not cut by a VOI face; regions below MIN_VOX are counted as slivers."""
    n = reg.shape[0]
    rows, slivers = [], 0
    for p in regionprops(reg):
        z0, y0, x0, z1, y1, x1 = p.bbox
        if min(z0, y0, x0) == 0 or max(z1, y1, x1) >= n:
            continue                     # cut by a face
        if p.area < MIN_VOX:
            slivers += 1
            continue
        rows.append({"id": s.id, "volume_vox": int(p.area),
                     "d_eq_mm": float((6 * p.area / np.pi) ** (1 / 3) * s.voxel_mm),
                     "axis_major_mm": float(p.axis_major_length * s.voxel_mm),
                     "axis_minor_mm": float(p.axis_minor_length * s.voxel_mm),
                     "aspect": float(p.axis_major_length / max(p.axis_minor_length, 1e-9))})
    return rows, slivers


def summary(rows: list[dict], slivers: int, s) -> dict:
    d = pd.DataFrame(rows)
    return {"id": s.id, "n_grains": len(d), "n_slivers": slivers, "nominal_mm": s.grain_mm,
            "d_eq_median_mm": float(d.d_eq_mm.median()), "d_eq_p10_mm": float(d.d_eq_mm.quantile(0.1)),
            "d_eq_p90_mm": float(d.d_eq_mm.quantile(0.9)), "aspect_median": float(d.aspect.median()),
            "aspect_p90": float(d.aspect.quantile(0.9)), "frac_aspect_gt_1_2": float((d.aspect > 1.2).mean())}


def run(cfg: Config, only=None, log=print) -> None:
    out = cfg.outdir("grains")
    for s in cfg.samples:
        if s.group != "granular" or (only and s.id not in only):
            continue
        seeds = sorted(SEED_SENSITIVITY + (EROSION_VOX,))
        found = {e: ([], 0) for e in seeds}               # seed -> (beads of all VOIs, slivers)
        ks = replicates(cfg, s.id)
        for k in ks:
            pore = load_pore(cfg, s.id, k)
            sd = solid_dt(pore)
            for e in seeds:
                rows_k, sl_k = measure(beads(pore, e, sd), s)
                for row in rows_k:
                    row["replicate"] = k
                found[e] = (found[e][0] + rows_k, found[e][1] + sl_k)
            del sd, pore
        rows, slivers = found[EROSION_VOX]
        r = {**summary(rows, slivers, s), "n_vois": len(ks)}
        sens = [{**summary(*found[e], s), "seed_vox": e} for e in seeds]
        write_json(out / f"{s.id}_grains.json", {"summary": r, "grains": rows, "seed_sensitivity": sens})
        log(f"  {s.id}: {r['n_grains']} beads ({slivers} slivers left out), d_eq median {r['d_eq_median_mm']:.2f} mm "
            f"(nominal {s.grain_mm}), aspect median {r['aspect_median']:.2f}, "
            f"{100 * r['frac_aspect_gt_1_2']:.0f}% with aspect > 1.2")
    done = [read_json(out / f"{s.id}_grains.json") for s in cfg.samples if (out / f"{s.id}_grains.json").exists()]
    pd.DataFrame([g for d in done for g in d["grains"]]).to_csv(out / "grains.csv", index=False)
    pd.DataFrame([d["summary"] for d in done]).to_csv(out / "grains_summary.csv", index=False)
    pd.DataFrame([x for d in done for x in d.get("seed_sensitivity", [])]).to_csv(
        out / "grains_seed_sensitivity.csv", index=False)
    write_json(out / "grains_summary.json", [d["summary"] for d in done])
