"""Stage `budget`: porosity budget of every specimen against its laboratory porosity.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Three image porosities are set against the laboratory porosity, all from the pooled grey-level histogram of
the median-filtered VOIs of a specimen (outputs/segment/<id>_hist_pooled.npz), so that each refers to all its
VOIs, like a laboratory measurement of the plug:

* phi_T50, phi_T50int: the fraction of voxels below the ISO-50 threshold and below the internal-air
  threshold (exactly the porosity of the segmented VOIs, since the pore phase is filtered < T);
* phi_mb, phi_mb_int: the threshold-free mass-balance porosity (I_s - mean grey) / (I_s - I_air), which
  counts every partial-volume voxel in proportion to its air content under linear mixing, with the
  outside air and with the internal air (I_air + offset (I_s - I_air)) as the pore end member. The solid
  end member is the solid mode of the specimen, so for a microporous rock matrix phi_mb measures the
  porosity darker than the matrix, not the porosity inside it; grey values above I_s (denser minerals,
  the second glass of CF10) lower it;
* T_lab, f_lab: the threshold at which the pooled image porosity equals the laboratory porosity, and its
  position f_lab = (T_lab - I_air) / (I_s - I_air) on the outside air-solid contrast (f = 0.5 is ISO-50;
  f near or above 1 means that the laboratory porosity is reached only by calling most of the solid mode
  pore).

For the rocks with a laboratory porosity, the main VOI is thresholded at T_lab and its connectivity and
percolation are measured, which asks whether the image would percolate if the threshold were tuned to
the laboratory porosity. Laboratory porosities come from data/lab_porosity.csv: helium porosity of the
rocks, and for the bead packs the mean of the four methods (whose central VOIs are denser than the column,
see the wall stage, so their f_lab is not a segmentation bias).

Outputs (outputs/budget/): porosity_budget.csv, one row per specimen.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import Config, read_json
from .morph import connectivity
from .segment import load_filtered, seg_paths

LAB_METHODS_BEADS = ("direct", "gravimetric", "indirect", "helium")


def pooled(cfg: Config, sid: str):
    z = np.load(seg_paths(cfg, sid, 0)["pooled"])
    return z["centres"].astype(float), z["hist"].astype(float)


def fraction_below(c: np.ndarray, h: np.ndarray, t: float) -> float:
    """Share of the voxels with grey value below t, from a histogram of bin centres c (uniform bins)."""
    w = c[1] - c[0]
    upper = c + w / 2
    cum = np.cumsum(h) / h.sum()
    return float(np.interp(t, upper, cum, left=0.0, right=1.0))


def threshold_for(c: np.ndarray, h: np.ndarray, phi: float) -> float:
    """Grey value below which a share phi of the voxels lies (inverse of fraction_below)."""
    w = c[1] - c[0]
    upper = c + w / 2
    cum = np.cumsum(h) / h.sum()
    keep = np.r_[True, np.diff(cum) > 0]               # strictly increasing for the interpolation
    return float(np.interp(phi, cum[keep], upper[keep]))


def mass_balance(c: np.ndarray, h: np.ndarray, i_air: float, i_s: float) -> float:
    """Threshold-free porosity (I_s - mean) / (I_s - I_air) under linear mixing of two end members."""
    mean = float((c * h).sum() / h.sum())
    return (i_s - mean) / (i_s - i_air)


def lab_porosity(lab: pd.DataFrame, sid: str, group: str):
    l = lab[lab.id == sid]
    if group == "granular":
        v = l[l.method.isin(LAB_METHODS_BEADS)].porosity_pct
        return (float(v.mean()) / 100, "mean of the four laboratory methods") if len(v) == len(LAB_METHODS_BEADS) else (np.nan, "")
    he = l[l.method == "helium"]
    return (float(he.porosity_pct.iloc[0]) / 100, str(he.source.iloc[0])) if len(he) else (np.nan, "")


def sample_row(cfg: Config, s, lab: pd.DataFrame, log=print) -> dict:
    ref = read_json(seg_paths(cfg, s.id, 0)["sample"])
    c, h = pooled(cfg, s.id)
    i_air, i_s = float(ref["I_air"]), float(ref["I_solid"])
    off = ref.get("air_int_offset")
    i_air_int = i_air + off * (i_s - i_air) if off is not None and np.isfinite(off) else np.nan
    phi_lab, src = lab_porosity(lab, s.id, s.group)
    t50int = ref.get("T50int")
    r = {"id": s.id, "group": s.group, "I_air": i_air, "I_solid": i_s, "T50": ref["T50"],
         "phi_lab": phi_lab, "lab_source": src,
         "phi_T50": fraction_below(c, h, ref["T50"]),
         "phi_T50int": fraction_below(c, h, t50int) if t50int is not None and np.isfinite(t50int) else np.nan,
         "phi_mb": mass_balance(c, h, i_air, i_s),
         "phi_mb_int": mass_balance(c, h, i_air_int, i_s) if np.isfinite(i_air_int) else np.nan}
    nan = float("nan")
    r.update(T_lab=nan, f_lab=nan, phi_main_at_lab=nan, perc_main_at_lab="", frac_connected_main_at_lab=nan)
    if np.isfinite(phi_lab):
        t_lab = threshold_for(c, h, phi_lab)
        r.update(T_lab=t_lab, f_lab=(t_lab - i_air) / (i_s - i_air))
        if s.group != "granular":
            t0 = time.time()
            pore = load_filtered(cfg, s.id, 0) < t_lab
            cn = connectivity(pore)
            del pore
            r.update(phi_main_at_lab=cn["phi"],
                     perc_main_at_lab="".join(str(int(cn[f"perc_{a}"])) for a in "zyx"),
                     frac_connected_main_at_lab=cn["frac_connected"])
            log(f"  {s.id}: lab {100 * phi_lab:.2f}% at f = {r['f_lab']:.3f}; main VOI there phi "
                f"{100 * cn['phi']:.2f}% perc zyx {r['perc_main_at_lab']} connected {100 * cn['frac_connected']:.1f}% "
                f"({time.time() - t0:.0f} s)")
    log(f"  {s.id}: phi T50 {100 * r['phi_T50']:.2f}%  mass balance {100 * r['phi_mb']:.2f}% "
        f"(internal air {100 * r['phi_mb_int']:.2f}%)  lab {100 * phi_lab:.2f}%")
    return r


def run(cfg: Config, only=None, log=print) -> None:
    out = cfg.outdir("budget")
    lab = pd.read_csv(cfg.p("lab_csv"))
    rows = [sample_row(cfg, s, lab, log) for s in cfg.samples if not only or s.id in only]
    f = out / "porosity_budget.csv"
    if only and f.exists():                            # keep the other specimens of an earlier run
        old = pd.read_csv(f)
        old = old[~old.id.isin([r["id"] for r in rows])]
        df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
    order = {s.id: k for k, s in enumerate(cfg.samples)}
    df = df.sort_values("id", key=lambda x: x.map(order))
    df.to_csv(f, index=False)
    log(f"budget: {len(df)} specimens -> {f}")
