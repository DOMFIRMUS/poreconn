"""Stage `transport`: formation factor (diffusion) and pore-network permeability of the main VOIs.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

For the main VOI of each sample (replicate 0, ISO-50 segmentation) and every axis along which it
percolates: formation factor F, tortuosity factor tau = phi F and Archie exponent m from the diffusion
solver (fd.py, with its diagnostics fd_iterations, fd_residual, fd_flux_balance and fd_converged), and
permeability from the pore network (pnm.py): inscribed (k_pnm) and equivalent sizes (k_pnmeq) with the
VOI faces as symmetry planes, and the same with the faces as walls (k_pnmw, k_pnmeqw). A failed network
extraction or solve gives NaN and an error string (<tag>_error) instead of stopping the stage.
Two image-based estimates are added for comparison, from quantities measured on the same VOI:

    Kozeny-Carman   k_KC = phi^3 / (5 S_V^2)       (S_V: pore-solid surface per bulk volume)
    Katz-Thompson   k_KT = d_c^2 / (226 F)         (d_c: critical diameter of drainage)

k_KT is reported only when d_c spans at least 3 voxels; otherwise d_c is set by the voxel, not by the
pore space, and k_KT is left empty (the d_c columns keep the value and its flag).

Each sample is written to <id>_transport.json and transport.csv is always rebuilt from the per-sample
files present (in the order of config.yaml), so that a re-run of some samples keeps the table consistent.

Threshold sensitivity (rocks): the main VOI is segmented again at the thresholds in
transport.threshold_variants (the bounds T40 and T60, the internal-air reference T50int and, for
completeness, T50) and, for every axis that percolates there, F, the network permeability (inscribed
and equivalent sizes, mirror faces), k_KC from the whole pore phase and k_KC_span from the cluster that
spans the axis (its own porosity and surface density) are recomputed (<id>_transport_thr.json,
transport_thresholds.csv; run_thresholds() redoes only this part). The bead packs are left out: their
percolation and porosity hardly change between the bounds and their networks are the most expensive.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import Config, read_json, write_json
from .fd import formation_factor
from .morph import crofton_sv, spanning_mask
from .pnm import network_permeability
from .segment import load_filtered, load_pore, seg_paths

MD = 9.869233e-16   # m^2 per millidarcy
TAGS = ("pnm", "pnmeq", "pnmw", "pnmeqw")


def _network(pore, dx_m, log=print, faces=("mirror", "wall")) -> dict:
    """network_permeability with a failure turned into NaN permeabilities and error strings."""
    try:
        return network_permeability(pore, dx_m, faces=faces)
    except Exception as e:     # noqa: BLE001 - any failure of the extraction or of OpenPNM
        msg = f"{type(e).__name__}: {e}"
        log(f"  PNM failed: {msg}")
        tags = [t for t in TAGS if ("wall" if t.endswith("w") else "mirror") in faces]
        out = {f"k_{t}_{a}_m2": np.nan for t in tags for a in "zyx"}
        out.update({f"{t}_error": msg for t in tags})
        return out


def threshold_rows(cfg: Config, s, variants, log=print) -> list:
    """F, k_PNM (mirror faces) and k_KC of the main VOI segmented at the threshold bounds."""
    seg = read_json(seg_paths(cfg, s.id, 0)["json"])
    filt = load_filtered(cfg, s.id, 0)
    tcfg = cfg["transport"]
    dx_m = s.voxel_mm * 1e-3
    rows = []
    for var in variants:
        thr = seg.get(var)
        if thr is None or not np.isfinite(thr):
            continue
        pore = filt < thr
        perc = [bool(spanning_mask(pore, a).any()) for a in range(3)]
        sv = crofton_sv(pore, s.voxel_mm)["S_V"] if pore.any() else np.nan
        pnm = _network(pore, dx_m, log, faces=("mirror",)) if any(perc) and tcfg.get("pnm", True) else {}
        for ax, a in enumerate("zyx"):
            r = {"id": s.id, "group": s.group, "variant": var, "threshold": thr, "axis": a,
                 "phi": float(pore.mean()), "perc": perc[ax]}
            if perc[ax]:
                ff = formation_factor(pore, ax, tol=tcfg["fd_tol"], maxiter=tcfg["fd_maxiter"])
                span = spanning_mask(pore, ax)
                phi_s = float(span.mean())
                sv_s = crofton_sv(span, s.voxel_mm)["S_V"]
                del span
                r.update(F=ff["F"], fd_converged=bool(ff["iterations"] < tcfg["fd_maxiter"] or ff["residual"] <= tcfg["fd_tol"]),
                         phi_span=phi_s, S_V_span_per_mm=sv_s,
                         k_pnm_m2=pnm.get(f"k_pnm_{a}_m2", np.nan), k_pnmeq_m2=pnm.get(f"k_pnmeq_{a}_m2", np.nan),
                         k_KC_m2=r["phi"] ** 3 / (5 * (sv * 1e3) ** 2),
                         k_KC_span_m2=phi_s ** 3 / (5 * (sv_s * 1e3) ** 2))
            else:
                r.update(F=np.inf, fd_converged=True, phi_span=0.0, S_V_span_per_mm=np.nan, k_pnm_m2=0.0,
                         k_pnmeq_m2=0.0, k_KC_m2=np.nan, k_KC_span_m2=np.nan)
            for key in ("k_pnm", "k_pnmeq", "k_KC", "k_KC_span"):
                r[f"{key}_mD"] = r[f"{key}_m2"] / MD
            r.update({k: v for k, v in pnm.items() if k.endswith("_error")})
            rows.append(r)
        log(f"  {s.id} {var}: phi {100 * pore.mean():.2f}% perc {''.join(str(int(p)) for p in perc)} "
            + " ".join(f"{x['axis']}: F {x['F']:.3g} k_pnm {x['k_pnm_mD']:.4g} mD" for x in rows[-3:] if x["perc"]))
        del pore
    return rows


def run(cfg: Config, only=None, log=print) -> None:
    out = cfg.outdir("transport")
    topo_dir = cfg.outdir("topology")
    tcfg = cfg["transport"]
    for s in cfg.samples:
        if only and s.id not in only:
            continue
        t0 = time.time()
        pore = load_pore(cfg, s.id, 0)
        topo = [r for r in read_json(topo_dir / f"{s.id}_v0_topology.json") if r["variant"] == "T50"][0]
        dx_m = s.voxel_mm * 1e-3
        pnm = _network(pore, dx_m, log=log) if tcfg.get("pnm", True) else {}
        srows = []
        for ax, a in enumerate("zyx"):
            r = {"id": s.id, "group": s.group, "axis": a, "phi": topo["phi"], "perc": topo[f"perc_{a}"],
                 "phi_span": topo[f"phi_span_{a}"], "S_V_per_mm": topo["S_V"]}
            if topo[f"perc_{a}"]:
                ff = formation_factor(pore, ax, tol=tcfg["fd_tol"], maxiter=tcfg["fd_maxiter"])
                r.update(F=ff["F"], tau=ff["tau"], m_archie=ff["m_archie"], fd_iterations=ff["iterations"],
                         fd_residual=ff["residual"], fd_flux_balance=ff["flux_balance"],
                         fd_converged=bool(ff["iterations"] < tcfg["fd_maxiter"] or ff["residual"] <= tcfg["fd_tol"]))
                for t in TAGS:
                    r[f"k_{t}_m2"] = pnm.get(f"k_{t}_{a}_m2", np.nan)
                r["k_KC_m2"] = topo["phi"] ** 3 / (5 * (topo["S_V"] * 1e3) ** 2)
                dc = topo.get(f"d_c_{a}", np.nan)
                r["d_c_mm"] = dc
                r["d_c_vox"] = topo.get(f"d_c_vox_{a}", np.nan)
                r["d_c_resolved"] = bool(topo.get(f"d_c_resolved_{a}", False))
                r["k_KT_m2"] = (dc * 1e-3) ** 2 / (226 * ff["F"]) if (np.isfinite(dc) and r["d_c_resolved"]) else np.nan
            else:
                r.update(F=np.inf, tau=np.inf, m_archie=np.nan, **{f"k_{t}_m2": 0.0 for t in TAGS},
                         k_KC_m2=np.nan, k_KT_m2=np.nan, d_c_mm=np.nan, d_c_vox=np.nan, d_c_resolved=False)
            for key in [f"k_{t}" for t in TAGS] + ["k_KC", "k_KT"]:
                r[f"{key}_mD"] = r[f"{key}_m2"] / MD
            r.update({k: v for k, v in pnm.items() if not k.startswith("k_")})
            srows.append(r)
            if r["perc"]:
                log(f"  {s.id} {a}: F {r['F']:.2f} tau {r['tau']:.2f} m {r['m_archie']:.2f} "
                    f"(it {r['fd_iterations']}) | k mD: pnm {r['k_pnm_mD']:.4g} pnm-eq {r['k_pnmeq_mD']:.4g} "
                    f"walls {r['k_pnmw_mD']:.4g}/{r['k_pnmeqw_mD']:.4g} KC {r['k_KC_mD']:.4g} KT {r['k_KT_mD']:.4g}")
        write_json(out / f"{s.id}_transport.json", srows)
        if s.group != "granular":
            write_json(out / f"{s.id}_transport_thr.json",
                       threshold_rows(cfg, s, tcfg.get("threshold_variants", ["T40", "T60"]), log=log))
        log(f"  {s.id}: done in {time.time() - t0:.0f} s")
    _rebuild(cfg, out)


def run_thresholds(cfg: Config, only=None, log=print) -> None:
    """Only the threshold-sensitivity rows of the rocks (no new main-VOI networks)."""
    out = cfg.outdir("transport")
    for s in cfg.samples:
        if (only and s.id not in only) or s.group == "granular":
            continue
        write_json(out / f"{s.id}_transport_thr.json",
                   threshold_rows(cfg, s, cfg["transport"].get("threshold_variants", ["T40", "T60"]), log=log))
    _rebuild(cfg, out)


def _rebuild(cfg: Config, out) -> None:
    for suffix, name in (("transport", "transport.csv"), ("transport_thr", "transport_thresholds.csv")):
        rows = []
        for s in cfg.samples:
            f = out / f"{s.id}_{suffix}.json"
            if f.exists():
                rows.extend(read_json(f))
        if rows:
            pd.DataFrame(rows).to_csv(out / name, index=False)
