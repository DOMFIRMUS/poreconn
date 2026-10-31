"""Stage `estimates`: Kozeny-Carman, Katz-Thompson and pore networks on the domains of the LBM runs.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

The transport stage evaluates the estimates on the whole main VOI, while the lattice Boltzmann
permeability of the porous samples is computed on the central lbm_edge^3 cube; to compare them on the
same pore space, each LBM domain is rebuilt here (the same rule as lbm.run_variant) and

    k_KC      = phi^3 / (5 S_V^2)          (phi and Crofton S_V of the whole pore phase of the domain)
    k_KC_span = phi_s^3 / (5 S_V,s^2)      (the same for the cluster spanning the axis only, the pore
                                            space the LBM, FD and PNM solvers see)
    k_KT      = d_c^2 / (226 F)            (critical diameter of drainage on the domain along the axis, F of
                                            the same domain from the lbm stage; only when d_c spans >= 3 voxels)

are written to <id>_estimates<suffix>.json; estimates<suffix>.csv is rebuilt from all of them. Where the
lbm stage also simulated the whole VOI block-averaged by two (cube domains), the pore networks are
extracted on that same coarsened image (k_<tag>_whole2_mD), so that networks and lattice Boltzmann can
be compared on a domain several grains wide and at the same voxel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, read_json, write_json
from .morph import crofton_sv, spanning_mask
from .topology import D_C_MIN_VOX, critical_diameter_vox

MD = 9.869233e-16
TAGS = ("pnm", "pnmeq", "pnmw", "pnmeqw")


def run_variant(cfg: Config, variant: str = "T50", only=None, log=print, full: bool = False) -> None:
    from .lbm import _coarse_whole, _suffix
    from .pnm import network_permeability
    from .rev import variant_pore
    from .segment import seg_paths
    ldir = cfg.outdir("lbm")
    sfx = _suffix(variant, full)
    for s in cfg.samples:
        if only and s.id not in only:
            continue
        f = ldir / f"{s.id}_lbm{sfx}.json"
        rows_lbm = read_json(f) if f.exists() else []
        if not rows_lbm:
            continue
        pore = variant_pore(cfg, s.id, 0, variant)
        rows = []
        whole2_pnm = None
        if any(np.isfinite(r.get("k_lbm_whole2_mD", np.nan) or np.nan) for r in rows_lbm):
            w2 = _coarse_whole(cfg, s.id, read_json(seg_paths(cfg, s.id, 0)["json"])[variant])
            axes = tuple("zyx".index(r["axis"]) for r in rows_lbm)
            try:
                whole2_pnm = network_permeability(w2, 2 * s.voxel_mm * 1e-3, axes=axes)
            except Exception as e:                  # noqa: BLE001
                whole2_pnm = {f"{t}_error": f"{type(e).__name__}: {e}"[:300] for t in TAGS}
            del w2
        for r in rows_lbm:
            e = int(r["edge"])
            if r["domain"] == "full VOI":
                dom = pore
            else:
                c = [(n - e) // 2 for n in pore.shape]
                dom = np.ascontiguousarray(pore[c[0]:c[0] + e, c[1]:c[1] + e, c[2]:c[2] + e])
            ax = "zyx".index(r["axis"])
            phi = float(dom.mean())
            sv = crofton_sv(dom, s.voxel_mm)["S_V"] * 1e3                     # 1/m
            span = spanning_mask(dom, ax)
            phi_s = float(span.mean())
            sv_s = crofton_sv(span, s.voxel_mm)["S_V"] * 1e3 if span.any() else np.nan
            del span
            dv = critical_diameter_vox(dom, ax) if r.get("crop_percolates", True) else np.nan
            F = r.get("F", np.nan)
            k_kc = phi ** 3 / (5 * sv ** 2) if sv > 0 else np.nan
            k_kcs = phi_s ** 3 / (5 * sv_s ** 2) if np.isfinite(sv_s) and sv_s > 0 else np.nan
            ok_dc = bool(np.isfinite(dv) and dv >= D_C_MIN_VOX)
            k_kt = (dv * s.voxel_mm * 1e-3) ** 2 / (226 * F) if ok_dc and np.isfinite(F) else np.nan
            row = {"id": s.id, "group": s.group, "variant": variant, "axis": r["axis"], "domain": r["domain"],
                   "edge": e, "phi": phi, "S_V_per_m": sv, "phi_span": phi_s, "S_V_span_per_m": sv_s,
                   "d_c_vox": dv, "d_c_resolved": ok_dc, "F": F,
                   "k_KC_m2": k_kc, "k_KC_span_m2": k_kcs, "k_KT_m2": k_kt,
                   "k_KC_mD": k_kc / MD, "k_KC_span_mD": k_kcs / MD, "k_KT_mD": k_kt / MD}
            if whole2_pnm is not None:
                for t in TAGS:
                    row[f"k_{t}_whole2_mD"] = whole2_pnm.get(f"k_{t}_{r['axis']}_m2", np.nan) / MD
                    row[f"{t}_whole2_error"] = whole2_pnm.get(f"{t}_error", "")
            rows.append(row)
            log(f"  {s.id} {r['axis']} {variant} ({r['domain']}): KC {k_kc / MD:.4g} mD, KC span {k_kcs / MD:.4g} mD, "
                f"d_c {dv:.1f} vox, KT {k_kt / MD:.4g} mD"
                + (f" | whole/2 PNM {row['k_pnm_whole2_mD']:.4g} / {row['k_pnmeq_whole2_mD']:.4g} mD"
                   if whole2_pnm is not None else ""))
        write_json(ldir / f"{s.id}_estimates{sfx}.json", rows)
        del pore
    rows = []
    for s in cfg.samples:
        f = ldir / f"{s.id}_estimates{sfx}.json"
        if f.exists():
            rows.extend(read_json(f))
    if rows:
        pd.DataFrame(rows).to_csv(ldir / f"estimates{sfx}.csv", index=False)


def run(cfg: Config, only=None, log=print) -> None:
    for v in cfg.raw.get("air_variants", ["T50"]):
        run_variant(cfg, v, only=only, log=log)
