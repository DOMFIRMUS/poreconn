"""Stage `validation`: the transport solvers on synthetic media (no micro-CT data needed).

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Two synthetic media, generated with PoreSpy and a fixed seed:
  * simple-cubic lattice of solid spheres (radius 12 voxels, spacing 31, centres at 15 + 31 i, so that
    every VOI face is a mirror plane of the lattice and the mirror boundaries of the solvers reproduce
    the infinite lattice exactly); 3 cells, 93^3 voxels;
  * overlapping spheres (radius 10 voxels, porosity 0.40, 120^3 voxels).
On each: lattice Boltzmann (reference; also with Lambda = 1/4 and with tau+ = 0.55), pore networks with
inscribed and with equivalent sizes, each with mirror and with wall faces, the pore network with SNOW-2
boundary pores (to document why they are not used), formation factor, and the Kozeny-Carman estimate.

Window study: the same estimators on windows of increasing size. For the lattice (2 to 4 cells) the
exact answer does not depend on the window, so any change measures the treatment of the faces; the
LBM reference is the one-cell domain, which with mirror boundaries is the infinite lattice. For the
overlapping spheres, centred windows of one 128^3 realization, each compared with the LBM on the same
window. Permeabilities in voxel^2 units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, write_json
from .fd import formation_factor
from .lbm import permeability
from .morph import crofton_sv
from .pnm import TAG, network_permeability
from .topology import critical_diameter_vox

TAGS = [TAG[s, f] for f in ("mirror", "wall") for s in ("inscribed", "equivalent")]


def _pnm_with_boundary_pores(pore: np.ndarray, axis: int = 0) -> float:
    """SNOW-2 with boundary pores (boundary_width=3) and inscribed sizes, flow between the boundary pores."""
    import openpnm as op
    import porespy as ps
    from .pnm import DROP
    ps.settings.tqdm["disable"] = True
    snow = ps.networks.snow2(pore, boundary_width=3, voxel_size=1.0, parallelization=None)
    pn = op.io.network_from_porespy(snow.network)
    pn["pore.diameter"] = pn["pore.inscribed_diameter"]
    dmin = pn["pore.diameter"][pn.conns].min(axis=1)
    pn["throat.diameter"] = np.minimum(pn["throat.inscribed_diameter"], 0.99 * dmin)
    geo = {k: v for k, v in op.models.collections.geometry.spheres_and_cylinders.items() if k not in DROP}
    pn.add_model_collection(geo)
    pn.regenerate_models()
    w = op.phase.Phase(network=pn)
    w["pore.viscosity"] = 1.0
    w.add_model("throat.hydraulic_conductance", op.models.physics.hydraulic_conductance.generic_hydraulic)
    w.regenerate_models()
    sf = op.algorithms.StokesFlow(network=pn, phase=w)
    lo, hi = ("xmin", "xmax")
    sf.set_value_BC(pores=pn.pores(lo), values=1.0)
    sf.set_value_BC(pores=pn.pores(hi), values=0.0)
    sf.run()
    Q = float(abs(sf.rate(pores=pn.pores(lo), mode="group")[0]))
    L = pore.shape[axis]
    A = float(np.prod(np.delete(pore.shape, axis)))
    return Q * L / A


def _pnm(pore: np.ndarray) -> dict:
    """The four network variants along axis 0, in voxel^2 (viscosity cancels: k = Q mu L / A dP)."""
    r = network_permeability(pore, 1.0, axes=(0,))
    return {f"k_{t}": r[f"k_{t}_z_m2"] for t in TAGS}


def lattice(cells: int, r: int = 12, spacing: int = 31) -> np.ndarray:
    """Pore space of a simple-cubic lattice of solid spheres whose faces are mirror planes."""
    n = cells * spacing
    c = np.arange(spacing // 2, n, spacing)
    zz, yy, xx = np.ogrid[:n, :n, :n]
    d2 = np.full((n, n, n), np.inf)
    for z0 in c:
        for y0 in c:
            for x0 in c:
                d2 = np.minimum(d2, (zz - z0) ** 2 + (yy - y0) ** 2 + (xx - x0) ** 2)
    return np.ascontiguousarray(d2 > r * r)


def window_study(log=print, quick: bool = False) -> list:
    import porespy as ps
    rows = []
    ref = permeability(lattice(1), 0, tol=1e-8, maxiter=80000)
    for cells in ((2, 3) if quick else (2, 3, 4)):
        pore = lattice(cells)
        r = {"medium": "simple_cubic_spheres", "window_vox": pore.shape[0], "n_cells": cells,
             "phi": float(pore.mean()), "k_lbm": ref["k_lu"], "lbm_ok": bool(ref.get("ok", ref["converged"])),
             "k_lbm_window": np.nan}
        if cells <= 3:
            lb = permeability(pore, 0, tol=1e-8, maxiter=80000)
            r["k_lbm_window"] = lb["k_lu"]
        r.update(_pnm(pore))
        rows.append(r)
    np.random.seed(2)
    big = np.ascontiguousarray(ps.generators.overlapping_spheres([128] * 3, r=10, porosity=0.4).astype(bool))
    for w in ((48, 64) if quick else (48, 64, 96, 128)):
        o = (128 - w) // 2
        pore = np.ascontiguousarray(big[o:o + w, o:o + w, o:o + w])
        lb = permeability(pore, 0, tol=1e-7, maxiter=80000)
        r = {"medium": "overlapping_spheres", "window_vox": w, "n_cells": np.nan, "phi": float(pore.mean()),
             "k_lbm": lb["k_lu"], "lbm_ok": bool(lb.get("ok", lb["converged"])), "k_lbm_window": lb["k_lu"]}
        r.update(_pnm(pore))
        rows.append(r)
    for r in rows:
        for t in TAGS:
            r[f"ratio_k_{t}_lbm"] = r[f"k_{t}"] / r["k_lbm"] if r["k_lbm"] > 0 else np.nan
        log(f"  window {r['medium']} {r['window_vox']}: k_lbm {r['k_lbm']:.4g} "
            + " ".join(f"{t}/lbm {r[f'ratio_k_{t}_lbm']:.3f}" for t in TAGS))
    return rows


def kt_synthetic(cfg: Config, log=print) -> None:
    """Katz-Thompson on the two synthetic media of run() (same generators and seed), using k_LBM and F
    from synthetic.csv; adds d_c_vox, k_KT, ratio_k_KT_lbm and the implied prefactor to synthetic.csv."""
    import porespy as ps
    ps.settings.tqdm["disable"] = True
    out = cfg.outdir("validation")
    syn = pd.read_csv(out / "synthetic.csv")
    np.random.seed(1)
    media = {"simple_cubic_spheres": lattice(3),
             "overlapping_spheres": ps.generators.overlapping_spheres([120, 120, 120], r=10, porosity=0.4)}
    for name, pore in media.items():
        pore = np.ascontiguousarray(pore.astype(bool))
        i = syn.index[syn.medium == name][0]
        if abs(float(pore.mean()) - syn.loc[i, "phi"]) > 1e-9:
            raise RuntimeError(f"{name}: regenerated medium differs from synthetic.csv")
        dc = critical_diameter_vox(pore, 0)
        F = syn.loc[i, "F"]
        syn.loc[i, "d_c_vox"] = dc
        syn.loc[i, "k_KT"] = dc ** 2 / (226 * F)
        syn.loc[i, "ratio_k_KT_lbm"] = syn.loc[i, "k_KT"] / syn.loc[i, "k_lbm"]
        syn.loc[i, "KT_prefactor_implied"] = syn.loc[i, "k_lbm"] * F / dc ** 2
        log(f"  {name}: d_c {dc:.1f} vox, KT/LBM {syn.loc[i, 'ratio_k_KT_lbm']:.3f}, "
            f"implied prefactor 1/{1 / syn.loc[i, 'KT_prefactor_implied']:.1f}")
    syn.to_csv(out / "synthetic.csv", index=False)
    write_json(out / "synthetic.json", syn.to_dict(orient="records"))


def run(cfg: Config, only=None, log=print) -> None:
    import porespy as ps
    ps.settings.tqdm["disable"] = True
    out = cfg.outdir("validation")
    np.random.seed(1)
    media = {
        "simple_cubic_spheres": lattice(3),
        "overlapping_spheres": ps.generators.overlapping_spheres([120, 120, 120], r=10, porosity=0.4),
    }
    rows = []
    for name, pore in media.items():
        pore = np.ascontiguousarray(pore.astype(bool))
        lb = permeability(pore, 0, tol=1e-7, maxiter=80000)
        lb_l4 = permeability(pore, 0, tol=1e-7, maxiter=80000, lam=0.25)
        lb_t = permeability(pore, 0, tol=1e-7, maxiter=80000, tau=0.55)
        pn = _pnm(pore)
        ff = formation_factor(pore, 0, tol=1e-8)
        sv = crofton_sv(pore, 1.0)["S_V"]
        phi = float(pore.mean())
        dc = critical_diameter_vox(pore, 0)
        r = {"medium": name, "phi": phi, "k_lbm": lb["k_lu"], "lbm_iterations": lb["iterations"],
             "lbm_converged": lb["converged"], "lbm_ok": bool(lb.get("ok", lb["converged"])),
             "k_lbm_lambda_1_4": lb_l4["k_lu"], "k_lbm_tau_0_55": lb_t["k_lu"], **pn,
             "k_pnm_boundary_pores": _pnm_with_boundary_pores(pore), "F": ff["F"], "S_V": sv,
             "k_KC": phi ** 3 / (5 * sv ** 2), "d_c_vox": dc, "k_KT": dc ** 2 / (226 * ff["F"])}
        r["KT_prefactor_implied"] = r["k_lbm"] * ff["F"] / dc ** 2     # c in k = c d_c^2 / F (KT: 1/226)
        for kname in [f"k_{t}" for t in TAGS] + ["k_pnm_boundary_pores", "k_KC", "k_KT", "k_lbm_lambda_1_4",
                                                  "k_lbm_tau_0_55"]:
            r[f"ratio_{kname}_lbm"] = r[kname] / r["k_lbm"]
        rows.append(r)
        log(f"  {name}: phi {phi:.3f} k_lbm {r['k_lbm']:.4g} "
            + " ".join(f"{t}/lbm {r[f'ratio_k_{t}_lbm']:.2f}" for t in TAGS)
            + f" boundary-pores/lbm {r['ratio_k_pnm_boundary_pores_lbm']:.3g} KC/lbm {r['ratio_k_KC_lbm']:.2f} F {r['F']:.2f}")
    pd.DataFrame(rows).to_csv(out / "synthetic.csv", index=False)
    write_json(out / "synthetic.json", rows)
    ws = window_study(log, quick=cfg.quick)
    pd.DataFrame(ws).to_csv(out / "window_study.csv", index=False)
    write_json(out / "window_study.json", ws)
