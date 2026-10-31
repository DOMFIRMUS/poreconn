"""Stage `lbm`: single-phase permeability by a two-relaxation-time lattice Boltzmann solver (D3Q19).

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Stokes flow is driven by a uniform body force g along the chosen axis. Only the pore voxels of the
clusters spanning that axis are simulated (sparse storage, pull streaming); half-way bounce-back is
applied at every pore-solid link. The pore phase is 6-connected, as in the labelling and the other
solvers: a diagonal link is open only when one of the two face neighbours it passes between is pore,
so no flow crosses an edge-only voxel contact. The lateral faces of the VOI are symmetry planes
(specular reflection, i.e. the geometry continued by its mirror image), the same no-flux condition used
by the diffusion and pore-network solvers; closed no-slip side walls (`lateral="wall"`) would add a
spurious wall drag in domains only a few grains wide. To make the flow axis periodic without
inlet/outlet artefacts, the sample is mirrored along that axis (the mirrored copy has the same
resistance, so the series pair has the permeability of the sample).
Linear equilibrium (exact Stokes limit), Guo forcing and TRT collision with the "magic" parameter
Lambda = (tau+ - 1/2)(tau- - 1/2) = 3/16, for which the bounce-back wall sits half-way between nodes
and the permeability does not depend on viscosity:

    k = nu q / g   (lattice units; x dx^2 for physical units)

with q the superficial velocity averaged over the whole (mirrored) domain. A run is accepted (`ok`)
when the relative change of q has fallen below `tol` and the mass flux through the cross-sections of
the sample is uniform (flux_cv <= flux_tol), as it must be at steady state.
The legacy solver of the dissertation returned k = 0 for every sample; this one is tested against the
analytical solutions for a square duct and a slit (tests/test_lbm.py).

The stage computes LBM, pore-network (pnm.network_permeability) and diffusion (fd.formation_factor)
permeability and formation factor on the same domain, for every axis along which the main VOI
percolates. The domain is the whole main VOI when its mirrored spanning pore space has at most
transport.lbm_max_nodes nodes, otherwise the central cube of edge `transport.lbm_edge` (memory limit of
the sparse solver; run_variant(full=True) forces the whole VOI). The
PNM faces are symmetry planes, like the LBM and FD ones (tags pnm, pnmeq); the wall-face variant (pnmw,
pnmeqw) is kept for comparison. For a central-cube domain (the bead packs, whose 150^3 cube spans only
about two bead diameters) the LBM is also run on the whole VOI block-averaged by 2 and re-thresholded
at the same threshold (as in the coarsen stage), a check of the domain size. Every air reference in
`air_variants` is simulated (the T50int runs are written with the suffix _T50int); lambda_sweep() reruns
the rock cases with several values of Lambda.
"""
from __future__ import annotations

import time

import numba as nb
import numpy as np
import pandas as pd

from .config import Config, read_json, write_json
from .morph import block_mean, label6, spanning_mask

MD = 9.869233e-16   # m^2 per millidarcy
TAGS = ("pnm", "pnmeq", "pnmw", "pnmeqw")
MAX_NODES = 3.0e6   # largest mirrored spanning pore space simulated on the whole VOI

C = np.array([[0, 0, 0],
              [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
              [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
              [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
              [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1]], dtype=np.int64)
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=np.int64)


@nb.njit(cache=True)
def _fluid(index, a, b, c, mirror_lateral):
    """Is node (a, b, c) fluid, with axis 0 periodic and the lateral faces mirrors (or walls)?"""
    n0, n1, n2 = index.shape
    if b < 0 or b >= n1:
        if not mirror_lateral:
            return False
        b = -b - 1 if b < 0 else 2 * n1 - b - 1
    if c < 0 or c >= n2:
        if not mirror_lateral:
            return False
        c = -c - 1 if c < 0 else 2 * n2 - c - 1
    return index[a % n0, b, c] >= 0


@nb.njit(cache=True)
def _neighbours(index, C, OPP, R1, R2, mirror_lateral):
    """Pull table: population q of node i comes from population srcq[q, i] of node src[q, i].

    Axis 0 is periodic. Across a lateral face the source is reflected (specular reflection, R1/R2
    flip the velocity component normal to the face) when mirror_lateral is True; a solid source, or
    a lateral face with mirror_lateral False, gives half-way bounce-back (src = i, srcq = OPP[q]).
    The pore phase is 6-connected: a diagonal link is open only if one of the two nodes displaced
    along a single axis of the link is fluid, so no flow passes through edge-only voxel contacts.
    """
    n0, n1, n2 = index.shape
    n = int(index.max()) + 1
    src = np.empty((19, n), np.int32)
    srcq = np.empty((19, n), np.int8)
    for a in range(n0):
        for b in range(n1):
            for c in range(n2):
                i = index[a, b, c]
                if i < 0:
                    continue
                for q in range(19):
                    aa = (a - C[q, 0]) % n0
                    bb = b - C[q, 1]
                    cc = c - C[q, 2]
                    qq = q
                    wall = False
                    if bb < 0 or bb >= n1:
                        if mirror_lateral:
                            bb = -bb - 1 if bb < 0 else 2 * n1 - bb - 1
                            qq = R1[qq]
                        else:
                            wall = True
                    if cc < 0 or cc >= n2:
                        if mirror_lateral:
                            cc = -cc - 1 if cc < 0 else 2 * n2 - cc - 1
                            qq = R2[qq]
                        else:
                            wall = True
                    j = -1 if wall else index[aa, bb, cc]
                    if j >= 0 and q >= 7:
                        # diagonal link: the two intermediate nodes of the 2 x 2 square it crosses
                        open_ = False
                        if C[q, 0] != 0 and _fluid(index, a - C[q, 0], b, c, mirror_lateral):
                            open_ = True
                        if C[q, 1] != 0 and _fluid(index, a, b - C[q, 1], c, mirror_lateral):
                            open_ = True
                        if C[q, 2] != 0 and _fluid(index, a, b, c - C[q, 2], mirror_lateral):
                            open_ = True
                        if not open_:
                            j = -1
                    if j >= 0:
                        src[q, i] = j
                        srcq[q, i] = qq
                    else:
                        src[q, i] = i
                        srcq[q, i] = OPP[q]
    return src, srcq


def _reflections():
    r1 = np.zeros(19, np.int64)
    r2 = np.zeros(19, np.int64)
    for q in range(19):
        for t, ax in ((r1, 1), (r2, 2)):
            v = C[q].copy()
            v[ax] = -v[ax]
            t[q] = int(np.where((C == v).all(axis=1))[0][0])
    return r1, r2


R1, R2 = _reflections()


@nb.njit(parallel=True, cache=True)
def _step(fa, fb, src, srcq, C, W, OPP, wp, wm, g):
    n = fa.shape[1]
    for i in nb.prange(n):
        f = np.empty(19)
        for q in range(19):
            f[q] = fa[srcq[q, i], src[q, i]]
        rho = 0.0
        ux = 0.0
        for q in range(19):
            rho += f[q]
            ux += f[q] * C[q, 0]
        u0 = ux + 0.5 * g
        uy = 0.0
        uz = 0.0
        for q in range(19):
            uy += f[q] * C[q, 1]
            uz += f[q] * C[q, 2]
        for q in range(19):
            p = OPP[q]
            cu = C[q, 0] * u0 + C[q, 1] * uy + C[q, 2] * uz
            cup = C[p, 0] * u0 + C[p, 1] * uy + C[p, 2] * uz
            eq = W[q] * (rho + 3.0 * cu)
            eqp = W[p] * (rho + 3.0 * cup)
            fp = 0.5 * (f[q] + f[p])
            fm = 0.5 * (f[q] - f[p])
            ep = 0.5 * (eq + eqp)
            em = 0.5 * (eq - eqp)
            # Guo forcing: equilibrium with u + g/2 and source scaled by (1 - omega-/2)
            fb[q, i] = f[q] - wp * (fp - ep) - wm * (fm - em) + (1.0 - 0.5 * wm) * 3.0 * W[q] * C[q, 0] * g


@nb.njit(parallel=True, cache=True)
def _mean_u0(fa, C, g, sel):
    # fa holds post-collision populations, whose momentum exceeds the physical one by g/2
    s = 0.0
    for i in nb.prange(fa.shape[1]):
        if sel[i]:
            u = 0.0
            for q in range(19):
                u += fa[q, i] * C[q, 0]
            s += u - 0.5 * g
    return s


@nb.njit(parallel=True, cache=True)
def _plane_flux(fa, src, srcq, C, off):
    """Populations streaming into each slice from the previous (plus) and the next (minus) slice.

    Nodes are numbered slice by slice, slice a holding nodes off[a] to off[a + 1] - 1; bounced-back
    and laterally reflected populations stay in their slice and are not counted.
    """
    L = off.shape[0] - 1
    plus = np.zeros(L)
    minus = np.zeros(L)
    for a in nb.prange(L):
        sp = 0.0
        sm = 0.0
        for i in range(off[a], off[a + 1]):
            for q in range(1, 19):
                if C[q, 0] == 0:
                    continue
                j = src[q, i]
                if j >= off[a] and j < off[a + 1]:
                    continue
                if C[q, 0] > 0:
                    sp += fa[srcq[q, i], j]
                else:
                    sm += fa[srcq[q, i], j]
        plus[a] = sp
        minus[a] = sm
    return plus, minus


def permeability(pore: np.ndarray, axis: int, dx: float = 1.0, tau: float = 1.0, mirror: bool = True,
                 lateral: str = "mirror", g: float = 1e-4, tol: float = 1e-6, maxiter: int = 40000,
                 check: int = 250, span_only: bool = True, lam: float = 3.0 / 16.0, flux_tol: float = 1e-3,
                 log=None) -> dict:
    """Permeability of `pore` along `axis` (0 = z, 1 = y, 2 = x); returns k in units of dx^2.

    `converged`: relative change of q over `check` steps below `tol` before `maxiter` (maxiter is never
    extended). `flux_cv`: std / |mean| of the mass flux (populations streamed across the plane) through
    the cross-sections between consecutive slices of the non-mirrored half; mass conservation makes them
    equal at steady state. Straight ducts and slits give < 1e-12, converged runs on random sphere packs
    and on 96^3 rock sub-volumes 1e-9 to 3e-5, runs stopped after a few hundred steps 6e-4 to 1e-2;
    hence flux_tol = 1e-3.
    `ok` = converged and flux_cv <= flux_tol.
    """
    p = np.ascontiguousarray(np.moveaxis(pore, axis, 0))
    m = spanning_mask(p, 0) if span_only else p.copy()
    out = {"axis": "zyx"[axis], "phi": float(pore.mean()), "phi_span": float(m.mean())}
    if not m.any():
        out.update(k=0.0, k_lu=0.0, iterations=0, converged=True, rel_change=0.0, flux_cv=0.0,
                   flux_tol=flux_tol, ok=True)
        return out
    geom = np.concatenate([m, m[::-1]], axis=0) if mirror else m
    L = geom.shape[0]
    index = np.full(geom.shape, -1, np.int32)
    index[geom] = np.arange(int(geom.sum()), dtype=np.int32)
    src, srcq = _neighbours(index, C, OPP, R1, R2, lateral == "mirror")
    n = int(geom.sum())
    off = np.concatenate([[0], np.cumsum(geom.sum(axis=(1, 2)))]).astype(np.int64)
    sample_sel = np.ones(n, np.bool_)
    del index
    tp = tau
    tm = 0.5 + lam / (tp - 0.5)
    nu = (tp - 0.5) / 3.0
    fa = np.repeat(W[:, None], n, axis=1).astype(np.float64)
    fb = np.empty_like(fa)
    A = p.shape[1] * p.shape[2]
    q_old = 0.0
    t0 = time.time()
    it = 0
    rel = np.inf
    converged = False
    while it < maxiter:
        for _ in range(check):
            _step(fa, fb, src, srcq, C, W, OPP, 1.0 / tp, 1.0 / tm, g)
            fa, fb = fb, fa
        it += check
        q = _mean_u0(fa, C, g, sample_sel) / (A * L)
        rel = abs(q - q_old) / max(abs(q), 1e-300)
        q_old = q
        if log and it % (check * 20) == 0:
            log(f"    lbm it {it} q {q:.4e} rel {rel:.2e} ({time.time() - t0:.0f} s)")
        if rel < tol:
            converged = True
            break
    k_lu = nu * q / g
    plus, minus = _plane_flux(fa, src, srcq, C, off)
    flux = plus - np.roll(minus, 1)                 # through the plane between slices a - 1 and a
    sec = flux[1:m.shape[0]] if m.shape[0] > 1 else flux
    flux_cv = float(np.std(sec) / abs(np.mean(sec))) if np.mean(sec) != 0 else np.inf
    out.update(k_lu=float(k_lu), k=float(k_lu * dx * dx), iterations=it, converged=converged,
               rel_change=float(rel), flux_cv=flux_cv, flux_tol=flux_tol,
               ok=bool(converged and flux_cv <= flux_tol), n_fluid=n, seconds=time.time() - t0,
               tau_plus=tp, tau_minus=tm, mirrored=bool(mirror), lateral=lateral, Lambda=lam)
    return out


def _coarse_whole(cfg: Config, sid: str, threshold: float) -> np.ndarray:
    """Main VOI block-averaged by 2 and re-thresholded at the given absolute threshold (as in coarsen)."""
    from .segment import load_filtered
    return block_mean(load_filtered(cfg, sid, 0), 2) < threshold


def _suffix(variant: str, full: bool = False) -> str:
    return ("" if variant == "T50" else f"_{variant}") + ("_full" if full else "")


def run_variant(cfg: Config, variant: str = "T50", only=None, log=print, full: bool = False) -> None:
    """LBM, PNM and FD on the same domain, for every axis along which the main VOI percolates.

    `variant` is the air reference (T50: stored ISO-50 mask; T50int: the median-filtered VOI
    re-thresholded with the internal-air reference; samples without it are skipped). The domain is the
    whole main VOI when its mirrored spanning pore space has at most transport.lbm_max_nodes nodes (or
    always, with `full`), otherwise the central cube of edge `transport.lbm_edge` (memory limit of the
    sparse solver). An axis along which that cube does not percolate still gets a row, with every k NaN
    (crop_percolates False). For a cube domain the LBM is repeated on the whole VOI binned by 2
    (k_lbm_whole2_*). PNM: the four variants of pnm.network_permeability (mirror faces pnm, pnmeq; wall
    faces pnmw, pnmeqw); a PNM failure gives NaN and <tag>_error. Results go to <id>_lbm<suffix>.json
    and lbm<suffix>.csv is rebuilt from every such file present (suffix '' for T50, '_<variant>'
    otherwise, plus '_full' for forced whole-VOI runs).

    tau+ only sets how fast the run converges (with Lambda fixed the TRT steady state does not depend on
    it): transport.lbm_tau for the bead packs, whose wide pores need a large viscosity to develop the
    flow quickly, and transport.lbm_tau_rock for the rocks (every domain), where the slow part is the
    equilibration of pressure along the domain, about (L^2 / 4 pi^2) nu phi / (k c_s^2) steps.
    """
    from .fd import formation_factor
    from .pnm import network_permeability
    from .rev import variant_pore
    from .segment import seg_paths
    tr = cfg["transport"]
    out_dir = cfg.outdir("lbm")
    edge = tr["lbm_edge"] if not cfg.quick else 40
    max_nodes = float(tr.get("lbm_max_nodes", MAX_NODES))
    lbm_kw = {"tol": tr["lbm_tol"], "maxiter": tr["lbm_maxiter"]}
    sfx = _suffix(variant, full)
    for s in cfg.samples:
        if only and s.id not in only:
            continue
        pore = variant_pore(cfg, s.id, 0, variant)
        if pore is None:
            continue
        lab, n = label6(pore)
        spans = [spanning_mask(pore, a, lab, n) for a in range(3)]
        del lab
        axes = [a for a in range(3) if spans[a].any()]
        n_span = max(int(m.sum()) for m in spans)
        del spans
        rows = []
        if not axes:
            log(f"  {s.id} {variant}: no percolating axis")
            write_json(out_dir / f"{s.id}_lbm{sfx}.json", rows)
            continue
        whole2 = None
        if full or 2 * n_span <= max_nodes:
            dom, where = pore, "full VOI"
        else:
            c = [(e - edge) // 2 for e in pore.shape]
            dom = np.ascontiguousarray(pore[c[0]:c[0] + edge, c[1]:c[1] + edge, c[2]:c[2] + edge])
            where = f"central {edge}^3"
            whole2 = _coarse_whole(cfg, s.id, read_json(seg_paths(cfg, s.id, 0)["json"])[variant])
        try:
            pnm = network_permeability(dom, s.voxel_mm * 1e-3, axes=tuple(axes))
        except Exception as e:                      # a PNM failure must not abort the sample
            pnm = {f"{t}_error": f"{type(e).__name__}: {e}"[:300] for t in TAGS}
        tau_p = tr["lbm_tau"] if s.group == "granular" else tr.get("lbm_tau_rock", tr["lbm_tau"])
        for ax in axes:
            a = "zyx"[ax]
            r = permeability(dom, ax, dx=s.voxel_mm, tau=tau_p, **lbm_kw)
            ff = formation_factor(dom, ax, tol=tr["fd_tol"], maxiter=tr["fd_maxiter"])
            crop_perc = r["phi_span"] > 0
            r.update(id=s.id, group=s.group, variant=variant, domain=where, edge=dom.shape[0], main_percolates=True,
                     crop_percolates=crop_perc, F=ff["F"], tau=ff["tau"], fd_iterations=ff["iterations"],
                     fd_residual=ff["residual"], fd_flux_balance=ff["flux_balance"],
                     fd_converged=bool(ff["iterations"] < tr["fd_maxiter"] or ff["residual"] <= tr["fd_tol"]),
                     **{f"k_{t}_m2": pnm.get(f"k_{t}_{a}_m2", np.nan) for t in TAGS})
            if not crop_perc:                       # the cube says nothing about the sample along this axis
                r.update(k=np.nan, k_lu=np.nan, converged=False, rel_change=np.nan, flux_cv=np.nan, ok=False,
                         F=np.nan, tau=np.nan, fd_converged=False, **{f"k_{t}_m2": np.nan for t in TAGS})
            r.update(k_lbm_m2=r["k"] * 1e-6, k_lbm_mD=r["k"] * 1e-6 / MD)
            for t in TAGS:
                r[f"k_{t}_mD"] = r[f"k_{t}_m2"] / MD
                r[f"{t}_error"] = pnm.get(f"{t}_error", "")
            r.update({k: v for k, v in pnm.items() if not k.startswith("k_") and not k.endswith("_error")})
            if whole2 is not None:
                w = permeability(whole2, ax, dx=2 * s.voxel_mm, tau=tau_p, **lbm_kw)
                r.update(k_lbm_whole2_m2=w["k"] * 1e-6, k_lbm_whole2_mD=w["k"] * 1e-6 / MD,
                         whole2_phi_span=w["phi_span"], whole2_iterations=w["iterations"],
                         whole2_converged=w["converged"], whole2_flux_cv=w["flux_cv"],
                         whole2_ok=bool(w["ok"] and w["phi_span"] > 0), whole2_voxel_mm=2 * s.voxel_mm)
                if w["phi_span"] == 0:                  # coarsened pore space does not span this axis
                    r.update(k_lbm_whole2_m2=np.nan, k_lbm_whole2_mD=np.nan)
            else:
                r.update(dict.fromkeys(("k_lbm_whole2_m2", "k_lbm_whole2_mD", "whole2_phi_span", "whole2_iterations",
                                        "whole2_converged", "whole2_flux_cv", "whole2_ok", "whole2_voxel_mm"), np.nan))
            rows.append(r)
            log(f"  {s.id} {a} {variant} ({where}): phi_span {100 * r['phi_span']:.2f}% k_lbm {r['k_lbm_mD']:.4g} mD "
                f"(it {r['iterations']}, ok {r['ok']}, flux_cv {r['flux_cv']:.1e}) | "
                + " ".join(f"{t} {r[f'k_{t}_mD']:.4g}" for t in TAGS)
                + f" | F {r['F']:.2f} (fd conv {r['fd_converged']})"
                + (f" | whole/2 {r['k_lbm_whole2_mD']:.4g} mD (ok {r['whole2_ok']})" if whole2 is not None else ""))
        for t in TAGS:
            if rows[0][f"{t}_error"]:
                log(f"  {s.id}: {t} failed: {rows[0][f'{t}_error']}")
        write_json(out_dir / f"{s.id}_lbm{sfx}.json", rows)
        del pore, dom, whole2
    rows = []
    for s in cfg.samples:
        f = out_dir / f"{s.id}_lbm{sfx}.json"
        if f.exists():
            rows.extend(read_json(f))
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / f"lbm{sfx}.csv", index=False)


def lambda_sweep(cfg: Config, only=None, log=print, lams=(1 / 12, 3 / 16, 1 / 4, 1 / 2)) -> None:
    """Dependence of k_LBM on the TRT parameter Lambda for the rock cases (same domain as the lbm stage).

    The rock throats are a few voxels wide, the regime in which TRT permeabilities depend most on
    Lambda; every percolating axis of the rocks' main VOI (T50) is rerun with each Lambda in `lams`
    (tau+ = transport.lbm_tau_rock) and lbm_lambda.csv is written.
    """
    from .segment import load_pore
    tr = cfg["transport"]
    out_dir = cfg.outdir("lbm")
    edge = tr["lbm_edge"]
    max_nodes = float(tr.get("lbm_max_nodes", MAX_NODES))
    rows = []
    for s in cfg.samples:
        if (only and s.id not in only) or s.group == "granular":
            continue
        pore = load_pore(cfg, s.id, 0)
        spans = [spanning_mask(pore, a) for a in range(3)]
        axes = [a for a in range(3) if spans[a].any()]
        n_span = max(int(m.sum()) for m in spans)
        del spans
        if not axes:
            continue
        if 2 * n_span <= max_nodes:
            dom, where = pore, "full VOI"
        else:
            c = [(e - edge) // 2 for e in pore.shape]
            dom, where = np.ascontiguousarray(pore[c[0]:c[0] + edge, c[1]:c[1] + edge, c[2]:c[2] + edge]), f"central {edge}^3"
        for ax in axes:
            for lam in lams:
                r = permeability(dom, ax, dx=s.voxel_mm, tau=tr.get("lbm_tau_rock", tr["lbm_tau"]), lam=lam,
                                 tol=tr["lbm_tol"], maxiter=tr["lbm_maxiter"])
                rows.append({"id": s.id, "axis": "zyx"[ax], "domain": where, "Lambda": lam, "k_lbm_mD": r["k"] * 1e-6 / MD,
                             "iterations": r["iterations"], "ok": r["ok"], "flux_cv": r["flux_cv"]})
                log(f"  {s.id} {'zyx'[ax]} Lambda {lam:.4f}: k {rows[-1]['k_lbm_mD']:.4g} mD (it {r['iterations']}, ok {r['ok']})")
        del pore, dom
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "lbm_lambda.csv", index=False)


def run(cfg: Config, only=None, log=print) -> None:
    """Stage entry: run_variant for every air reference in `air_variants` (bead packs at T50 only)."""
    for v in cfg.raw.get("air_variants", ["T50"]):
        run_variant(cfg, v, only=only, log=log)
