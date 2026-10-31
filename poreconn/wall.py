"""Stage `wall`: radial porosity profile of the glass-bead columns (wall effect).

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

The laboratory porosity of a packed column refers to the whole column, including the loosely packed
layer next to the wall, whereas the VOIs are central cubes. To test whether the wall effect explains the
difference, full cross-sections of each column are read over the z-range of the main VOI, the inner wall
of the acrylic tube is located, and the porosity is measured as a function of the distance from the wall
with the same median filter and the same ISO-50 threshold as the VOI. The whole-section porosity
(area-weighted over the disc inside the tube) is then compared with the laboratory values.

Grid: the slice range and the binning come from the meta file of the main VOI (slice_index_start,
shape, bin), the section is binned with the same phase as the VOI crop (from crop_y0 and crop_x0 modulo
the bin) and rounded to uint16 as in extract.read_voi, so that inside the VOI the voxels are those of
the VOI. Binned pixel j covers raw pixels o + b j ... o + b j + b - 1 and is centred at o + b j + (b-1)/2,
which is used to place the column axis (VOI meta centre_y, centre_x, raw pixels) on the binned grid.

Inner wall: the acrylic has its own grey level, between those of air and glass, and a small spread.
Beads touch the wall only at points, so the packing ends in a thin layer that is almost all air: the
azimuthal mean grey value falls to near the air level just inside the wall and rises in one step to the
acrylic plateau, while the grey-level spread already drops inside that air layer. The spread is used
only for a first estimate (start of the first annulus, beyond 0.7 times the nominal radius and at least
1 mm wide, where the spread stays below a fifth of its median in the packing). The wall is then placed,
in each of 36 angular sectors, at the half-way crossing between the near-wall minimum and the acrylic
plateau of the mean grey value (0.25-voxel radial bins, linear interpolation). A two-harmonic series
r_w(t) = c0 + c1 cos t + s1 sin t + c2 cos 2t + s2 sin 2t fitted to the sector edges corrects the axis
position (first harmonic; the detection is repeated once about the corrected axis) and describes a slight
ovality (second harmonic). The same fit on three slabs along z gives the slopes of the coefficients
(tilt of the tube relative to the scan axis, taper); the wall of each slice is the fit over the whole
range shifted by these slopes, and the distance from the wall of a pixel at (r, t) is r_w(t, z) - r. The
mean inner diameter 2 c0 is reported next to the tube's nominal inner diameter (Silva 2024, Table 1) as a
check. If the sector fit fails, the half-way crossing of the azimuthally averaged profile is used (a
circle about the VOI axis); if that fails too, the nominal radius is used, the detected diameter is NaN
and the result is flagged (wall_detected = false).

Outputs: <id>_wall.json and <id>_wall_profile.csv per column; wall.csv and wall_profiles.csv are rebuilt
from all per-column files on every run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from .config import Config, read_json, write_json
from .extract import list_slices, voi_paths
from .segment import seg_paths

INNER_DIAMETER_MM = {"CF4": 25.55, "CF5": 26.36, "CF7": 25.63, "CF10": 26.16}  # Silva (2024), Table 1
N_SECTORS = 36
DR = 0.25                 # radial bin of the grey profiles used to find the wall (binned voxels)
LOW_SPREAD = 0.2          # wall annulus: grey spread below this fraction of the packing's
MIN_WALL_MM = 1.0         # minimum width of that annulus (the tube wall is about 3 mm thick)
DIP_MM = (-0.5, 0.2)      # search window of the near-wall minimum, relative to the current wall estimate
PLATEAU_MM = (0.3, 0.8)   # acrylic plateau, relative to the current wall estimate
MAX_DEV_MM = 0.25         # sector edges farther than this from the azimuthal-profile edge are rejected
MIN_SECTORS = 24
MAX_REL_DEV = 0.10        # detected radius must be within 10 % of the nominal one
N_ZSLABS = 3              # slabs along z used to measure the tilt and taper of the wall


def section_grid(meta: dict, raw_shape) -> dict:
    """Binned section grid with the binning phase of the VOI crop, and the VOI axis on that grid."""
    b = meta["bin"]
    oy, ox = meta["crop_y0"] % b, meta["crop_x0"] % b
    return {"bin": b, "origin_y": oy, "origin_x": ox,
            "ny": (raw_shape[0] - oy) // b, "nx": (raw_shape[1] - ox) // b,
            "cy": (meta["centre_y"] - oy - (b - 1) / 2) / b, "cx": (meta["centre_x"] - ox - (b - 1) / 2) / b}


def read_section(files, z0: int, nz: int, g: dict) -> np.ndarray:
    """nz binned slices from raw slice z0, block-averaged and rounded as in extract.read_voi."""
    b, oy, ox, ny, nx = g["bin"], g["origin_y"], g["origin_x"], g["ny"], g["nx"]
    out = np.empty((nz, ny, nx), np.uint16)
    acc = np.zeros((ny, nx), np.float64)
    for k in range(nz * b):
        img = tifffile.imread(files[z0 + k])[oy:oy + ny * b, ox:ox + nx * b].astype(np.float64)
        acc += img.reshape(ny, b, nx, b).mean(axis=(1, 3))
        if (k + 1) % b == 0:
            out[k // b] = np.clip(np.rint(acc / b), 0, 65535).astype(np.uint16)
            acc[:] = 0
    return out


def median_inplace(vol: np.ndarray, size: int, chunk: int = 24) -> np.ndarray:
    """ndi.median_filter(vol, size, mode="nearest"), computed in place chunk by chunk along axis 0."""
    h = size // 2
    before = vol[:0].copy()                      # unfiltered slices just before the current chunk
    for c0 in range(0, len(vol), chunk):
        c1 = min(c0 + chunk, len(vol))
        block = np.concatenate([before, vol[c0:min(c1 + h, len(vol))]])
        nb = len(before)
        before = vol[max(c1 - h, 0):c1].copy()
        vol[c0:c1] = ndi.median_filter(block, size=size, mode="nearest")[nb:nb + c1 - c0]
    return vol


def polar(shape, cy: float, cx: float):
    yy, xx = np.indices(shape)
    return np.hypot(yy - cy, xx - cx), np.arctan2(yy - cy, xx - cx)


def grey_sums(slab: np.ndarray, cy: float, cx: float, r_max: float):
    """Count, sum and sum of squares of the grey value per angular sector and radial bin (DR)."""
    r, t = polar(slab.shape[1:], cy, cx)
    nb = int(r_max / DR)
    rb = (r / DR).astype(int)
    sec = ((t + np.pi) / (2 * np.pi) * N_SECTORS).astype(int) % N_SECTORS
    keep = rb < nb
    idx = (sec * nb + rb)[keep]
    m = N_SECTORS * nb
    n = np.bincount(idx, minlength=m).astype(float) * len(slab)
    s1, s2 = np.zeros(m), np.zeros(m)
    for z in range(len(slab)):
        v = slab[z][keep].astype(np.float64)
        s1 += np.bincount(idx, weights=v, minlength=m)
        s2 += np.bincount(idx, weights=v * v, minlength=m)
    return n.reshape(N_SECTORS, nb), s1.reshape(N_SECTORS, nb), s2.reshape(N_SECTORS, nb)


def half_way_edge(m: np.ndarray, rc: np.ndarray, dip, plateau):
    """Radius where the mean grey value, moving outwards from its minimum in `dip`, crosses half-way
    to the median over `plateau`; (edge, minimum, plateau), edge NaN if there is no crossing."""
    w = np.flatnonzero((rc >= dip[0]) & (rc <= dip[1]) & np.isfinite(m))
    p = (rc >= plateau[0]) & (rc <= plateau[1]) & np.isfinite(m)
    if len(w) < 4 or p.sum() < 4:
        return np.nan, np.nan, np.nan
    k = w[np.argmin(m[w])]
    lo, hi = float(m[k]), float(np.median(m[p]))
    half = 0.5 * (lo + hi)
    for j in range(k, np.flatnonzero(p)[0]):
        if m[j] < half <= m[j + 1]:
            return float(rc[j] + (half - m[j]) / (m[j + 1] - m[j]) * DR), lo, hi
    return np.nan, lo, hi


def harmonics(t):
    t = np.asarray(t, float)
    return np.stack([np.ones_like(t), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)], axis=-1)


def fit_wall(t: np.ndarray, e: np.ndarray):
    """Least-squares two-harmonic fit of the sector edges, after one pass of outlier rejection."""
    X = harmonics(t)
    c = np.linalg.lstsq(X, e, rcond=None)[0]
    res = e - X @ c
    keep = np.abs(res) <= max(3 * 1.4826 * np.median(np.abs(res - np.median(res))), 0.5)
    c = np.linalg.lstsq(X[keep], e[keep], rcond=None)[0]
    return c, float(np.sqrt(np.mean((e[keep] - X[keep] @ c) ** 2))), int(keep.sum())


def detect_wall(n, s1, s2, r_nom: float, vox_mm: float) -> dict:
    """Locate the inner wall from the sector sums of `grey_sums` (radii in binned voxels)."""
    rc = (np.arange(n.shape[1]) + 0.5) * DR
    N = n.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s1.sum(0) / N
        sd = np.sqrt(np.maximum(s2.sum(0) / N - m ** 2, 0))
        msec = s1 / n
    out = {"ok": False, "sector_edges": np.full(N_SECTORS, np.nan)}
    core = sd[(rc < 0.6 * r_nom) & (N > 0)]
    low = (sd < LOW_SPREAD * np.median(core)) & (rc >= 0.7 * r_nom) & (N > 0)
    # first low-spread annulus at least MIN_WALL_MM wide
    edges = np.flatnonzero(np.diff(np.r_[0, low.astype(int), 0]))
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if (b - a) * DR * vox_mm >= MIN_WALL_MM]
    if not runs:
        out["reason"] = "no low-spread wall annulus"
        return out
    r_a = rc[runs[0][0]]
    mm = 1.0 / vox_mm
    r0, air, plat = half_way_edge(m, rc, (r_a + DIP_MM[0] * mm, r_a + DIP_MM[1] * mm),
                                  (r_a + PLATEAU_MM[0] * mm, r_a + PLATEAU_MM[1] * mm))
    out.update(r_spread=float(r_a), r_profile=r0, grey_near_wall_min=air, grey_wall=plat)
    if not np.isfinite(r0) or abs(r0 / r_nom - 1) > MAX_REL_DEV:
        out["reason"] = "no plausible edge on the azimuthal profile"
        return out
    out["ok"] = True
    e = np.array([half_way_edge(msec[k], rc, (r0 + DIP_MM[0] * mm, r0 + DIP_MM[1] * mm),
                                (r0 + PLATEAU_MM[0] * mm, r0 + PLATEAU_MM[1] * mm))[0] for k in range(N_SECTORS)])
    e[np.abs(e - r0) > MAX_DEV_MM * mm] = np.nan
    out["sector_edges"] = e
    t = -np.pi + (np.arange(N_SECTORS) + 0.5) * 2 * np.pi / N_SECTORS
    ok = np.isfinite(e)
    if ok.sum() < MIN_SECTORS:
        out["coef"] = np.array([r0, 0, 0, 0, 0])
        out["method"] = f"azimuthal-profile half-way crossing (only {int(ok.sum())} sector edges)"
        return out
    c, rms, used = fit_wall(t[ok], e[ok])
    out.update(coef=c, rms=rms, n_sectors=used,
               method=f"half-way crossing of the mean grey value in {N_SECTORS} sectors, two-harmonic fit")
    if abs(c[0] / r_nom - 1) > MAX_REL_DEV:
        out.update(ok=False, reason="fitted radius implausible")
    return out


def _hist_median(h: np.ndarray) -> float:
    c = np.cumsum(h)
    if c[-1] == 0:
        return float("nan")
    tot = c[-1]
    lo = np.searchsorted(c, (tot - 1) // 2 + 1)
    hi = np.searchsorted(c, tot // 2 + 1)
    return 0.5 * (lo + hi)


def sample_wall(cfg: Config, s, log=print) -> tuple[dict, pd.DataFrame]:
    meta = read_json(voi_paths(cfg, s.id, 0)[1])
    seg = read_json(seg_paths(cfg, s.id, 0)["json"])
    files = list_slices(cfg.p("raw_root") / s.raw)
    b = meta["bin"]
    vox = meta["voxel_um"] / 1000.0
    z0, nz = meta["slice_index_start"], meta["shape"][0]
    g = section_grid(meta, tifffile.imread(files[z0]).shape)
    slab = median_inplace(read_section(files, z0, nz, g), cfg["segmentation"]["median_size"])
    r_nom = INNER_DIAMETER_MM[s.id] / 2 / vox
    # wall detection about the VOI axis, repeated once about the axis corrected by the first harmonic
    cy, cx = g["cy"], g["cx"]
    det = detect_wall(*grey_sums(slab, cy, cx, 1.3 * r_nom), r_nom, vox)
    if det["ok"] and "rms" in det:
        cy, cx = cy + det["coef"][2], cx + det["coef"][1]
        det2 = detect_wall(*grey_sums(slab, cy, cx, 1.3 * r_nom), r_nom, vox)
        if det2["ok"] and "rms" in det2:
            det = det2
        else:
            cy, cx = g["cy"], g["cx"]
    detected = bool(det["ok"])
    coef = det["coef"] if detected else np.array([r_nom, 0, 0, 0, 0])
    method = det["method"] if detected else f"nominal radius about the VOI axis (detection failed: {det['reason']})"
    # tilt and taper: the coefficients vary linearly along z (slopes from N_ZSLABS slabs, mean = coef)
    slope = np.zeros(5)
    z_model = "constant along z"
    if detected and "rms" in det:
        zc, cz = [], []
        for a in range(N_ZSLABS):
            z1, z2 = a * nz // N_ZSLABS, (a + 1) * nz // N_ZSLABS
            dz = detect_wall(*grey_sums(slab[z1:z2], cy, cx, 1.3 * r_nom), r_nom, vox)
            if dz["ok"] and "rms" in dz:
                zc.append(0.5 * (z1 + z2 - 1))
                cz.append(dz["coef"])
        if len(cz) == N_ZSLABS:
            slope = np.polyfit(zc, np.array(cz), 1)[0]
            z_model = f"linear along z (slopes from {N_ZSLABS} slabs)"
        else:
            z_model = f"constant along z (wall found in {len(cz)} of {N_ZSLABS} slabs)"
    # porosity as a function of the distance from the wall
    r, t = polar(slab.shape[1:], cy, cx)
    H = harmonics(t)
    grain = (s.grain_mm or 3.5) / vox
    disc = r < 0.5 * coef[0]
    nbin = int(1.5 * r_nom)
    pc, nc = np.zeros(nbin), np.zeros(nbin)
    n_in = n_core = n_near = p_in = p_core = p_near = 0
    h_core, h_shell = np.zeros(65536), np.zeros(65536)
    for z in range(nz):
        d = H @ (coef + slope * (z - (nz - 1) / 2)) - r     # binned voxels from the inner wall
        inside = d > 0
        core, near, shell = d > grain, inside & (d < grain / 2), inside & (d < grain)
        db = d[inside].astype(int)
        f = slab[z]
        p = f < seg["T50"]
        pc += np.bincount(db, weights=p[inside], minlength=nbin)
        nc += np.bincount(db, minlength=nbin)
        n_in, n_core, n_near = n_in + int(inside.sum()), n_core + int(core.sum()), n_near + int(near.sum())
        p_in, p_core, p_near = p_in + int(p[inside].sum()), p_core + int(p[core].sum()), p_near + int(p[near].sum())
        h_core += np.bincount(f[disc & ~p], minlength=65536)
        h_shell += np.bincount(f[shell & ~p], minlength=65536)
    nbin = int(np.flatnonzero(nc)[-1]) + 1
    phi_d = pc[:nbin] / np.maximum(nc[:nbin], 1)
    d_mm = (np.arange(nbin) + 0.5) * vox
    amp2 = float(np.hypot(coef[3], coef[4]))
    off = (cy + coef[2] - g["cy"], cx + coef[1] - g["cx"]) if detected and "rms" in det else (np.nan, np.nan)
    nan = float("nan")
    res = {"id": s.id, "wall_detected": detected, "detection_method": method,
           "inner_radius_px": float(coef[0]) if detected else nan,
           "inner_diameter_mm": float(2 * coef[0] * vox) if detected else nan,
           "inner_diameter_min_mm": float(2 * (coef[0] - amp2) * vox) if detected else nan,
           "inner_diameter_max_mm": float(2 * (coef[0] + amp2) * vox) if detected else nan,
           "inner_diameter_nominal_mm": INNER_DIAMETER_MM[s.id],
           "inner_radius_used_px": float(coef[0]),
           "phi_section": p_in / n_in, "phi_core": p_core / n_core, "phi_within_half_grain_of_wall": p_near / n_near,
           "voi_phi": seg["phi_T50"], "T50": seg["T50"], "z_range": [z0, z0 + nz * b - 1],
           "grey_bead_core": _hist_median(h_core), "grey_bead_near_wall": _hist_median(h_shell),
           "grey_near_wall_min": det.get("grey_near_wall_min", nan), "grey_wall_plateau": det.get("grey_wall", nan),
           "inner_radius_spread_rule_px": det.get("r_spread", nan),
           "inner_radius_profile_px": det.get("r_profile", nan),
           "wall_sectors_used": det.get("n_sectors", 0), "wall_fit_rms_px": det.get("rms", nan),
           "axis_offset_y_px": float(off[0]), "axis_offset_x_px": float(off[1]),
           "wall_z_model": z_model, "wall_tilt_deg": float(np.degrees(np.arctan(np.hypot(slope[1], slope[2])))),
           "inner_diameter_change_mm": float(2 * slope[0] * (nz - 1) * vox),
           "voxel_mm": vox, "bin_origin_yx": [g["origin_y"], g["origin_x"]],
           "axis_yx_px": [float(cy), float(cx)], "wall_coef_px": [float(v) for v in coef],
           "wall_coef_slope_px": [float(v) for v in slope],
           "wall_sector_edges_px": [float(v) for v in det["sector_edges"]]}
    prof = pd.DataFrame({"id": s.id, "distance_from_wall_mm": d_mm, "phi": phi_d})
    flag = "" if detected else "  [WALL NOT DETECTED: nominal radius used]"
    log(f"  {s.id}: inner D {res['inner_diameter_mm']:.3f} mm (nominal {INNER_DIAMETER_MM[s.id]}), "
        f"phi section {100 * res['phi_section']:.2f}% core {100 * res['phi_core']:.2f}% "
        f"near wall {100 * res['phi_within_half_grain_of_wall']:.2f}% VOI {100 * res['voi_phi']:.2f}%{flag}")
    return res, prof


def run(cfg: Config, only=None, log=print) -> None:
    out_dir = cfg.outdir("wall")
    ids = [s.id for s in cfg.samples if s.group == "granular" and s.id in INNER_DIAMETER_MM]
    for s in cfg.samples:
        if s.id not in ids or (only and s.id not in only):
            continue
        res, prof = sample_wall(cfg, s, log=log)
        write_json(out_dir / f"{s.id}_wall.json", res)
        prof.to_csv(out_dir / f"{s.id}_wall_profile.csv", index=False)
    # the summary tables are always rebuilt from all per-sample files
    rows, profs = [], []
    for sid in ids:
        if (out_dir / f"{sid}_wall.json").exists():
            r = read_json(out_dir / f"{sid}_wall.json")
            rows.append({k: v for k, v in r.items() if not isinstance(v, list) or k == "z_range"})
        if (out_dir / f"{sid}_wall_profile.csv").exists():
            profs.append(pd.read_csv(out_dir / f"{sid}_wall_profile.csv"))
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "wall.csv", index=False)
    if profs:
        pd.concat(profs, ignore_index=True).to_csv(out_dir / "wall_profiles.csv", index=False)
