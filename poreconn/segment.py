"""Stage `segment`: median filtering and partial-volume (ISO-50) thresholding of every VOI.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

A voxel is labelled pore when its filtered grey value is closer to air than to the solid, i.e. when it
is more than half void. The threshold is

    T_f = I_air + f (I_solid - I_air),   f = 0.5 (ISO-50), with f = 0.4 and 0.6 as bounds.

The references are fixed per sample (the replicate VOIs of a specimen come from the same scan), so all
VOIs of a sample share the same thresholds:

* I_solid: mode of the solid phase in the histogram pooled over the sample's VOIs - the rock matrix, or
  the dominant glass mode of a bead pack; for the mixed pack CF10 (`solid_mode: lightest` in
  config.yaml) the main mode of the less attenuating beads.
* I_air: for the bead packs, the pore-air mode (the lowest substantial peak of the pooled histogram;
  the pores are millimetric); for the rocks, the median grey value of the air in the annulus 20-100 px
  beyond the plug surface (extract.outside_air), median over the sample's VOIs. The air grey value
  falls slowly with the distance from the plug (scatter and beam-hardening halo); the median of the air
  farther out (air_far) is recorded for the QA.

Inside the rocks, the voxels deep in large resolved pores are imaged brighter than the air around the
plug. The cause cannot be established from these data (instrument artefact or partial infill of the
pores), so this "internal air" is not used for the main segmentation; it defines an upper bound,
T50int = I_air,int + 0.5 (I_solid - I_air,int), with I_air,int the median grey value of the voxels at
least `deep` voxels inside a provisional pore phase (f = 0.6), pooled over the sample's VOIs and
excluding a shell of `deep` voxels at the VOI faces (where the distance transform does not see the
solid beyond the face). Otsu's threshold of each filtered VOI is recorded as well, for comparison
with the dissertation, which used it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tifffile
from edt import edt
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu

from .config import Config, read_json, write_json
from .extract import voi_paths


def median3(vol: np.ndarray, size: int = 3) -> np.ndarray:
    return ndi.median_filter(vol, size=size, mode="nearest")


def smoothed_histogram(vals: np.ndarray, lo: float, hi: float, nbins: int = 512, sigma: float = 3.0):
    h, e = np.histogram(vals, bins=nbins, range=(lo, hi))
    c = 0.5 * (e[1:] + e[:-1])
    hs = ndi.gaussian_filter1d(h.astype(float), sigma)
    return c, h, hs


def seg_paths(cfg: Config, sid: str, k: int) -> dict[str, Path]:
    d = cfg.outdir("segment")
    c = cfg.outdir("cache")
    return {"pore": d / f"{sid}_v{k}_pore.tif", "json": d / f"{sid}_v{k}_seg.json",
            "hist": d / f"{sid}_v{k}_hist.npz", "filt": c / f"{sid}_v{k}_filtered.tif",
            "sample": d / f"{sid}_sample_references.json", "pooled": d / f"{sid}_hist_pooled.npz"}


def filter_key(cfg: Config, sid: str, k: int) -> dict:
    """What the cached filtered VOI depends on: the grey VOI file, its cut and the filter size."""
    fgrey, fmeta = voi_paths(cfg, sid, k)
    st = fgrey.stat()
    m = read_json(fmeta)
    return {"grey_bytes": st.st_size, "grey_mtime_ns": st.st_mtime_ns, "median_size": cfg["segmentation"]["median_size"],
            **{c: m.get(c) for c in ("raw_folder", "slice_index_start", "shape", "crop_y0", "crop_x0", "crop_width", "bin")}}


def load_filtered(cfg: Config, sid: str, k: int) -> np.ndarray:
    """Median-filtered grey VOI, cached in outputs/cache; recomputed when its sidecar JSON does not
    match the current grey VOI (size, modification time, slice range and crop) and filter size."""
    p = seg_paths(cfg, sid, k)["filt"]
    side = p.with_suffix(".json")
    key = filter_key(cfg, sid, k)
    if p.exists() and side.exists():
        try:
            old = read_json(side)
        except ValueError:
            old = {}
        if old.get("key") == key and old.get("filtered_bytes") == p.stat().st_size:
            return tifffile.imread(p)
    grey = tifffile.imread(voi_paths(cfg, sid, k)[0])
    filt = median3(grey, cfg["segmentation"]["median_size"])
    del grey
    tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.tif")
    tifffile.imwrite(tmp, filt, compression="zlib")
    try:
        os.replace(tmp, p)
    except PermissionError:         # another process is reading the cache file (Windows): skip caching
        tmp.unlink(missing_ok=True)
        return filt
    write_json(side, {"key": key, "filtered_bytes": p.stat().st_size})
    return filt


def load_pore(cfg: Config, sid: str, k: int) -> np.ndarray:
    return tifffile.imread(seg_paths(cfg, sid, k)["pore"]).astype(bool)


def replicates(cfg: Config, sid: str) -> list[int]:
    ks = []
    k = 0
    while voi_paths(cfg, sid, k)[0].exists():
        ks.append(k)
        k += 1
    return ks


# ------------------------------------------------------------------ references
def histogram_peaks(vals: np.ndarray, lo: float | None = None, hi: float | None = None):
    plo, phi_ = np.percentile(vals, [0.05, 99.95])
    lo = plo if lo is None else min(lo, plo)
    hi = phi_ if hi is None else hi
    c, h, hs = smoothed_histogram(vals, lo, hi)
    peaks, _ = find_peaks(hs, prominence=0.02 * hs.max())
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(hs))])
    return c, h, hs, peaks, lo, hi


def granular_modes(c, hs, peaks, lo, hi, solid_mode: str = "dominant") -> tuple[float, float]:
    """Pore-air and solid modes of a bead-pack histogram.

    Air: the lowest-grey peak with at least 20 % of the tallest peak. Solid: among the peaks more than
    a quarter of the grey range above the air with at least 20 % of the tallest peak, the tallest
    ('dominant') or, for a pack of two kinds of beads ('lightest'), the lowest-grey one with at least
    half the height of the tallest solid peak.
    """
    top = hs[peaks].max()
    big = [p for p in peaks if hs[p] >= 0.2 * top]
    air = min(big, key=lambda p: c[p])
    solids = [p for p in big if c[p] > c[air] + 0.25 * (hi - lo)]
    if not solids:
        raise ValueError("no solid peak found above the pore-air mode")
    if solid_mode == "dominant":
        solid = max(solids, key=lambda p: hs[p])
    elif solid_mode == "lightest":
        stop = max(hs[p] for p in solids)
        solid = min((p for p in solids if hs[p] >= 0.5 * stop), key=lambda p: c[p])
    else:
        raise ValueError(f"unknown solid_mode {solid_mode}")
    return float(c[air]), float(c[solid])


def internal_air_offset(filts: list[np.ndarray], air_out: float, solid: float, deep: int = 5,
                        min_deep: int = 2000) -> dict:
    """Median (grey - outside air) / (solid - outside air) of voxels deep inside resolved pores."""
    vals = []
    contrast = solid - air_out
    for f in filts:
        prov = f < air_out + 0.6 * contrast
        if not prov.any():
            continue
        dt = edt(prov)          # exact EDT in float32 (scipy's needs ~5x the memory)
        m = dt >= deep
        m[:deep] = m[-deep:] = False
        m[:, :deep] = m[:, -deep:] = False
        m[:, :, :deep] = m[:, :, -deep:] = False
        if m.any():
            vals.append(f[m].astype(np.float64))
    n = int(sum(len(v) for v in vals))
    if n < min_deep:
        return {"air_int_offset": float("nan"), "n_deep": n}
    off = (np.median(np.concatenate(vals)) - air_out) / contrast
    return {"air_int_offset": float(off), "n_deep": n}


def sample_references(cfg: Config, s, ks: list[int]) -> dict:
    """Air and solid references of a sample, from its pooled VOIs."""
    filts = [load_filtered(cfg, s.id, k) for k in ks]
    vals = np.concatenate([f[::2, ::2, ::2].ravel() for f in filts])
    if s.group == "granular":
        c, h, hs, peaks, lo, hi = histogram_peaks(vals)
        I_air, I_solid = granular_modes(c, hs, peaks, lo, hi, getattr(s, "solid_mode", "dominant"))
        ref = {"I_air": I_air, "I_solid": I_solid, "air_source": "pore-air mode of the pooled histogram",
               "air_outside": float("nan"), "air_far": float("nan"), "air_int_offset": float("nan"), "n_deep": 0}
    else:
        metas = [read_json(voi_paths(cfg, s.id, k)[1]) for k in ks]

        def pooled(key):
            v = np.array([m.get(key, np.nan) for m in metas], float)
            return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")

        air_out = pooled("air_outside_median")
        if not np.isfinite(air_out) or "air_far_median" not in metas[0]:
            raise ValueError(f"{s.id}: no outside-air reference of the current method; run the 'air' stage")
        c, h, hs, peaks, lo, hi = histogram_peaks(vals, lo=air_out - 0.02 * (np.percentile(vals, 99.95) - air_out))
        I_solid = float(c[peaks[int(np.argmax(hs[peaks]))]])
        ref = {"I_air": air_out, "I_solid": I_solid,
               "air_source": "air annulus 20-100 px beyond the plug surface (median over the sample's VOIs)",
               "air_outside": air_out, "air_far": pooled("air_far_median"),
               **internal_air_offset(filts, air_out, I_solid)}
    ref.update(hist_centres=c, hist=h, peaks=c[peaks].tolist(), peak_heights=hs[peaks].tolist())
    return ref


def thresholds(ref: dict, seg_cfg: dict) -> dict:
    a, s = ref["I_air"], ref["I_solid"]
    fr = [seg_cfg["iso_fraction"]] + list(seg_cfg["sensitivity"])
    thr = {f"T{int(round(100 * f))}": a + f * (s - a) for f in fr}
    off = ref.get("air_int_offset", float("nan"))
    if np.isfinite(off):
        ai = a + off * (s - a)
        thr["T50int"] = ai + 0.5 * (s - ai)
    else:
        thr["T50int"] = float("nan")
    for k, v in thr.items():
        if k != "T50int" and not np.isfinite(v):
            raise ValueError(f"threshold {k} is not finite")
    return thr


# ------------------------------------------------------------------ stage
def run(cfg: Config, only=None, log=print) -> None:
    for s in cfg.samples:
        if only and s.id not in only:
            continue
        ks = replicates(cfg, s.id)
        ref = sample_references(cfg, s, ks)
        thr = thresholds(ref, cfg["segmentation"])
        p0 = seg_paths(cfg, s.id, 0)
        np.savez_compressed(p0["pooled"], centres=ref.pop("hist_centres"), hist=ref.pop("hist"))
        write_json(p0["sample"], {"id": s.id, "group": s.group, "replicates": ks, **ref, **thr})
        for k in ks:
            filt = load_filtered(cfg, s.id, k)
            t_otsu = float(threshold_otsu(filt[::2, ::2, ::2]))
            pore = filt < thr["T50"]
            p = seg_paths(cfg, s.id, k)
            tifffile.imwrite(p["pore"], pore.astype(np.uint8), compression="zlib")
            sub = filt[::2, ::2, ::2].ravel()
            hc, hh, _ = smoothed_histogram(sub, *np.percentile(sub, [0.05, 99.95]))
            np.savez_compressed(p["hist"], centres=hc, hist=hh)
            res = {"id": s.id, "replicate": k, "group": s.group, **ref, **thr, "T_otsu": t_otsu,
                   "phi_T50": float(pore.mean()), "phi_T40": float((filt < thr["T40"]).mean()),
                   "phi_T60": float((filt < thr["T60"]).mean()),
                   "phi_T50int": float((filt < thr["T50int"]).mean()) if np.isfinite(thr["T50int"]) else float("nan"),
                   "phi_otsu": float((filt < t_otsu).mean()), "contrast": float(ref["I_solid"] - ref["I_air"])}
            write_json(p["json"], res)
            log(f"  {s.id} v{k}: air {ref['I_air']:.0f} solid {ref['I_solid']:.0f} T50 {thr['T50']:.0f} "
                f"phi50 {100 * res['phi_T50']:.2f}% (T40 {100 * res['phi_T40']:.2f}, T60 {100 * res['phi_T60']:.2f}, "
                f"T50int {100 * res['phi_T50int']:.2f}; int-air offset {ref.get('air_int_offset', float('nan')):.3f}) "
                f"Otsu {100 * res['phi_otsu']:.2f}%")
