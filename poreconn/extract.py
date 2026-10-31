"""Stage `extract`: cut grey-level volumes of interest (VOIs) from the raw reconstructions.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

For every sample the VOI is a cube of `voi.edge` voxels (after binning), centred in the plane on the
plug or column axis found on the central slice, and placed along the core axis (z) away from the end
faces. Up to `voi.max_per_sample` non-overlapping replicate VOIs are stacked along z. Slices that cannot
be decoded (several trachyte stacks contain corrupted TIFF files) are never used: a window is accepted
only if all its slices are readable. The crop box, slice range, binning and an air reference measured
outside the sample are written next to each VOI, so that the cut can be repeated exactly.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu, threshold_otsu

from .config import Config, Sample, write_json


def list_slices(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() == ".tif"]

    def key(p: Path):
        m = re.search(r"(\d+)\.tif$", p.name, re.IGNORECASE)
        return int(m.group(1)) if m else -1

    return sorted(files, key=key)


def readable(f: Path) -> bool:
    """True if the slice can be fully decoded (header and pixel data)."""
    try:
        with tifffile.TiffFile(f) as tf:
            _ = tf.pages[0].asarray()
        return True
    except Exception:
        return False


def sample_geometry(img: np.ndarray, granular: bool) -> dict:
    """Centre and radius of the plug (or of the outer wall of a bead column) on one slice."""
    step = 4
    small = img[::step, ::step].astype(np.float32)
    nz = small[small > 0]
    t = threshold_multiotsu(nz, classes=3)[0] if granular else threshold_otsu(nz)
    m = ndi.binary_closing(small > t, iterations=3)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    cy, cx = ndi.center_of_mass(m)
    r = np.sqrt(m.sum() / np.pi)
    return {"cy": float(cy * step), "cx": float(cx * step), "r": float(r * step), "t_mask": float(t)}


def outside_air(img: np.ndarray, geo: dict, margins=(20.0, 12.0, 6.0), band: float = 100.0,
                min_px: int = 5000) -> dict:
    """Median grey value of the air in an annulus around the plug, and of the air farther away.

    The air reference is the median of the pixels between r + margin and r + band from the plug axis
    (r = plug radius), anywhere in the slice. The whole slice is reconstructed (the scans' reconstruction
    mask radius exceeds the slice half-diagonal), so the corners are valid air; but the air grey value
    falls with the distance from the plug (scatter and beam-hardening halo), so a fixed band next to the
    plug is used rather than all the air in the slice. The margin from the plug surface is reduced
    (20, 12, 6 px) only when fewer than `min_px` pixels qualify; the margin used is returned (NaN, with a
    NaN median, when none works). `far_median` is the median beyond r + band (NaN below `min_px` pixels).
    """
    ny, nx = img.shape
    yy, xx = np.ogrid[:ny, :nx]
    d = np.hypot(yy - geo["cy"], xx - geo["cx"]) - geo["r"]
    far = d > band
    n_far = int(far.sum())
    out = {"median": float("nan"), "n_px": 0, "margin_px": float("nan"),
           "far_median": float(np.median(img[far])) if n_far >= min_px else float("nan"), "far_n_px": n_far}
    for margin in margins:
        sel = (d > margin) & ~far
        n = int(sel.sum())
        out["n_px"] = n
        if n >= min_px:
            out.update(median=float(np.median(img[sel])), margin_px=float(margin))
            break
    return out


AIR_METHOD = "annulus 20-100 px beyond the plug surface"
AIR_NA = {"air_outside_median": float("nan"), "air_outside_pixels_per_slice": 0,
          "air_outside_margin_px": float("nan"), "air_far_median": float("nan"), "air_far_pixels_per_slice": 0,
          "air_outside_method": "not applicable (the column fills the field of view)"}


def window_air(files: list[Path], z0: int, h: int, geo: dict) -> dict:
    """Outside-air fields of a VOI's meta: outside_air on 5 raw slices spread over slices z0..z0+h-1."""
    a = [outside_air(tifffile.imread(files[i]), geo) for i in np.linspace(z0, z0 + h - 1, 5).astype(int)]

    def med(key):
        v = np.array([x[key] for x in a], float)
        return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")

    mrg = np.array([x["margin_px"] for x in a], float)
    return {"air_outside_median": med("median"),
            "air_outside_pixels_per_slice": int(np.median([x["n_px"] for x in a])),
            "air_outside_margin_px": float(np.nanmin(mrg)) if np.isfinite(mrg).any() else float("nan"),
            "air_far_median": med("far_median"),
            "air_far_pixels_per_slice": int(np.median([x["far_n_px"] for x in a])),
            "air_outside_method": AIR_METHOD}


def candidate_windows(n: int, h: int, margin_frac: float, gap: int):
    """Window starts: centre first, then alternately above and below."""
    lo, hi = int(round(margin_frac * n)), int(round((1 - margin_frac) * n)) - h
    if hi < lo:
        return []
    centre = (n - h) // 2
    centre = min(max(centre, lo), hi)
    starts = [centre]
    k = 1
    while True:
        added = False
        for s in (centre + k * (h + gap), centre - k * (h + gap)):
            if lo <= s <= hi:
                starts.append(s)
                added = True
        if not added:
            break
        k += 1
    return starts


def find_windows(files: list[Path], h: int, cfg_voi: dict) -> tuple[list[int], dict]:
    """Choose up to max_per_sample fully readable, non-overlapping windows of h slices.

    The first window is the one closest to the middle of the stack (the main VOI, replicate 0).
    """
    n = len(files)
    cache: dict[int, bool] = {}

    def ok(i: int) -> bool:
        if i not in cache:
            cache[i] = readable(files[i])
        return cache[i]

    def window_ok(s: int) -> bool:
        # check the ends first (cheap rejection), then every slice
        if not (ok(s) and ok(s + h - 1)):
            return False
        return all(ok(i) for i in range(s, s + h))

    chosen: list[int] = []
    for margin in (cfg_voi["z_margin_frac"], 0.02):
        lo_m, hi_m = int(round(margin * n)), int(round((1 - margin) * n)) - h
        for s0 in candidate_windows(n, h, margin, cfg_voi["gap"]):
            if len(chosen) >= cfg_voi["max_per_sample"]:
                break
            # slide the window (up to half its height) to dodge corrupted slices, inside the margins
            for shift in [0] + [d * sgn for d in range(10, h // 2 + 1, 10) for sgn in (1, -1)]:
                s = s0 + shift
                if s < lo_m or s > hi_m:
                    continue
                if any(abs(s - c) < h + cfg_voi["gap"] for c in chosen):
                    continue
                if window_ok(s):
                    chosen.append(s)
                    break
        if chosen:
            break
    fallback = False
    if not chosen:
        # last resort (stacks with many corrupted slices): the readable window closest to the middle,
        # anywhere in the stack, with no end margin
        for s in sorted(range(0, n - h + 1, 5), key=lambda v: abs(v - (n - h) // 2)):
            if window_ok(s):
                chosen.append(s)
                fallback = True
                break
    n_bad = sum(1 for v in cache.values() if not v)
    return chosen, {"slices_checked": len(cache), "slices_unreadable_seen": n_bad,
                    "window_fallback_no_margin": fallback}


def read_voi(files: list[Path], z0: int, h: int, y0: int, x0: int, w: int, b: int) -> np.ndarray:
    """Read slices z0..z0+h-1, crop [y0:y0+w, x0:x0+w] and block-average by b (in all three axes)."""
    out = np.empty((h // b, w // b, w // b), dtype=np.uint16)
    acc = np.zeros((w // b, w // b), dtype=np.float64)
    for k in range(h):
        img = tifffile.imread(files[z0 + k])[y0:y0 + w, x0:x0 + w].astype(np.float64)
        if img.shape != (w, w):
            raise ValueError(f"crop outside the image: {files[z0 + k].name} {img.shape}")
        if b > 1:
            img = img.reshape(w // b, b, w // b, b).mean(axis=(1, 3))
        acc += img
        if (k + 1) % b == 0:
            out[k // b] = np.clip(np.rint(acc / b), 0, 65535).astype(np.uint16)
            acc[:] = 0
    return out


def voi_paths(cfg: Config, sid: str, k: int) -> tuple[Path, Path]:
    d = cfg.p("voi_dir")
    return d / f"{sid}_v{k}_grey.tif", d / f"{sid}_v{k}_meta.json"


def extract_sample(cfg: Config, s: Sample, log=print) -> list[dict]:
    t0 = time.time()
    folder = cfg.p("raw_root") / s.raw
    files = list_slices(folder)
    granular = s.group == "granular"
    edge = cfg["voi"]["edge"] if not cfg.quick else 100
    h = w = edge * s.bin

    # geometry and outside-air reference from the first readable slice near the middle
    mid = len(files) // 2
    for off in range(0, len(files) // 2, 7):
        for i in (mid + off, mid - off):
            if 0 <= i < len(files) and readable(files[i]):
                mid = i
                break
        else:
            continue
        break
    img = tifffile.imread(files[mid])
    geo = sample_geometry(img, granular)
    y0 = int(round(geo["cy"])) - w // 2
    x0 = int(round(geo["cx"])) - w // 2
    half_diag = (w / 2) * np.sqrt(2)
    # bead columns: the radius found is that of the acrylic tube; the inner radius is ~81 % of it
    r_free = geo["r"] * (25.6 / 31.5) if granular else geo["r"]
    if half_diag > r_free:
        log(f"  WARNING {s.id}: VOI corners {half_diag:.0f} px from the axis, free radius {r_free:.0f} px")

    starts, chk = find_windows(files, h, cfg["voi"])
    if not starts:
        raise RuntimeError(f"{s.id}: no readable window of {h} slices")
    metas = []
    for k, z0 in enumerate(starts):
        vol = read_voi(files, z0, h, y0, x0, w, s.bin)
        # outside-air reference on 5 slices of the window
        air = window_air(files, z0, h, geo) if not granular else AIR_NA
        fgrey, fmeta = voi_paths(cfg, s.id, k)
        fgrey.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(fgrey, vol, compression="zlib")
        meta = {
            "id": s.id, "replicate": k, "group": s.group, "raw_folder": s.raw,
            "first_slice": files[z0].name, "last_slice": files[z0 + h - 1].name,
            "slice_index_start": int(z0), "n_slices_stack": len(files),
            "crop_y0": y0, "crop_x0": x0, "crop_width": w, "bin": s.bin,
            "voxel_um_raw": s.voxel_um, "voxel_um": s.voxel_um * s.bin, "shape": list(vol.shape),
            "axis_order": "z (core axis), y, x", "centre_y": geo["cy"], "centre_x": geo["cx"],
            "sample_radius_px": geo["r"], "sample_diameter_mm": 2 * geo["r"] * s.voxel_um / 1000,
            **air,
            "dtype": "uint16",
            **chk,
        }
        write_json(fmeta, meta)
        metas.append(meta)
        log(f"  {s.id} v{k}: slices {z0}-{z0 + h - 1} of {len(files)}, crop y{y0} x{x0} w{w}, "
            f"bin {s.bin} -> {vol.shape}, air {meta['air_outside_median']:.0f}")
    log(f"  {s.id}: {len(metas)} VOI(s) in {time.time() - t0:.0f} s")
    return metas


def run(cfg: Config, only: list[str] | None = None, log=print) -> None:
    rows = []
    for s in cfg.samples:
        if only and s.id not in only:
            continue
        rows.extend(extract_sample(cfg, s, log=log))
    write_json(cfg.outdir("extract") / ("vois.json" if not only else f"vois_{'_'.join(only)}.json"), rows)
