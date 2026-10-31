"""Morphological and topological descriptors of a binary pore image (True = pore).

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Conventions
-----------
* The pore phase is 6-connected (face neighbours), consistently in cluster labelling, percolation,
  the Euler characteristic and the transport solvers; the solid is then 26-connected.
* Euler characteristic chi = beta0 - beta1 + beta2 of the pore phase. Two quantities are computed:
  - `euler`: the integer chi of the pore phase inside the VOI closed by solid (scikit-image pads the
    image with background) - a topological count of this finite window;
  - `euler_density`: the edge-corrected estimator of the Euler number density chi_V of the (stationary)
    pore space, counting only the 2x2x2 voxel configurations lying entirely inside the window
    (Ohser-Nagel-Schladitz): chi_V = (1/8) sum_c w_c h_c / (V of the interior cells). Unlike chi/V of
    the closed window, it does not depend on the window size, so windows of different sizes and samples
    imaged with different voxels can be compared.
* Surface density S_V by Crofton's formula, S_V = 2 P_L, with P_L the number of pore-solid
  transitions per unit length averaged over the three axes (unbiased for isotropic structures). The
  mean chord lengths are the Crofton-consistent l_p = 4 phi / S_V and l_s = 4 (1 - phi) / S_V, i.e.
  2 phi / P_L with the same mean P_L, so that they match the reported S_V; the per-axis chords
  2 phi / P_L_axis are kept separately (their arithmetic mean agrees with l_p only for isotropic VOIs).
  Counting runs cut by the VOI faces as whole chords would bias the chords low by about l/L.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import euler_number
from skimage.measure._regionprops_utils import EULER_COEFS3D_26

STRUCT6 = ndi.generate_binary_structure(3, 1)


def label6(pore: np.ndarray):
    return ndi.label(pore, structure=STRUCT6)


def face_labels(lab: np.ndarray, axis: int, side: int) -> np.ndarray:
    sl = [slice(None)] * 3
    sl[axis] = 0 if side == 0 else -1
    u = np.unique(lab[tuple(sl)])
    return u[u > 0]


def connectivity(pore: np.ndarray, lab=None, n=None) -> dict:
    """Cluster statistics and percolation along the three axes (z, y, x)."""
    if lab is None:
        lab, n = label6(pore)
    sizes = np.bincount(lab.ravel(), minlength=n + 1).astype(np.int64)
    sizes[0] = 0
    V = pore.size
    span = {}
    touching = set()
    for ax, name in enumerate("zyx"):
        a, b = face_labels(lab, ax, 0), face_labels(lab, ax, 1)
        span[name] = np.intersect1d(a, b)
        touching.update(a.tolist())
        touching.update(b.tolist())
    span_any = np.unique(np.concatenate([span[a] for a in "zyx"])) if n else np.array([], int)
    interior = np.setdiff1d(np.arange(1, n + 1), np.fromiter(touching, int, len(touching)))
    phi = float(pore.mean())
    out = {
        "phi": phi, "n_clusters": int(n),
        "largest_cluster_frac": float(sizes.max() / max(pore.sum(), 1)),
        "phi_span_any": float(sizes[span_any].sum() / V),
        "phi_isolated": float(sizes[interior].sum() / V),
    }
    for a in "zyx":
        out[f"perc_{a}"] = bool(len(span[a]) > 0)
        out[f"phi_span_{a}"] = float(sizes[span[a]].sum() / V)
    out["perc_any"] = bool(len(span_any) > 0)
    out["frac_connected"] = out["phi_span_any"] / phi if phi > 0 else 0.0
    out["frac_isolated"] = out["phi_isolated"] / phi if phi > 0 else 0.0
    # clusters that touch a face but do not span the VOI along any axis (dead ends cut by the window)
    out["frac_boundary_nonspanning"] = max(0.0, 1.0 - out["frac_connected"] - out["frac_isolated"]) if phi > 0 else 0.0
    return out


def spanning_mask(pore: np.ndarray, axis: int, lab=None, n=None) -> np.ndarray:
    """Pore voxels that belong to clusters touching both faces normal to `axis`."""
    if lab is None:
        lab, n = label6(pore)
    s = np.intersect1d(face_labels(lab, axis, 0), face_labels(lab, axis, 1))
    if len(s) == 0:
        return np.zeros_like(pore, dtype=bool)
    keep = np.zeros(n + 1, dtype=bool)
    keep[s] = True
    return keep[lab]


def euler(pore: np.ndarray, conn: int = 6) -> int:
    """Integer Euler characteristic of the pore phase (6- or 26-connected) in the window closed by solid."""
    return int(euler_number(pore, connectivity=1 if conn == 6 else 3))


_CONFIG = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 1, 4], [0, 2, 8]],
                    [[0, 0, 0], [0, 16, 64], [0, 32, 128]]])


def euler_density(pore: np.ndarray, dx: float = 1.0, conn: int = 6) -> float:
    """Edge-corrected Euler number density (per dx^3) of the pore phase (6- or 26-connected).

    Uses the same 2x2x2 configuration weights as scikit-image (connectivity 1) but only the cells
    entirely inside the image: with scipy's convolution, output voxel i encodes the cell made of
    voxels i-1 and i along each axis, so the interior cells are XF[1:, 1:, 1:].
    """
    XF = ndi.convolve(pore.astype(np.int32), _CONFIG, mode="constant", cval=0)
    h = np.bincount(XF[1:, 1:, 1:].ravel(), minlength=256)
    coefs = EULER_COEFS3D_26[::-1] if conn == 6 else EULER_COEFS3D_26
    chi = 0.125 * float(coefs @ h)
    n_cells = (pore.shape[0] - 1) * (pore.shape[1] - 1) * (pore.shape[2] - 1)
    return chi / (n_cells * dx ** 3)


def crofton_sv(pore: np.ndarray, dx: float) -> dict:
    """S_V (1/length) from transition counts along z, y, x; also mean pore and solid chord lengths."""
    out = {}
    phi = float(pore.mean())
    pls, lp = [], []
    for ax, a in enumerate("zyx"):
        p = np.moveaxis(pore, ax, -1)
        trans = np.count_nonzero(p[..., 1:] != p[..., :-1])
        length = p.shape[0] * p.shape[1] * (p.shape[2] - 1) * dx
        pl = trans / length
        pls.append(pl)
        lp.append(2 * phi / pl if pl > 0 else np.inf)
        out[f"PL_{a}"] = pl
        out[f"chord_pore_{a}"] = lp[-1]
        out[f"chord_solid_{a}"] = 2 * (1 - phi) / pl if pl > 0 else np.inf
    sv = 2.0 * float(np.mean(pls))
    out["S_V"] = sv
    out["chord_pore_mean"] = 4 * phi / sv if sv > 0 else np.inf
    out["chord_solid_mean"] = 4 * (1 - phi) / sv if sv > 0 else np.inf
    out["chord_anisotropy"] = float(max(lp) / max(min(lp), 1e-12))
    return out


def block_mean(vol: np.ndarray, f: int) -> np.ndarray:
    """Block-average by an integer factor (the volume is cropped to a multiple of f)."""
    if f == 1:
        return vol.astype(np.float32)
    n = [s // f * f for s in vol.shape]
    v = vol[: n[0], : n[1], : n[2]].astype(np.float32)
    return v.reshape(n[0] // f, f, n[1] // f, f, n[2] // f, f).mean(axis=(1, 3, 5))
