"""Stage `imaging`: imaging diagnostics of the segmentation (effective resolution, filter, glass modes,
enclosed air, VOI footprint, synthetic bead packs).

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Six analyses, each a function called from `run` (outputs in outputs/imaging/):

1. Effective resolution (`esf_sample`, main VOI of every specimen). The edge-spread function (ESF) is
   measured across the ISO-50 pore/solid interface, on the unfiltered grey VOI and on the median-filtered
   one. The signed distance of a voxel to the T50 interface is d = EDT_pore - 0.5 inside the pore and
   d = 0.5 - EDT_solid in the solid (d > 0 in the pore; a voxel face-adjacent to the other phase is at
   0.5 voxel). Only interfaces between LARGE features are used, so that both plateaus exist: a pore
   voxel is kept if it lies in the opening of the pore by a ball of radius ESF_LARGE_VOX (it belongs to
   a pore region whose inscribed radius is at least ESF_LARGE_VOX) and its nearest solid voxel belongs to
   the opening of the solid by the same ball (to within ESF_TOL_VOX); a solid voxel symmetrically. A shell
   of ESF_DMAX voxels at the VOI faces is excluded. Grey values are binned by d in ESF_BIN-voxel bins over
   |d| <= ESF_DMAX; the median per bin (abscissa: mean d of the bin) is fitted with
   g(d) = a + b * 0.5 * (1 + erf((d - d0) / (sqrt(2) sigma))), bins with >= ESF_MIN_BIN_VOX voxels.
   A VOI is skipped (NaN, with the reason) when fewer than ESF_MIN_VOX voxels qualify or when either side
   has fewer than ESF_MIN_PLATEAU_BINS populated bins beyond |d| >= ESF_PLATEAU_VOX. The plateau noise is
   the median, over those plateau bins, of the robust spread (IQR / 1.349) of the grey values of a bin.
   The EDT distance distorts the profile (near an oblique interface the nearest voxel of the other phase
   lies closer to it than half a voxel), so the raw fitted sigma of the unfiltered profile is biased: on the
   synthetic pore below (noise 0-16 % of the contrast) it is 16-27 % high at sigma = 0.4 voxel, within 4 %
   at sigma = 0.7-0.8 (the bead packs), 6-12 % low at sigma = 1-2 voxels and 12-21 % low at 4 voxels.
   It is therefore calibrated (`esf_calibration`): a spherical pore of radius ESF_CAL_R voxels with a
   point-sampled Gaussian edge of known sigma (ESF_CAL_SIGMAS) and white noise at the VOI's measured
   plateau noise is processed exactly like a VOI (median filter, ISO-50 mask of the filtered volume,
   selection, fit), and the fitted sigma of the VOI is mapped back by interpolation (sigma_cal_vox). For the
   unfiltered profile the calibration does not depend on the curvature (pore radius 12-45 voxels, convex or
   concave) and little on the noise: from 0 to 16 % of the contrast (the range of the VOIs; trachytes
   0.14-0.16) the fitted sigma changes by less than 3 % for sigma = 0.7-2 voxels and by up to about 10 % at
   0.4 and at 4 voxels, and each VOI is calibrated at its own noise. The filtered one depends strongly on
   the noise, and real noise is spatially correlated, so the unfiltered sigma_cal is the reference value. It is the width of the
   Gaussian equivalent to the whole edge response, voxel aperture included; the 10-90 % edge width is
   2.563 sigma. No standard error of sigma is exported (the curve_fit covariance of bin medians with
   systematic, correlated residuals is not an uncertainty; use the spread between VOIs). A slow tail that
   the erf does not describe (scatter, beam hardening, graded walls) is measured on the residuals of the
   fit: on each side, the mean residual at |d| in TAIL_NEAR minus that at |d| >= TAIL_FAR, as a fraction of
   the contrast (tail_pore_frac, tail_solid_frac; > 0: the grey value still approaches that side's plateau
   beyond the erf). A pure blurred step gives 0 whatever sigma and d0: the part of the raw profile change
   that the fitted erf explains is not a tail. d0 is the offset (voxels) of the local 50 % point from the
   T50 interface (d0 < 0: the local mid-grey lies on the solid side, i.e. T50 is below the local mid-grey).
   A profile with a residual tail above ESF_TAIL_MAX of the contrast on either side is flagged (step_like =
   false, `step_like`): the erf misses a systematic part of it, its sigma is not an imaging resolution and
   the group summaries leave it out. The real VOIs give residual tails of 0.01-0.05, and an erf fitted to a
   step with an added exponential tail of up to half the contrast absorbs most of it into a larger sigma and
   d0 (residual tail <= 0.09 for decay lengths of 1-4 voxels): the flag removes only profiles that are not
   erf-like, and the fitted sigma of a step-like profile still includes any material gradient at the interface (optics and gradients cannot be
   separated from one profile). The image noise (noise_frac) is the smaller of the two plateau spreads over
   the fitted contrast: the other phase adds material heterogeneity (e.g. the two glasses of CF10).
2. Median-filter sensitivity (`filter_sample`, main VOIs). The unfiltered grey VOI is thresholded at the
   same T50 (and at T50int where it is finite) and compared with the median-filtered segmentation:
   porosity, cluster connectivity and percolation (morph.connectivity), chi_V (morph.euler_density, per
   mm^3) and, along every percolating axis, the critical diameter (topology.critical_diameter_vox).
3. Two glasses of CF10 (`two_glass`, specimens with `solid_mode: lightest`) and secondary solid modes of
   every bead pack (`solid_modes`). Peaks of the pooled histogram (outputs/segment/<id>_hist_pooled.npz,
   smoothed with a Gaussian of HIST_SMOOTH_BINS bins as in segment.smoothed_histogram) are found with a
   prominence of at least PEAK_PROMINENCE times the tallest smoothed count (segment.histogram_peaks'
   rule). A solid peak other than the reference I_solid is a secondary mode; it is split from the main mode
   at the histogram minimum between them (`mode_split`), and the beads of all VOIs (grains.beads) whose core
   median grey (`bead_cores`) lies on its side are counted, so that a second kind of glass (whole beads) can
   be told from grey-level variation inside beads. For CF10 the porosity is computed at (a) T50 (light-glass
   reference), (b) the ISO-50 threshold of the dense glass (highest-grey peak), (c) a bead-wise local ISO-50
   (`two_glass_voi`): the T50 solid is partitioned with grains.beads, every bead is classified dense when the
   median grey of its core voxels (distance from the pore phase > CORE_VOX voxels) is >= the same split as
   in solid_modes (the histogram minimum between the main mode and the dense peak, passed by `run`), and
   the voxels within SHELL_VOX voxels of a bead (nearest bead class) are re-thresholded at
   I_air + 0.5 (I_bead_mode - I_air). vox_rethresholded_dense counts the voxels whose class changed from
   (a) to (c); vox_shell_dense the voxels whose nearest bead (within SHELL_VOX) is dense.
4. Enclosed-air control. Two air references are in use and the tables label every row with its own
   (air_reference): the bead packs' offsets are relative to the pore-air mode inside the column (the open
   pore space, so the bead-pack open-pore offset is about 0 by construction), the rocks' to the air outside
   the plug (segment's I_air of the rocks). Absolute offsets of beads and rocks are therefore not
   comparable; the comparison is the difference (enclosed - open) within each sample, in which the air
   reference cancels (only the normalising contrast differs). It tests a local, cavity-scale effect (air
   enclosed by solid imaged brighter than open pore air in the same object), not a global one (all air
   inside a dense object raised relative to outside air), which the bead packs cannot test.
   (a) Bead packs (`enclosed_air_sample`, all VOIs). The gas bubbles in the glass (T50 pore clusters,
   6-connected, that touch no VOI face: the components grains.solid_dt fills) are air enclosed by glass.
   For every bubble of at least BUBBLE_MIN_VOX voxels: volume, equivalent diameter, inscribed radius
   (maximum EDT) and the median filtered grey of its deepest voxels (EDT >= max - BUBBLE_DEEP_TOL) as an
   offset (g - I_air) / (I_solid - I_air), and whether it lies inside a single bead (grains.beads labels of
   its voxels; pockets of the pore space between beads cut off by 6-connectivity straddle two or more and
   are left out of the summary). The deepest voxels of small bubbles are raised by the blur and the median
   filter alone, so the 'all' row of the summary uses only bubbles with r_in_vox >= BUBBLE_MIN_R_IN_VOX
   (n reported), and a blur-only reference (`bubble_blur_reference`: blurred_pore at the pack's sigma_cal and
   noise, median3, T50, bubbles_voi) gives the offset expected without any enclosed-air effect against the
   bubble size; it is interpolated at every bubble's d_eq (offset_deep_blur_ref) and subtracted
   (offset_deep_excess). At sigma = 0.70 voxel (noise 0.04-0.055) it is about 0.38, 0.12, 0.03-0.04, 0.015
   and < 0.01 for true radii of 2, 2.5, 3, 3.5 and >= 4 voxels (the measured r_in is 0.4-0.8 voxel below the
   true radius, so r_in >= 3.5 keeps bubbles of true radius above about 4.2 voxels, whose blur-only offset
   is < 0.01). The open pore space
   (clusters touching a face) and the enclosed clusters are also measured with the definition of
   qa.deep_offsets: provisional pore phase cut at f = 0.6, voxels at EDT >= 2, 3, 5, 8 voxels away from a
   shell of the same depth at the faces, median grey as an offset (pooled over the VOIs of the pack, from
   grey-value histograms; `depth_histograms`).
   (b) Rocks (`rock_enclosed_sample`, main VOI): the same depth_histograms with the outside-air reference,
   split into open pores (clusters of the provisional phase touching a VOI face) and enclosed pores
   (touching none). The qa offsets of all rock pore air (outputs/qa/<id>_qa.json, `rock_depth_rows`) are
   kept as rows 'rock pore'.
   Depths are compared in units of the edge width: depth_over_sigma = depth_vox / sigma_cal_vox (unfiltered
   ESF of the sample, item 1; the group median where the sample's fit failed, sigma_cal_source), since the
   blur tail at depth d is about Phi(-d / sigma) (0.002 for the beads but 0.13 for the trachytes at depth 2).
   The summary compares groups only at depth_over_sigma >= DEPTH_MIN_SIGMA.
5. Bead VOI footprint (`footprint_sample`). The radial porosity profile of the column (wall stage; bins of
   one binned voxel of distance from the inner wall) is averaged over the square footprint of the main VOI
   on a grid of FOOT_SUB^2 points per voxel and FOOT_NZ levels along z, with the distance from the wall of
   every point computed exactly as in wall.sample_wall (two-harmonic wall about the detected axis, with
   its slopes along z). Frame: the wall stage works on the binned section grid whose binned pixel J covers
   raw pixels o + b J ... o + b J + b - 1 (o = crop0 mod b, `bin_origin_yx`), and the axis `axis_yx_px` is
   in that frame; VOI pixel j is section pixel (crop0 - o) / b + j. The prediction is not independent of
   the VOI: pi/4 of the footprint lies inside the disc inscribed in it, whose annuli are measured on voxels
   of the VOI itself; it tests the absence of an azimuthal anomaly in the VOI and locates the difference
   between the VOI and the core (beyond one bead of the wall) in the profile averaged over the inscribed
   disc and over the core outside it.
6. Synthetic validation of the bead measurement (`synthetic_beads`). Equal spheres of diameter SYN_D_VOX
   voxels (SYN_VOX_MM mm voxels) are packed in a periodic SYN_EDGE^3 box: random centres relaxed as soft
   spheres of diameter SYN_D_VOX / SYN_GROWTH until no pair overlaps (solid fraction SYN_FILL), then drawn at
   the full diameter, so neighbours closer than a few % of the diameter touch through small lenses. The
   union is digitised with partial volume (SYN_SUPER^3 sub-samples per voxel), converted to grey (SYN_AIR,
   SYN_GLASS), blurred with the calibrated unfiltered ESF sigma of the single-glass bead packs (median over
   the step-like packs, item 1, CF10 and any other `solid_mode: lightest` pack excluded because its profile
   mixes two glasses; 1 voxel if unavailable) less the aperture already applied by the digitisation
   (variance (1 - 1/SYN_SUPER^2) / 12 voxel^2, removed in quadrature), given Gaussian noise of the bead
   packs' unfiltered image noise (noise_frac of item 1, a fraction of the fitted contrast; median over the
   same packs), median filtered (segment.median3), thresholded at the midpoint and measured with
   grains.beads + grains.measure with the stage's parameters. Prolate spheroids of axis ratio SYN_ASPECT
   and the same volume test that elongation is detected in a dense pack: the same number of particles,
   with random fixed orientations, is relaxed at the volume-equivalent size (/ SYN_GROWTH) with the exact
   contact distance of two spheroids along their centre line (Perram-Wertheim contact function,
   `spheroid_contact_distance`, `relax_spheroids`) and drawn at full size. Every relaxed packing is checked
   on its digitised union (`digitised_overlap`: voxel centres inside two relaxed particles; 0 in all tested
   packs) and its true partial-volume solid fraction is reported (solid_frac_true; about 0.61 for both
   shapes at SYN_FILL = 0.58, at the low end of the real VOIs' 0.60-0.70).

Each analysis writes per-sample JSON files and the CSV tables are rebuilt from all of them; the summary
imaging_summary.json is rebuilt from the tables. `run(cfg, only, log, parts)` runs a subset of the
analyses with `parts` (names in PARTS); part 'enclosed_air' runs the bead packs and the rocks.

Outputs (outputs/imaging/) and the columns added in this version:
* esf.csv: tail_pore_frac, tail_solid_frac now on the fit residuals; step_like from them; sigma_se_vox
  dropped (not an uncertainty).
* filter_sensitivity.csv, solid_modes.csv, voi_footprint.csv: unchanged.
* cf10_two_glass.csv: split_grey is now the histogram minimum (as solid_modes); vox_rethresholded_dense
  counts the voxels whose class changed; new vox_shell_dense (voxels whose nearest bead is dense).
* enclosed_air_bubbles.csv: new n_bead_labels, in_single_bead, offset_deep_blur_ref, offset_deep_excess.
* enclosed_air_summary.csv: bins hold the single-bead bubbles; the 'all' row the single-bead bubbles with
  r_in_vox >= BUBBLE_MIN_R_IN_VOX. New selection, offset_deep_blur_ref_median, offset_deep_excess_median;
  on the 'all' row r_in_min_vox, n_bubbles_recorded (all bubbles >= BUBBLE_MIN_VOX), n_inter_bead,
  n_below_r_min, sigma_blur_ref_vox (phi_bubbles: all recorded bubbles, as before).
* enclosed_air_bubble_blur_reference.csv (new): per bead pack and true radius r_true_vox: sigma_vox,
  noise_frac, sigma_source, n_rep, r_in_vox_median, d_eq_vox_median, offset_deep_median/_q25/_q75.
* enclosed_air_depth.csv: new sources 'rock open pore' and 'rock enclosed pore' (<id>_enclosed_air_rock.json);
  new columns air_reference, I_air, contrast, sigma_cal_vox, sigma_cal_source, depth_over_sigma.
* enclosed_air_difference.csv (new): id, group, kind ('bead pack' / 'rock'), depth_vox, depth_mm,
  sigma_cal_vox, sigma_cal_source, depth_over_sigma, air_reference, contrast, offset_open,
  offset_enclosed, enclosed_minus_open, n_vox_open, n_vox_enclosed.
* synthetic_beads.csv: new relax_overlap_vox, relax_shape (per seed); solid_frac_true (mean),
  solid_frac_true_min, solid_frac_true_max, phi_true on the 'all' rows.
* imaging_summary.json: esf_step_like_rule; enclosed_air_depth entries restricted to depth_over_sigma >=
  DEPTH_MIN_SIGMA, with air_reference, ids, n_below_min_depth_sigma; new enclosed_air_difference (same
  keys 'group | kind | depth d') and enclosed_air_rule; enclosed_air_bubbles with selection, n_bubbles_recorded,
  n_inter_bead, offset_deep_blur_ref_median, offset_deep_excess_median, air_reference; synthetic with
  solid_frac_true(_min, _max) and relax_overlap_vox_max.
"""
from __future__ import annotations

import itertools
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import tifffile
from edt import edt
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.special import erf, ndtr

from .config import Config, read_json, write_json
from .extract import voi_paths
from .grains import beads, measure
from .morph import connectivity, euler_density, face_labels, label6
from .segment import load_filtered, load_pore, median3, replicates, seg_paths
from .topology import critical_diameter_vox
from .wall import harmonics

PARTS = ("esf", "filter", "two_glass", "enclosed_air", "footprint", "synthetic")

# 1. edge-spread function
ESF_DMAX = 6.0            # |d| range of the profile (voxels)
ESF_BIN = 0.25            # bin width (voxels)
ESF_LARGE_VOX = 5         # inscribed radius required on both sides of the interface (voxels)
ESF_TOL_VOX = 1.0         # nearest voxel of the other phase must belong to its large part to within this
ESF_MIN_VOX = 10000       # fewer qualifying voxels: VOI skipped
ESF_MIN_BIN_VOX = 20      # bins with fewer voxels are not fitted
ESF_PLATEAU_VOX = 4.0     # |d| from which a bin counts as plateau
ESF_MIN_PLATEAU_BINS = 3  # populated plateau bins required on each side
TAIL_NEAR = (1.5, 2.5)    # |d| range (voxels) of the near end of the slow tail
TAIL_FAR = 5.0            # |d| from which bins form the far end of the slow tail
ESF_TAIL_MAX = 0.15       # a profile whose residual slow tail exceeds this fraction of the contrast is not step-like
ESF_CAL_SIGMAS = (0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)   # calibration blurs (voxels)
ESF_CAL_R = 30.0          # radius of the calibration pore (voxels)
ESF_CAL_N = 96            # edge of the calibration volume (voxels)
# 3. histogram modes, two glasses
HIST_SMOOTH_BINS = 3.0    # Gaussian smoothing of the pooled histogram (segment.smoothed_histogram)
PEAK_PROMINENCE = 0.02    # peak prominence >= this x tallest smoothed count (segment.histogram_peaks)
CORE_VOX = 5              # bead core: distance from the pore phase > CORE_VOX voxels
SHELL_VOX = 3             # voxels within SHELL_VOX of a bead are re-thresholded with its glass mode
# 4. enclosed air
BUBBLE_MIN_VOX = 20       # smallest bubble recorded (voxels)
BUBBLE_DEEP_TOL = 0.5     # deepest voxels of a bubble: EDT >= max EDT - BUBBLE_DEEP_TOL
DEPTHS = (2, 3, 5, 8)     # depths of qa.deep_offsets (voxels)
F_PROV = 0.6              # provisional pore phase of qa.deep_offsets
DEPTH_MIN_N = 200         # qa.deep_offsets: fewer voxels at a depth -> NaN
BUBBLE_D_BINS_VOX = (0.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, np.inf)   # equivalent-diameter bins (voxels)
BUBBLE_MIN_R_IN_VOX = 3.5  # bubble summary ('all'): inscribed radius (max EDT) >= this, deepest voxels free of blur
BUBBLE_REF_R_VOX = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0)   # true radii of the blur-only reference
BUBBLE_REF_REPS = 16      # noise seeds / sub-voxel centre positions per radius of the blur-only reference
DEPTH_MIN_SIGMA = 3.0     # summary: groups compared only at depth >= this x sigma_cal of the sample
AIR_REF_BEADS = "pore-air mode inside the column (bead pack)"
AIR_REF_ROCKS = "outside air, annulus beyond the plug (rock)"
DIFF_PAIRS = {"bead pack": ("bead-pack open pore", "bead-pack enclosed bubble"),
              "rock": ("rock open pore", "rock enclosed pore")}       # (open, enclosed) sources of the difference
# 5. footprint
FOOT_SUB = 4              # sub-points per binned voxel and axis
FOOT_NZ = 9               # levels along z
# 6. synthetic packs
SYN_EDGE = 300
SYN_VOX_MM = 0.05
SYN_D_VOX = 74.0          # 3.7 mm, about the measured median bead size of CF4/CF5
SYN_GROWTH = 1.015        # drawn diameter / relaxed (non-overlapping) diameter
SYN_FILL = 0.58           # solid fraction of the relaxed packing (of the relaxation spheres)
SYN_SUPER = 2             # partial-volume supersampling per axis
SYN_AIR, SYN_GLASS = 8000.0, 13800.0
SYN_ASPECT = 1.25         # axis ratio of the prolate spheroids
SYN_PW_ITER = 40          # golden-section steps of the Perram-Wertheim contact distance of two spheroids
SYN_SEEDS = (1, 2, 3)
SYN_SIGMA_FALLBACK = 1.0  # blur (voxels) when no bead-pack ESF is available
SYN_NOISE_FALLBACK = 0.05  # noise sd / contrast when no bead-pack ESF is available


# ------------------------------------------------------------------ 1. edge-spread function
def esf_model(d, a, b, d0, sigma):
    """Blurred step: a on the solid side (d -> -inf), a + b on the pore side (d -> +inf)."""
    return a + b * 0.5 * (1.0 + erf((d - d0) / (np.sqrt(2.0) * sigma)))


def signed_distance(pore: np.ndarray):
    """Signed distance to the pore/solid interface (voxels, > 0 in the pore) and the two EDTs."""
    dp = edt(pore)
    ds = edt(~pore)
    d = np.where(pore, dp - 0.5, 0.5 - ds).astype(np.float32)
    return d, dp, ds


def large_interface(pore: np.ndarray, dp: np.ndarray, ds: np.ndarray, r: float = ESF_LARGE_VOX,
                    tol: float = ESF_TOL_VOX) -> np.ndarray:
    """Voxels of large features whose nearest voxel of the other phase belongs to a large feature too.

    Large pore: opening of the pore by a ball of radius r (voxels within r of a voxel with EDT >= r);
    large solid likewise.
    """
    pore_big = pore & (edt(dp < r) <= r)
    solid_big = ~pore & (edt(ds < r) <= r)
    ok = np.zeros(pore.shape, bool)
    if pore_big.any() and solid_big.any():
        t = edt(~solid_big)
        ok |= pore_big & (t <= dp + tol)
        del t
        t = edt(~pore_big)
        ok |= solid_big & (t <= ds + tol)
        del t
    return ok


def esf_selection(pore: np.ndarray, dmax: float = ESF_DMAX):
    """Flat indices and signed distances of the voxels used for the ESF."""
    d, dp, ds = signed_distance(pore)
    ok = large_interface(pore, dp, ds)
    del dp, ds
    ok &= np.abs(d) <= dmax
    m = int(np.ceil(dmax))
    ok[:m] = ok[-m:] = False
    ok[:, :m] = ok[:, -m:] = False
    ok[:, :, :m] = ok[:, :, -m:] = False
    idx = np.flatnonzero(ok)
    return idx, d.ravel()[idx]


def esf_profile(d: np.ndarray, g: np.ndarray, dmax: float = ESF_DMAX, step: float = ESF_BIN) -> pd.DataFrame:
    """Per-bin count, mean distance, median and quartiles of the grey values."""
    nb = int(round(2 * dmax / step))
    b = np.clip(np.floor((d + dmax) / step).astype(np.int64), 0, nb - 1)
    n = np.bincount(b, minlength=nb)
    dsum = np.bincount(b, weights=d, minlength=nb)
    order = np.lexsort((g, b))
    gs = g[order]
    del order
    edges = np.r_[0, np.cumsum(n)]
    rows = []
    for i in range(nb):
        v = gs[edges[i]:edges[i + 1]]
        q = np.percentile(v, [25, 50, 75]) if len(v) else [np.nan] * 3
        rows.append({"bin": i, "d_lo": -dmax + i * step, "n": int(n[i]),
                     "d_mean": dsum[i] / n[i] if n[i] else np.nan,
                     "g_q25": float(q[0]), "g_median": float(q[1]), "g_q75": float(q[2])})
    p = pd.DataFrame(rows)
    p["g_sd_robust"] = (p.g_q75 - p.g_q25) / 1.349
    return p


def fit_esf(prof: pd.DataFrame, min_bin: int = ESF_MIN_BIN_VOX, plateau: float = ESF_PLATEAU_VOX,
            min_plateau_bins: int = ESF_MIN_PLATEAU_BINS, min_vox: int = ESF_MIN_VOX) -> dict:
    """Fit esf_model to the bin medians; NaN with a reason when the data do not support a fit.

    The slow tails are measured on the residuals of the fit (bin median minus fitted erf): on each side,
    the mean residual at |d| in TAIL_NEAR minus the mean residual at |d| >= TAIL_FAR, signed so that > 0
    means the grey value is still approaching that side's plateau beyond what the erf describes, as a
    fraction of the fitted contrast. A pure blurred step gives 0 whatever sigma and d0. No standard error
    of sigma is returned: the curve_fit covariance of the bin medians, whose residuals are systematic and
    correlated, is not an uncertainty of the resolution (use the spread between VOIs).
    """
    nan = float("nan")
    out = {"n_vox": int(prof.n.sum()), "n_bins": 0, "sigma_vox": nan, "d0_vox": nan,
           "plateau_solid": nan, "plateau_pore": nan, "contrast_fit": nan, "fit_rmse": nan,
           "noise_solid_sd": nan, "noise_pore_sd": nan, "tail_pore_frac": nan, "tail_solid_frac": nan, "reason": ""}
    use = prof[prof.n >= min_bin]
    sol, por = use[use.d_mean <= -plateau], use[use.d_mean >= plateau]
    out["n_bins"] = int(len(use))
    if out["n_vox"] < min_vox:
        out["reason"] = f"fewer than {min_vox} interface voxels of large features"
        return out
    if len(sol) < min_plateau_bins or len(por) < min_plateau_bins:
        out["reason"] = f"fewer than {min_plateau_bins} populated plateau bins on a side"
        return out
    x, y = use.d_mean.to_numpy(), use.g_median.to_numpy()
    a0 = float(sol.g_median.median())
    b0 = float(por.g_median.median()) - a0
    try:
        p, _ = curve_fit(esf_model, x, y, p0=[a0, b0, 0.0, 1.0],
                         bounds=([-np.inf, -np.inf, -3.0, 0.05], [np.inf, np.inf, 3.0, 6.0]), maxfev=20000)
    except (RuntimeError, ValueError) as e:
        out["reason"] = f"fit failed: {e}"
        return out
    res = y - esf_model(x, *p)
    out.update(sigma_vox=float(p[3]), d0_vox=float(p[2]),
               plateau_solid=float(p[0]), plateau_pore=float(p[0] + p[1]), contrast_fit=float(-p[1]),
               fit_rmse=float(np.sqrt(np.mean(res ** 2))),
               noise_solid_sd=float(sol.g_sd_robust.median()), noise_pore_sd=float(por.g_sd_robust.median()))
    # slow tail beyond the erf: change of the fit residual between |d| in TAIL_NEAR and |d| >= TAIL_FAR, as
    # a fraction of the fitted contrast (> 0: the grey value keeps approaching the plateau of that side
    # beyond what the fitted erf describes)
    ad = np.abs(x)
    for side, sgn in (("pore", 1), ("solid", -1)):
        on = np.sign(x) == sgn
        near = res[on & (ad >= TAIL_NEAR[0]) & (ad <= TAIL_NEAR[1])]
        far = res[on & (ad >= TAIL_FAR)]
        out[f"tail_{side}_frac"] = float(sgn * (near.mean() - far.mean()) / out["contrast_fit"]) \
            if len(near) and len(far) else nan
    return out


def step_like(f: dict, tail_max: float = ESF_TAIL_MAX) -> bool:
    """True when the profile is a single blurred step: at least one finite residual tail and none above
    tail_max of the contrast (fit_esf's tail_pore_frac, tail_solid_frac)."""
    tails = [v for v in (f["tail_pore_frac"], f["tail_solid_frac"]) if np.isfinite(v)]
    return bool(tails) and max(abs(v) for v in tails) <= tail_max


def blurred_pore(sigma: float, noise_frac: float, seed: int = 0, n: int = ESF_CAL_N, R: float = ESF_CAL_R,
                 air: float = SYN_AIR, solid: float = SYN_GLASS, shift=0.31) -> np.ndarray:
    """uint16 grey volume of a spherical air pore of radius R in glass with a Gaussian edge of width sigma
    (voxels, point-sampled; centre `shift` voxels off the volume centre, a scalar or one value per axis) and
    white noise of noise_frac x contrast."""
    sz, sy, sx = np.broadcast_to(np.asarray(shift, np.float32), (3,))
    z, y, x = np.indices((n, n, n), dtype=np.float32) - (n - 1) / 2
    z += sz
    y += sy
    x += sx
    r = np.sqrt(z * z + y * y + x * x)
    del z, y, x
    g = solid + (air - solid) * ndtr((R - r) / sigma)
    g += np.random.default_rng(seed).normal(0, noise_frac * (solid - air), g.shape)
    return np.clip(np.rint(g), 0, 65535).astype(np.uint16)


def esf_calibration(noise_frac: float, sigmas=ESF_CAL_SIGMAS, seed: int = 0) -> pd.DataFrame:
    """Fitted sigma (unfiltered and median-filtered profiles) against the true blur of a synthetic pore
    processed like a VOI: median filter, ISO-50 mask of the filtered volume, large-feature selection, fit.

    The EDT distance distorts the profile (the nearest voxel of the other phase lies closer to an oblique
    interface than half a voxel), so the fitted sigma of the unfiltered profile is 16-27 % high at sigma =
    0.4 voxel, within 4 % at 0.7-0.8, 6-12 % low at 1-2 voxels and 12-21 % low at 4 voxels (noise 0-16 %
    of the contrast). Over that noise range the unfiltered calibration moves by less than 3 % for sigma =
    0.7-2 voxels and by up to about 10 % at 0.4 and at 4 voxels; the median-filtered one depends strongly
    on the noise (at 16 % noise its fit is 66 % high at 0.4 voxel and 10 % high at 1 voxel).
    """
    noise_frac = noise_frac if np.isfinite(noise_frac) else 0.0
    rows = []
    for st in sigmas:
        grey = blurred_pore(st, noise_frac, seed)
        filt = median3(grey)
        idx, d = esf_selection(filt < 0.5 * (SYN_AIR + SYN_GLASS))
        r = {"sigma_true": st, "noise_frac": noise_frac}
        for name, v in (("none", grey), ("median3", filt)):
            f = fit_esf(esf_profile(d, v.ravel()[idx].astype(np.float32)), min_vox=1000)
            r[f"sigma_fit_{name}"] = f["sigma_vox"]
        rows.append(r)
    return pd.DataFrame(rows)


def calibrate_sigma(sigma_fit: float, cal: pd.DataFrame, name: str) -> float:
    """True blur for a fitted sigma, by linear interpolation of the calibration (NaN outside its range)."""
    c = cal[np.isfinite(cal[f"sigma_fit_{name}"])].sort_values(f"sigma_fit_{name}")
    x, y = c[f"sigma_fit_{name}"].to_numpy(), c.sigma_true.to_numpy()
    if not np.isfinite(sigma_fit) or len(x) < 2 or np.any(np.diff(x) <= 0) or not x[0] <= sigma_fit <= x[-1]:
        return float("nan")
    return float(np.interp(sigma_fit, x, y))


def esf_sample(cfg: Config, s, log=print) -> dict:
    """ESF of the main VOI, unfiltered and median-filtered, with the fitted sigma calibrated on a
    synthetic pore of the same noise."""
    t0 = time.time()
    pore = load_pore(cfg, s.id, 0)
    idx, d = esf_selection(pore)
    del pore
    rows, profs = [], []
    for name in ("none", "median3"):
        grey = tifffile.imread(voi_paths(cfg, s.id, 0)[0]) if name == "none" else load_filtered(cfg, s.id, 0)
        g = grey.ravel()[idx].astype(np.float32)
        del grey
        prof = esf_profile(d, g)
        del g
        f = fit_esf(prof)
        um = s.voxel_mm * 1000.0
        r = {"id": s.id, "group": s.group, "replicate": 0, "filter": name, "voxel_um": um, **f,
             "sigma_um": f["sigma_vox"] * um, "edge_10_90_vox": 2 * 1.2816 * f["sigma_vox"],
             "edge_10_90_um": 2 * 1.2816 * f["sigma_vox"] * um,
             "noise_frac": min(f["noise_solid_sd"], f["noise_pore_sd"]) / f["contrast_fit"]
             if np.isfinite(f["contrast_fit"]) and f["contrast_fit"] > 0 else float("nan")}
        r["step_like"] = step_like(f)
        rows.append(r)
        prof.insert(0, "filter", name)
        prof.insert(0, "id", s.id)
        if np.isfinite(f["sigma_vox"]):
            prof["g_fit"] = esf_model(prof.d_mean.to_numpy(), f["plateau_solid"], f["plateau_pore"] - f["plateau_solid"],
                                      f["d0_vox"], f["sigma_vox"])
        profs.append(prof)
    del idx, d
    cal = esf_calibration(rows[0]["noise_frac"]) if np.isfinite(rows[0]["sigma_vox"]) else pd.DataFrame()
    for r in rows:
        sc = calibrate_sigma(r["sigma_vox"], cal, r["filter"]) if len(cal) else float("nan")
        r.update(sigma_cal_vox=sc, sigma_cal_um=sc * r["voxel_um"], edge_10_90_cal_um=2 * 1.2816 * sc * r["voxel_um"],
                 cal_noise_frac=rows[0]["noise_frac"] if len(cal) else float("nan"))
        log(f"  {s.id} ESF ({r['filter']}): {r['n_vox']} voxels, sigma {r['sigma_vox']:.3f} vox (calibrated "
            f"{sc:.3f} vox = {r['sigma_cal_um']:.1f} um), plateaus {r['plateau_solid']:.0f}/{r['plateau_pore']:.0f}, "
            f"noise {r['noise_solid_sd']:.0f}/{r['noise_pore_sd']:.0f} {r['reason']}")
    log(f"  {s.id} ESF done ({time.time() - t0:.0f} s)")
    return {"rows": rows, "profiles": pd.concat(profs, ignore_index=True).to_dict(orient="list"),
            "calibration": cal.to_dict(orient="list")}


# ------------------------------------------------------------------ 2. median-filter sensitivity
def segmentation_metrics(pore: np.ndarray, dx: float, d_c: bool = True) -> dict:
    """Porosity, connectivity, percolation, chi_V and critical diameters of a pore mask."""
    lab, n = label6(pore)
    r = connectivity(pore, lab, n)
    del lab
    r["chi_V"] = euler_density(pore, dx)
    for ax, a in enumerate("zyx"):
        dv = critical_diameter_vox(pore, ax) if d_c and r[f"perc_{a}"] else float("nan")
        r[f"d_c_vox_{a}"] = dv
        r[f"d_c_mm_{a}"] = dv * dx
    return r


def filter_sample(cfg: Config, s, log=print) -> list[dict]:
    """Main VOI thresholded with and without the 3x3x3 median filter, at T50 and T50int."""
    import porespy as ps
    ps.settings.tqdm["disable"] = True
    seg = read_json(seg_paths(cfg, s.id, 0)["json"])
    thr = {"T50": seg["T50"], "T50int": seg.get("T50int")}
    thr = {k: float(v) for k, v in thr.items() if v is not None and np.isfinite(v)}
    rows = []
    for name in ("median3", "none"):
        t0 = time.time()
        grey = load_filtered(cfg, s.id, 0) if name == "median3" else tifffile.imread(voi_paths(cfg, s.id, 0)[0])
        for tn, t in thr.items():
            pore = load_pore(cfg, s.id, 0) if (name == "median3" and tn == "T50") else grey < t
            r = {"id": s.id, "group": s.group, "replicate": 0, "threshold_name": tn, "threshold": t, "filter": name,
                 "voxel_mm": s.voxel_mm, **segmentation_metrics(pore, s.voxel_mm)}
            del pore
            rows.append(r)
            log(f"  {s.id} {tn} {name}: phi {100 * r['phi']:.3f}% conn {100 * r['frac_connected']:.1f}% "
                f"perc zyx {int(r['perc_z'])}{int(r['perc_y'])}{int(r['perc_x'])} chiV {r['chi_V']:.4g}/mm3 "
                f"d_c zyx {r['d_c_vox_z']:.2f} {r['d_c_vox_y']:.2f} {r['d_c_vox_x']:.2f} vox ({time.time() - t0:.0f} s)")
        del grey
    ref = {r["threshold_name"]: r for r in rows if r["filter"] == "median3"}
    for r in rows:
        m = ref[r["threshold_name"]]
        r["dphi_vs_median3"] = r["phi"] - m["phi"]
        r["phi_rel_change_vs_median3"] = (r["phi"] - m["phi"]) / m["phi"] if m["phi"] > 0 else float("nan")
        r["chi_V_ratio_vs_median3"] = r["chi_V"] / m["chi_V"] if m["chi_V"] != 0 else float("nan")
        r["perc_any_same_as_median3"] = bool(r["perc_any"] == m["perc_any"])
        for a in "zyx":
            r[f"d_c_ratio_{a}_vs_median3"] = r[f"d_c_vox_{a}"] / m[f"d_c_vox_{a}"] \
                if np.isfinite(m[f"d_c_vox_{a}"]) and m[f"d_c_vox_{a}"] > 0 else float("nan")
    return rows


# ------------------------------------------------------------------ 3. glass modes and the CF10 bound
def pooled_peaks(cfg: Config, sid: str):
    """Centres, raw and smoothed counts of the pooled histogram, its peaks and their prominences."""
    z = np.load(seg_paths(cfg, sid, 0)["pooled"])
    c, h = z["centres"].astype(float), z["hist"].astype(float)
    hs = ndi.gaussian_filter1d(h, HIST_SMOOTH_BINS)
    pk, prop = find_peaks(hs, prominence=PEAK_PROMINENCE * hs.max())
    return c, h, hs, pk, prop["prominences"]


def mode_split(c: np.ndarray, hs: np.ndarray, p: int, q: int) -> float:
    """Grey value of the minimum of the smoothed histogram between peaks p and q (indices): the split
    between two solid modes, used for every classification of beads by glass (solid_modes, two_glass)."""
    lo, hi = sorted((int(p), int(q)))
    return float(c[lo + int(np.argmin(hs[lo:hi + 1]))])


def dense_split(rows: list[dict]) -> float | None:
    """split_grey of the highest-grey secondary solid mode above I_solid in solid_modes' rows (the dense
    glass of a two-glass pack), None when there is none."""
    sec = [r for r in rows if r["role"] == "solid_secondary" and r["grey"] > r["I_solid"]]
    return max(sec, key=lambda r: r["grey"])["split_grey"] if sec else None


def bead_cores(filt: np.ndarray, pore: np.ndarray):
    """Beads of the T50 solid (grains.beads) and the median filtered grey of each bead's core (voxels
    farther than CORE_VOX from the pore phase, bubbles included). Returns the label image, the bead
    labels, their core medians and core voxel counts, the voxel count of every label, and the grey-value
    histogram of all core voxels."""
    reg = beads(pore)
    core = edt(~pore) > CORE_VOX
    sel = core & (reg > 0)
    del core
    ids, g = reg[sel], filt[sel]
    del sel
    core_hist = np.bincount(g, minlength=65536)
    o = np.lexsort((g, ids))
    ids, g = ids[o], g[o]
    del o
    lab_u, start, cnt = np.unique(ids, return_index=True, return_counts=True)
    med = np.array([np.median(g[a:a + n]) for a, n in zip(start, cnt)])
    vol = np.bincount(reg.ravel())
    return reg, lab_u, med, cnt, vol, core_hist


def solid_modes(cfg: Config, s, log=print) -> list[dict]:
    """Peaks of a bead pack's pooled histogram with their role; for every secondary solid mode, the beads
    (all VOIs) whose core median lies on its side of the histogram minimum between it and the main mode."""
    ref = read_json(seg_paths(cfg, s.id, 0)["sample"])
    c, h, hs, pk, prom = pooled_peaks(cfg, s.id)
    I_air, I_solid, T50 = ref["I_air"], ref["I_solid"], ref["T50"]
    air = pk[np.argmin(np.abs(c[pk] - I_air))]
    main = pk[np.argmin(np.abs(c[pk] - I_solid))]
    nan = float("nan")
    rows = []
    for p, pr in zip(pk, prom):
        role = "air" if p == air else "solid_main" if p == main else "solid_secondary" if c[p] >= T50 else "other"
        r = {"id": s.id, "group": s.group, "grey": float(c[p]), "height": float(hs[p]),
             "height_rel_main": float(hs[p] / hs[main]), "prominence_rel_max": float(pr / hs.max()),
             "contrast_frac": float((c[p] - I_air) / (I_solid - I_air)), "role": role,
             "I_air": I_air, "I_solid": I_solid, "T50": T50, "split_grey": nan, "n_beads": 0,
             "n_beads_this_mode": 0, "bead_volume_frac_this_mode": nan, "core_vox_frac_this_mode": nan,
             "bead_core_median_this_mode": nan, "bead_core_median_other": nan}
        if role == "solid_secondary":
            r["split_grey"] = mode_split(c, hs, p, main)
        rows.append(r)
    sec = [r for r in rows if r["role"] == "solid_secondary"]
    if not sec:
        return rows
    meds, vols, chist = [], [], np.zeros(65536, np.int64)
    for k in replicates(cfg, s.id):
        filt = load_filtered(cfg, s.id, k)
        pore = load_pore(cfg, s.id, k)
        reg, lab_u, med, cnt, vol, ch = bead_cores(filt, pore)
        del reg, filt, pore
        meds.append(med)
        vols.append(vol[lab_u])
        chist += ch
    med, vol = np.concatenate(meds), np.concatenate(vols)
    g = np.arange(65536)
    for r in sec:
        this = (med < r["split_grey"]) if r["grey"] < I_solid else (med >= r["split_grey"])
        side = (g < r["split_grey"]) if r["grey"] < I_solid else (g >= r["split_grey"])
        r.update(n_beads=int(len(med)), n_beads_this_mode=int(this.sum()),
                 bead_volume_frac_this_mode=float(vol[this].sum() / vol.sum()),
                 core_vox_frac_this_mode=float(chist[side].sum() / chist.sum()),
                 bead_core_median_this_mode=float(np.median(med[this])) if this.any() else nan,
                 bead_core_median_other=float(np.median(med[~this])) if (~this).any() else nan)
        log(f"  {s.id}: secondary solid mode {r['grey']:.0f} ({r['contrast_frac']:.3f} of the contrast): "
            f"{r['n_beads_this_mode']} of {r['n_beads']} beads ({100 * r['bead_volume_frac_this_mode']:.1f}% of bead "
            f"volume), {100 * r['core_vox_frac_this_mode']:.1f}% of bead-core voxels beyond {r['split_grey']:.0f}")
    return rows


def two_glass_voi(filt: np.ndarray, pore: np.ndarray, T_light: float, T_dense: float, split: float):
    """Porosities (a) at T_light (the T50 mask `pore`), (b) at T_dense and (c) bead-wise of one VOI.

    Beads (bead_cores) whose core median grey is >= split are dense; the voxels within SHELL_VOX of a bead
    take the threshold of the nearest bead's glass. Returns the VOI's numbers and, per bead, its label,
    core median, core voxels, volume and class. vox_rethresholded_dense counts the voxels whose class
    changed from (a) to (c); vox_shell_dense the voxels whose nearest bead (within SHELL_VOX) is dense.
    """
    phi_a = float(pore.mean())
    phi_b = float((filt < T_dense).mean())
    reg, lab_u, med, cnt, vol, _ = bead_cores(filt, pore)
    dense = med >= split
    is_dense = np.zeros(vol.size, bool)
    is_dense[lab_u[dense]] = True
    dmask = is_dense[reg]
    lmask = (reg > 0) & ~dmask
    del reg
    dist_d = edt(~dmask) if dmask.any() else np.full(pore.shape, np.inf, np.float32)
    del dmask
    dist_l = edt(~lmask) if lmask.any() else np.full(pore.shape, np.inf, np.float32)
    del lmask
    near = np.minimum(dist_d, dist_l) <= SHELL_VOX
    to_dense = dist_d < dist_l
    del dist_d, dist_l
    pore_c = pore.copy()
    pore_c[near] = np.where(to_dense[near], filt[near] < T_dense, filt[near] < T_light)
    phi_c = float(pore_c.mean())
    n_changed = int((pore_c != pore).sum())
    n_shell_dense = int((near & to_dense).sum())
    del pore_c, near, to_dense
    nd = int(dense.sum())
    nan = float("nan")
    r = {"phi_a_T50": phi_a, "phi_b_T50_dense": phi_b, "phi_c_beadwise": phi_c, "n_beads": int(len(lab_u)),
         "n_dense": nd, "n_light": int(len(lab_u) - nd),
         "dense_volume_frac": float(vol[lab_u[dense]].sum() / max(vol[lab_u].sum(), 1)),
         "vox_rethresholded_dense": n_changed, "vox_shell_dense": n_shell_dense,
         "core_grey_median_light": float(np.median(med[~dense])) if (~dense).any() else nan,
         "core_grey_median_dense": float(np.median(med[dense])) if nd else nan}
    beads_ = {"label": lab_u, "core_grey_median": med, "core_vox": cnt, "volume_vox": vol[lab_u], "dense": dense}
    return r, beads_


def two_glass(cfg: Config, s, log=print, split: float | None = None) -> dict:
    """Porosity of a two-glass pack at the light-glass T50, the dense-glass ISO-50 and a bead-wise ISO-50.

    split: grey value separating light from dense beads; by default (and as passed by `run`, from
    solid_modes) the histogram minimum between the main solid mode and the highest-grey peak (mode_split),
    so that both tables classify the beads alike.
    """
    ref = read_json(seg_paths(cfg, s.id, 0)["sample"])
    c, h, hs, pk, prom = pooled_peaks(cfg, s.id)
    I_air, I_light = ref["I_air"], ref["I_solid"]
    main = pk[np.argmin(np.abs(c[pk] - I_light))]
    dense_pk = pk[np.argmax(c[pk])]
    I_dense = float(c[dense_pk])
    T_light, T_dense = ref["T50"], I_air + 0.5 * (I_dense - I_air)
    if split is None:
        split = mode_split(c, hs, dense_pk, main) if dense_pk != main else float("inf")
    if dense_pk == main:
        log(f"  {s.id}: the highest-grey peak is the main solid mode; no dense glass")
    rows, bead_rows = [], []
    for k in replicates(cfg, s.id):
        t0 = time.time()
        filt = load_filtered(cfg, s.id, k)
        pore = load_pore(cfg, s.id, k)
        v, b = two_glass_voi(filt, pore, T_light, T_dense, split)
        del filt, pore
        for lb, m, n, vv, dd in zip(b["label"], b["core_grey_median"], b["core_vox"], b["volume_vox"], b["dense"]):
            bead_rows.append({"id": s.id, "replicate": k, "label": int(lb), "volume_vox": int(vv),
                              "core_vox": int(n), "core_grey_median": float(m), "dense": bool(dd)})
        r = {"id": s.id, "replicate": k, "I_air": I_air, "I_light": I_light, "I_dense": I_dense, "split_grey": split,
             "T_light": T_light, "T_dense": T_dense, **v}
        rows.append(r)
        log(f"  {s.id} v{k}: phi (a) T50 {100 * v['phi_a_T50']:.2f}%  (b) dense ISO-50 {100 * v['phi_b_T50_dense']:.2f}%  "
            f"(c) bead-wise {100 * v['phi_c_beadwise']:.2f}%; {v['n_dense']} dense of {v['n_beads']} beads "
            f"({100 * v['dense_volume_frac']:.0f}% of bead volume), {v['vox_rethresholded_dense']} voxels "
            f"reclassified ({time.time() - t0:.0f} s)")
    return {"rows": rows, "beads": bead_rows}


# ------------------------------------------------------------------ 4. enclosed air
def _hist_quantile(hist: np.ndarray, q: float) -> float:
    c = np.cumsum(hist)
    if c[-1] == 0:
        return float("nan")
    return float(np.searchsorted(c, q * c[-1]))


def depth_histograms(filt: np.ndarray, air: float, contrast: float, f_prov: float = F_PROV, depths=DEPTHS) -> dict:
    """Grey-value histograms of the voxels >= d inside the provisional pore phase (qa.deep_offsets'
    definition), split into clusters that touch a VOI face (open) and clusters that do not (enclosed)."""
    prov = filt < air + f_prov * contrast
    lab, n = label6(prov)
    outer = np.unique(np.concatenate([face_labels(lab, a, sd) for a in range(3) for sd in (0, 1)]))
    is_open = np.zeros(n + 1, bool)
    is_open[outer] = True
    op = is_open[lab]
    del lab
    dtp = edt(prov)
    del prov
    out = {}
    for dd in depths:
        mm = dtp >= dd
        mm[:dd] = mm[-dd:] = False
        mm[:, :dd] = mm[:, -dd:] = False
        mm[:, :, :dd] = mm[:, :, -dd:] = False
        out[dd] = {"open": np.bincount(filt[mm & op], minlength=65536),
                   "enclosed": np.bincount(filt[mm & ~op], minlength=65536)}
        del mm
    return out


def depth_rows(hist: dict, sid: str, group: str, dx: float, air: float, contrast: float, names, air_ref: str) -> list[dict]:
    """Rows of the depth table from depth_histograms' output (pooled or not): the median grey of the open
    and of the enclosed voxels at every depth as (g - air) / contrast (NaN below DEPTH_MIN_N voxels).
    names: source labels of the open and the enclosed rows; air_ref: what `air` is."""
    rows = []
    for dd in sorted(hist):
        for src, key in zip(names, ("open", "enclosed")):
            hh = hist[dd][key]
            n = int(hh.sum())
            rows.append({"id": sid, "group": group, "source": src, "depth_vox": dd, "depth_mm": dd * dx,
                         "offset": (_hist_quantile(hh, 0.5) - air) / contrast if n >= DEPTH_MIN_N else float("nan"),
                         "n_vox": n, "air_reference": air_ref, "I_air": float(air), "contrast": float(contrast)})
    return rows


def bubbles_voi(filt: np.ndarray, pore: np.ndarray, min_vox: int = BUBBLE_MIN_VOX, tol: float = BUBBLE_DEEP_TOL,
                reg: np.ndarray | None = None) -> pd.DataFrame:
    """Volume, inscribed radius and median grey of the deepest voxels of every bubble (T50 pore cluster
    touching no face) of at least min_vox voxels.

    reg: bead label image of the same VOI (grains.beads(pore), which fills the bubbles). When given, every
    bubble gets n_bead_labels (distinct bead labels > 0 among its voxels) and in_single_bead (all its voxels
    in one bead): a bubble in the glass lies inside one bead, whereas a pocket of the pore space between
    beads that 6-connectivity cuts off from the open pores straddles the watershed lines of two or more.
    """
    cols = ["label", "volume_vox", "r_in_vox", "n_deep", "g_deep", "g_all"]
    if reg is not None:
        cols += ["n_bead_labels", "in_single_bead"]
    lab, n = label6(pore)
    outer = np.unique(np.concatenate([face_labels(lab, a, sd) for a in range(3) for sd in (0, 1)]))
    size = np.bincount(lab.ravel(), minlength=n + 1)
    keep = size >= min_vox
    keep[0] = False
    keep[outer] = False
    sel = keep[lab]
    ids = lab[sel]
    del lab
    dt = edt(pore)[sel]
    g = filt[sel]
    bead = reg[sel] if reg is not None else None
    del sel
    if ids.size == 0:
        return pd.DataFrame(columns=cols)
    o = np.argsort(ids, kind="stable")
    ids, dt, g = ids[o], dt[o], g[o]
    if bead is not None:
        bead = bead[o]
    lab_u, start, cnt = np.unique(ids, return_index=True, return_counts=True)
    rows = []
    for lb, a, m in zip(lab_u, start, cnt):
        t, v = dt[a:a + m], g[a:a + m]
        deep = t >= t.max() - tol
        r = {"label": int(lb), "volume_vox": int(m), "r_in_vox": float(t.max()), "n_deep": int(deep.sum()),
             "g_deep": float(np.median(v[deep])), "g_all": float(np.median(v))}
        if bead is not None:
            u = np.unique(bead[a:a + m])
            r["n_bead_labels"] = int((u > 0).sum())
            r["in_single_bead"] = bool(len(u) == 1 and u[0] > 0)
        rows.append(r)
    return pd.DataFrame(rows, columns=cols)


def bubble_blur_reference(sigma: float, noise_frac: float, radii=BUBBLE_REF_R_VOX, reps: int = BUBBLE_REF_REPS,
                          seed: int = 0) -> pd.DataFrame:
    """Deepest-voxel offset of a bubble with no enclosed-air effect, against its size.

    A spherical air pore of true radius r_true_vox in glass is imaged with the pack's edge (blurred_pore:
    point-sampled Gaussian edge of width sigma = the calibrated unfiltered ESF sigma, white noise of
    noise_frac x contrast, random sub-voxel centre), median filtered (segment.median3), thresholded at the
    mid-grey and measured with bubbles_voi; the offset (g_deep - air) / contrast is then due to the blur
    and the median filter alone. reps noise seeds and centre positions per radius; medians (and quartiles of
    the offset) over them, with the measured inscribed radius and equivalent diameter.
    """
    rng = np.random.default_rng(seed)
    t, con = 0.5 * (SYN_AIR + SYN_GLASS), SYN_GLASS - SYN_AIR
    rows = []
    for R in radii:
        n = 2 * int(np.ceil(R + 4 * sigma + 3)) + 1
        v = []
        for _ in range(reps):
            grey = blurred_pore(sigma, noise_frac, seed=int(rng.integers(2 ** 31)), n=n, R=R,
                                shift=rng.uniform(-0.5, 0.5, 3))
            filt = median3(grey)
            b = bubbles_voi(filt, filt < t, min_vox=1)
            if len(b):
                b = b.loc[b.volume_vox.idxmax()]
                v.append((b.r_in_vox, (6 * b.volume_vox / np.pi) ** (1 / 3), (b.g_deep - SYN_AIR) / con))
        v = np.array(v, float).reshape(-1, 3)
        q = np.percentile(v[:, 2], [25, 50, 75]) if len(v) else [np.nan] * 3
        rows.append({"r_true_vox": float(R), "n_rep": int(len(v)), "sigma_vox": float(sigma), "noise_frac": float(noise_frac),
                     "r_in_vox_median": float(np.median(v[:, 0])) if len(v) else np.nan,
                     "d_eq_vox_median": float(np.median(v[:, 1])) if len(v) else np.nan,
                     "offset_deep_median": float(q[1]), "offset_deep_q25": float(q[0]), "offset_deep_q75": float(q[2])})
    return pd.DataFrame(rows)


def blur_offset_at(d_eq_vox, ref: pd.DataFrame) -> np.ndarray:
    """Blur-only deepest-voxel offset (bubble_blur_reference) interpolated at the bubbles' equivalent
    diameters; NaN below the smallest reference bubble, the largest one's value above it."""
    d_eq_vox = np.asarray(d_eq_vox, float)
    r = ref[np.isfinite(ref.d_eq_vox_median) & np.isfinite(ref.offset_deep_median)].sort_values("d_eq_vox_median")
    if len(r) < 2:
        return np.full(d_eq_vox.shape, np.nan)
    x, y = r.d_eq_vox_median.to_numpy(), r.offset_deep_median.to_numpy()
    return np.where(d_eq_vox >= x[0], np.interp(d_eq_vox, x, y), np.nan)


def bubble_summary(bub: pd.DataFrame, sid: str, group: str, dx: float, r_min: float = BUBBLE_MIN_R_IN_VOX) -> list[dict]:
    """Bubble offsets per equivalent-diameter bin and over the bubbles large enough for their deepest voxels
    to be free of the blur (the 'all' row: inscribed radius >= r_min voxels). Inter-bead pockets
    (in_single_bead false) are left out of every row."""
    nan = float("nan")
    one = bub[bub.in_single_bead.astype(bool)] if "in_single_bead" in bub else bub
    big = one[one.r_in_vox >= r_min]

    def stats(m):
        has_ref = "offset_deep_blur_ref" in m
        return {"n_bubbles": int(len(m)),
                "d_eq_median_mm": float(m.d_eq_mm.median()) if len(m) else nan,
                "r_in_median_vox": float(m.r_in_vox.median()) if len(m) else nan,
                "offset_deep_median": float(m.offset_deep.median()) if len(m) else nan,
                "offset_deep_q25": float(m.offset_deep.quantile(0.25)) if len(m) else nan,
                "offset_deep_q75": float(m.offset_deep.quantile(0.75)) if len(m) else nan,
                "offset_deep_blur_ref_median": float(m.offset_deep_blur_ref.median()) if has_ref and len(m) else nan,
                "offset_deep_excess_median": float(m.offset_deep_excess.median()) if has_ref and len(m) else nan}

    summ = []
    edges = np.array(BUBBLE_D_BINS_VOX)
    for i in range(len(edges) - 1):
        m = one[(one.d_eq_vox >= edges[i]) & (one.d_eq_vox < edges[i + 1])]
        if len(m) == 0:
            continue
        summ.append({"id": sid, "group": group, "d_bin": f"{edges[i]:g}-{edges[i + 1]:g} vox",
                     "d_bin_lo_vox": edges[i], "d_bin_hi_vox": edges[i + 1],
                     "d_bin_lo_mm": edges[i] * dx, "d_bin_hi_mm": edges[i + 1] * dx, **stats(m),
                     "selection": "bubbles inside a single bead"})
    summ.append({"id": sid, "group": group, "d_bin": "all", "d_bin_lo_vox": float(edges[0]), "d_bin_hi_vox": float("inf"),
                 "d_bin_lo_mm": 0.0, "d_bin_hi_mm": float("inf"), **stats(big),
                 "selection": f"bubbles inside a single bead with inscribed radius >= {r_min:g} vox",
                 "r_in_min_vox": float(r_min), "n_bubbles_recorded": int(len(bub)),
                 "n_inter_bead": int(len(bub) - len(one)), "n_below_r_min": int(len(one) - len(big))})
    return summ


def esf_scale(cfg: Config, out) -> dict:
    """Edge width and noise of every sample from esf.csv (unfiltered profile): sigma_cal_vox and noise_frac
    of the sample when its fit is finite, else the median of its group, with the source."""
    f = out / "esf.csv"
    if not f.exists():
        return {}
    e = pd.read_csv(f)
    e = e[(e["filter"] == "none") & np.isfinite(e.sigma_cal_vox)]
    grp = e.groupby("group")[["sigma_cal_vox", "noise_frac"]].median()
    res = {}
    for s in cfg.samples:
        m = e[e.id == s.id]
        if len(m):
            res[s.id] = {"sigma_cal_vox": float(m.sigma_cal_vox.iloc[0]), "noise_frac": float(m.noise_frac.iloc[0]),
                         "sigma_cal_source": "sample"}
        elif s.group in grp.index:
            res[s.id] = {"sigma_cal_vox": float(grp.loc[s.group, "sigma_cal_vox"]),
                         "noise_frac": float(grp.loc[s.group, "noise_frac"]), "sigma_cal_source": "group median"}
    return res


def enclosed_air_sample(cfg: Config, s, log=print) -> dict:
    """Bubbles and depth offsets of all VOIs of a bead pack (air reference: the pore-air mode inside the
    column), with the blur-only reference of the bubble offsets at the pack's edge width."""
    ref = read_json(seg_paths(cfg, s.id, 0)["sample"])
    I_air, I_solid = ref["I_air"], ref["I_solid"]
    con = I_solid - I_air
    dx = s.voxel_mm
    sc = esf_scale(cfg, cfg.outdir("imaging")).get(s.id, {})
    sig, noise = sc.get("sigma_cal_vox", float("nan")), sc.get("noise_frac", float("nan"))
    src = sc.get("sigma_cal_source", "fallback")
    if not np.isfinite(sig):
        sig, src = SYN_SIGMA_FALLBACK, "fallback (no ESF)"
    noise = noise if np.isfinite(noise) else SYN_NOISE_FALLBACK
    t0 = time.time()
    bref = bubble_blur_reference(sig, noise)
    bref.insert(0, "sigma_source", src)
    bref.insert(0, "id", s.id)
    log(f"  {s.id}: blur-only bubble reference at sigma {sig:.3f} vox, noise {noise:.3f} ({src}): deepest offset "
        + " ".join(f"r{r.r_true_vox:g}:{r.offset_deep_median:.3f}" for r in bref.itertuples())
        + f" ({time.time() - t0:.0f} s)")
    bubbles = []
    hist = {dd: {"open": np.zeros(65536, np.int64), "enclosed": np.zeros(65536, np.int64)} for dd in DEPTHS}
    n_vox = phi_bub = 0
    for k in replicates(cfg, s.id):
        t0 = time.time()
        filt = load_filtered(cfg, s.id, k)
        pore = load_pore(cfg, s.id, k)
        reg = beads(pore)
        b = bubbles_voi(filt, pore, reg=reg)
        del reg
        b.insert(0, "replicate", k)
        b.insert(0, "group", s.group)
        b.insert(0, "id", s.id)
        b["d_eq_vox"] = (6 * b.volume_vox.astype(float) / np.pi) ** (1 / 3)
        b["d_eq_mm"] = b.d_eq_vox * dx
        b["r_in_mm"] = b.r_in_vox * dx
        b["offset_deep"] = (b.g_deep - I_air) / con
        b["offset_all"] = (b.g_all - I_air) / con
        b["offset_deep_blur_ref"] = blur_offset_at(b.d_eq_vox, bref)
        b["offset_deep_excess"] = b.offset_deep - b.offset_deep_blur_ref
        bubbles.append(b)
        n_vox += pore.size
        phi_bub += int(b.volume_vox.sum())
        del pore
        for dd, hh in depth_histograms(filt, I_air, con).items():
            hist[dd]["open"] += hh["open"]
            hist[dd]["enclosed"] += hh["enclosed"]
        del filt
        ok = b[b.in_single_bead.astype(bool) & (b.r_in_vox >= BUBBLE_MIN_R_IN_VOX)]
        log(f"  {s.id} v{k}: {len(b)} bubbles >= {BUBBLE_MIN_VOX} vox ({int((~b.in_single_bead.astype(bool)).sum())} "
            f"inter-bead), {len(ok)} in one bead with r_in >= {BUBBLE_MIN_R_IN_VOX:g} vox: median deepest offset "
            f"{ok.offset_deep.median():.3f} (blur alone {ok.offset_deep_blur_ref.median():.3f}) ({time.time() - t0:.0f} s)")
    bub = pd.concat(bubbles, ignore_index=True)
    depth = depth_rows(hist, s.id, s.group, dx, I_air, con, ("bead-pack open pore", "bead-pack enclosed bubble"),
                       AIR_REF_BEADS)
    summ = bubble_summary(bub, s.id, s.group, dx)
    summ[-1].update(phi_bubbles=phi_bub / n_vox, n_vois=len(bubbles), sigma_blur_ref_vox=sig)
    return {"bubbles": bub.to_dict(orient="records"), "summary": summ, "depth": depth,
            "blur_reference": bref.to_dict(orient="records")}


def rock_enclosed_sample(cfg: Config, s, log=print) -> dict:
    """Open and enclosed pore air at depth in the main VOI of a rock, with the outside-air reference of
    the segmentation (I_air of the rocks: the air annulus beyond the plug), as qa.deep_offsets."""
    t0 = time.time()
    ref = read_json(seg_paths(cfg, s.id, 0)["sample"])
    air = ref.get("air_outside")
    if air is None or not np.isfinite(air):
        log(f"  {s.id}: no outside-air reference; skipped")
        return {"depth": [], "reason": "no outside-air reference"}
    con = ref["I_solid"] - air
    filt = load_filtered(cfg, s.id, 0)
    h = depth_histograms(filt, air, con)
    del filt
    rows = depth_rows(h, s.id, s.group, s.voxel_mm, air, con, ("rock open pore", "rock enclosed pore"), AIR_REF_ROCKS)
    off = {(r["source"], r["depth_vox"]): r["offset"] for r in rows}
    log(f"  {s.id}: enclosed - open offset by depth (outside-air reference) "
        + " ".join(f"{dd}:{off[('rock enclosed pore', dd)] - off[('rock open pore', dd)]:+.3f}" for dd in DEPTHS)
        + f" ({time.time() - t0:.0f} s)")
    return {"depth": rows, "air_reference": AIR_REF_ROCKS, "I_air": air, "contrast": con}


def rock_depth_rows(cfg: Config) -> list[dict]:
    """Internal-air offsets at depth of the rocks (qa stage, main VOI; outside-air reference)."""
    rows = []
    for s in cfg.samples:
        f = cfg.outdir("qa") / f"{s.id}_qa.json"
        if s.group == "granular" or not f.exists():
            continue
        q = read_json(f)
        for dd in DEPTHS:
            off = q.get(f"offset_depth{dd}", float("nan"))
            rows.append({"id": s.id, "group": s.group, "source": "rock pore", "depth_vox": dd,
                         "depth_mm": dd * s.voxel_mm, "offset": float("nan") if off is None else off,
                         "n_vox": q.get(f"n_depth{dd}", 0), "air_reference": AIR_REF_ROCKS,
                         "I_air": q.get("I_air", float("nan")),
                         "contrast": q.get("I_solid", float("nan")) - q.get("I_air", float("nan"))})
    return rows


def depth_table(cfg: Config, out) -> pd.DataFrame:
    """enclosed_air_depth.csv: bead-pack rows, rock open/enclosed rows and the qa rock rows, each labelled
    with its air reference, with the sample's edge width and the depth in units of it."""
    rows = _json_rows(out, "*_enclosed_air.json", "depth") + _json_rows(out, "*_enclosed_air_rock.json", "depth")
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows + rock_depth_rows(cfg))
    for col in ("I_air", "contrast"):                    # absent from files written before this version
        if col not in d:
            d[col] = np.nan
    ref = d.source.map(lambda v: AIR_REF_BEADS if str(v).startswith("bead-pack") else AIR_REF_ROCKS)
    d["air_reference"] = d.air_reference.fillna(ref) if "air_reference" in d else ref
    sc = esf_scale(cfg, out)
    d["sigma_cal_vox"] = d.id.map(lambda i: sc.get(i, {}).get("sigma_cal_vox", np.nan)).astype(float)
    d["sigma_cal_source"] = d.id.map(lambda i: sc.get(i, {}).get("sigma_cal_source", "none"))
    d["depth_over_sigma"] = d.depth_vox / d.sigma_cal_vox
    return d


def difference_table(depth: pd.DataFrame) -> pd.DataFrame:
    """enclosed_air_difference.csv: (enclosed - open) offset per sample and depth, bead packs (bubbles in the
    glass minus open pore space, pore-air reference) and rocks (enclosed minus open pores, outside-air
    reference). The air reference cancels in the difference; only the normalising contrast differs."""
    rows = []
    for kind, (so, se) in DIFF_PAIRS.items():
        o = depth[depth.source == so].set_index(["id", "depth_vox"])
        e = depth[depth.source == se].set_index(["id", "depth_vox"])
        for key in o.index.intersection(e.index):
            ro, re_ = o.loc[key], e.loc[key]
            rows.append({"id": key[0], "group": ro.group, "kind": kind, "depth_vox": key[1], "depth_mm": ro.depth_mm,
                         "sigma_cal_vox": ro.sigma_cal_vox, "sigma_cal_source": ro.sigma_cal_source,
                         "depth_over_sigma": ro.depth_over_sigma, "air_reference": ro.air_reference,
                         "contrast": ro.contrast, "offset_open": ro.offset, "offset_enclosed": re_.offset,
                         "enclosed_minus_open": re_.offset - ro.offset,
                         "n_vox_open": int(ro.n_vox), "n_vox_enclosed": int(re_.n_vox)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5. VOI footprint
def footprint_phi(phi_bins: np.ndarray, coef, slope, axis_yx, origin_yx, n: int, nz: int,
                  sub: int = FOOT_SUB, nzs: int = FOOT_NZ) -> dict:
    """Area-weighted mean of a radial porosity profile over a square VOI footprint.

    phi_bins[k]: porosity at a distance from the wall in [k, k+1) voxels; coef, slope: two-harmonic wall
    (wall.harmonics) and its change per slice; axis_yx: column axis; origin_yx: position of the VOI's first
    pixel; n: VOI edge; nz: slices (the wall model is centred on slice (nz - 1) / 2). All in voxels of the
    same grid.
    """
    coef, slope = np.asarray(coef, float), np.asarray(slope, float)
    u = (np.arange(n * sub) + 0.5) / sub - 0.5
    yy = origin_yx[0] + u[:, None] - axis_yx[0]
    xx = origin_yx[1] + u[None, :] - axis_yx[1]
    r = np.hypot(yy, xx)
    H = harmonics(np.arctan2(yy, xx))
    del yy, xx
    acc = cnt = out = 0
    dmin = np.inf
    for z in np.linspace(0, nz - 1, nzs):
        d = H @ (coef + slope * (z - (nz - 1) / 2)) - r
        inside = d > 0
        k = np.clip(np.floor(d[inside]).astype(int), 0, len(phi_bins) - 1)
        acc += phi_bins[k].sum()
        cnt += int(inside.sum())
        out += int((~inside).sum())
        dmin = min(dmin, float(d.min()))
    return {"phi_pred": acc / cnt if cnt else float("nan"), "frac_outside_wall": out / (out + cnt),
            "d_min_vox": dmin}


def voi_section_origin(meta: dict, bin_origin_yx) -> tuple[float, float]:
    """Position of the VOI's first pixel on the binned section grid of the wall stage (binned pixel J covers
    raw pixels o + b J ... o + b J + b - 1, o = bin_origin_yx = crop0 mod b): VOI pixel j, which covers raw
    pixels crop0 + b j ... crop0 + b j + b - 1, is section pixel (crop0 - o) / b + j."""
    b = meta["bin"]
    oy, ox = bin_origin_yx
    return (meta["crop_y0"] - oy) / b, (meta["crop_x0"] - ox) / b


def footprint_sample(cfg: Config, s, log=print) -> dict:
    """Porosity of the main VOI predicted from the radial profile of the column alone."""
    w = read_json(cfg.outdir("wall") / f"{s.id}_wall.json")
    pf = cfg.outdir("wall") / f"{s.id}_wall_profile.csv"
    if pf.exists():
        prof = pd.read_csv(pf)
    else:
        prof = pd.read_csv(cfg.outdir("wall") / "wall_profiles.csv")
        prof = prof[prof.id == s.id]
    meta = read_json(voi_paths(cfg, s.id, 0)[1])
    t = pd.read_csv(cfg.outdir("topology") / "topology.csv")
    t = t[(t.id == s.id) & (t.variant == "T50") & (t.replicate == 0)]
    phi_meas = float(t.phi.iloc[0]) if len(t) else float("nan")
    r = footprint_from(w, prof, meta, phi_meas, s)
    log(f"  {s.id}: VOI phi measured {100 * phi_meas:.2f}%, predicted from the radial profile "
        f"{100 * r['phi_voi_pred_radial']:.2f}%, core {100 * w['phi_core']:.2f}% (inscribed disc "
        f"{100 * r['phi_profile_inscribed_disc']:.2f}%, core outside it {100 * r['phi_profile_core_outside_inscribed']:.2f}%); "
        f"footprint corner {r['d_min_wall_mm']:.2f} mm from the wall, "
        f"{100 * r['frac_footprint_within_grain_of_wall']:.2f}% of it within one bead of the wall")
    return r


def footprint_from(w: dict, prof: pd.DataFrame, meta: dict, phi_meas: float, s) -> dict:
    """footprint_sample's numbers from the wall result w (<id>_wall.json), its radial profile prof
    (distance_from_wall_mm, phi), the meta of the main VOI and its measured porosity."""
    vox = w["voxel_mm"]
    k = np.rint(prof.distance_from_wall_mm.to_numpy() / vox - 0.5).astype(int)
    phi_bins = np.full(k.max() + 1, np.nan)
    phi_bins[k] = prof.phi.to_numpy()
    if np.isnan(phi_bins).any():
        good = np.flatnonzero(np.isfinite(phi_bins))
        phi_bins = np.interp(np.arange(len(phi_bins)), good, phi_bins[good])
    origin = voi_section_origin(meta, w["bin_origin_yx"])
    n, nz = meta["shape"][1], meta["shape"][0]
    f = footprint_phi(phi_bins, w["wall_coef_px"], w["wall_coef_slope_px"], w["axis_yx_px"], origin, n, nz)
    grain = (s.grain_mm or 3.5) / vox
    near = footprint_phi((np.arange(len(phi_bins)) < grain).astype(float), w["wall_coef_px"],
                         w["wall_coef_slope_px"], w["axis_yx_px"], origin, n, nz)["phi_pred"]
    # where the core porosity differs from the VOI's: the profile averaged (weight r) over the disc inscribed
    # in the VOI (r < half edge; its annuli are measured on voxels of the VOI) and over the core outside it
    d_mm = (np.arange(len(phi_bins)) + 0.5) * vox
    r_mm = w["inner_radius_used_px"] * vox - d_mm
    half = n * vox / 2
    inner, ring = (r_mm > 0) & (r_mm < half), (r_mm >= half) & (d_mm > grain * vox)
    wt = np.clip(r_mm, 0, None)
    r = {"id": s.id, "group": s.group, "phi_voi_measured": phi_meas, "phi_voi_pred_radial": f["phi_pred"],
         "pred_minus_measured": f["phi_pred"] - phi_meas, "phi_core": w["phi_core"], "phi_section": w["phi_section"],
         "measured_minus_core": phi_meas - w["phi_core"], "pred_minus_core": f["phi_pred"] - w["phi_core"],
         "phi_profile_inscribed_disc": float((wt * phi_bins)[inner].sum() / wt[inner].sum()),
         "phi_profile_core_outside_inscribed": float((wt * phi_bins)[ring].sum() / wt[ring].sum())
         if ring.any() else float("nan"),
         "frac_footprint_in_inscribed_disc": float(np.pi / 4),
         "voi_axis_offset_y_vox": origin[0] + (n - 1) / 2 - w["axis_yx_px"][0],
         "voi_axis_offset_x_vox": origin[1] + (n - 1) / 2 - w["axis_yx_px"][1],
         "inner_radius_mm": w["inner_radius_used_px"] * vox, "voi_edge_mm": n * vox,
         "d_min_wall_mm": f["d_min_vox"] * vox, "frac_footprint_within_grain_of_wall": near,
         "grain_mm": s.grain_mm, "frac_footprint_outside_wall": f["frac_outside_wall"],
         "voxel_mm": vox, "voi_origin_section_yx": list(origin), "sub_points_per_voxel": FOOT_SUB, "z_levels": FOOT_NZ}
    return r


# ------------------------------------------------------------------ 6. synthetic bead packs
def relax_packing(n: int, L: float, diam: float, rng, max_iter: int = 20000, tol: float = 1e-4):
    """n soft spheres of diameter diam in a periodic box, pushed apart until no pair overlaps by more
    than tol * diam. Returns the centres, the iterations and the largest remaining overlap / diam."""
    x = rng.uniform(0, L, (n, 3))
    target = diam * (1 + tol)
    worst = np.inf
    for it in range(1, max_iter + 1):
        dv = x[:, None, :] - x[None, :, :]
        dv -= L * np.round(dv / L)
        dist = np.sqrt((dv ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        ov = np.clip(target - dist, 0, None)
        worst = float(np.clip(diam - dist, 0, None).max() / diam)
        if worst <= tol:
            return x, it, worst
        x += 0.5 * ((ov / np.maximum(dist, 1e-9))[..., None] * dv).sum(1)
        x %= L
    return x, max_iter, worst


def spheroid_contact_distance(u: np.ndarray, ei: np.ndarray, ej: np.ndarray, a: float, b: float,
                              n_iter: int = SYN_PW_ITER) -> np.ndarray:
    """Centre distance at which two equal prolate spheroids (semi-axes a >= b, unit long axes ei, ej; rows)
    touch when one is translated along the unit vector u (rows): the Perram-Wertheim contact function
    F(l) = l (1 - l) u^T [(1 - l) A^-1 + l B^-1]^-1 u, A^-1 = b^2 I + (a^2 - b^2) e e^T, is concave in
    l in [0, 1] and the spheroids at centre distance r overlap iff r^2 max_l F(l) < 1, so the contact
    distance is 1 / sqrt(max_l F) (exact; golden-section search over l)."""
    u, ei, ej = (np.atleast_2d(np.asarray(v, float)) for v in (u, ei, ej))
    eye = np.eye(3)
    Ai = b * b * eye + (a * a - b * b) * ei[:, :, None] * ei[:, None, :]
    Bi = b * b * eye + (a * a - b * b) * ej[:, :, None] * ej[:, None, :]

    def F(lam):
        M = (1 - lam)[:, None, None] * Ai + lam[:, None, None] * Bi
        return lam * (1 - lam) * np.einsum("pi,pi->p", u, np.linalg.solve(M, u[:, :, None])[:, :, 0])

    g = (np.sqrt(5) - 1) / 2
    lo, hi = np.zeros(len(u)), np.ones(len(u))
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = F(c), F(d)
    for _ in range(n_iter):
        left = fc > fd                                   # maximum in [lo, d]
        hi = np.where(left, d, hi)
        lo = np.where(left, lo, c)
        c_new, d_new = hi - g * (hi - lo), lo + g * (hi - lo)
        f_new = F(np.where(left, c_new, d_new))
        fd, fc = np.where(left, fc, f_new), np.where(left, f_new, fd)
        c, d = c_new, d_new
    return 1.0 / np.sqrt(np.maximum(np.maximum(fc, fd), 1e-300))


def relax_spheroids(n: int, L: float, a: float, b: float, axes: np.ndarray, rng, max_iter: int = 20000,
                    tol: float = 1e-4):
    """n soft prolate spheroids (semi-axes a >= b, fixed long axes `axes`) in a periodic box, pushed apart
    along the centre line until no pair overlaps by more than tol x its contact distance
    (spheroid_contact_distance). Returns the centres, the iterations and the largest remaining overlap
    (fraction of the contact distance)."""
    x = rng.uniform(0, L, (n, 3))
    iu, ju = np.triu_indices(n, 1)
    worst = np.inf
    for it in range(1, max_iter + 1):
        dv = x[iu] - x[ju]
        dv -= L * np.round(dv / L)
        dist = np.sqrt((dv ** 2).sum(-1))
        c = np.flatnonzero(dist < 2 * a)                  # farther pairs cannot touch
        if c.size == 0:
            return x, it, 0.0
        dc = np.maximum(dist[c], 1e-9)
        sig = spheroid_contact_distance(dv[c] / dc[:, None], axes[iu[c]], axes[ju[c]], a, b)
        worst = float((np.clip(sig - dist[c], 0, None) / sig).max())
        if worst <= tol:
            return x, it, worst
        ov = np.clip(sig * (1 + tol) - dist[c], 0, None)
        push = 0.5 * (ov / dc)[:, None] * dv[c]
        dx = np.zeros_like(x)
        np.add.at(dx, iu[c], push)
        np.add.at(dx, ju[c], -push)
        x += dx
        x %= L
    return x, max_iter, worst


def digitised_overlap(centres: np.ndarray, L: int, a: float, b: float, axes=None) -> int:
    """Voxel centres of a periodic box of L voxels that lie inside two or more of the spheroids (semi-axes
    a >= b, long axes `axes`; spheres when a == b): an overlap check of a packing on its digitised union."""
    cnt = np.zeros((L, L, L), np.uint8)
    u = np.arange(L, dtype=float)
    for i, c0 in enumerate(centres):
        for sh in itertools.product((-L, 0, L), repeat=3):
            c = c0 + np.array(sh, float)
            lo = np.searchsorted(u, c - a, "left")
            hi = np.searchsorted(u, c + a, "right")
            if np.any(hi <= lo):
                continue
            z = u[lo[0]:hi[0], None, None] - c[0]
            y = u[None, lo[1]:hi[1], None] - c[1]
            x = u[None, None, lo[2]:hi[2]] - c[2]
            r2 = z * z + y * y + x * x
            if a == b:
                inside = r2 <= a * a
            else:
                e = axes[i]
                t2 = (z * e[0] + y * e[1] + x * e[2]) ** 2
                inside = t2 / (a * a) + (r2 - t2) / (b * b) <= 1.0
            cnt[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] += inside
    return int((cnt >= 2).sum())


def random_axes(n: int, rng) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def digitise(centres: np.ndarray, L: int, a: float, b: float, axes=None, sup: int = SYN_SUPER) -> np.ndarray:
    """Partial-volume occupancy (0-1, float32) of the union of prolate spheroids (semi-axes a >= b, long
    axis `axes`; spheres when a == b) in a periodic box of L voxels, from sup^3 sub-samples per voxel."""
    S = L * sup
    occ = np.zeros((S, S, S), bool)
    u = (np.arange(S) + 0.5) / sup - 0.5
    for i, c0 in enumerate(centres):
        for sh in itertools.product((-L, 0, L), repeat=3):
            c = c0 + np.array(sh, float)
            lo = np.searchsorted(u, c - a, "left")
            hi = np.searchsorted(u, c + a, "right")
            if np.any(hi <= lo):
                continue
            z = u[lo[0]:hi[0], None, None] - c[0]
            y = u[None, lo[1]:hi[1], None] - c[1]
            x = u[None, None, lo[2]:hi[2]] - c[2]
            r2 = z * z + y * y + x * x
            if a == b:
                inside = r2 <= a * a
            else:
                e = axes[i]
                t2 = (z * e[0] + y * e[1] + x * e[2]) ** 2
                inside = t2 / (a * a) + (r2 - t2) / (b * b) <= 1.0
            occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= inside
    cnt = np.zeros((L, L, L), np.uint8)
    for dz, dy, dx in itertools.product(range(sup), repeat=3):
        cnt += occ[dz::sup, dy::sup, dx::sup]
    del occ
    return cnt.astype(np.float32) / sup ** 3


def synthetic_pack(L: int, d_vox: float, aspect: float, seed: int, sigma: float, noise_sd: float,
                   fill: float = SYN_FILL, growth: float = SYN_GROWTH, air: float = SYN_AIR,
                   glass: float = SYN_GLASS) -> tuple[np.ndarray, dict]:
    """Pore mask of a synthetic pack imaged like the bead columns, and the facts of its construction.

    n particles of the volume of a sphere of diameter d_vox fill `fill` of the box when shrunk by `growth`
    (n = fill L^3 / (pi/6 (d_vox / growth)^3), the same n for spheres and spheroids); they are relaxed at
    the shrunk size (spheres: relax_packing; spheroids: random fixed orientations, relax_spheroids with the
    exact contact distance) and drawn at the full size. relax_overlap_vox: voxel centres inside two shrunk
    particles (digitised_overlap; 0 when the relaxation converged); solid_frac_true: partial-volume solid
    fraction of the drawn union.
    """
    rng = np.random.default_rng(seed)
    R = d_vox / 2
    b = R / aspect ** (1 / 3)
    a = aspect * b                                   # same volume as the sphere of diameter d_vox
    d_relax = d_vox / growth                         # volume-equivalent diameter of the relaxed particles
    n = int(round(fill * L ** 3 / (np.pi / 6 * d_relax ** 3)))
    if aspect == 1:
        x, it, worst = relax_packing(n, L, d_relax, rng)
        axes = None
    else:
        axes = random_axes(n, rng)
        x, it, worst = relax_spheroids(n, L, a / growth, b / growth, axes, rng)
    ov_vox = digitised_overlap(x, L, a / growth, b / growth, axes)
    vol = digitise(x, L, a, b, axes)
    solid_frac = float(vol.mean())
    vol *= glass - air
    vol += air
    if sigma > 0:
        vol = ndi.gaussian_filter(vol, sigma, mode="wrap", output=np.float32)
    for z0 in range(0, L, 50):
        vol[z0:z0 + 50] += noise_sd * rng.standard_normal(vol[z0:z0 + 50].shape, dtype=np.float32)
    grey = np.clip(np.rint(vol), 0, 65535).astype(np.uint16)
    del vol
    filt = median3(grey)
    del grey
    pore = filt < 0.5 * (air + glass)
    del filt
    info = {"n_particles": n, "relax_iterations": it, "relax_max_overlap_frac": worst, "relax_overlap_vox": ov_vox,
            "relax_shape": "spheres" if aspect == 1 else "spheroids (Perram-Wertheim contact distance)",
            "solid_frac_true": solid_frac, "semi_axis_a_vox": a, "semi_axis_b_vox": b}
    return pore, info


def measure_synthetic(pore: np.ndarray, vox_mm: float, name: str) -> tuple[list[dict], int]:
    """grains.beads + grains.measure on a synthetic pore mask."""
    s = SimpleNamespace(id=name, voxel_mm=vox_mm)
    return measure(beads(pore), s)


def synthetic_params(out, exclude=()) -> dict:
    """Blur and noise of the synthetic packs, from the unfiltered ESF of the bead packs.

    exclude: ids left out (the two-glass packs: their profile mixes the interfaces of two glasses and is
    not a single erf). The digitisation (SYN_SUPER point samples per axis, averaged) already applies an
    aperture of variance (1 - 1/SYN_SUPER^2) / 12 voxel^2, which is removed in quadrature.
    """
    f = out / "esf.csv"
    if f.exists():
        e = pd.read_csv(f)
        e = e[(e.group == "granular") & (e["filter"] == "none") & np.isfinite(e.sigma_cal_vox) & e.step_like.astype(bool)
              & ~e.id.isin(list(exclude))]
        if len(e):
            sc = float(e.sigma_cal_vox.median())
            ap = (1 - 1 / SYN_SUPER ** 2) / 12
            return {"sigma_esf_vox": sc, "sigma_vox": float(np.sqrt(max(sc ** 2 - ap, 0.0))),
                    "noise_frac": float(e.noise_frac.median()), "aperture_var_vox2": ap,
                    "excluded": list(exclude),
                    "source": f"median of the calibrated unfiltered ESF of {len(e)} bead packs ({', '.join(e.id)}), "
                              f"digitisation aperture ({ap:.4f} vox^2) removed in quadrature"
                              + (f"; two-glass packs excluded ({', '.join(exclude)})" if len(exclude) else "")}
    return {"sigma_esf_vox": float("nan"), "sigma_vox": SYN_SIGMA_FALLBACK, "noise_frac": SYN_NOISE_FALLBACK,
            "aperture_var_vox2": float("nan"), "excluded": list(exclude), "source": "fallback (no bead-pack ESF)"}


def bead_summary(rows: list[dict], d_true_mm: float) -> dict:
    d = pd.DataFrame(rows)
    if d.empty:
        return {"n_beads": 0}
    return {"n_beads": int(len(d)), "d_eq_median_mm": float(d.d_eq_mm.median()),
            "d_eq_ratio_median": float(d.d_eq_mm.median() / d_true_mm),
            "d_eq_ratio_p10": float(d.d_eq_mm.quantile(0.1) / d_true_mm),
            "d_eq_ratio_p90": float(d.d_eq_mm.quantile(0.9) / d_true_mm),
            "aspect_median": float(d.aspect.median()), "aspect_p90": float(d.aspect.quantile(0.9)),
            "frac_aspect_gt_1_2": float((d.aspect > 1.2).mean())}


def synthetic_beads(cfg: Config, log=print) -> dict:
    """Spheres and prolate spheroids imaged and measured like the bead packs."""
    out = cfg.outdir("imaging")
    par = synthetic_params(out, exclude=tuple(s.id for s in cfg.samples if s.solid_mode == "lightest"))
    noise_sd = par["noise_frac"] * (SYN_GLASS - SYN_AIR)
    d_true_mm = SYN_D_VOX * SYN_VOX_MM
    rows, beads_all = [], []
    for shape, aspect in (("sphere", 1.0), ("prolate", SYN_ASPECT)):
        pooled, sl_all, solid = [], 0, []
        for seed in SYN_SEEDS:
            t0 = time.time()
            pore, info = synthetic_pack(SYN_EDGE, SYN_D_VOX, aspect, seed, par["sigma_vox"], noise_sd)
            phi = float(pore.mean())
            bead_rows, sl = measure_synthetic(pore, SYN_VOX_MM, f"synthetic_{shape}_{seed}")
            del pore
            for r in bead_rows:
                r.update(shape=shape, seed=seed)
            pooled += bead_rows
            sl_all += sl
            r = {"shape": shape, "seed": seed, "aspect_true": aspect, "d_true_mm": d_true_mm, "edge_vox": SYN_EDGE,
                 "voxel_mm": SYN_VOX_MM, "sigma_esf_vox": par["sigma_esf_vox"], "sigma_blur_vox": par["sigma_vox"],
                 "noise_sd": noise_sd,
                 "noise_frac": par["noise_frac"], **info, "phi_segmented": phi, "phi_true": 1 - info["solid_frac_true"],
                 "n_slivers": sl, **bead_summary(bead_rows, d_true_mm)}
            rows.append(r)
            solid.append(info["solid_frac_true"])
            log(f"  synthetic {shape} seed {seed}: {info['n_particles']} particles (relaxed in {info['relax_iterations']} "
                f"iterations, {info['relax_overlap_vox']} overlapping voxels), solid fraction {info['solid_frac_true']:.3f}, "
                f"phi {100 * phi:.1f}% (true {100 * r['phi_true']:.1f}%), {r['n_beads']} beads measured, "
                f"d_eq/d {r.get('d_eq_ratio_median', np.nan):.3f}, "
                f"aspect median {r.get('aspect_median', np.nan):.3f} p90 {r.get('aspect_p90', np.nan):.3f} "
                f"({time.time() - t0:.0f} s)")
        rows.append({"shape": shape, "seed": "all", "aspect_true": aspect, "d_true_mm": d_true_mm, "edge_vox": SYN_EDGE,
                     "voxel_mm": SYN_VOX_MM, "sigma_esf_vox": par["sigma_esf_vox"], "sigma_blur_vox": par["sigma_vox"],
                     "noise_sd": noise_sd, "noise_frac": par["noise_frac"],
                     "solid_frac_true": float(np.mean(solid)), "solid_frac_true_min": float(np.min(solid)),
                     "solid_frac_true_max": float(np.max(solid)), "phi_true": float(1 - np.mean(solid)),
                     "n_slivers": sl_all, **bead_summary(pooled, d_true_mm)})
        beads_all += pooled
    return {"params": par, "rows": rows, "beads": beads_all}


# ------------------------------------------------------------------ tables and summary
def _json_rows(out, pattern: str, key: str | None = None) -> list[dict]:
    rows = []
    for f in sorted(out.glob(pattern)):
        d = read_json(f)
        v = d if key is None else d.get(key, [])
        rows.extend(v if isinstance(v, list) else [v])
    return rows


def _order(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "id" not in df:
        return df
    order = {s.id: i for i, s in enumerate(cfg.samples)}
    return df.assign(_o=df.id.map(order)).sort_values(["_o"], kind="stable").drop(columns="_o")


def rebuild_tables(cfg: Config, out) -> None:
    """All CSV tables from the per-sample JSON files."""
    def save(rows, name):
        if rows:
            _order(cfg, pd.DataFrame(rows)).to_csv(out / name, index=False)

    save(_json_rows(out, "*_esf.json", "rows"), "esf.csv")
    profs = []
    for f in sorted(out.glob("*_esf.json")):
        profs.append(pd.DataFrame(read_json(f)["profiles"]))
    if profs:
        _order(cfg, pd.concat(profs, ignore_index=True)).to_csv(out / "esf_profiles.csv", index=False)
    save(_json_rows(out, "*_filter.json"), "filter_sensitivity.csv")
    save(_json_rows(out, "*_two_glass.json", "rows"), "cf10_two_glass.csv")
    save(_json_rows(out, "*_two_glass.json", "beads"), "cf10_two_glass_beads.csv")
    save(_json_rows(out, "*_solid_modes.json"), "solid_modes.csv")
    save(_json_rows(out, "*_enclosed_air.json", "bubbles"), "enclosed_air_bubbles.csv")
    save(_json_rows(out, "*_enclosed_air.json", "summary"), "enclosed_air_summary.csv")
    save(_json_rows(out, "*_enclosed_air.json", "blur_reference"), "enclosed_air_bubble_blur_reference.csv")
    depth = depth_table(cfg, out)
    if len(depth):
        _order(cfg, depth).to_csv(out / "enclosed_air_depth.csv", index=False)
        diff = difference_table(depth)
        if len(diff):
            _order(cfg, diff).to_csv(out / "enclosed_air_difference.csv", index=False)
    save(_json_rows(out, "*_footprint.json"), "voi_footprint.csv")
    if (out / "synthetic_beads.json").exists():
        syn = read_json(out / "synthetic_beads.json")
        pd.DataFrame(syn["rows"]).to_csv(out / "synthetic_beads.csv", index=False)
        pd.DataFrame(syn["beads"]).to_csv(out / "synthetic_beads_grains.csv", index=False)


def _stats(v: pd.Series) -> dict:
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "median": float(v.median()), "min": float(v.min()), "max": float(v.max())}


def summarise(cfg: Config, out) -> dict:
    """Key numbers of every analysis, from the CSV tables."""
    S: dict = {}
    f = out / "esf.csv"
    if f.exists():
        e = pd.read_csv(f)
        S["esf"] = {}
        for (grp, flt), d in e.groupby(["group", "filter"]):
            ok = d[np.isfinite(d.sigma_vox) & d.step_like.astype(bool)]
            S["esf"][f"{grp}_{flt}"] = {"n_vois": int(len(d)), "n_fitted": int(len(ok)),
                                        "fitted": ok.id.tolist(), "sigma_fit_vox": _stats(ok.sigma_vox),
                                        "sigma_cal_vox": _stats(ok.sigma_cal_vox), "sigma_cal_um": _stats(ok.sigma_cal_um),
                                        "edge_10_90_cal_um": _stats(ok.edge_10_90_cal_um), "d0_vox": _stats(ok.d0_vox),
                                        "tail_pore_frac": _stats(ok.tail_pore_frac),
                                        "tail_solid_frac": _stats(ok.tail_solid_frac),
                                        "noise_frac": _stats(ok.noise_frac),
                                        "not_step_like": d[np.isfinite(d.sigma_vox) & ~d.step_like.astype(bool)].id.tolist(),
                                        "skipped": dict(zip(d[~np.isfinite(d.sigma_vox)].id, d[~np.isfinite(d.sigma_vox)].reason))}
        S["esf_step_like_rule"] = (f"residual slow tail (fit residual at |d| in {TAIL_NEAR} minus at |d| >= {TAIL_FAR} "
                                   f"voxels) <= {ESF_TAIL_MAX} of the contrast on both sides")
    f = out / "filter_sensitivity.csv"
    if f.exists():
        e = pd.read_csv(f)
        raw = e[e["filter"] == "none"]
        S["filter"] = {}
        for (grp, tn), d in raw.groupby(["group", "threshold_name"]):
            S["filter"][f"{grp}_{tn}"] = {
                "n_vois": int(len(d)), "dphi_abs_pct_points": _stats(100 * d.dphi_vs_median3),
                "phi_rel_change": _stats(d.phi_rel_change_vs_median3),
                "chi_V_ratio": _stats(d.chi_V_ratio_vs_median3),
                "n_perc_any_changed": int((~d.perc_any_same_as_median3.astype(bool)).sum()),
                "perc_changed_ids": d[~d.perc_any_same_as_median3.astype(bool)].id.tolist(),
                "d_c_ratio": _stats(pd.concat([d[f"d_c_ratio_{a}_vs_median3"] for a in "zyx"]))}
    f = out / "cf10_two_glass.csv"
    if f.exists():
        e = pd.read_csv(f)
        S["two_glass"] = {sid: {"I_light": float(d.I_light.iloc[0]), "I_dense": float(d.I_dense.iloc[0]),
                                "phi_a_T50": d.phi_a_T50.tolist(), "phi_b_T50_dense": d.phi_b_T50_dense.tolist(),
                                "phi_c_beadwise": d.phi_c_beadwise.tolist(),
                                "phi_mean": {k: float(d[k].mean()) for k in ("phi_a_T50", "phi_b_T50_dense", "phi_c_beadwise")},
                                "n_dense": d.n_dense.tolist(), "n_beads": d.n_beads.tolist(),
                                "dense_volume_frac": d.dense_volume_frac.tolist()} for sid, d in e.groupby("id")}
    f = out / "solid_modes.csv"
    if f.exists():
        e = pd.read_csv(f)
        sec = e[e.role == "solid_secondary"]
        S["solid_modes"] = {sid: {k: d[k].tolist() for k in ("grey", "contrast_frac", "height_rel_main",
                                                              "prominence_rel_max", "split_grey", "n_beads",
                                                              "n_beads_this_mode", "bead_volume_frac_this_mode",
                                                              "core_vox_frac_this_mode")}
                            for sid, d in sec.groupby("id")}
        S["solid_modes_prominence_rule"] = f"prominence >= {PEAK_PROMINENCE} x tallest count, smoothing {HIST_SMOOTH_BINS} bins"
    f = out / "enclosed_air_summary.csv"
    if f.exists():
        e = pd.read_csv(f)
        allr = e[e.d_bin == "all"]
        nan = float("nan")

        def count(v):
            return int(v) if v is not None and np.isfinite(v) else nan

        S["enclosed_air_bubbles"] = {
            r["id"]: {"n_bubbles": int(r["n_bubbles"]), "offset_deep_median": r["offset_deep_median"],
                      "offset_deep_iqr": [r["offset_deep_q25"], r["offset_deep_q75"]],
                      "d_eq_median_mm": r["d_eq_median_mm"], "phi_bubbles": r["phi_bubbles"],
                      "selection": r.get("selection", "all bubbles"), "n_bubbles_recorded": count(r.get("n_bubbles_recorded")),
                      "n_inter_bead": count(r.get("n_inter_bead")),
                      "offset_deep_blur_ref_median": r.get("offset_deep_blur_ref_median", nan),
                      "offset_deep_excess_median": r.get("offset_deep_excess_median", nan),
                      "air_reference": AIR_REF_BEADS}
            for r in allr.to_dict(orient="records")}
    rule = (f"groups compared only at depths >= {DEPTH_MIN_SIGMA:g} sigma_cal (unfiltered ESF of the sample, its group "
            f"median when its fit failed); bead-pack offsets are relative to the {AIR_REF_BEADS}, rock offsets to the "
            f"{AIR_REF_ROCKS}: compare (enclosed - open) between them, not the absolute offsets")
    for name, col, src_col in (("enclosed_air_depth", "offset", "source"),
                               ("enclosed_air_difference", "enclosed_minus_open", "kind")):
        f = out / f"{name}.csv"
        if not f.exists():
            continue
        e = pd.read_csv(f)
        if "depth_over_sigma" not in e:
            e["depth_over_sigma"] = np.nan
        if "air_reference" not in e:
            e["air_reference"] = e[src_col].map(lambda v: AIR_REF_BEADS if str(v).startswith("bead") else AIR_REF_ROCKS)
        S[name] = {}
        for (grp, src, dd), d in e.groupby(["group", src_col, "depth_vox"]):
            dq = d[d.depth_over_sigma >= DEPTH_MIN_SIGMA]
            st = _stats(dq[col])
            st.update(air_reference=str(d.air_reference.iloc[0]), ids=dq[np.isfinite(dq[col])].id.tolist(),
                      n_below_min_depth_sigma=int(len(d) - len(dq)))
            S[name][f"{grp} | {src} | depth {dd}"] = st
        S["enclosed_air_rule"] = rule
    f = out / "voi_footprint.csv"
    if f.exists():
        e = pd.read_csv(f)
        S["footprint"] = {r.id: {"phi_measured": r.phi_voi_measured, "phi_pred_radial": r.phi_voi_pred_radial,
                                 "phi_core": r.phi_core, "pred_minus_measured": r.pred_minus_measured,
                                 "d_min_wall_mm": r.d_min_wall_mm} for r in e.itertuples()}
    f = out / "synthetic_beads.csv"
    if f.exists():
        e = pd.read_csv(f)
        per_seed = e[e.seed.astype(str) != "all"]
        S["synthetic"] = {r["shape"]: {k: r.get(k, float("nan")) for k in (
            "aspect_true", "n_beads", "d_eq_ratio_median", "aspect_median", "aspect_p90", "frac_aspect_gt_1_2",
            "sigma_esf_vox", "sigma_blur_vox", "noise_sd", "solid_frac_true", "solid_frac_true_min", "solid_frac_true_max")}
            | {"relax_overlap_vox_max": float(per_seed[per_seed["shape"] == r["shape"]].relax_overlap_vox.max())
               if "relax_overlap_vox" in per_seed else float("nan")}
            for r in e[e.seed.astype(str) == "all"].to_dict(orient="records")}
    return S


# ------------------------------------------------------------------ stage
def run(cfg: Config, only=None, log=print, parts=None) -> None:
    out = cfg.outdir("imaging")
    parts = PARTS if parts is None else [p for p in PARTS if p in parts]
    sel = [s for s in cfg.samples if not only or s.id in only]
    packs = [s for s in sel if s.group == "granular"]
    for part in parts:
        log(f" imaging: {part}")
        if part == "esf":
            for s in sel:
                write_json(out / f"{s.id}_esf.json", esf_sample(cfg, s, log))
        elif part == "filter":
            for s in sel:
                write_json(out / f"{s.id}_filter.json", filter_sample(cfg, s, log))
        elif part == "two_glass":
            for s in packs:
                rows = solid_modes(cfg, s, log)
                write_json(out / f"{s.id}_solid_modes.json", rows)
                log(f"  {s.id}: modes " + ", ".join(f"{r['role']} {r['grey']:.0f} ({r['contrast_frac']:.2f})" for r in rows))
                if s.solid_mode == "lightest":
                    write_json(out / f"{s.id}_two_glass.json", two_glass(cfg, s, log, split=dense_split(rows)))
        elif part == "enclosed_air":
            for s in packs:
                write_json(out / f"{s.id}_enclosed_air.json", enclosed_air_sample(cfg, s, log))
            for s in sel:
                if s.group != "granular":
                    write_json(out / f"{s.id}_enclosed_air_rock.json", rock_enclosed_sample(cfg, s, log))
        elif part == "footprint":
            for s in packs:
                if (cfg.outdir("wall") / f"{s.id}_wall.json").exists():
                    write_json(out / f"{s.id}_footprint.json", footprint_sample(cfg, s, log))
        elif part == "synthetic" and not only:
            write_json(out / "synthetic_beads.json", synthetic_beads(cfg, log))
        rebuild_tables(cfg, out)
    write_json(out / "imaging_summary.json", summarise(cfg, out))
