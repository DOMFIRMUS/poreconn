"""Stage `predict`: does porosity or the Euler number density predict whether a sub-volume percolates?

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Cases. Every non-overlapping cube of stage `rev` and every axis is one case (y = percolation along that
axis); cubes with phi = 0 are kept (chi_V = 0 there) and counted (`n_empty`). rev.csv must have the column
S_V (chi_cube is the primary feature) and, when config.yaml lists more than one air reference
(`air_variants`), the column `variant`; otherwise the stage stops with an error (stale stage rev output).
Each variant (T50, T50int) is analysed separately. The 3 axis rows of a cube share its features, so every
model is fitted on cubes (binomial: share of percolating axes, weight 3 - the same likelihood as the 3
rows) and every permutation moves whole cubes; AUCs are computed on the cases.

Features (asinh compresses the heavy tails of chi_V; s0 = CHI_SCALE sets where it starts to act: the
dimensionless chi_V l^3 of the cubes spans about 1e-4 to 1, so asinh with unit scale would be the
identity over the data range):
  phi        porosity of the cube (fraction)
  chi_main   asinh(chi_V l_p,main^3 / s0), l_p,main = 4 phi / S_V of the specimen's main VOI (replicate 0)
             at the same air variant as the cases: topology.csv, or (T50int: topology has S_V only at T50)
             the whole-VOI cube of rev.csv (edge N; at T50 it equals the topology chord); the T50 chord only
             where neither has that variant for the specimen - recorded in `note` and in the summary. The
             same length for every cube of a specimen, so across specimens this feature also encodes the
             specimen's pore size (the normalisation of report.py)
  chi_cube   asinh(chi_V l_c^3 / s0), l_c = 4 phi / S_V of the cube itself (0 for empty cubes). If it is
             non-finite in some case of a cell (S_V = 0 at phi = 1), the chi_cube models of that cell are
             dropped and the conditional primary becomes chi_main; `note` says so.
  chi_raw    asinh(chi_V / s1), s1 = median |chi_V| of the cases of the cell (non-zero values if that
             median is 0); only for subsets imaged with a single voxel size (coquina, trachyte)
Porosity basis. Wherever porosity enters a model it enters as phi, sqrt(phi), phi^2 (PHI_BASIS; each
column standardised - within the training fold for the LOSO models, by its within-specimen SD for the
fixed-effects models), so that a non-linear P(y | phi) is not credited to chi, which correlates with phi
(and chi_cube contains phi through l_c). Model names keep 'phi'; column `design` lists the fitted columns.

Analyses, per variant x subset (all, rocks = not granular, coquina, trachyte) x cube edge (voxels):
  (a) pooled leave-one-specimen-out (LOSO) AUC of logistic models (report.loso_auc: out-of-fold scores of
      all folds pooled into one ROC curve), with a specimen-bootstrap interval of the fixed scores. Each
      fold has its own intercept (the training prevalence) and the held-out prevalence anti-correlates
      with it, so for an uninformative feature this AUC lies below 0.5. auc_pooled_slope ranks the held-out
      cases of the same folds (loso_fast, raw units) by the linear predictor without the fold constant,
      X_te b; that removes the intercept artefact, but for a feature clustered by specimen a smaller
      leave-one-out bias of the slope remains (uninformative clustered feature: about 0.46 in simulation),
      so for both pooled AUCs the reference is the structured null (b), not 0.5;
  (b) structured permutation null of both pooled AUCs: at cube level (the 3 axis rows of a cube move
      together) every raw feature is split into the specimen mean and the cube deviations from it; the
      specimen means are permuted across specimens and the deviations among the cubes of each specimen,
      and the whole LOSO is refitted. This keeps the clustering of the feature by specimen; a permutation
      of single rows destroys it (for a clustered, uninformative feature it rejected in 35-52 % of simulated
      data sets at a nominal 5 %; this null: about 5 % or less, tests/test_predict.py). The phi basis is
      rebuilt from the permuted phi (sqrt of max(phi, 0): a swapped mean can make a few values negative).
      p = (1 + #null >= obs) / (1 + n), one-sided; its floor is 1 / (1 + n), so report p at the floor as
      "p < 1 / (1 + n)";
  (c) within-specimen AUC of out-of-fold scores of LOSO folds fitted on WITHIN-specimen variation: fold k is
      a fixed-effects logistic regression (one intercept per specimen, common slopes; fe_loso) on the other
      specimens with both outcomes, and specimen k is scored X_te b without intercept (which does not change
      the ranking inside a specimen). Folds fitted on pooled data are dominated by differences between
      specimens and can reverse the within-specimen direction (Simpson's paradox). auc_within: concordant
      pairs (percolating vs non-percolating case of the SAME specimen, ties 1/2) summed over specimens / all
      such pairs - this weights a specimen by n1 n0, so auc_within_specmean (mean of the per-specimen AUCs)
      is given as a sensitivity check. Specimen-cluster bootstrap intervals; null: cubes permuted within
      each specimen (no mean swap), LOSO refitted; null centre about 0.5;
  (d) two-feature models: paired specimen-bootstrap intervals of the change of within-specimen and of
      pooled AUC relative to phi alone, and p_d_within, the one-sided permutation p of the change of within
      AUC (chi improves it) when only chi is permuted (Freedman-Lane: chi = fitted value of its regression
      on the phi basis and specimen intercepts + residual; the residuals are permuted among the cubes of
      each specimen, so chi keeps its relation to phi and to the specimen and loses only its own
      information on y).
  A cell is skipped (status) when fewer than MIN_BOTH specimens have both outcomes or a LOSO training fold
  has a single class. With MIN_BOTH <= n < MIN_INFER specimens with both outcomes the cluster bootstrap has
  too few distinct resamples for nominal coverage (n_distinct_resamples = C(2n - 1, n): 35 for n = 4, 126
  for n = 5): only point estimates are given, the inferential columns (intervals, nulls, p-values) are NaN
  and status is 'descriptive only (n specimens with both outcomes)'.
Conditional analysis: fixed-effects logistic regression y ~ phi basis + chi + specimen intercepts on the
specimens with both outcomes (the others carry no conditional information), covariates standardised by
their WITHIN-specimen SD (fixed effects use only within-specimen variation, so a total SD would make the
"per SD" coefficients depend on the spread between specimens of the subset). beta_chi: chi coefficient per
within-SD; beta_phi: median over the cubes used (phi > 0) of d(linear predictor)/d(phi) times the
within-SD of phi. Tests of chi given phi and specimen (the cases are not independent - 3 axes per cube,
spatially correlated cubes - so the chi2 LR p-value is far too small and is kept only as p_lr_naive):
p_boot, two-sided specimen-cluster bootstrap p = max(1 / (B + 1), 2 min(share_chi_pos, 1 - share_chi_pos));
p_perm_cond, Freedman-Lane permutation p of the LR statistic (chi residualised on the phi basis and the
specimen, residuals permuted among the cubes of each specimen, full model refitted; the reduced model does
not change). A ridge L2_DUMMY on the intercepts is used only if the unpenalised fit does not converge
(separation; column `l2_dummy`). Same MIN_INFER guard.
phi50: porosity at which a logistic fit P(y | phi) crosses 0.5 (as report.phi50), per group with a
percolation transition (report.auc_design) and edge, with a specimen bootstrap that records the discarded
resamples. phi50_extrapolated flags an estimate outside the central 95 % of the cases' phi (phi_q025,
phi_q975), with fewer than PHI50_MIN_NEAR cases within +-CAL_BIN of it (phi50_n_near), without both
outcomes among those, or with a negative lower bound (phi50_lo_negative) - also when the slope is not
positive (status 'slope not positive', phi50 NaN). p_fit_phi0: fitted P at phi = 0. Goodness of fit from
the calibration bins with >= CAL_GOF_MIN_N cases: cal_max_abs_dev (largest |observed - fitted|) and
cal_mean_abs_dev (case-weighted mean). With fewer than MIN_INFER specimens the interval is NaN (status
'descriptive only (n specimens)'); phi50_lo_negative still uses it. Calibration table: observed share and
mean fitted probability in porosity bins.

Outputs (outputs/predict/); phi as a fraction; the stage pools all specimens, so it always rebuilds every
table (`only` is ignored). Columns marked + are new in this version:
  predict_auc.csv          one row per variant x subset x edge x model. Cell description: n_cases, n_pos,
                           n_specimens(_pos/_neg/_both), n_empty (phi = 0 cases) and n_empty_pos, edge_mm range,
                           chi_raw_scale_s1, +n_distinct_resamples (C(2n - 1, n), n = specimens with both
                           outcomes), +note (non-fatal: chi_cube non-finite, chord of another variant).
                           Model: features (raw), +design (fitted columns), +n_perm (permutations drawn).
                           auc_pooled(_lo/_hi): pooled LOSO AUC (report.loso_auc) and specimen-bootstrap
                           interval of the fixed out-of-fold scores; null_mean/_lo/_hi, null_n_perm, p_perm: its
                           structured permutation null (2.5/97.5 %) and one-sided p;
                           +auc_pooled_slope(_lo/_hi), +null_slope_mean/_lo/_hi, +p_perm_slope: the same for the
                           ranking without fold constants (reference: its null, near but below 0.5);
                           auc_within(_lo/_hi), within_pairs: within-specimen AUC of the within-fitted folds;
                           +null_within_n_perm, null_within_mean/_lo/_hi, p_perm_within: its null (cubes
                           permuted within specimens); +auc_within_specmean(_lo/_hi): mean of per-specimen AUCs;
                           auc_within_expected(+_lo/+_hi): within-specimen AUC of the raw feature in the
                           expected direction (phi up, chi down), no fit; auc_pooled_nonempty,
                           auc_within_nonempty: the same scores on phi > 0 cases only; d_auc_within/
                           d_auc_pooled(_lo/_hi): two-feature model minus phi alone, paired specimen bootstrap;
                           +p_d_within, +d_within_n_perm: Freedman-Lane permutation p of d_auc_within; boot_n:
                           usable resamples; status: 'ok', 'descriptive only (...)' or why skipped.
  predict_conditional.csv  one row per variant x subset x edge x chi feature (primary = chi_cube if available,
                           else chi_main): n_cases_used, n_specimens_used/_dropped, +n_distinct_resamples,
                           beta_phi, beta_chi (per within-specimen SD, see above) with bootstrap intervals,
                           share_chi_pos, lr_stat, +p_boot, +p_perm_cond, +perm_cond_n, +p_lr_naive (renamed
                           from p_lr: chi2(1) LR p treating cases as independent - not for inference),
                           l2_dummy, boot_used, status.
  predict_phi50.csv        one row per variant x group x edge: phi50(_lo/_hi), observed phi range (phi_min,
                           phi_max, +phi_q025, +phi_q975), +phi50_n_near, +phi50_n_near_pos, +phi50_lo_negative,
                           phi50_extrapolated (criteria above), slope (per unit phi) and intercept of the full
                           fit, +p_fit_phi0, converged, +cal_bins_gof, +cal_max_abs_dev, +cal_mean_abs_dev,
                           +n_distinct_resamples (all specimens of the group), bootstrap resamples used /
                           discarded (non-positive slope, one class) / not converged but kept, status.
  predict_calibration.csv  porosity bins (bin_lo, bin_hi): n, n_pos, phi_mean, observed share, mean fitted P.
  predict_summary.json     {"params", "chord_fallback": {variant: [ids]}, "cells": {variant: {subset: {edge:
                           {counts, status, note, "auc": {model: {"pooled": [est, lo, hi], "null": [mean, lo,
                           hi, n_perm], "p_perm", "pooled_slope": [est, lo, hi], "null_slope": [mean, lo, hi],
                           "p_perm_slope", "within": [est, lo, hi], "null_within": [mean, lo, hi],
                           "p_perm_within", "within_specmean": [est, lo, hi], "within_expected", "nonempty":
                           [pooled, within], "d_within": [est, lo, hi], "p_d_within", "d_pooled": [..]}},
                           "conditional": {chi: {"beta_chi": [est, lo, hi], "share_chi_pos", "p_boot",
                           "p_perm_cond", "p_lr_naive", "status"}}}}}}, "phi50": {variant: {group: {edge:
                           {"phi50": [est, lo, hi], "extrapolated", "n_near", "p_fit_phi0", "cal_max_abs_dev",
                           "status", ...}}}}}
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.special import expit

from .config import Config, write_json
from .report import GROUP_ORDER, auc_design, flag, loso_auc, pore_chord

CHI_SCALE = 0.01          # s0 of asinh(chi_V l^3 / s0), dimensionless
L2_LOSO = 1e-3            # ridge on the standardised slopes, as report._fit_logit
L2_DUMMY = 1e-2           # ridge on the specimen intercepts, used only when needed (separation)
MIN_BOTH = 3              # specimens with both outcomes needed for a cell (point estimates)
MIN_INFER = 8             # ... needed for intervals and p-values (cluster bootstrap, permutation nulls)
N_PERM = 200              # permutations of the AUC nulls (pooled, within, chi given phi)
N_PERM_EDGE = {50: 100, 75: 100}   # fewer for the largest cells (runtime; p floor 1/101); recorded (n_perm)
N_PERM_COND = 200         # Freedman-Lane permutations of the conditional LR test
N_PERM_COND_EDGE = {50: 100, 75: 100}
N_BOOT = 1000             # specimen bootstrap of the AUCs
N_BOOT_COND = 500         # specimen bootstrap of the fixed-effects coefficients (refitted)
N_BOOT_PHI50 = 2000       # specimen bootstrap of phi50
CAL_BIN = 0.025           # width of the porosity bins of the calibration table
CAL_MIN_N = 6             # minimum cases per calibration bin
CAL_GOF_MIN_N = 20        # minimum cases per bin for the goodness-of-fit summary of the phi50 fit
PHI50_MIN_NEAR = 20       # cases within +-CAL_BIN of phi50 needed for it not to be flagged as extrapolated
PHI_BASIS = ("phi", "phi_sqrt", "phi_sq")   # porosity enters every model as phi, sqrt(phi), phi^2
SEED = 20261027
SUBSETS = ("all", "rocks", "coquina", "trachyte")
EXPECTED_SIGN = {"phi": 1.0, "chi_main": -1.0, "chi_cube": -1.0, "chi_raw": -1.0}


# ------------------------------------------------------------------ cases and features
def main_chord(topo: pd.DataFrame, variant: str) -> pd.Series:
    """Mean pore chord 4 phi / S_V of every specimen's main VOI (replicate 0) at an air variant, by id
    (chord_pore_mean where S_V gives none); empty if topology.csv has no row of that variant."""
    m = topo[(topo.variant == variant) & (topo.replicate == 0)].drop_duplicates("id").set_index("id")
    if not len(m):
        return pd.Series(dtype=float)
    lp = pore_chord(m.phi, m.S_V) if "S_V" in m else np.full(len(m), np.nan)
    alt = m["chord_pore_mean"].to_numpy(float) if "chord_pore_mean" in m else np.full(len(m), np.nan)
    return pd.Series(np.where(np.isfinite(lp), lp, alt), index=m.index).dropna()


def voi_chord(rev: pd.DataFrame, variant: str) -> pd.Series:
    """Mean pore chord 4 phi / S_V of every specimen's main VOI (replicate 0) at an air variant from stage
    rev: its whole-VOI cube (edge N, the only non-overlapping cube of its edge), by id. topology.csv has S_V
    and the chords only at T50 (for T50 the two agree exactly); this gives the T50int chord."""
    r = rev[(rev.variant == variant) & (rev.replicate == 0)]
    if not len(r):
        return pd.Series(dtype=float)
    r = r[r.groupby(["id", "edge"]).cube.transform("size") == 1]
    r = r[r.edge == r.groupby("id").edge.transform("max")].drop_duplicates("id").set_index("id")
    return pd.Series(pore_chord(r.phi, r.S_V), index=r.index).dropna()


def case_table(cfg: Config) -> pd.DataFrame:
    """One row per non-overlapping cube of rev.csv and axis, with phi, chi_V and the chi features."""
    rev = pd.read_csv(cfg.outdir("rev") / "rev.csv")
    air = list(cfg.raw.get("air_variants", ["T50"]))
    if len(air) > 1 and "variant" not in rev:
        raise ValueError(f"predict: config air_variants = {air} but rev.csv has no 'variant' column (stale "
                         "output of stage rev) - rerun stage rev for all samples")
    if "S_V" not in rev:
        raise ValueError("predict: rev.csv has no 'S_V' column, so chi_cube (the primary feature) cannot be "
                         "computed (stale output of stage rev) - rerun stage rev for all samples")
    rev = rev[~flag(rev, "overlapping")].copy()
    rev["variant"] = rev["variant"].fillna("T50").astype(str) if "variant" in rev else "T50"
    rev["voxel_mm"] = rev.id.map({s.id: s.voxel_mm for s in cfg.samples})
    keep = ["variant", "id", "group", "replicate", "edge", "edge_mm", "cube", "voxel_mm", "phi", "chi_V", "S_V"]
    long = pd.concat([rev[keep].assign(axis=a, y=flag(rev, f"perc_{a}").astype(int)) for a in "zyx"],
                     ignore_index=True)
    long.loc[long.phi <= 0, "chi_V"] = 0.0
    topo = pd.read_csv(cfg.outdir("topology") / "topology.csv")
    lp50 = main_chord(topo, "T50")
    long["l_p_main"] = np.nan
    long["l_p_main_variant"] = "T50"
    for v in long.variant.unique():
        sel = (long.variant == v).to_numpy()
        ids = long.loc[sel, "id"]
        lp = ids.map(main_chord(topo, v) if v != "T50" else lp50)
        if v != "T50":                    # topology.csv has no chord at T50int: whole-VOI cube of stage rev
            lp = lp.where(lp.notna(), ids.map(voi_chord(rev, v)))
        fb = lp.isna().to_numpy()
        lp = lp.where(~fb, ids.map(lp50))
        long.loc[sel, "l_p_main"] = lp.to_numpy(float)
        long.loc[sel, "l_p_main_variant"] = np.where(fb, "T50", v)
    miss = sorted(long.loc[long.l_p_main.isna(), "id"].unique())
    if miss:
        raise ValueError(f"predict: no main-VOI pore chord in topology.csv for {miss} - rerun stage topology")
    long["chi_main"] = np.arcsinh(long.chi_V * long.l_p_main ** 3 / CHI_SCALE)
    long["l_c"] = pore_chord(long.phi, long.S_V)
    long["chi_cube"] = np.where(long.phi <= 0, 0.0, np.arcsinh(long.chi_V * long.l_c ** 3 / CHI_SCALE))
    return long


def subset_mask(df: pd.DataFrame, subset: str) -> pd.Series:
    if subset == "all":
        return pd.Series(True, index=df.index)
    if subset == "rocks":
        return df.group != "granular"
    return df.group == subset


def add_chi_raw(d: pd.DataFrame) -> float:
    """Adds chi_raw = asinh(chi_V / s1) to d (in place); returns s1 (median |chi_V| of the cell)."""
    a = d.chi_V.abs().to_numpy(float)
    s1 = float(np.median(a))
    if not s1 > 0:
        s1 = float(np.median(a[a > 0])) if (a > 0).any() else np.nan
    d["chi_raw"] = np.arcsinh(d.chi_V / s1)
    return s1


def phi_basis(phi) -> np.ndarray:
    """Columns phi, sqrt(phi), phi^2 (sqrt of max(phi, 0): a swapped specimen mean in the structured
    permutation can push a few permuted values below 0)."""
    phi = np.asarray(phi, float)
    return np.column_stack([phi, np.sqrt(np.maximum(phi, 0.0)), phi ** 2])


def add_phi_basis(d: pd.DataFrame) -> None:
    """Adds the columns phi_sqrt and phi_sq of PHI_BASIS to d (in place)."""
    b = phi_basis(d.phi.to_numpy(float))
    d["phi_sqrt"], d["phi_sq"] = b[:, 1], b[:, 2]


def design_cols(feats) -> list:
    """Fitted columns of a model: phi expanded to PHI_BASIS."""
    return [c for f in feats for c in (PHI_BASIS if f == "phi" else (f,))]


def design_matrix(F, feats) -> np.ndarray:
    """Design matrix from the raw features F (columns in the order of feats), phi expanded to its basis."""
    F = np.asarray(F, float).reshape(len(F), -1)
    return np.column_stack([phi_basis(F[:, j]) if f == "phi" else F[:, j] for j, f in enumerate(feats)])


def cube_index(d: pd.DataFrame):
    """Cube of every case (its 3 axis rows share the features): (key = cube code of each case, ordered by
    specimen; first = one case per cube; w = cases per cube). Without the columns replicate and cube every
    case is its own cube."""
    if {"replicate", "cube"} <= set(d.columns):
        key = d.groupby(["id", "replicate", "cube"], sort=True, dropna=False).ngroup().to_numpy()
    else:
        key = np.arange(len(d))
    first = np.unique(key, return_index=True)[1]
    return key, first, np.bincount(key).astype(float)


def n_distinct(n: int) -> int:
    """Distinct multisets of n specimens drawn with replacement from n: C(2n - 1, n)."""
    return int(math.comb(2 * n - 1, n)) if n > 0 else 0


# ------------------------------------------------------------------ permutations
def perm_within(rng, gc) -> np.ndarray:
    """Random permutation of positions that keeps every position inside its specimen (gc: specimen codes);
    x[perm_within(rng, gc)] shuffles x among the members of each specimen."""
    gc = np.asarray(gc)
    p = np.empty(len(gc), int)
    p[np.argsort(gc, kind="stable")] = np.lexsort((rng.random(len(gc)), gc))
    return p


def perm_struct(rng, F, gc, G=None, swap_means=True) -> np.ndarray:
    """Structured permutation of the cube-level feature matrix F (rows = cubes, gc = specimen codes): the
    deviations from the specimen mean are permuted among the cubes of each specimen and, with swap_means,
    the specimen means are permuted across specimens. Without swap_means it is a plain permutation of the
    cubes within each specimen (values unchanged). The columns of F move together."""
    gc = np.asarray(gc)
    F = np.asarray(F, float).reshape(len(gc), -1)
    G = G or int(gc.max()) + 1
    p = perm_within(rng, gc)
    if not swap_means:
        return F[p]
    n = np.maximum(np.bincount(gc, minlength=G), 1).astype(float)
    m = np.stack([np.bincount(gc, weights=F[:, j], minlength=G) for j in range(F.shape[1])], 1) / n[:, None]
    return F[p] - m[gc] + m[rng.permutation(G)][gc]


def fl_split(x, Z, gc):
    """Freedman-Lane split of x (cube level): fitted value of the least-squares regression of x on the
    columns Z and specimen intercepts, and its residual (mean 0 within every specimen)."""
    x = np.asarray(x, float)
    Z = np.asarray(Z, float).reshape(len(x), -1)
    gc = np.asarray(gc)
    G = int(gc.max()) + 1
    n = np.maximum(np.bincount(gc, minlength=G), 1).astype(float)

    def dm(v):
        return v - (np.bincount(gc, weights=v, minlength=G) / n)[gc]
    Zd = np.column_stack([dm(Z[:, j]) for j in range(Z.shape[1])])
    xd = dm(x)
    res = xd - Zd @ np.linalg.lstsq(Zd, xd, rcond=None)[0]
    return x - res, res


# ------------------------------------------------------------------ fitting
def _tdot(A, B):
    """A.T @ B for tall A and B (sum over the cases). einsum, not BLAS: a multithreaded BLAS product with a
    long inner dimension was 30-100x slower on a machine whose cores are busy with other jobs."""
    return np.einsum("ni,nj->ij", A, B)


def _newton_batch(X1, XX, pairs, y, T, pen, B, tol=1e-12, maxit=100, far=1.0):
    """Penalised logistic fits for several training masks at once (columns of T), by Newton's method from B.

    Objective per column f: sum_i T_if [log(1 + e^z) - y z] + sum_j pen_jf b_j^2 / 2 (intercept free).
    A step is checked, and halved until the objective decreases, only when it may change some linear
    predictor by more than `far` (bound |d0| + sum_j |d_j| max|x_j|); smaller Newton steps are taken in full.
    """
    k1 = X1.shape[1]
    yT = y[:, None]
    idx = np.arange(1, k1)
    xmax = np.abs(X1).max(0)[:, None]

    def objective(B):
        Z = X1 @ B
        return ((np.logaddexp(0, Z) - yT * Z) * T).sum(0) + 0.5 * (pen * B[1:] ** 2).sum(0)

    for _ in range(maxit):
        P = expit(X1 @ B)
        G = _tdot(X1, T * (P - yT))
        G[1:] += pen * B[1:]
        Hm = _tdot(XX, T * (P * (1 - P)))
        H = np.empty((B.shape[1], k1, k1))
        for m, (a, b) in enumerate(pairs):
            H[:, a, b] = H[:, b, a] = Hm[m]
        H[:, idx, idx] += pen.T
        D = np.linalg.solve(H, G.T[..., None])[..., 0].T
        dec = (G * D).sum(0)
        Bn = B - D
        if dec.max() < tol:
            return Bn
        chk = (np.abs(D) * xmax).sum(0) > far
        if chk.any():
            f0, fn = objective(B), objective(Bn)
            s = np.ones(B.shape[1])
            bad = chk & (fn > f0)
            while bad.any() and s.min() > 1e-8:
                s[bad] *= 0.5
                Bn[:, bad] = B[:, bad] - s[bad] * D[:, bad]
                fn = objective(Bn)
                bad = chk & (fn > f0)
        B = Bn
    return B


def loso_fast(X, y, g, l2=L2_LOSO, w=None, slope_scores=False):
    """Leave-one-specimen-out logistic scores, all folds solved together by Newton's method.

    Same estimator as report.loso_auc: in each fold the features are standardised with the training mean
    and SD (+1e-12) and l2 |b|^2 penalises the standardised slopes. Here every fold is fitted in raw units
    with the equivalent penalty l2 sum (b_j sd_j)^2, so all folds share one design matrix; the folds start
    from the fit to all cases. Case weights w (e.g. 3 for a cube with the share of percolating axes as y)
    give the same fit as the replicated cases. Returns (out-of-fold scores, folds skipped because their
    training set has a single class) and, with slope_scores, also the raw-unit linear predictor of the
    held-out cases without the fold constant, X_te b.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    N, k = X.shape
    w = np.ones(N) if w is None else np.asarray(w, float)
    _, gi = np.unique(np.asarray(g), return_inverse=True)
    F = gi.max() + 1
    T = (gi[:, None] != np.arange(F)[None, :]) * w[:, None]          # N x F: training weight
    ntr = T.sum(0)
    pos = y @ T
    ok = (pos > 1e-9) & (ntr - pos > 1e-9)
    scores, sl = np.full(N, np.nan), np.full(N, np.nan)
    if not ok.any():
        return (scores, int(F), sl) if slope_scores else (scores, int(F))
    T, ntr = T[:, ok], ntr[ok]
    mu = _tdot(X, T) / ntr
    sd = np.sqrt(np.maximum(_tdot(X ** 2, T) / ntr - mu ** 2, 0)) + 1e-12
    X1 = np.column_stack([np.ones(N), X])
    pairs = [(a, b) for a in range(k + 1) for b in range(a, k + 1)]
    XX = np.column_stack([X1[:, a] * X1[:, b] for a, b in pairs])
    ybar = np.average(y, weights=w)
    sd_all = np.sqrt(np.average((X - np.average(X, axis=0, weights=w)) ** 2, axis=0, weights=w))
    b0 = np.zeros((k + 1, 1))
    b0[0] = np.log(ybar / (1 - ybar))
    b_all = _newton_batch(X1, XX, pairs, y, w[:, None], 2 * l2 * (sd_all[:, None] + 1e-12) ** 2, b0)
    B = _newton_batch(X1, XX, pairs, y, T, 2 * l2 * sd ** 2, np.repeat(b_all, T.shape[1], axis=1))
    col = np.full(F, -1)
    col[np.flatnonzero(ok)] = np.arange(T.shape[1])
    have = col[gi] >= 0
    Bh = B[:, col[gi[have]]].T
    scores[have] = expit((X1[have] * Bh).sum(1))
    if not slope_scores:
        return scores, int(F - T.shape[1])
    sl[have] = (X[have] * Bh[:, 1:]).sum(1)
    return scores, int(F - T.shape[1]), sl


def fe_logit(X, y, gi, w=None, l2_dummy=0.0, tol=1e-10, maxit=200, bound=50.0, init=None):
    """Fixed-effects logistic regression: one intercept per group (codes 0..G-1) and common slopes.

    Newton's method with the intercept block eliminated (Schur complement), step halving, case weights w
    (bootstrap multiplicities, or cases per cube with y the share of positives) and an optional ridge
    l2_dummy on the intercepts; init = (intercepts, slopes) starts from a previous fit. Returns (slopes,
    intercepts, log-likelihood without the penalty, converged); not converged includes |parameter| >
    bound (separation).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    gi = np.asarray(gi)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    G = int(gi.max()) + 1
    k = X.shape[1]
    if init is None:
        nw = np.bincount(gi, weights=w, minlength=G)
        pw = np.clip(np.bincount(gi, weights=w * y, minlength=G) / np.maximum(nw, 1e-300), 1e-4, 1 - 1e-4)
        a, b = np.log(pw / (1 - pw)), np.zeros(k)
    else:
        a, b = np.array(init[0], float).copy(), np.array(init[1], float).copy()

    def pll(a, b):
        z = a[gi] + X @ b
        ll = np.sum(w * (y * z - np.logaddexp(0, z)))
        return ll, ll - l2_dummy * np.sum(a ** 2)

    ll, f = pll(a, b)
    conv = False
    for _ in range(maxit):
        p = expit(a[gi] + X @ b)
        r, v = w * (y - p), w * p * (1 - p)
        ga = np.bincount(gi, weights=r, minlength=G) - 2 * l2_dummy * a
        gb = np.einsum("ni,n->i", X, r)
        haa = np.bincount(gi, weights=v, minlength=G) + 2 * l2_dummy + 1e-300
        hab = np.column_stack([np.bincount(gi, weights=v * X[:, j], minlength=G) for j in range(k)])
        S = _tdot(X, v[:, None] * X) - hab.T @ (hab / haa[:, None])
        try:
            db = np.linalg.solve(S, gb - hab.T @ (ga / haa))
        except np.linalg.LinAlgError:     # no within-group variation of some column: slopes not identified
            break
        da = (ga - hab @ db) / haa
        dec = ga @ da + gb @ db
        s = 1.0
        while True:
            ll_n, f_n = pll(a + s * da, b + s * db)
            if f_n >= f - 1e-12 * abs(f) or s < 1e-8:
                break
            s *= 0.5
        a, b, ll, f = a + s * da, b + s * db, ll_n, f_n
        if dec < tol:
            conv = True
            break
    conv = conv and np.all(np.abs(a) < bound) and np.all(np.abs(b) < bound)
    return b, a, float(ll), bool(conv)


def fe_loso(X, y, gi, w=None, tol=1e-10, maxit=60, bound=50.0):
    """Fixed-effects logistic fits (as fe_logit, unpenalised) of every leave-one-specimen-out fold at once.

    Fold f holds out specimen f; its training set is every other specimen with both outcomes (the others
    carry no information on the common slopes). The folds that hold out such a specimen are solved together
    by Newton's method (intercept block eliminated per fold) from the fit to all specimens with both
    outcomes, which is also the fit of every fold that holds out a specimen without both outcomes. A fold
    that does not converge (separation) is refitted alone with the ridge L2_DUMMY on its intercepts.
    Returns (slopes k x G, column f = fold f, NaN when the fold has no training specimen; converged without
    the ridge, G).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    gi = np.asarray(gi)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    k = X.shape[1]
    G = int(gi.max()) + 1
    nw = np.bincount(gi, weights=w, minlength=G)
    n1 = np.bincount(gi, weights=w * y, minlength=G)
    use = (n1 > 1e-9) & (nw - n1 > 1e-9)
    Bout, conv = np.full((k, G), np.nan), np.zeros(G, bool)
    ug = np.flatnonzero(use)
    U = len(ug)
    if U == 0:
        return Bout, conv
    m = use[gi]
    Xu, yu, wu = X[m], y[m], w[m]
    gu = np.searchsorted(ug, gi[m])
    b_all, a_all, _, c_all = fe_logit(Xu, yu, gu, wu)
    if not c_all:
        b_all, a_all, _, _ = fe_logit(Xu, yu, gu, wu, l2_dummy=L2_DUMMY)
    Bout[:, ~use] = b_all[:, None]
    conv[~use] = c_all
    if U < 2:
        return Bout, conv
    if not c_all:                       # start the folds from neutral values, not from a diverged fit
        b_all = np.zeros(k)
        pw = np.clip(np.bincount(gu, weights=wu * yu, minlength=U) / np.bincount(gu, weights=wu, minlength=U),
                     1e-4, 1 - 1e-4)
        a_all = np.log(pw / (1 - pw))
    Nu = len(yu)
    M = sparse.csr_matrix((np.ones(Nu), (gu, np.arange(Nu))), shape=(U, Nu))
    act = ~np.eye(U, dtype=bool)        # act[j, f]: specimen j is in the training set of fold f
    W = wu[:, None] * act[gu]           # Nu x U
    A = np.repeat(a_all[:, None], U, axis=1)
    B = np.repeat(b_all[:, None], U, axis=1)
    pairs = [(a, b) for a in range(k) for b in range(a, k)]
    XX = np.column_stack([Xu[:, a] * Xu[:, b] for a, b in pairs])
    yT = yu[:, None]
    xmax = np.abs(Xu).max(0)
    running, ok = np.ones(U, bool), np.zeros(U, bool)

    def loglik(Ac, Bc, Wc):
        Z = Xu @ Bc + Ac[gu]
        return (Wc * (yT * Z - np.logaddexp(0, Z))).sum(0)

    for _ in range(maxit):
        cols = np.flatnonzero(running)
        if not len(cols):
            break
        Wc, Ac, Bc = W[:, cols], A[:, cols], B[:, cols]
        Z = Xu @ Bc + Ac[gu]
        P = expit(Z)
        R = Wc * (yT - P)
        V = Wc * P * (1 - P)
        ga = M @ R                                                    # U x c
        gb = _tdot(Xu, R)                                             # k x c
        haa = M @ V + 1e-300
        hab = np.stack([M @ (V * Xu[:, [j]]) for j in range(k)])     # k x U x c
        Hm = _tdot(XX, V)
        S = np.empty((len(cols), k, k))
        for q, (a, b) in enumerate(pairs):
            S[:, a, b] = S[:, b, a] = Hm[q]
        S -= np.einsum("igc,jgc->cij", hab, hab / haa)
        rhs = gb - np.einsum("igc,gc->ic", hab, ga / haa)
        dg = np.abs(np.diagonal(S, axis1=1, axis2=2)).max(1) + 1e-300
        sing = np.linalg.cond(S / dg[:, None, None]) > 1e12   # slopes not identified in the fold
        if sing.any():                                          # (no within variation): stop it, refit
            S[sing] += 1e-6 * dg[sing, None, None] * np.eye(k)  # alone below
        db = np.linalg.solve(S, rhs.T[..., None])[..., 0].T
        da = (ga - np.einsum("igc,ic->gc", hab, db)) / haa
        dec = (ga * da).sum(0) + (gb * db).sum(0)
        An, Bn = Ac + da, Bc + db
        # as _newton_batch: a step is checked (and halved until the log-likelihood does not decrease) only if
        # it may change some linear predictor by more than 1; smaller Newton steps are taken in full
        chk = np.abs(np.where(act[:, cols], da, 0.0)).max(0) + (np.abs(db) * xmax[:, None]).sum(0) > 1.0
        if chk.any():
            f0 = (Wc * (yT * Z - np.logaddexp(0, Z))).sum(0)
            s = np.ones(len(cols))
            bad = chk & (loglik(An, Bn, Wc) < f0 - 1e-12 * np.abs(f0))
            while bad.any():
                s[bad] *= 0.5
                An[:, bad] = Ac[:, bad] + s[bad] * da[:, bad]
                Bn[:, bad] = Bc[:, bad] + s[bad] * db[:, bad]
                bad = bad & (loglik(An, Bn, Wc) < f0 - 1e-12 * np.abs(f0)) & (s >= 1e-8)
        A[:, cols], B[:, cols] = An, Bn
        blown = (np.abs(np.where(act[:, cols], An, 0.0)) >= bound).any(0) | (np.abs(Bn) >= bound).any(0) | sing
        done = dec < tol
        ok[cols[done & ~blown]] = True
        running[cols[done | blown]] = False
    Bout[:, ug] = B
    conv[ug] = ok
    for j in np.flatnonzero(~ok):
        tr = gu != j
        g2 = np.unique(gu[tr], return_inverse=True)[1]
        Bout[:, ug[j]] = fe_logit(Xu[tr], yu[tr], g2, wu[tr], l2_dummy=L2_DUMMY)[0]
    return Bout, conv


def loso_within_scores(X, y, gi, w=None) -> np.ndarray:
    """Out-of-fold within-specimen scores: specimen k is scored X b (no intercept) with the slopes b of the
    fixed-effects fit to the other specimens with both outcomes (fe_loso). Only the ranking inside a
    specimen is meaningful. X is standardised once (mean and SD of the specimens with both outcomes), which
    leaves the unpenalised fits and the rankings unchanged.

    One column: only the sign of b matters, and it is exact without a fit - the profile log-likelihood of
    the fixed-effects model is concave in b and its slope at b = 0 (intercepts at the specimen logits) is
    the summed within-specimen covariance sum_j sum_i w (x - xbar_j)(y - ybar_j) of the training specimens.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    gi = np.asarray(gi)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    G = int(gi.max()) + 1
    nw = np.bincount(gi, weights=w, minlength=G)
    n1 = np.bincount(gi, weights=w * y, minlength=G)
    use = (n1 > 1e-9) & (nw - n1 > 1e-9)
    if X.shape[1] == 1:
        x = X[:, 0]
        xb = np.bincount(gi, weights=w * x, minlength=G) / np.maximum(nw, 1e-300)
        cov = np.bincount(gi, weights=w * (x - xb[gi]) * (y - (n1 / np.maximum(nw, 1e-300))[gi]), minlength=G)
        cov = np.where(use, cov, 0.0)
        S = cov.sum() - cov                                  # fold k: all training specimens but k
        sgn = np.where(np.abs(S) > 1e-12 * (np.abs(cov).sum() + 1e-300), np.sign(S), 0.0)
        if use.sum() < 2:
            sgn = np.where(use, np.nan, sgn) if use.any() else np.full(G, np.nan)
        return sgn[gi] * x
    m = use[gi]
    m = m if m.any() else np.ones(len(y), bool)
    mu = np.average(X[m], axis=0, weights=w[m])
    sd = np.sqrt(np.average((X[m] - mu) ** 2, axis=0, weights=w[m])) + 1e-12
    Xs = (X - mu) / sd
    B, _ = fe_loso(Xs, y, gi, w)
    return (Xs * B[:, gi].T).sum(1)


def phi50_fit(phi, y, w=None):
    """Porosity at P = 0.5 of a logistic fit P(y | phi) (as report.phi50, unpenalised); also the slope
    (per unit phi) and whether the fit converged. NaN when the slope is not positive or y has one class.
    y may be a share of positives with w the number of cases (cube level)."""
    phi, y = np.asarray(phi, float), np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    m = w > 0
    phi, y, w = phi[m], y[m], w[m]
    n1 = float(np.sum(w * y))
    if n1 <= 1e-9 or float(np.sum(w)) - n1 <= 1e-9:
        return np.nan, np.nan, np.nan, False
    mu = np.average(phi, weights=w)
    sd = np.sqrt(np.average((phi - mu) ** 2, weights=w)) + 1e-12
    b, a, _, conv = fe_logit(((phi - mu) / sd)[:, None], y, np.zeros(len(y), int), w)
    slope = b[0] / sd
    est = float(mu - a[0] / b[0] * sd) if b[0] > 0 else np.nan
    return est, float(slope), float(a[0] - b[0] * mu / sd), conv


# ------------------------------------------------------------------ AUCs
def within_parts(y, s, gi, G=None):
    """Per specimen: Mann-Whitney concordant pairs U (ties 1/2) and number of pairs n1 n0."""
    y = np.asarray(y, float)
    gi = np.asarray(gi)
    G = G or int(gi.max()) + 1
    r = pd.Series(np.asarray(s, float)).groupby(gi).rank(method="average").to_numpy()
    n1 = np.bincount(gi, weights=y, minlength=G)
    n = np.bincount(gi, minlength=G).astype(float)
    U = np.bincount(gi, weights=r * y, minlength=G) - n1 * (n1 + 1) / 2
    return U, n1 * (n - n1)


def within_auc(y, s, gi):
    """Within-specimen AUC (pairs only within a specimen) and the number of specimens with both outcomes."""
    U, P = within_parts(y, s, gi)
    return (float(U.sum() / P.sum()) if P.sum() > 0 else np.nan), int((P > 0).sum())


def within_parts_w(n1, n0, s, gi, G=None):
    """within_parts for cases grouped in units that share a score (cubes): n1, n0 = positive and negative
    cases of every unit, s = its score, gi = its specimen. Same U and P as within_parts on the cases."""
    n1, n0, s = (np.asarray(v, float) for v in (n1, n0, s))
    gi = np.asarray(gi)
    G = G or int(gi.max()) + 1
    P = np.bincount(gi, weights=n1, minlength=G) * np.bincount(gi, weights=n0, minlength=G)
    if not len(s):
        return np.zeros(G), P
    o = np.lexsort((s, gi))
    g, sv = gi[o], s[o]
    new = np.r_[True, (g[1:] != g[:-1]) | (sv[1:] != sv[:-1])]       # blocks of tied scores per specimen
    b = np.cumsum(new) - 1
    B1, B0 = np.bincount(b, weights=n1[o]), np.bincount(b, weights=n0[o])
    bg = g[new]
    c0 = np.cumsum(B0) - B0                                           # negatives in all earlier blocks ...
    start = np.r_[True, bg[1:] != bg[:-1]]
    c0 = c0 - c0[start][np.cumsum(start) - 1]                         # ... of the same specimen
    U = np.bincount(bg, weights=B1 * (c0 + 0.5 * B0), minlength=G)
    U[np.bincount(gi, weights=~np.isfinite(s), minlength=G) > 0] = np.nan
    return U, P


def _wauc(yc, s, w):
    """AUC of cube-level scores (yc = share of positive cases, w = cases per cube); NaN when undefined."""
    s = np.asarray(s, float)
    if not len(s) or not np.isfinite(s).all():
        return np.nan
    return auc_weighted(yc, s, w)


def _wwithin(n1, n0, s, gi, G):
    U, P = within_parts_w(n1, n0, s, gi, G)
    return float(U.sum() / P.sum()) if P.sum() > 0 else np.nan


def auc_weighted(y, s, w, levels=None):
    """AUC with case weights (ties 1/2); `levels` = np.unique(s, return_inverse=True)[1] may be passed."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    inv = np.unique(s, return_inverse=True)[1] if levels is None else levels
    K = int(inv.max()) + 1
    wp = np.bincount(inv, weights=w * y, minlength=K)
    wn = np.bincount(inv, weights=w * (1 - y), minlength=K)
    W1, W0 = wp.sum(), wn.sum()
    if W1 <= 0 or W0 <= 0:
        return np.nan
    return float((wp * (np.cumsum(wn) - 0.5 * wn)).sum() / (W1 * W0))


def cluster_counts(rng, G, n):
    """n x G matrix of how often each of G specimens is drawn in a resample of G with replacement."""
    draws = rng.integers(0, G, size=(n, G))
    return np.stack([np.bincount(r, minlength=G) for r in draws]).astype(float)


def _ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def _nanmean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _p_upper(null, obs):
    """One-sided permutation p = (1 + #null >= obs) / (1 + n finite)."""
    null = np.asarray(null, float)
    n = int(np.isfinite(null).sum())
    return float((1 + np.sum(null >= obs)) / (1 + n)) if n and np.isfinite(obs) else np.nan


# ------------------------------------------------------------------ one cell
def models_for(d: pd.DataFrame, single_voxel: bool) -> dict:
    """Model name -> raw features ('phi' enters as PHI_BASIS)."""
    m = {"phi": ["phi"], "chi_main": ["chi_main"]}
    cube = "chi_cube" in d and np.isfinite(d["chi_cube"]).all()
    if cube:
        m["chi_cube"] = ["chi_cube"]
    if single_voxel:
        m["chi_raw"] = ["chi_raw"]
    m["phi+chi_main"] = ["phi", "chi_main"]
    if cube:
        m["phi+chi_cube"] = ["phi", "chi_cube"]
    if single_voxel:
        m["phi+chi_raw"] = ["phi", "chi_raw"]
    return m


AUC_COLS = ("model", "features", "design", "n_perm",
            "auc_pooled", "auc_pooled_lo", "auc_pooled_hi", "null_n_perm", "null_mean", "null_lo", "null_hi",
            "p_perm", "auc_pooled_slope", "auc_pooled_slope_lo", "auc_pooled_slope_hi", "null_slope_mean",
            "null_slope_lo", "null_slope_hi", "p_perm_slope",
            "auc_within", "auc_within_lo", "auc_within_hi", "within_pairs", "null_within_n_perm",
            "null_within_mean", "null_within_lo", "null_within_hi", "p_perm_within",
            "auc_within_specmean", "auc_within_specmean_lo", "auc_within_specmean_hi",
            "auc_within_expected", "auc_within_expected_lo", "auc_within_expected_hi",
            "auc_pooled_nonempty", "auc_within_nonempty", "boot_n",
            "d_auc_within", "d_auc_within_lo", "d_auc_within_hi", "p_d_within", "d_within_n_perm",
            "d_auc_pooled", "d_auc_pooled_lo", "d_auc_pooled_hi")


def auc_cell(d: pd.DataFrame, models: dict, seed, n_perm: int, log=print, infer: bool = True) -> list:
    """Pooled LOSO AUC (with and without fold constants), its structured permutation null, within-specimen
    AUC of within-fitted folds and its null, and paired deltas vs phi alone (with a Freedman-Lane null) for
    every model. With infer=False only the point estimates are computed (inferential columns NaN)."""
    d = d.copy()
    add_phi_basis(d)
    y = d.y.to_numpy(float)
    ids, gi = np.unique(d.id.to_numpy(), return_inverse=True)
    G = len(ids)
    # every score is constant over the 3 axis rows of a cube: fits and AUCs on cubes, weighted by their cases
    key, first, wc = cube_index(d)
    n1c = np.bincount(key, weights=y)
    n0c = wc - n1c
    yc = n1c / wc
    gc = gi[first]
    ne = d.phi.to_numpy(float)[first] > 0
    C = cluster_counts(np.random.default_rng([*seed, 1]), G, N_BOOT) if infer else None
    keep, rows = {}, []
    for name, feats in models.items():
        t0 = time.time()
        cols = design_cols(feats)
        Fc = d[feats].to_numpy(float)[first]
        Xc = design_matrix(Fc, feats)
        a, sc, _ = loso_auc(d, cols)
        sc = sc[first]
        sl = loso_fast(Xc, yc, gc, w=wc, slope_scores=True)[2]
        a_sl = _wauc(yc, sl, wc)
        sw = loso_within_scores(Xc, yc, gc, wc)
        U, P = within_parts_w(n1c, n0c, sw, gc, G)
        wa = float(U.sum() / P.sum()) if P.sum() > 0 else np.nan
        okp = P > 0
        spec = float((U[okp] / P[okp]).mean()) if okp.any() else np.nan
        r = dict.fromkeys(AUC_COLS, np.nan)
        r.update({"model": name, "features": "+".join(feats), "design": "+".join(cols), "n_perm": 0,
                  "auc_pooled": a, "auc_pooled_slope": a_sl, "auc_within": wa, "within_pairs": float(P.sum()),
                  "auc_within_specmean": spec, "auc_pooled_nonempty": _wauc(yc[ne], sc[ne], wc[ne]),
                  "auc_within_nonempty": _wwithin(n1c[ne], n0c[ne], sw[ne], gc[ne], G), "boot_n": 0})
        Ue = Pe = None
        if len(feats) == 1:          # raw feature oriented in the expected direction, no model
            Ue, Pe = within_parts_w(n1c, n0c, EXPECTED_SIGN.get(feats[0], 1.0) * Fc[:, 0], gc, G)
            r["auc_within_expected"] = float(Ue.sum() / Pe.sum()) if Pe.sum() > 0 else np.nan
        two = len(feats) == 2 and "phi" in keep and "phi" in feats
        if two:
            r["d_auc_within"] = wa - keep["phi"]["wa"]
            r["d_auc_pooled"] = a - keep["phi"]["a"]
        pooled_b = None
        if infer:
            lev = np.unique(sc, return_inverse=True)[1]
            pooled_b = np.array([auc_weighted(yc, sc, wc * C[b][gc], lev) for b in range(len(C))])
            lev = np.unique(sl, return_inverse=True)[1]
            slope_b = np.array([auc_weighted(yc, sl, wc * C[b][gc], lev) for b in range(len(C))])
            den = C @ P
            within_b = _ratio(C @ U, den)
            spec_b = _ratio(C[:, okp] @ (U[okp] / P[okp]), C[:, okp].sum(1))
            # permutation nulls: the same permutations for every model of the cell
            rp, rw, rd = (np.random.default_rng([*seed, s]) for s in (2, 5, 6))
            null_p, null_s, null_w, null_d = (np.full(n_perm, np.nan) for _ in range(4))
            if two:
                j = 1 - feats.index("phi")
                fit_c, res_c = fl_split(Fc[:, j], phi_basis(Fc[:, feats.index("phi")]), gc)
            for i in range(n_perm):
                s_p, sk, s_ps = loso_fast(design_matrix(perm_struct(rp, Fc, gc, G, True), feats), yc, gc, w=wc,
                                          slope_scores=True)
                if not sk:
                    null_p[i], null_s[i] = _wauc(yc, s_p, wc), _wauc(yc, s_ps, wc)
                Xw = design_matrix(perm_struct(rw, Fc, gc, G, False), feats)
                null_w[i] = _wwithin(n1c, n0c, loso_within_scores(Xw, yc, gc, wc), gc, G)
                if two:
                    Fd = Fc.copy()
                    Fd[:, j] = fit_c + res_c[perm_within(rd, gc)]
                    s_d = loso_within_scores(design_matrix(Fd, feats), yc, gc, wc)
                    null_d[i] = _wwithin(n1c, n0c, s_d, gc, G) - keep["phi"]["wa"]
            r.update({"n_perm": n_perm,
                      "auc_pooled_lo": _ci(pooled_b)[0], "auc_pooled_hi": _ci(pooled_b)[1],
                      "null_n_perm": int(np.isfinite(null_p).sum()), "null_mean": _nanmean(null_p),
                      "null_lo": _ci(null_p)[0], "null_hi": _ci(null_p)[1], "p_perm": _p_upper(null_p, a),
                      "auc_pooled_slope_lo": _ci(slope_b)[0], "auc_pooled_slope_hi": _ci(slope_b)[1],
                      "null_slope_mean": _nanmean(null_s), "null_slope_lo": _ci(null_s)[0],
                      "null_slope_hi": _ci(null_s)[1], "p_perm_slope": _p_upper(null_s, a_sl),
                      "auc_within_lo": _ci(within_b)[0], "auc_within_hi": _ci(within_b)[1],
                      "null_within_n_perm": int(np.isfinite(null_w).sum()),
                      "null_within_mean": _nanmean(null_w), "null_within_lo": _ci(null_w)[0],
                      "null_within_hi": _ci(null_w)[1], "p_perm_within": _p_upper(null_w, wa),
                      "auc_within_specmean_lo": _ci(spec_b)[0], "auc_within_specmean_hi": _ci(spec_b)[1],
                      "boot_n": int(np.isfinite(within_b).sum())})
            if Ue is not None:
                eb = _ratio(C @ Ue, C @ Pe)
                r["auc_within_expected_lo"], r["auc_within_expected_hi"] = _ci(eb)
            if two:
                dw = _ratio(C @ (U - keep["phi"]["U"]), den)
                dp = pooled_b - keep["phi"]["pooled_b"]
                r.update({"d_auc_within_lo": _ci(dw)[0], "d_auc_within_hi": _ci(dw)[1],
                          "p_d_within": _p_upper(null_d, r["d_auc_within"]),
                          "d_within_n_perm": int(np.isfinite(null_d).sum()),
                          "d_auc_pooled_lo": _ci(dp)[0], "d_auc_pooled_hi": _ci(dp)[1]})
        keep[name] = {"U": U, "a": a, "wa": wa, "pooled_b": pooled_b}
        rows.append(r)
        log(f"      {name:13s} pooled {a:.3f} (null {r['null_mean']:.3f} [{r['null_lo']:.3f}, {r['null_hi']:.3f}])"
            f"  slope-only {a_sl:.3f} (p {r['p_perm_slope']:.3f})  within {wa:.3f} [{r['auc_within_lo']:.3f}, "
            f"{r['auc_within_hi']:.3f}] p {r['p_perm_within']:.3f}  spec-mean {spec:.3f}"
            + (f"  d_within {r['d_auc_within']:+.3f} p {r['p_d_within']:.3f}" if two else "")
            + f"  {time.time() - t0:.0f} s")
    return rows


COND_COLS = ("beta_phi", "beta_phi_lo", "beta_phi_hi", "beta_chi", "beta_chi_lo", "beta_chi_hi", "share_chi_pos",
             "lr_stat", "p_boot", "p_perm_cond", "perm_cond_n", "p_lr_naive", "l2_dummy", "boot_used")


def conditional_cell(d: pd.DataFrame, chi: str, seed, n_perm: int = N_PERM_COND) -> dict:
    """Fixed-effects logit y ~ phi basis + chi + specimen (covariates per within-specimen SD): chi
    coefficient, cluster bootstrap (interval, two-sided p) and Freedman-Lane permutation test of the LR
    statistic against y ~ phi basis + specimen."""
    both = d.groupby("id").y.transform(lambda v: v.min() != v.max()).astype(bool)
    d2 = d[both]
    nu = d2.id.nunique()
    r = {"chi_feature": chi, "n_cases_used": len(d2), "n_specimens_used": nu,
         "n_specimens_dropped": d.id.nunique() - nu, "n_distinct_resamples": n_distinct(nu)}
    r.update(dict.fromkeys(COND_COLS, np.nan))
    if nu < MIN_BOTH:
        r["status"] = f"only {nu} specimen(s) with both outcomes (< {MIN_BOTH})"
        return r
    y = d2.y.to_numpy(float)
    _, gi = np.unique(d2.id.to_numpy(), return_inverse=True)
    key, first, wc = cube_index(d2)
    yc = np.bincount(key, weights=y) / wc
    gc = gi[first]
    G = int(gc.max()) + 1
    phi = d2.phi.to_numpy(float)[first]
    Xr = np.column_stack([phi_basis(phi), d2[chi].to_numpy(float)[first]])
    nw = np.bincount(gc, weights=wc, minlength=G)
    mg = np.stack([np.bincount(gc, weights=wc * Xr[:, j], minlength=G) for j in range(Xr.shape[1])], 1) / nw[:, None]
    sd = np.sqrt(np.average((Xr - mg[gc]) ** 2, axis=0, weights=wc))     # within-specimen SD
    if not (sd > 1e-12).all():
        r["status"] = "constant feature within specimens"
        return r
    X = (Xr - np.average(Xr, axis=0, weights=wc)) / sd
    l2 = 0.0
    b, a, ll1, conv = fe_logit(X, yc, gc, wc)
    if not conv:
        l2 = L2_DUMMY
        b, a, ll1, conv = fe_logit(X, yc, gc, wc, l2_dummy=l2)
    if not conv:
        r["status"] = "fixed-effects fit did not converge (separation or no within-specimen variation)"
        return r
    _, _, ll0, conv0 = fe_logit(X[:, :3], yc, gc, wc, l2_dummy=l2)
    lr = max(2 * (ll1 - ll0), 0.0)
    pp = phi > 0

    def phi_effect(bb):
        """Median over cubes (phi > 0) of d(eta)/d(phi) times the within-SD of phi."""
        if not pp.any():
            return np.nan
        q = phi[pp]
        return float(np.median(bb[0] + bb[1] * sd[0] / (2 * sd[1] * np.sqrt(q)) + bb[2] * 2 * q * sd[0] / sd[2]))

    r.update({"beta_phi": phi_effect(b), "beta_chi": float(b[3]), "lr_stat": lr,
              "p_lr_naive": float(stats.chi2.sf(lr, 1)), "l2_dummy": l2})
    status = "ok" if conv0 else "phi-only fit did not converge"
    if nu < MIN_INFER:
        r["p_lr_naive"] = np.nan
        r["status"] = f"descriptive only ({nu} specimens with both outcomes)"
        return r
    C = cluster_counts(np.random.default_rng([*seed, 3]), G, N_BOOT_COND)
    bs = []
    for c in C:
        wb = wc * c[gc]
        m = wb > 0
        n1 = float(np.sum(wb * yc))
        if n1 <= 1e-9 or float(wb.sum()) - n1 <= 1e-9:
            continue
        gs, g2 = np.unique(gc[m], return_inverse=True)
        bb, _, _, cv = fe_logit(X[m], yc[m], g2, wb[m], l2_dummy=l2, init=(a[gs], b))
        if cv:
            bs.append(bb)
    bs = np.array(bs).reshape(-1, X.shape[1])
    share = float((bs[:, 3] > 0).mean()) if len(bs) else np.nan
    eff = np.array([phi_effect(bb) for bb in bs])
    # Freedman-Lane: chi = fitted(phi basis, specimen) + residual; residuals permuted among the cubes of
    # every specimen; the reduced model does not contain chi, so only the full model is refitted
    fit_c, res_c = fl_split(X[:, 3], X[:, :3], gc)
    rng = np.random.default_rng([*seed, 7])
    lrs = np.full(n_perm, np.nan)
    Xp = X.copy()
    for i in range(n_perm):
        Xp[:, 3] = fit_c + res_c[perm_within(rng, gc)]
        _, _, llp, _ = fe_logit(Xp, yc, gc, wc, l2_dummy=l2, init=(a, b))
        lrs[i] = max(2 * (llp - ll0), 0.0)
    r.update({"beta_phi_lo": _ci(eff)[0], "beta_phi_hi": _ci(eff)[1],
              "beta_chi_lo": _ci(bs[:, 3])[0], "beta_chi_hi": _ci(bs[:, 3])[1], "share_chi_pos": share,
              "p_boot": float(max(1 / (len(bs) + 1), min(1.0, 2 * min(share, 1 - share)))) if len(bs) else np.nan,
              "p_perm_cond": _p_upper(lrs, lr), "perm_cond_n": int(np.isfinite(lrs).sum()),
              "boot_used": len(bs), "status": status})
    return r


def calibration_rows(d: pd.DataFrame, slope, icpt) -> list:
    """Observed share and mean fitted P(y | phi) in porosity bins of width CAL_BIN with >= CAL_MIN_N cases."""
    phi, y = d.phi.to_numpy(float), d.y.to_numpy(float)
    edges = np.arange(0, phi.max() + CAL_BIN, CAL_BIN)
    fit = expit(icpt + slope * phi) if np.isfinite(slope) else np.full(len(phi), np.nan)
    k = np.clip(np.digitize(phi, edges) - 1, 0, len(edges) - 2)
    rows = []
    for j in range(len(edges) - 1):
        m = k == j
        if m.sum() < CAL_MIN_N:
            continue
        rows.append({"bin_lo": float(edges[j]), "bin_hi": float(edges[j + 1]), "n": int(m.sum()),
                     "n_pos": int(y[m].sum()), "phi_mean": float(phi[m].mean()), "observed": float(y[m].mean()),
                     "fitted": float(fit[m].mean())})
    return rows


def phi50_cell(d: pd.DataFrame, seed) -> dict:
    """phi50 with a specimen bootstrap (counts of discarded resamples by reason), extrapolation flag,
    fitted P at phi = 0 and a goodness-of-fit summary from the calibration bins."""
    y, phi = d.y.to_numpy(float), d.phi.to_numpy(float)
    _, gi = np.unique(d.id.to_numpy(), return_inverse=True)
    key, first, wc = cube_index(d)
    yc, phc, gc = np.bincount(key, weights=y) / wc, phi[first], gi[first]
    est, slope, icpt, conv = phi50_fit(phc, yc, wc)
    G = int(gi.max()) + 1
    C = cluster_counts(np.random.default_rng([*seed, 4]), G, N_BOOT_PHI50)
    bs, one, nonpos, nonconv = [], 0, 0, 0
    for c in C:
        w = wc * c[gc]
        n1 = float(np.sum(w * yc))
        if n1 <= 1e-9 or float(w.sum()) - n1 <= 1e-9:
            one += 1
            continue
        e, s, _, cv = phi50_fit(phc, yc, w)
        if not np.isfinite(e):
            nonpos += 1
            continue
        nonconv += not cv          # kept (as report.phi50, which keeps the optimiser's estimate)
        bs.append(e)
    lo, hi = _ci(bs)
    q_lo, q_hi = (float(v) for v in np.percentile(phi, [2.5, 97.5]))
    near = np.abs(phi - est) <= CAL_BIN if np.isfinite(est) else np.zeros(len(phi), bool)
    n_near, pos_near = int(near.sum()), int(y[near].sum())
    lo_neg = bool(np.isfinite(lo) and lo < 0)
    extrap = bool(not np.isfinite(est) or not q_lo <= est <= q_hi or n_near < PHI50_MIN_NEAR
                  or pos_near in (0, n_near) or lo_neg)
    gof = [c for c in calibration_rows(d, slope, icpt) if c["n"] >= CAL_GOF_MIN_N and np.isfinite(c["fitted"])]
    dev = np.array([abs(c["observed"] - c["fitted"]) for c in gof])
    nb = np.array([c["n"] for c in gof], float)
    status = []
    if not (np.isfinite(slope) and slope > 0):
        status.append("slope not positive")
    if G < MIN_INFER:
        status.append(f"descriptive only ({G} specimens)")
        lo = hi = np.nan
    return {"phi50": est, "phi50_lo": lo, "phi50_hi": hi, "phi_min": float(phi.min()), "phi_max": float(phi.max()),
            "phi_q025": q_lo, "phi_q975": q_hi, "phi50_n_near": n_near, "phi50_n_near_pos": pos_near,
            "phi50_lo_negative": lo_neg, "phi50_extrapolated": extrap,
            "slope": slope, "intercept": icpt, "p_fit_phi0": float(expit(icpt)) if np.isfinite(icpt) else np.nan,
            "converged": conv, "cal_bins_gof": len(gof),
            "cal_max_abs_dev": float(dev.max()) if len(dev) else np.nan,
            "cal_mean_abs_dev": float((dev * nb).sum() / nb.sum()) if len(dev) else np.nan,
            "n_distinct_resamples": n_distinct(G),
            "boot_n": N_BOOT_PHI50, "boot_used": len(bs), "boot_discarded_slope": nonpos,
            "boot_discarded_one_class": one, "boot_discarded": nonpos + one, "boot_not_converged_kept": nonconv,
            "status": "; ".join(status) or "ok"}


# ------------------------------------------------------------------ main
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run(cfg: Config, only=None, log=print) -> None:
    out = cfg.outdir("predict")
    t_start = time.time()
    if only:
        log("  predict pools all specimens: `only` ignored, every table rebuilt")
    cases = case_table(cfg)
    variants = [v for v in ("T50", "T50int") if v in set(cases.variant)]
    variants += sorted(set(cases.variant) - set(variants))
    edges = sorted(int(e) for e in cases.edge.unique())
    fallback = {v: sorted(cases.loc[(cases.variant == v) & (cases.l_p_main_variant != v), "id"].unique())
                for v in variants}
    log(f"  predict: {len(cases)} cases, variants {variants}, edges {edges}")
    for v, f in fallback.items():
        if f:
            log(f"  predict: {v} chi_main of {f} uses the T50 chord (no {v} row in topology.csv)")
    auc_rows, cond_rows, p50_rows, cal_rows = [], [], [], []
    summary = {"params": {"chi_scale_s0": CHI_SCALE, "l2_loso": L2_LOSO, "l2_dummy": L2_DUMMY,
                          "min_specimens_both": MIN_BOTH, "min_specimens_inference": MIN_INFER,
                          "n_perm": N_PERM, "n_perm_edge": N_PERM_EDGE, "n_perm_cond": N_PERM_COND,
                          "n_perm_cond_edge": N_PERM_COND_EDGE, "n_boot": N_BOOT, "n_boot_cond": N_BOOT_COND,
                          "n_boot_phi50": N_BOOT_PHI50, "cal_bin": CAL_BIN, "cal_min_n": CAL_MIN_N,
                          "cal_gof_min_n": CAL_GOF_MIN_N, "phi50_min_near": PHI50_MIN_NEAR,
                          "phi_basis": list(PHI_BASIS), "seed": SEED, "phi_unit": "fraction",
                          "pooled_null": "specimen means permuted across specimens + cube deviations within",
                          "within_null": "cubes permuted within specimens",
                          "within_folds": "fixed-effects logit on the training specimens with both outcomes"},
               "chord_fallback": fallback, "cells": {}, "phi50": {}}
    for vi, variant in enumerate(variants):
        cv = cases[cases.variant == variant]
        for si, subset in enumerate(SUBSETS):
            for e in edges:
                d = cv[subset_mask(cv, subset) & (cv.edge == e)].copy()
                if not len(d):
                    continue
                seed = (SEED, vi, si, int(e))
                single = d.voxel_mm.nunique() == 1
                s1 = add_chi_raw(d) if single else np.nan
                per = d.groupby("id").y.agg(["min", "max"])
                nboth = int((per["min"] != per["max"]).sum())
                notes = []
                nbad = int((~np.isfinite(d.chi_cube.to_numpy(float))).sum())
                if nbad:
                    notes.append(f"chi_cube non-finite in {nbad} cases: chi_cube models dropped, conditional "
                                 "primary chi_main")
                fb = sorted(d.loc[d.l_p_main_variant != variant, "id"].unique())
                if fb:
                    notes.append(f"chi_main of {', '.join(fb)} uses the T50 chord")
                info = {"variant": variant, "subset": subset, "edge": int(e),
                        "edge_mm_min": float(d.edge_mm.min()), "edge_mm_max": float(d.edge_mm.max()),
                        "n_cases": len(d), "n_pos": int(d.y.sum()), "n_specimens": int(len(per)),
                        "n_specimens_pos": int((per["max"] == 1).sum()),
                        "n_specimens_neg": int((per["min"] == 0).sum()),
                        "n_specimens_both": nboth,
                        "n_empty": int((d.phi <= 0).sum()), "n_empty_pos": int(d.y[d.phi <= 0].sum()),
                        "chi_raw_scale_s1": s1, "n_distinct_resamples": n_distinct(nboth),
                        "note": "; ".join(notes)}
                status = ""
                if nboth < MIN_BOTH:
                    status = f"only {nboth} specimen(s) with both outcomes (< {MIN_BOTH})"
                else:
                    tr_one = [sid for sid in per.index
                              if d.loc[d.id != sid, "y"].nunique() < 2]
                    if tr_one:
                        status = f"{len(tr_one)} LOSO fold(s) with a single-class training set"
                infer = nboth >= MIN_INFER
                models = models_for(d, single)
                cell = {k: v for k, v in info.items() if k not in ("variant", "subset", "edge")}
                cell["status"] = status or "ok"
                log(f"    {variant} {subset} {e}: {info['n_cases']} cases, {info['n_pos']} percolating, "
                    f"{nboth}/{info['n_specimens']} specimens with both outcomes"
                    + (f" - skipped: {status}" if status else "" if infer else " - descriptive only")
                    + (f" ({info['note']})" if notes else ""))
                if status:
                    for name, feats in models.items():
                        auc_rows.append({**info, "model": name, "features": "+".join(feats),
                                         "design": "+".join(design_cols(feats)), "status": status})
                else:
                    st = "ok" if infer else f"descriptive only ({nboth} specimens with both outcomes)"
                    cell["status"] = st
                    rows = auc_cell(d, models, seed, N_PERM_EDGE.get(int(e), N_PERM), log, infer=infer)
                    cell["auc"] = {}
                    for r in rows:
                        auc_rows.append({**info, **r, "status": st})
                        cell["auc"][r["model"]] = {
                            "pooled": [r["auc_pooled"], r["auc_pooled_lo"], r["auc_pooled_hi"]],
                            "null": [r["null_mean"], r["null_lo"], r["null_hi"], r["null_n_perm"]],
                            "p_perm": r["p_perm"],
                            "pooled_slope": [r["auc_pooled_slope"], r["auc_pooled_slope_lo"],
                                             r["auc_pooled_slope_hi"]],
                            "null_slope": [r["null_slope_mean"], r["null_slope_lo"], r["null_slope_hi"]],
                            "p_perm_slope": r["p_perm_slope"],
                            "within": [r["auc_within"], r["auc_within_lo"], r["auc_within_hi"]],
                            "null_within": [r["null_within_mean"], r["null_within_lo"], r["null_within_hi"]],
                            "p_perm_within": r["p_perm_within"],
                            "within_specmean": [r["auc_within_specmean"], r["auc_within_specmean_lo"],
                                                r["auc_within_specmean_hi"]],
                            "within_expected": r["auc_within_expected"],
                            "nonempty": [r["auc_pooled_nonempty"], r["auc_within_nonempty"]],
                            "d_within": [r["d_auc_within"], r["d_auc_within_lo"], r["d_auc_within_hi"]],
                            "p_d_within": r["p_d_within"],
                            "d_pooled": [r["d_auc_pooled"], r["d_auc_pooled_lo"], r["d_auc_pooled_hi"]]}
                # conditional analysis
                chis = (["chi_cube"] if "chi_cube" in models else []) + ["chi_main"] + (["chi_raw"] if single else [])
                cell["conditional"] = {}
                for ci, chi in enumerate(chis):
                    r = conditional_cell(d, chi, (*seed, ci), N_PERM_COND_EDGE.get(int(e), N_PERM_COND))
                    r["primary"] = chi == chis[0]
                    cond_rows.append({**info, **r})
                    cell["conditional"][chi] = {"beta_chi": [r["beta_chi"], r["beta_chi_lo"], r["beta_chi_hi"]],
                                                "share_chi_pos": r["share_chi_pos"], "p_boot": r["p_boot"],
                                                "p_perm_cond": r["p_perm_cond"], "p_lr_naive": r["p_lr_naive"],
                                                "status": r["status"]}
                    if r["status"] == "ok":
                        log(f"      FE {chi:9s} beta_chi {r['beta_chi']:+.3f} [{r['beta_chi_lo']:+.3f}, "
                            f"{r['beta_chi_hi']:+.3f}]  P(>0) {r['share_chi_pos']:.2f}  p_boot {r['p_boot']:.3f}"
                            f"  p_perm {r['p_perm_cond']:.3f}")
                summary["cells"].setdefault(variant, {}).setdefault(subset, {})[str(int(e))] = cell
        # phi50 per group with a transition, and calibration
        for gidx, g in enumerate(GROUP_ORDER):
            for e in edges:
                d = cv[(cv.group == g) & (cv.edge == e)]
                if not len(d):
                    continue
                st = auc_design(d)
                base = {"variant": variant, "group": g, "edge": int(e), "edge_mm": float(d.edge_mm.mean()),
                        "n_cases": len(d), "n_pos": int(d.y.sum()), "n_specimens": d.id.nunique()}
                slope = icpt = np.nan
                if st:
                    p50_rows.append({**base, "status": st})
                else:
                    r = phi50_cell(d, (SEED, vi, 100 + gidx, int(e)))
                    slope, icpt = r["slope"], r["intercept"]
                    p50_rows.append({**base, **r})
                    summary["phi50"].setdefault(variant, {}).setdefault(g, {})[str(int(e))] = {
                        "phi50": [r["phi50"], r["phi50_lo"], r["phi50_hi"]], "boot_discarded": r["boot_discarded"],
                        "boot_used": r["boot_used"], "phi_range": [r["phi_min"], r["phi_max"]],
                        "phi_q": [r["phi_q025"], r["phi_q975"]], "extrapolated": r["phi50_extrapolated"],
                        "n_near": r["phi50_n_near"], "p_fit_phi0": r["p_fit_phi0"],
                        "cal_max_abs_dev": r["cal_max_abs_dev"], "status": r["status"]}
                    log(f"    {variant} phi50 {g} {e}: {100 * r['phi50']:.1f} % [{100 * r['phi50_lo']:.1f}, "
                        f"{100 * r['phi50_hi']:.1f}]  discarded {r['boot_discarded']}/{N_BOOT_PHI50}"
                        f"  extrapolated {r['phi50_extrapolated']} (n near {r['phi50_n_near']})"
                        f"  max |obs - fit| {r['cal_max_abs_dev']:.2f}  {r['status']}")
                if g == "coquina" or not st:
                    for c in calibration_rows(d, slope, icpt):
                        cal_rows.append({**base, **c})
    pd.DataFrame(auc_rows).to_csv(out / "predict_auc.csv", index=False)
    pd.DataFrame(cond_rows).to_csv(out / "predict_conditional.csv", index=False)
    pd.DataFrame(p50_rows).to_csv(out / "predict_phi50.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(out / "predict_calibration.csv", index=False)
    summary["runtime_s"] = round(time.time() - t_start, 1)
    write_json(out / "predict_summary.json", _clean(summary))
    log(f"  predict: tables in {out} ({time.time() - t_start:.0f} s)")
