"""Stage `report`: tables, statistics and the key numbers quoted in the manuscript.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Outputs (outputs/report/):
  tables/*.csv           tables of the article and of the supplementary material
  key_numbers.json       {name: {"value", "unit", "description"}} read by the manuscript builder
  RESULTS_DIGEST.md      plain-language digest of the numbers, for the authors

Percolation-prediction analysis: every sub-volume of edge e (stage `rev`) and axis is a case whose
outcome is "percolates along that axis". Logistic models with porosity alone, with the dimensionless
Euler number density chi_V * l_p^3 alone (l_p = 4 phi / S_V: mean pore chord of the sample's main VOI,
so that samples imaged with different voxels are comparable; transformed with asinh) and with both are
compared by the area under the ROC curve (AUC) in leave-one-sample-out cross-validation, with a
sample-level bootstrap confidence interval. The AUC is reported as not estimable (NaN, with the reason)
when the percolating or the non-percolating cases come from fewer than 3 samples.

Method comparison: every PNM/LBM ratio uses the pore network computed on the same domain as the LBM
(stage `lbm`: the whole main VOI or its central sub-volume), only for converged LBM runs (ok). In the
transport table (T4) the columns of the stage `transport` refer to the whole main VOI and those prefixed
or labelled `lbm` to the LBM domain; Kozeny-Carman and Katz-Thompson are divided by k_LBM only where
the LBM domain is the whole VOI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .config import Config, read_json, write_json
from .extract import voi_paths
from .topology import D_C_MIN_VOX

GROUP_LABEL = {"granular": "Glass-bead pack", "limestone": "Reef limestone", "coquina": "Coquina",
               "trachyte": "Trachyte"}
GROUP_ORDER = ["granular", "limestone", "coquina", "trachyte"]
MD = 9.869233e-16
PNM_TAGS = {"pnm": "inscribed sizes, mirror faces", "pnmeq": "volume-/area-equivalent sizes, mirror faces",
            "pnmw": "inscribed sizes, wall faces", "pnmeqw": "volume-/area-equivalent sizes, wall faces"}
LAB_METHODS = ["direct", "gravimetric", "indirect", "helium"]   # physical laboratory methods, bead packs
AUC_MIN_SAMPLES = 3


class Keys:
    def __init__(self):
        self.d = {}

    def add(self, name, value, unit="", description=""):
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            value = value.item()
        if isinstance(value, (list, tuple)):
            value = [v.item() if isinstance(v, (np.floating, np.integer, np.bool_)) else v for v in value]
        self.d[name] = {"value": value, "unit": unit, "description": description}


MD_M2 = 9.869233e-16   # m^2 per millidarcy


def flag(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    """Boolean column as read back from a CSV (bool, 0/1 or 'True'/'False'); missing values -> default."""
    if col not in df:
        return pd.Series(default, index=df.index, dtype=bool)
    v = df[col]
    if v.dtype == bool:
        return v
    out = v.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])
    return (out | (v.isna() & default)).astype(bool)


def pore_chord(phi, S_V):
    """Mean pore chord l_p = 4 phi / S_V (Crofton; S_V = 2 P_L), in the length unit of 1 / S_V."""
    phi, S_V = np.asarray(phi, float), np.asarray(S_V, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(S_V > 0, 4 * phi / S_V, np.nan)


def _range(x) -> list:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(x.min()), float(x.max())] if len(x) else [np.nan, np.nan]


def _gmean(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


# ------------------------------------------------------------------ logistic regression
def _fit_logit(X, y, l2=1e-3):
    X1 = np.column_stack([np.ones(len(X)), X])

    def nll(b):
        z = X1 @ b
        return np.sum(np.logaddexp(0, z) - y * z) + l2 * np.sum(b[1:] ** 2)

    def grad(b):
        p = 1 / (1 + np.exp(-(X1 @ b)))
        g = X1.T @ (p - y)
        g[1:] += 2 * l2 * b[1:]
        return g
    res = optimize.minimize(nll, np.zeros(X1.shape[1]), jac=grad, method="L-BFGS-B")
    return res.x


def _predict(b, X):
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ b)))


def auc(y, s):
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = stats.rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_design(df, ycol="y", group="id", min_samples=AUC_MIN_SAMPLES) -> str:
    """Why a LOSO AUC cannot be estimated ('' if it can): each class must occur in >= min_samples samples."""
    if df[ycol].nunique() < 2:
        return "a single outcome class"
    for val, name in ((1, "percolating"), (0, "non-percolating")):
        n = df.loc[df[ycol] == val, group].nunique()
        if n < min_samples:
            return f"{name} cases in {n} sample(s) only (< {min_samples})"
    return ""


def loso_auc(df, feats, ycol="y", group="id"):
    """Leave-one-sample-out predictions; returns pooled AUC, out-of-sample scores and folds skipped.

    A fold whose training set has a single class cannot be fitted; if any fold is skipped the pooled
    AUC would silently drop that sample, so it is returned as NaN.
    """
    scores = np.full(len(df), np.nan)
    X = df[feats].to_numpy(float)
    y = df[ycol].to_numpy(float)
    g = df[group].to_numpy()
    skipped = 0
    for sid in np.unique(g):
        tr, te = g != sid, g == sid
        if len(np.unique(y[tr])) < 2:
            skipped += 1
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12   # standardised within the training fold
        b = _fit_logit((X[tr] - mu) / sd, y[tr])
        scores[te] = _predict(b, (X[te] - mu) / sd)
    if skipped:
        return np.nan, scores, skipped
    return auc(y, scores), scores, 0


def boot_auc(df, scores, ycol="y", group="id", n=2000, seed=7):
    """Sample-level bootstrap percentile interval of the AUC; also the number of usable resamples."""
    rng = np.random.default_rng(seed)
    ids = df[group].unique()
    by = {i: np.where(df[group].to_numpy() == i)[0] for i in ids}
    y = df[ycol].to_numpy()
    out = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([by[i] for i in pick])
        a = auc(y[idx], scores[idx])
        if np.isfinite(a):
            out.append(a)
    lo, hi = np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)
    return float(lo), float(hi), len(out)


def phi50(df, ycol="y", group="id", n=1000, seed=11):
    """Porosity at which a logistic fit P(percolation | phi) crosses 0.5, with a sample-bootstrap 95 % interval."""
    def fit(d):
        y = d[ycol].to_numpy(float)
        if y.min() == y.max():
            return np.nan
        x = d.phi.to_numpy(float)
        mu, sd = x.mean(), x.std() + 1e-12
        b = _fit_logit(((x - mu) / sd)[:, None], y, l2=0.0)
        return float(mu - b[0] / b[1] * sd) if b[1] > 0 else np.nan
    est = fit(df)
    rng = np.random.default_rng(seed)
    ids = df[group].unique()
    parts = {i: df[df[group] == i] for i in ids}
    bs = [fit(pd.concat([parts[i] for i in rng.choice(ids, len(ids))])) for _ in range(n)]
    bs = np.array([b for b in bs if np.isfinite(b)])
    lo, hi = np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan)
    return est, float(lo), float(hi)


# ------------------------------------------------------------------ main
def run(cfg: Config) -> None:
    out = cfg.outdir("report")
    tdir = out / "tables"
    tdir.mkdir(exist_ok=True)
    K = Keys()
    notes = []
    samples = {s.id: s for s in cfg.samples}
    voi_edge = cfg["voi"]["edge"] if not cfg.quick else 100

    # ---- inputs
    metas = []
    for s in cfg.samples:
        k = 0
        while voi_paths(cfg, s.id, k)[1].exists():
            metas.append(read_json(voi_paths(cfg, s.id, k)[1]))
            k += 1
    meta = pd.DataFrame(metas)
    seg = pd.DataFrame([read_json(f) for f in sorted(cfg.outdir("segment").glob("*_seg.json"))])
    refs = pd.DataFrame([read_json(f) for f in sorted(cfg.outdir("segment").glob("*_sample_references.json"))])
    topo = pd.read_csv(cfg.outdir("topology") / "topology.csv")
    coars_all = pd.read_csv(cfg.outdir("coarsen") / "coarsen.csv")
    for c, v in (("replicate", 0), ("variant", "T50")):
        if c not in coars_all:
            coars_all[c] = v
    coars = coars_all[(coars_all.replicate == 0) & (coars_all.variant == "T50")].copy()   # main VOIs, ISO-50
    rev_all = pd.read_csv(cfg.outdir("rev") / "rev.csv")
    if "variant" not in rev_all:
        rev_all["variant"] = "T50"
    rev = rev_all[rev_all.variant == "T50"].copy()
    tra = pd.read_csv(cfg.outdir("transport") / "transport.csv")
    lbm = pd.read_csv(cfg.outdir("lbm") / "lbm.csv") if (cfg.outdir("lbm") / "lbm.csv").exists() else pd.DataFrame()
    wall_f = cfg.outdir("wall") / "wall.csv"
    wall = pd.read_csv(wall_f) if wall_f.exists() else pd.DataFrame()
    lab = pd.read_csv(cfg.p("lab_csv"))
    legacy = pd.read_csv(cfg.p("legacy_csv"))
    for a in ("any", "z", "y", "x"):
        topo[f"perc_{a}"] = flag(topo, f"perc_{a}")
    topo["l_p"] = pore_chord(topo.phi, topo.S_V) if "S_V" in topo else np.nan
    t50 = topo[topo.variant == "T50"].copy()
    main = t50[t50.replicate == 0].set_index("id")

    # ---- Table 1: data set
    rows = []
    for s in cfg.samples:
        m = meta[meta.id == s.id]
        lab_s = lab[lab.id == s.id]
        he = lab_s[lab_s.method == "helium"].porosity_pct
        rows.append({
            "Sample": s.id, "Group": GROUP_LABEL[s.group],
            "Scan voxel (um)": s.voxel_um, "Analysed voxel (um)": s.voxel_um * s.bin,
            "VOI edge (mm)": round(cfg["voi"]["edge"] * s.voxel_mm, 2), "VOIs": len(m),
            "Plug/column diameter (mm)": round(float(m.sample_diameter_mm.iloc[0]), 1) if s.group != "granular" else np.nan,
            "Helium porosity (%)": float(he.iloc[0]) if len(he) else np.nan,
        })
    t1 = pd.DataFrame(rows)
    t1.to_csv(tdir / "T1_dataset.csv", index=False)

    # ---- supplementary: acquisition settings, VOI placement, segmentation references and QA
    acq_f = cfg.p("lab_csv").parent / "acquisition.csv"
    if acq_f.exists():
        pd.read_csv(acq_f).to_csv(tdir / "S1_acquisition.csv", index=False)
    vcols = [c for c in ["id", "replicate", "first_slice", "last_slice", "slice_index_start", "n_slices_stack",
                         "crop_y0", "crop_x0", "crop_width", "bin", "voxel_um", "sample_diameter_mm",
                         "air_outside_median", "air_far_median", "slices_unreadable_seen",
                         "window_fallback_no_margin"] if c in meta.columns]
    meta[vcols].to_csv(tdir / "S2_vois.csv", index=False)
    if "window_fallback_no_margin" in meta:
        fb = flag(meta, "window_fallback_no_margin")
        K.add("n_vois_window_fallback", [int(fb.sum()), int(meta.window_fallback_no_margin.notna().sum())], "",
              "VOIs placed by the last-resort window search (no end margin), VOIs with the flag recorded")
        if fb.any():
            K.add("vois_window_fallback", [f"{i}_v{int(k)}" for i, k in zip(meta.id[fb], meta.replicate[fb])], "",
                  "VOIs placed without the end margin")
    refcols = [c for c in ["id", "group", "air_source", "I_air", "I_solid", "air_outside", "air_far", "air_int_offset", "n_deep",
                           "T40", "T50", "T60", "T50int"] if c in refs.columns]
    refs[refcols].to_csv(tdir / "S3_segmentation_references.csv", index=False)
    segcols = [c for c in ["id", "replicate", "T_otsu", "phi_T40", "phi_T50", "phi_T60", "phi_T50int", "phi_otsu"]
               if c in seg.columns]
    seg[segcols].to_csv(tdir / "S3_segmentation_porosity.csv", index=False)
    qa_f = cfg.outdir("qa") / "qa.csv"
    if qa_f.exists():
        qa = pd.read_csv(qa_f)
        qa.to_csv(tdir / "S3b_qa.csv", index=False)
        K.add("cupping_max_pct", float(100 * qa.cupping_frac.abs().max()), "%",
              "largest |solid median at VOI edge - centre| as % of the air-solid contrast, main VOIs")
        for g in ("coquina", "trachyte", "limestone"):
            q = qa[qa.group == g]
            if {"offset_depth2", "offset_depth8"} <= set(q.columns):
                pair = q[np.isfinite(q.offset_depth2.astype(float)) & np.isfinite(q.offset_depth8.astype(float))]
                if len(pair):
                    K.add(f"offset_depth_{g}", [float(pair.offset_depth2.median()), float(pair.offset_depth8.median())],
                          "", f"median internal-air offset at depth 2 and 8 voxels, {g}, samples with both depths")
                    K.add(f"offset_depth_{g}_n", [len(pair), len(q)], "",
                          f"samples of {g} with an internal-air offset at both depths 2 and 8, samples of {g}")
        qr = qa[qa.group != "granular"]
        if "air_far_shift_frac" in qr:
            K.add("air_far_shift_frac_range", _range(qr.air_far_shift_frac), "",
                  "far-field minus near outside-air median as fraction of the air-solid contrast, rocks")
        if {"offset_depth5", "offset_depth5_prov08"} <= set(qr.columns):
            K.add("offset_depth5_range", _range(qr.offset_depth5), "",
                  "internal-air offset at depth 5, provisional pore phase f = 0.6, rocks")
            K.add("offset_depth5_prov08_range", _range(qr.offset_depth5_prov08), "",
                  "internal-air offset at depth 5, provisional pore phase f = 0.8, rocks")
            K.add("offset_depth5_prov08_minus_06_range", _range(qr.offset_depth5_prov08 - qr.offset_depth5), "",
                  "change of the depth-5 internal-air offset from f = 0.6 to f = 0.8, rocks")
    rock = refs[refs.group != "granular"]
    if "air_int_offset" in rock and rock.air_int_offset.notna().any():
        K.add("air_offset_range", [float(rock.air_int_offset.min()), float(rock.air_int_offset.max())], "",
              "internal-air offset as fraction of the outside air-solid contrast (rock samples where measurable)")
        for g in ("coquina", "trachyte", "limestone"):
            q = rock[rock.group == g].air_int_offset.dropna()
            if len(q):
                K.add(f"air_offset_{g}_range", [float(q.min()), float(q.max())], "", f"internal-air offset range, {g}")
        K.add("n_samples_no_internal_air", int(rock.air_int_offset.isna().sum()), "",
              "rock samples with too few deep pore voxels for an internal-air estimate")
    K.add("n_samples", len(cfg.samples), "", "number of samples")
    K.add("n_vois", len(meta), "", "number of VOIs analysed")
    for g in GROUP_ORDER:
        K.add(f"n_{g}", int(sum(s.group == g for s in cfg.samples)), "", f"samples in group {g}")

    # ---- Table 2: porosity, connectivity and topology (main VOI + replicate spread)
    rows = []
    for s in cfg.samples:
        d = t50[t50.id == s.id]
        m = main.loc[s.id]
        sg = seg[(seg.id == s.id) & (seg.replicate == 0)].iloc[0]
        rows.append({
            "Sample": s.id, "Group": GROUP_LABEL[s.group],
            "phi (%)": 100 * m.phi, "phi replicate SD (pp)": 100 * d.phi.std(ddof=1) if len(d) > 1 else np.nan,
            "phi T40-T60 (%)": f"{100 * sg.phi_T40:.2f}-{100 * sg.phi_T60:.2f}",
            "phi T50int (%)": 100 * sg.phi_T50int if np.isfinite(sg.get("phi_T50int", np.nan)) else np.nan,
            "phi Otsu (%)": 100 * sg.phi_otsu,
            "Connected fraction (%)": 100 * m.frac_connected, "Isolated fraction (%)": 100 * m.frac_isolated,
            "Face-cut non-spanning fraction (%)": 100 * m.frac_boundary_nonspanning,
            "Percolating axes (z,y,x)": f"{int(m.perc_z)}{int(m.perc_y)}{int(m.perc_x)}",
            "Replicates percolating (any axis)": f"{int(d.perc_any.sum())}/{len(d)}",
            "chi_V (mm-3)": m.chi_V, "chi_V l_p^3": m.chi_V * m.l_p ** 3, "S_V (mm-1)": m.S_V,
            "Mean pore chord (mm)": m.l_p, "d50 (mm)": m.get("d50", np.nan),
            "Chord anisotropy": m.chord_anisotropy, "Clusters": int(m.n_clusters),
        })
    t2 = pd.DataFrame(rows)
    t2.to_csv(tdir / "T2_porosity_topology.csv", index=False)

    # group summaries for the text
    for g in GROUP_ORDER:
        d = main[main.group == g]
        K.add(f"phi_{g}_range", [100 * d.phi.min(), 100 * d.phi.max()], "%", f"ISO-50 porosity range, {g}")
        K.add(f"conn_{g}_range", [100 * d.frac_connected.min(), 100 * d.frac_connected.max()], "%",
              f"connected fraction range, {g}")
        K.add(f"chiV_{g}_range", [d.chi_V.min(), d.chi_V.max()], "mm-3", f"chi_V range, {g}")
        K.add(f"lp_{g}_range", _range(d.l_p), "mm", f"mean pore chord 4 phi / S_V range, {g}")
        K.add(f"iso_{g}_range", _range(100 * d.frac_isolated), "%",
              f"share of the resolved porosity in clusters touching no VOI face, {g}")
        K.add(f"facecut_{g}_range", _range(100 * d.frac_boundary_nonspanning), "%",
              f"share of the resolved porosity in clusters cut by a VOI face that do not span it, {g}")
        rep = t2.loc[t2.Group == GROUP_LABEL[g], "phi replicate SD (pp)"]
        K.add(f"phi_rep_sd_{g}_range", _range(rep), "pp", f"standard deviation of porosity between replicate VOIs, {g}")
        K.add(f"nperc_{g}", int(d.perc_any.sum()), "", f"samples whose main VOI percolates along any axis, {g}")
        K.add(f"nchi_neg_{g}", [int((d.chi_V < 0).sum()), len(d)], "", f"main VOIs with chi_V < 0, main VOIs, {g}")
        va = t50[t50.group == g]
        K.add(f"n_vois_perc_{g}", [int(va.perc_any.sum()), len(va)], "", f"VOIs (all replicates) percolating, VOIs, {g}")
        K.add(f"nperc3_{g}", int((d.perc_z & d.perc_y & d.perc_x).sum()), "", f"main VOIs percolating along all axes, {g}")
    allv = t50
    K.add("n_vois_percolating", int(allv.perc_any.sum()), "", "VOIs percolating along any axis")
    K.add("sign_test_chi_perc", [int(((allv.chi_V < 0) & allv.perc_any).sum()),
                                 int(((allv.chi_V >= 0) & ~allv.perc_any).sum()), len(allv)], "",
          "VOIs with chi<0 that percolate, chi>=0 that do not, total")

    # robustness of the conclusions to the segmentation (per VOI, relative to ISO-50)
    sens = seg.assign(span=100 * (seg.phi_T60 - seg.phi_T40))
    K.add("thr_span_pp_range", [float(sens.span.min()), float(sens.span.max())], "pp", "phi(T60) - phi(T40) across VOIs")
    for g in GROUP_ORDER:
        q = sens[sens.group == g]
        K.add(f"thr_span_{g}_range", [float(q.span.min()), float(q.span.max())], "pp", f"phi(T60) - phi(T40), {g}")
    ti = topo.assign(**{f"perc_{a}": topo[f"perc_{a}"].astype(int) for a in ("any", "z", "y", "x")})
    piv = {a: ti.pivot_table(index=["id", "replicate"], columns="variant", values=f"perc_{a}", aggfunc="first")
           for a in ("any", "z", "y", "x")}
    chi = ti.pivot_table(index=["id", "replicate"], columns="variant", values="chi_V", aggfunc="first")
    rob = []
    for var in ("T40", "T60", "T50int", "otsu"):
        if var not in piv["any"]:
            continue
        p = piv["any"]
        ok = p[var].notna() & p["T50"].notna()
        n_perc = int((p.loc[ok, var].astype(bool) != p.loc[ok, "T50"].astype(bool)).sum())
        n_sign = int((np.sign(chi.loc[ok, var]) != np.sign(chi.loc[ok, "T50"])).sum())
        # per axis: VOI-axis cases whose percolation differs from ISO-50
        n_ax = gained = lost = 0
        for a in "zyx":
            pa = piv[a]
            oka = pa[var].notna() & pa["T50"].notna()
            v, t = pa.loc[oka, var].astype(bool), pa.loc[oka, "T50"].astype(bool)
            n_ax += int(oka.sum())
            gained += int((v & ~t).sum())
            lost += int((~v & t).sum())
        rob.append({"variant": var, "vois_compared": int(ok.sum()), "percolation_changes": n_perc,
                    "chi_sign_changes": n_sign, "axis_cases_compared": n_ax,
                    "axis_percolation_changes": gained + lost, "axis_gained": gained, "axis_lost": lost})
        K.add(f"perc_changes_{var}", [n_perc, int(ok.sum())], "", f"VOIs whose percolation differs from ISO-50 at {var}")
        K.add(f"perc_axis_changes_{var}", [gained + lost, n_ax], "",
              f"VOI-axis cases whose percolation differs from ISO-50 at {var}, cases compared")
        K.add(f"perc_axis_gained_lost_{var}", [gained, lost], "",
              f"VOI-axis cases that percolate only at {var}, only at ISO-50")
        K.add(f"chi_sign_changes_{var}", [n_sign, int(ok.sum())], "", f"VOIs whose sign of chi_V differs from ISO-50 at {var}")
    pd.DataFrame(rob).to_csv(tdir / "S4_robustness.csv", index=False)
    seg_main = seg[seg.replicate == 0].set_index("id")
    dif = 100 * (seg_main.phi_otsu - seg_main.phi_T50)
    K.add("otsu_minus_iso_pp_range", [float(dif.min()), float(dif.max())], "pp", "Otsu minus ISO-50 porosity, main VOIs")
    for g in GROUP_ORDER:
        q = dif[[samples[i].group == g for i in dif.index]]
        K.add(f"otsu_minus_iso_{g}_range", [float(q.min()), float(q.max())], "pp", f"Otsu minus ISO-50 porosity, {g}")

    # ---- legacy comparison
    leg = legacy.set_index("id")
    rows = []
    n_zero = 0
    for s in cfg.samples:
        m = main.loc[s.id]
        l = leg.loc[s.id]
        k_leg = float(l.k_lbm_z_mD)
        zero = not (np.isfinite(k_leg) and k_leg > 0)
        n_zero += zero
        rows.append({"Sample": s.id, "Group": GROUP_LABEL[s.group],
                     "Voxel reported (um)": l.voxel_um_reported, "Voxel actual (um)": s.voxel_um,
                     "phi dissertation (%)": l.phi_total_pct, "phi this study (%)": 100 * m.phi,
                     "phi_eff dissertation (%)": l.phi_eff_pct, "connected this study (%)": 100 * m.phi_span_any,
                     "chi dissertation": l.euler, "chi this study": int(m.euler),
                     "perc dissertation (x,y,z)": f"{int(l.perc_x)}{int(l.perc_y)}{int(l.perc_z)}",
                     "perc this study (z,y,x)": f"{int(m.perc_z)}{int(m.perc_y)}{int(m.perc_x)}",
                     "k LBM dissertation (mD)": np.nan if zero else k_leg,
                     "k LBM dissertation note": "not available: reported as 0 by the dissertation's solver"
                                                if zero else ""})
    tl = pd.DataFrame(rows)
    tl.to_csv(tdir / "S_legacy_comparison.csv", index=False)
    vn = tl[tl.Group == GROUP_LABEL["limestone"]]
    K.add("legacy_vn_phi", list(vn["phi dissertation (%)"]), "%", "dissertation porosity VN01, VN02, VN04")
    K.add("new_vn_phi", list(vn["phi this study (%)"]), "%", "this study porosity VN01, VN02, VN04")
    K.add("legacy_k_lbm_zero", [int(n_zero), len(tl)], "",
          "samples whose dissertation LBM permeability is 0 (a solver failure, shown as not available), samples")
    if n_zero:
        notes.append(f"S_legacy_comparison: the dissertation reports k_LBM = 0 for {n_zero} of {len(tl)} samples; "
                     "this is a missing value (failed solver), shown as not available, not a result.")

    # ---- laboratory comparison
    rows = []
    wl = pd.DataFrame()
    if len(wall):
        wl = wall.copy()
        wl["wall_detected"] = flag(wl, "wall_detected") if "wall_detected" in wl else wl.inner_diameter_mm.notna()
        wl = wl.set_index("id")
    for sid in lab.id.unique():
        l = lab[lab.id == sid]
        r = {"Sample": sid, "VOI phi main (%)": 100 * main.loc[sid].phi,
             "VOI phi mean of replicates (%)": 100 * t50[t50.id == sid].phi.mean()}
        for _, x in l.iterrows():
            r[f"{x.method} (%)"] = x.porosity_pct
        if len(wl) and sid in wl.index:
            w = wl.loc[sid]
            r["Whole section (%)"] = 100 * w.phi_section
            r["Core, >1 grain from wall (%)"] = 100 * w.phi_core
            r["Within half a grain of wall (%)"] = 100 * w.phi_within_half_grain_of_wall
            r["Wall detected"] = bool(w.wall_detected)
            r["Wall detection method"] = w.get("detection_method", "")
            r["Inner diameter detected (mm)"] = w.inner_diameter_mm if w.wall_detected else np.nan
            r["Inner diameter nominal (mm)"] = w.inner_diameter_nominal_mm
        rows.append(r)
    tlab = pd.DataFrame(rows)
    beads = tlab[tlab.Sample.str.startswith("CF")]
    if all(f"{m} (%)" in beads for m in LAB_METHODS):
        tlab["Lab mean, 4 methods (%)"] = tlab[[f"{m} (%)" for m in LAB_METHODS]].mean(axis=1, skipna=False)
        beads = tlab[tlab.Sample.str.startswith("CF")]
    tlab.to_csv(tdir / "T3_lab_comparison.csv", index=False)
    if "Whole section (%)" in beads and "Lab mean, 4 methods (%)" in beads:
        phys = beads["Lab mean, 4 methods (%)"]
        K.add("beads_voi_minus_lab_pp", _range(beads["VOI phi mean of replicates (%)"] - phys), "pp",
              "VOI porosity minus mean of physical lab methods, bead packs")
        K.add("beads_section_minus_lab_pp", _range(beads["Whole section (%)"] - phys), "pp",
              "whole-section porosity minus mean of physical lab methods, bead packs")
        K.add("beads_nearwall_phi_range", _range(beads["Within half a grain of wall (%)"]), "%",
              "porosity within half a grain diameter of the wall")
        det = beads["Wall detected"].eq(True)
        K.add("beads_n_wall_detected", [int(det.sum()), len(beads)], "",
              "bead columns whose inner tube wall was detected in the images, bead columns")
        err = (beads["Inner diameter detected (mm)"] / beads["Inner diameter nominal (mm)"] - 1)[det].abs()
        K.add("beads_inner_diam_err_pct", float(100 * err.max()) if len(err) else np.nan,
              "%", "largest relative difference detected vs nominal tube inner diameter (columns with the wall detected)")
    gr_f = cfg.outdir("grains") / "grains_summary.csv"
    if gr_f.exists():
        gr = pd.read_csv(gr_f)
        gr.to_csv(tdir / "S_bead_geometry.csv", index=False)
        K.add("beads_deq_median_range", _range(gr.d_eq_median_mm), "mm",
              "median volume-equivalent bead diameter, bead packs (grains not cut by the VOI faces)")
        K.add("beads_deq_over_nominal_range", _range(gr.d_eq_median_mm / gr.nominal_mm), "",
              "median volume-equivalent bead diameter / nominal size")
        K.add("beads_aspect_median_range", _range(gr.aspect_median), "", "median major/minor inertia axis ratio of the beads")
        K.add("beads_frac_aspect_gt_1_2_range", _range(100 * gr.frac_aspect_gt_1_2), "%",
              "share of beads with axis ratio above 1.2")
        K.add("beads_n_grains_range", _range(gr.n_grains), "",
              "beads measured per pack (all VOIs of the column, beads not cut by a VOI face)")
        ss_f = cfg.outdir("grains") / "grains_seed_sensitivity.csv"
        if ss_f.exists() and ss_f.stat().st_size > 1:
            ss = pd.read_csv(ss_f)
            ss.to_csv(tdir / "S_bead_seed_sensitivity.csv", index=False)
            ref = gr.set_index("id")
            dev = {c: 100 * (ss[c] / ss.id.map(ref[c]) - 1).abs().max() for c in ("d_eq_median_mm", "aspect_median")}
            K.add("beads_seed_sens", [int(ss.seed_vox.min()), int(ss.seed_vox.max()), float(dev["d_eq_median_mm"]),
                                      float(dev["aspect_median"])], "",
                  "watershed seed distance range (voxels); largest change of the median bead diameter and of the "
                  "median axis ratio relative to the main seed (%)")
        if len(wl):
            g2 = gr.set_index("id")
            ids = [i for i in g2.index if i in wl.index]
            ratio = [wl.loc[i, "inner_diameter_nominal_mm"] / g2.loc[i, "d_eq_median_mm"] for i in ids]
            K.add("beads_tube_over_bead_range", _range(ratio), "",
                  "nominal inner tube diameter / median volume-equivalent bead diameter")
    vnl = tlab[tlab.Sample.str.startswith("VN")]
    K.add("vn_resolved_over_helium", list((vnl["VOI phi mean of replicates (%)"] / vnl["helium (%)"]).round(3)),
          "", "resolved / helium porosity, VN01 VN02 VN04")

    # ---- LBM runs: usable rows (converged and finite k), PNM on the same domain
    lc = pd.DataFrame()
    if len(lbm):
        lc = lbm.copy()
        ok = flag(lc, "ok") if "ok" in lc else flag(lc, "converged")
        k_ok = np.isfinite(lc.k_lbm_m2.astype(float)) & (lc.k_lbm_m2.astype(float) > 0)
        lc["used"] = ok & k_ok
        crop = flag(lc, "crop_percolates", default=True)
        lc["excluded_reason"] = np.where(~crop, "central sub-volume does not percolate along this axis",
                                         np.where(~ok, "LBM not ok (not converged or failed checks)",
                                                  np.where(~k_ok, "k_LBM not finite or not positive", "")))
        for t in PNM_TAGS:
            col = f"k_{t}_m2"
            lc[f"r_{t}"] = np.where(lc.used, lc[col].astype(float) / lc.k_lbm_m2.astype(float), np.nan) \
                if col in lc else np.nan
        est_f = cfg.outdir("lbm") / "estimates.csv"
        if est_f.exists():
            est = pd.read_csv(est_f).set_index(["id", "axis"])
            key = list(zip(lc.id, lc.axis))
            for kk, col in (("KC", "k_KC_m2"), ("KCspan", "k_KC_span_m2"), ("KT", "k_KT_m2")):
                if col not in est:
                    continue
                v = np.array([est.at[ka, col] if ka in est.index else np.nan for ka in key], float)
                lc[f"k_{kk}_dom_mD"] = v / MD_M2
                lc[f"r_{kk}_dom"] = np.where(lc.used, v / lc.k_lbm_m2.astype(float), np.nan)
            lc["d_c_vox_dom"] = [est.at[ka, "d_c_vox"] if ka in est.index else np.nan for ka in key]
            for t in PNM_TAGS:                      # networks on the whole VOI binned by 2, vs the LBM there
                col = f"k_{t}_whole2_mD"
                if col in est:
                    v = np.array([est.at[ka, col] if ka in est.index else np.nan for ka in key], float)
                    lc[col] = v
                    w2ok = flag(lc, "whole2_ok") & lc.used
                    lc[f"r_{t}_whole2"] = np.where(w2ok, v / lc.k_lbm_whole2_mD.astype(float), np.nan)
        if "k_lbm_whole2_mD" in lc:
            w2 = flag(lc, "whole2_ok") & lc.used & (lc.domain != "full VOI")
            lc["r_whole2_crop"] = np.where(w2, lc.k_lbm_whole2_mD.astype(float) / lc.k_lbm_mD.astype(float), np.nan)
        lc.to_csv(tdir / "S_method_comparison.csv", index=False)

    # ---- transport table (T4): full-VOI columns from `transport`, LBM-domain columns from `lbm`
    perc = tra[flag(tra, "perc")].copy()
    cols = [c for c in ["id", "group", "axis", "phi", "phi_span", "F", "tau", "m_archie", "fd_converged",
                        "k_pnm_mD", "k_pnmeq_mD", "k_pnmw_mD", "k_pnmeqw_mD", "k_KC_mD", "k_KT_mD",
                        "d_c_mm", "d_c_vox", "d_c_resolved"] if c in perc.columns]
    tt = perc[cols].copy()
    tt.insert(3, "transport_domain", f"full VOI {voi_edge}^3")
    if len(lc):
        lb = lc[~lc.set_index(["id", "axis"]).index.duplicated()].set_index(["id", "axis"])
        key = list(zip(tt.id, tt.axis))

        def pick(col, default=np.nan):
            return [lb.at[ka, col] if (ka in lb.index and col in lb) else default for ka in key]
        tt["lbm_domain"] = pick("domain", "")
        tt["lbm_edge"] = pick("edge")
        tt["lbm_ok"] = [bool(v) if isinstance(v, (bool, np.bool_)) else False for v in pick("used", False)]
        tt["k_lbm_mD"] = np.where(tt.lbm_ok, np.asarray(pick("k_lbm_mD"), float), np.nan)
        for t in PNM_TAGS:
            tt[f"k_{t}_lbmdom_mD"] = np.asarray(pick(f"k_{t}_mD"), float)
        same = tt.lbm_ok & (tt.lbm_domain == "full VOI")
        for kk in ("KC", "KT"):
            if f"k_{kk}_mD" in tt:
                tt[f"ratio_{kk}_lbm_same_domain"] = np.where(same, tt[f"k_{kk}_mD"] / tt.k_lbm_mD, np.nan)
                r = tt.loc[same, f"ratio_{kk}_lbm_same_domain"]
                K.add(f"ratio_{kk}_lbm_range", _range(r), "",
                      f"k_{kk}/k_LBM, cases whose LBM domain is the whole VOI (same domain as k_{kk})")
                K.add(f"ratio_{kk}_lbm_n", int(np.isfinite(r).sum()), "",
                      f"cases with k_{kk}/k_LBM on the same domain (whole VOI, LBM ok)")
    tt.to_csv(tdir / "T4_transport.csv", index=False)
    notes.append(f"T4_transport: the columns from phi to d_c_resolved (F, the four pore networks k_pnm*_mD, k_KC_mD, "
                 f"k_KT_mD) refer to the whole main VOI ({voi_edge}^3, column transport_domain); lbm_domain and "
                 "lbm_edge give the domain of k_lbm_mD and of the k_<tag>_lbmdom_mD columns (pore network on the LBM "
                 "domain); k_lbm_mD is empty when the LBM run is not ok; KC and KT are divided by k_LBM only where "
                 "the LBM domain is the whole VOI.")
    K.add("n_perc_axes", len(perc), "", "sample-axis cases that percolate (main VOI)")
    if "fd_converged" in perc:
        K.add("n_fd_not_converged", int((~flag(perc, "fd_converged", default=True)).sum()), "",
              "percolating sample-axis cases whose diffusion solve did not converge (main VOI)")
    for g in GROUP_ORDER:
        d = perc[perc.group == g]
        if len(d):
            K.add(f"F_{g}_range", [d.F.min(), d.F.max()], "", f"formation factor range, {g}")
            K.add(f"m_{g}_range", [d.m_archie.min(), d.m_archie.max()], "", f"Archie exponent range, {g}")
            if "d_c_vox" in d:
                K.add(f"d_c_vox_{g}_range", _range(d.d_c_vox), "voxel", f"critical diameter in voxels, percolating axes, {g}")
    rk = perc[perc.group != "granular"]
    if len(rk) and {"k_KC_mD", "k_pnm_mD", "k_pnmeq_mD"} <= set(rk.columns):
        K.add("kc_over_pnm_rocks_range", _range(rk.k_KC_mD / rk.k_pnm_mD), "",
              "k_KC / k_PNM(inscribed, mirror faces), whole VOI, percolating rock axes")
        K.add("kc_over_pnmeq_rocks_range", _range(rk.k_KC_mD / rk.k_pnmeq_mD), "",
              "k_KC / k_PNM(equivalent sizes, mirror faces), whole VOI, percolating rock axes")

    # ---- method comparison (same domain, LBM ok)
    if len(lc):
        u = lc[lc.used]
        K.add("n_lbm_ok", int(len(u)), "", "LBM sample-axis runs used (ok and finite k)")
        K.add("n_lbm_excluded", int((~lc.used).sum()), "", "LBM sample-axis runs excluded (not ok or k not finite)")
        if (~lc.used).any():
            K.add("lbm_excluded_cases", [f"{i} {a}" for i, a in zip(lc.id[~lc.used], lc.axis[~lc.used])], "",
                  "LBM runs excluded from the method comparison")
        K.add("n_lbm_domains", [int((u.domain == "full VOI").sum()), int((u.domain != "full VOI").sum())], "",
              "LBM runs used on the whole VOI, on the central sub-volume")
        for t, lab_t in PNM_TAGS.items():
            r = u[f"r_{t}"].to_numpy(float)
            if not np.isfinite(r).any():
                continue
            K.add(f"ratio_{t}_lbm_range", _range(r), "", f"k_PNM({lab_t})/k_LBM range, same domain, LBM ok")
            K.add(f"ratio_{t}_lbm_gmean", _gmean(r), "", f"geometric mean k_PNM({lab_t})/k_LBM, same domain, LBM ok")
            K.add(f"ratio_{t}_lbm_n", [int(np.isfinite(r).sum()), int((r == 0).sum())], "",
                  f"cases with k_PNM({lab_t})/k_LBM, of which k_PNM = 0 (excluded from the geometric mean)")
        if "d_c_vox_dom" in u:                  # cases whose critical throat is resolved (d_c >= D_C_MIN_VOX)
            res = u[u.d_c_vox_dom.astype(float) >= D_C_MIN_VOX]
            K.add("n_lbm_dc_resolved", [int(len(res)), int(len(u))], "",
                  f"LBM runs used whose critical diameter on the domain spans >= {D_C_MIN_VOX} voxels, of all used")
            for t, lab_t in PNM_TAGS.items():
                r = res[f"r_{t}"].to_numpy(float)
                if np.isfinite(r).any():
                    K.add(f"ratio_{t}_lbm_res_range", _range(r), "", f"k_PNM({lab_t})/k_LBM, resolved d_c only")
                    K.add(f"ratio_{t}_lbm_res_gmean", _gmean(r), "", f"geometric mean k_PNM({lab_t})/k_LBM, resolved d_c only")
        for _, rr in u[u.group != "granular"].iterrows():  # each rock case on its own
            K.add(f"lbm_case_{rr.id}_{rr.axis}",
                  [float(rr.k_lbm_mD) / 1000] + [float(rr.get(c, np.nan)) for c in
                                                 ("r_pnm", "r_pnmeq", "r_KC_dom", "r_KT_dom", "d_c_vox_dom", "F",
                                                  "r_whole2_crop", "r_KCspan_dom", "phi", "phi_span")], "",
                  f"{rr.id} {rr.axis}: k_LBM (D), k_PNM/k_LBM inscribed and equivalent, k_KC/k_LBM, k_KT/k_LBM, "
                  "d_c on the domain (voxels), F on the domain, k_LBM(whole VOI x2)/k_LBM, k_KC(spanning "
                  "cluster)/k_LBM, porosity and spanning porosity of the domain")
        whole2 = u[np.isfinite(u.get("r_pnm_whole2", pd.Series(np.nan, index=u.index)).astype(float))] \
            if "r_pnm_whole2" in u else u.iloc[0:0]
        for g, sub in (("granular", whole2[whole2.group == "granular"]), ("all", whole2)):
            for t, lab_t in PNM_TAGS.items():
                r = sub.get(f"r_{t}_whole2", pd.Series(dtype=float)).to_numpy(float)
                if np.isfinite(r).any():
                    K.add(f"ratio_{t}_lbm_whole2_{g}_range", _range(r), "",
                          f"k_PNM({lab_t})/k_LBM, both on the whole VOI block-averaged by 2, {g}")
                    K.add(f"ratio_{t}_lbm_whole2_{g}_gmean", _gmean(r), "",
                          f"geometric mean k_PNM({lab_t})/k_LBM on the whole VOI block-averaged by 2, {g}")
        if "pnm_face_pore_frac" in u:
            bu = u[u.group == "granular"]
            K.add("pnm_face_pore_frac_beads_range", _range(100 * bu.pnm_face_pore_frac.astype(float)), "%",
                  "share of network pores touching a face, bead-pack LBM domains (central 150^3)")
            K.add("pnm_n_pores_beads_range", _range(bu.pnm_n_pores.astype(float)), "",
                  "pores of the networks on the bead-pack LBM domains")
        for t in PNM_TAGS:
            c = f"{t}_throats_clipped"
            if c in u:
                K.add(f"pnm_clipped_{t}_range", _range(100 * u[c].astype(float)), "%",
                      f"throats clipped to 0.99 x the smaller pore diameter, {PNM_TAGS[t]}, LBM domains")
        if "r_whole2_crop" in u:
            bw = {}
            for sid, d in u[u.group == "granular"].groupby("id"):
                r = d.r_whole2_crop.to_numpy(float)
                if np.isfinite(r).any():
                    bw[sid] = [float(np.nanmin(r)), float(np.nanmax(r)), float(d.phi.iloc[0]),
                               float(d.whole2_phi_span.astype(float).iloc[0])]
            if bw:
                K.add("beads_whole2_by_sample", bw, "", "per bead pack: k_LBM(whole VOI x2)/k_LBM(central 150^3) min, "
                      "max; porosity of the central sub-volume; spanning porosity of the whole VOI x2")
                rest = [v for s, v in bw.items() if v[2] <= v[3] * 1.1]     # sub-volume as porous as the VOI
                if rest:
                    K.add("beads_whole2_over_crop_typical_range", [min(v[0] for v in rest), max(v[1] for v in rest)],
                          "", "k_LBM(whole VOI x2)/k_LBM(sub-volume), bead packs whose sub-volume porosity is within "
                          "10 % of that of the VOI")
        for kk in ("KC", "KCspan", "KT"):
            if f"r_{kk}_dom" in u:
                r = u[f"r_{kk}_dom"].to_numpy(float)
                if np.isfinite(r).any():
                    K.add(f"ratio_{kk}_lbm_dom_range", _range(r), "",
                          f"k_{kk}/k_LBM with both on the LBM domain, LBM ok")
                    K.add(f"ratio_{kk}_lbm_dom_n", int(np.isfinite(r).sum()), "", f"cases with k_{kk}/k_LBM on the LBM domain")
                    for g in GROUP_ORDER:
                        rg = u.loc[u.group == g, f"r_{kk}_dom"].to_numpy(float)
                        if np.isfinite(rg).any():
                            K.add(f"ratio_{kk}_lbm_dom_{g}_range", _range(rg), "", f"k_{kk}/k_LBM on the LBM domain, {g}")
        for t in PNM_TAGS:
            for g in GROUP_ORDER:
                rg = u.loc[u.group == g, f"r_{t}"].to_numpy(float)
                if np.isfinite(rg).any():
                    K.add(f"ratio_{t}_lbm_{g}_range", _range(rg), "", f"k_PNM({PNM_TAGS[t]})/k_LBM, same domain, {g}")
        b = u[u.group == "granular"]
        if len(b):
            K.add("k_lbm_beads_range_D", [b.k_lbm_mD.min() / 1000, b.k_lbm_mD.max() / 1000], "D",
                  "LBM permeability of the bead packs (domain: " + ", ".join(sorted(b.domain.unique())) + ")")
        if "r_whole2_crop" in lc:
            for sub, name, desc in ((lc[lc.group == "granular"], "beads_lbm_whole2_over_crop", "bead packs"),
                                    (lc, "lbm_whole2_over_crop", "all samples")):
                r = sub.r_whole2_crop.to_numpy(float)
                if np.isfinite(r).any():
                    K.add(f"{name}_range", _range(r), "", "k_LBM(whole VOI block-averaged x2) / k_LBM(central "
                          f"sub-volume), {desc}: domain-size check")
                    K.add(f"{name}_n", int(np.isfinite(r).sum()), "", f"cases in the domain-size check, {desc}")
        K.add("n_lbm_rock_cases", int((lc.group != "granular").sum()), "", "rock sample-axis cases simulated with LBM")
        K.add("n_lbm_rock_ok", int((u.group != "granular").sum()), "", "rock sample-axis LBM runs used (ok)")
        # bead packs: Carman-Kozeny for spheres, k = d^2 phi^3 / (180 (1 - phi)^2), with the Sauter mean
        # diameter d32 = sum d^3 / sum d^2 of the measured volume-equivalent bead diameters (all VOIs)
        gc_f = cfg.outdir("grains") / "grains.csv"
        if gc_f.exists() and len(b):
            gcs = pd.read_csv(gc_f)
            d32 = gcs.groupby("id").d_eq_mm.apply(lambda x: float((x ** 3).sum() / (x ** 2).sum()))
            K.add("beads_d32_range", _range(d32), "mm", "Sauter mean diameter of the beads, bead packs")
            bb = b[b.id.isin(d32.index)].copy()
            if len(bb):
                d = d32.loc[bb.id].to_numpy(float) * 1e-3
                phi_d = bb.phi.to_numpy(float)
                bb["d32_mm"] = d * 1e3
                bb["k_CK_spheres_m2"] = d ** 2 * phi_d ** 3 / (180 * (1 - phi_d) ** 2)
                bb["ratio_lbm_CK_spheres"] = bb.k_lbm_m2.astype(float) / bb.k_CK_spheres_m2
                bb[["id", "axis", "domain", "phi", "d32_mm", "k_lbm_m2", "k_CK_spheres_m2",
                    "ratio_lbm_CK_spheres"]].to_csv(tdir / "S_beads_carman_kozeny.csv", index=False)
                K.add("beads_lbm_over_CKspheres_range", _range(bb.ratio_lbm_CK_spheres), "",
                      "k_LBM / Carman-Kozeny for spheres (Sauter diameter of the beads, porosity of the LBM domain)")

    # ---- transport at the threshold bounds (rocks, main VOI)
    thr_f = cfg.outdir("transport") / "transport_thresholds.csv"
    if thr_f.exists() and thr_f.stat().st_size > 1:
        th = pd.read_csv(thr_f)
        th["perc"] = flag(th, "perc")
        if (th.variant == "T50").any():
            both = th.copy()
        else:                                       # older runs: ISO-50 rows from the transport stage
            base = tra[["id", "axis", "perc", "phi", "F", "k_pnm_mD", "k_pnmeq_mD"]].assign(
                variant="T50", perc=flag(tra, "perc"))
            both = pd.concat([th, base[base.id.isin(th.id.unique())]], ignore_index=True)
        both.to_csv(tdir / "S_transport_thresholds.csv", index=False)
        for var in ("T40", "T50", "T60", "T50int"):
            q = both[both.variant == var]
            if not len(q):
                continue
            K.add(f"thr_nperc_rocks_{var}", [int(q.groupby("id").perc.any().sum()), int(q.id.nunique()),
                                             int(q.perc.sum())], "",
                  f"rock samples whose main VOI percolates (any axis) at {var}, rock samples, percolating sample-axes")
            K.add(f"thr_perc_cases_{var}", [f"{i} {a}" for i, a in zip(q.id[q.perc], q.axis[q.perc])], "",
                  f"rock sample-axes whose main VOI percolates at {var}")
        if "k_KC_span_mD" in both:
            pc = both[both.perc & np.isfinite(both.k_KC_span_mD.astype(float))]
            K.add("kc_span_over_kc_rocks_range", _range(pc.k_KC_span_mD / pc.k_KC_mD), "",
                  "k_KC of the spanning cluster / k_KC of the whole pore phase, rock main VOIs, every threshold")
        piv = both.pivot_table(index=["id", "axis"], columns="variant", values="k_pnm_mD", aggfunc="first")
        for var in ("T40", "T60", "T50int"):
            if var in piv and "T50" in piv:
                ok = (piv["T50"] > 0) & (piv[var] > 0)
                if ok.any():
                    K.add(f"thr_kpnm_ratio_{var}_range", _range(piv.loc[ok, var] / piv.loc[ok, "T50"]), "",
                          f"k_PNM(inscribed, mirror faces) at {var} / at T50, rock sample-axes percolating at both")
        pivF = both.pivot_table(index=["id", "axis"], columns="variant", values="F", aggfunc="first")
        for var in ("T40", "T60", "T50int"):
            if var in pivF and "T50" in pivF:
                ok = np.isfinite(pivF["T50"]) & np.isfinite(pivF[var])
                if ok.any():
                    K.add(f"thr_F_ratio_{var}_range", _range(pivF.loc[ok, var] / pivF.loc[ok, "T50"]), "",
                          f"formation factor at {var} / at T50, rock sample-axes percolating at both")

    # ---- synthetic validation
    val_f = cfg.outdir("validation") / "synthetic.csv"
    if val_f.exists():
        v = pd.read_csv(val_f).set_index("medium")
        v.to_csv(tdir / "S5_synthetic_validation.csv")
        v_ok = flag(v, "lbm_ok", default=True)
        for med in v.index:
            for col in [c for c in v.columns if c.startswith("ratio_")]:
                K.add(f"val_{med}_{col}", float(v.loc[med, col]), "",
                      f"synthetic {med}: {col}" + ("" if v_ok[med] else " (reference LBM run not ok)"))
            if "KT_prefactor_implied" in v and np.isfinite(v.loc[med, "KT_prefactor_implied"]):
                K.add(f"val_{med}_KT_inv_prefactor", float(1 / v.loc[med, "KT_prefactor_implied"]), "",
                      f"synthetic {med}: 1/c with k_LBM = c d_c^2 / F (Katz-Thompson: 226)")
                K.add(f"val_{med}_d_c_vox", float(v.loc[med, "d_c_vox"]), "voxel", f"synthetic {med}: critical diameter")
    win_f = cfg.outdir("validation") / "window_study.csv"
    if win_f.exists():
        ws = pd.read_csv(win_f)
        ws["lbm_ok"] = flag(ws, "lbm_ok", default=True)
        kcol = "k_lbm_window" if "k_lbm_window" in ws else "k_lbm"
        ws["k_lbm_rel_largest"] = np.nan
        for med, d in ws.groupby("medium", sort=False):
            d = d[d.lbm_ok & np.isfinite(d[kcol].astype(float))]
            if len(d):
                ws.loc[d.index, "k_lbm_rel_largest"] = d[kcol] / d.loc[d.window_vox.idxmax(), kcol]
        ws.to_csv(tdir / "S_window_study.csv", index=False)
        for med, d in ws.groupby("medium", sort=False):
            K.add(f"val_window_{med}_windows", [int(w) for w in d.window_vox], "voxel", f"window edges, {med}")
            ok_d = d[d.lbm_ok]
            for t, lab_t in PNM_TAGS.items():
                c = f"ratio_k_{t}_lbm"
                if c in d:
                    K.add(f"val_window_{med}_{t}_range", _range(ok_d[c]), "",
                          f"range over windows of k_PNM({lab_t})/k_LBM, {med}")
            rel = d.k_lbm_rel_largest.to_numpy(float)
            if np.isfinite(rel).any():
                K.add(f"val_window_{med}_klbm_rel_range", _range(rel), "",
                      f"range over windows of k_LBM on the window ({kcol}) / its value on the largest window "
                      f"with an LBM run, {med}")
            for t, lab_t in PNM_TAGS.items():
                r = ok_d[f"ratio_k_{t}_lbm"].to_numpy(float) if f"ratio_k_{t}_lbm" in ok_d else np.array([])
                r = r[np.isfinite(r) & (r > 0)]
                if len(r) > 1:
                    K.add(f"val_window_{med}_{t}_spread_pct", float(100 * (r.max() / r.min() - 1)), "%",
                          f"spread over windows of k_PNM({lab_t})/k_LBM (max/min - 1), {med}")
            if d.n_cells.notna().any():
                # lattice: the one-cell LBM reference and the LBM on larger windows must coincide
                k = np.concatenate([d.k_lbm.to_numpy(float)[:1], d.get("k_lbm_window", d.k_lbm).to_numpy(float)])
                k = k[np.isfinite(k) & (k > 0)]
                if len(k) > 1:
                    K.add(f"val_window_{med}_lbm_spread_pct", float(100 * (k.max() / k.min() - 1)), "%",
                          f"spread of k_LBM between the one-cell domain and larger windows (max/min - 1), {med}")
    if val_f.exists():
        for med in v.index:
            K.add(f"val_{med}_phi", float(v.loc[med, "phi"]), "", f"porosity of the synthetic medium {med}")

    # ---- coarsening
    cs = coars.copy()
    cs["perc_any"] = flag(cs, "perc_any")
    cs.to_csv(tdir / "S_coarsening.csv", index=False)
    # voxel size at which each sample loses percolation along every axis (only samples percolating natively)
    rows = []
    for sid, d in cs.groupby("id"):
        d = d.sort_values("factor")
        native = bool(d.perc_any.iloc[0])
        lost = d[~d.perc_any]
        rows.append({"id": sid, "group": d.group.iloc[0],
                     "perc_native": native,
                     "voxel_loss_um": float(1000 * lost.voxel_mm.iloc[0]) if (native and len(lost)) else np.nan,
                     "conn_native": float(d.frac_connected.iloc[0]),
                     "conn_x2": float(d[d.factor == 2].frac_connected.iloc[0]) if (d.factor == 2).any() else np.nan,
                     "phi_native": float(d.phi.iloc[0]),
                     "phi_x4": float(d[d.factor == 4].phi.iloc[0]) if (d.factor == 4).any() else np.nan})
    loss = pd.DataFrame(rows)
    loss.to_csv(tdir / "S_coarsening_summary.csv", index=False)
    for g in GROUP_ORDER:
        d = loss[loss.group == g]
        dn = d[d.perc_native]
        K.add(f"coarsen_loss_{g}", int(d.voxel_loss_um.notna().sum()), "",
              f"samples of {g} that percolate natively and lose percolation when coarsened up to x4")
        K.add(f"coarsen_perc_native_{g}", [int(len(dn)), int(len(d))], "",
              f"samples of {g} that percolate at the native voxel, samples coarsened")
        K.add(f"coarsen_conn_drop_{g}", _range(100 * (dn.conn_native - dn.conn_x2)),
              "pp", f"drop of the connected fraction from native to x2 voxel, {g} (samples percolating natively)")

    # per axis: voxel at which a natively percolating sample-axis stops percolating (NaN: never up to x4)
    rows = []
    for (sid, g), d in cs.groupby(["id", "group"]):
        d = d.sort_values("factor")
        for a in "zyx":
            pa = flag(d, f"perc_{a}").to_numpy()
            if pa[0]:
                lost = np.where(~pa)[0]
                rows.append({"id": sid, "group": g, "axis": a,
                             "voxel_loss_um": float(1000 * d.voxel_mm.iloc[lost[0]]) if len(lost) else np.nan,
                             "conn_native": float(d.frac_connected.iloc[0]),
                             "conn_x4": float(d.frac_connected.iloc[-1])})
    ax_loss = pd.DataFrame(rows)
    ax_loss.to_csv(tdir / "S_coarsening_axes.csv", index=False)

    # every VOI and air reference: critical diameter in coarse voxels when an axis stops percolating
    rows = []
    for (sid, k, var), d in coars_all.groupby(["id", "replicate", "variant"]):
        d = d.sort_values("factor")
        if d.group.iloc[0] == "granular":
            continue
        for a in "zyx":
            pa = flag(d, f"perc_{a}").to_numpy()
            if not pa[0]:
                continue
            dc = float(d[f"d_c_vox_{a}"].iloc[0]) if f"d_c_vox_{a}" in d else np.nan
            lost = np.where(~pa)[0]
            b_loss = int(d.factor.iloc[lost[0]]) if len(lost) else np.nan
            b_max = int(d.factor.iloc[-1])
            rows.append({"id": sid, "replicate": int(k), "variant": var, "axis": a, "d_c_vox": dc,
                         "factor_loss": b_loss, "d_c_coarse_vox_at_loss": dc / b_loss if len(lost) else np.nan,
                         "d_c_coarse_vox_at_max": dc / b_max if not len(lost) else np.nan,
                         "d_c_coarse_vox_before_loss": dc / int(d.factor.iloc[lost[0] - 1]) if len(lost) else np.nan})
    cdiag = pd.DataFrame(rows)
    cdiag.to_csv(tdir / "S_coarsening_diagnostic.csv", index=False)
    if len(cdiag):
        lostd = cdiag[cdiag.factor_loss.notna()]
        kept = cdiag[cdiag.factor_loss.isna()]
        K.add("coarsen_diag_n", [len(cdiag), len(lostd), int(cdiag.id.nunique()), len(kept)], "",
              "rock VOI-axis cases percolating at the native voxel (all VOIs, both air references), cases lost "
              "when coarsened up to x4, rock samples involved, cases kept up to x4")
        K.add("coarsen_diag_dc_native_range", _range(cdiag.d_c_vox), "voxel",
              "native critical diameter of the natively percolating rock VOI-axes")
        K.add("coarsen_diag_lost_dc_coarse_range", _range(lostd.d_c_coarse_vox_at_loss), "coarse voxel",
              "critical diameter in coarse voxels at the factor where the axis stopped percolating")
        K.add("coarsen_diag_lost_dc_before_range", _range(lostd.d_c_coarse_vox_before_loss), "coarse voxel",
              "critical diameter in coarse voxels at the last factor where the axis still percolated")
        K.add("coarsen_diag_kept_dc_coarse_range", _range(kept.d_c_coarse_vox_at_max), "coarse voxel",
              "critical diameter in coarse voxels at x4, axes that never stopped percolating")
        K.add("coarsen_diag_lost_native_dc_range", _range(lostd.d_c_vox), "voxel",
              "native critical diameter of the axes that stopped percolating")
        K.add("coarsen_diag_kept_native_dc_range", _range(kept.d_c_vox), "voxel",
              "native critical diameter of the rock axes that kept percolating up to x4")
    # chi_V under coarsening (main VOIs, ISO-50)
    chi_w = coars.pivot_table(index="id", columns="factor", values="chi_V", aggfunc="first")
    grp = coars.groupby("id").group.first()
    if {1, 2, 4} <= set(chi_w.columns):
        bead = chi_w[grp.reindex(chi_w.index) == "granular"]
        K.add("coarsen_chi_beads_x2_over_x1_range", _range(bead[2] / bead[1]), "",
              "chi_V at twice the voxel / native chi_V, bead packs (main VOIs)")
        trc = chi_w[grp.reindex(chi_w.index) == "trachyte"]
        flips = [i for i in trc.index if np.sign(trc.loc[i, 1]) != np.sign(trc.loc[i, 2])]
        K.add("coarsen_chi_trachyte_sign_flip_x2", [len(flips), len(trc)], "",
              "trachyte main VOIs whose chi_V changes sign from the native to twice the voxel, trachytes")
        K.add("coarsen_chi_trachyte_flip_ids", flips, "", "trachytes whose chi_V changes sign at twice the voxel")
        coq = chi_w[grp.reindex(chi_w.index) == "coquina"]
        K.add("coarsen_chi_coquina_x1_over_x4_range", _range(coq[1] / coq[4]), "",
              "native chi_V / chi_V at four times the voxel, coquinas (main VOIs)")
    for g in GROUP_ORDER:
        d = ax_loss[ax_loss.group == g] if len(ax_loss) else ax_loss
        if len(d):
            K.add(f"coarsen_axes_lost_{g}", [int(d.voxel_loss_um.notna().sum()), len(d)], "",
                  f"natively percolating sample-axes of {g} that stop percolating when the voxel is coarsened up to x4")
    lostd = ax_loss[ax_loss.voxel_loss_um.notna()] if len(ax_loss) else ax_loss
    if len(lostd):
        K.add("coarsen_axes_lost_cases", [f"{i} {a} at {v:.0f} um" for i, a, v in
                                          zip(lostd.id, lostd.axis, lostd.voxel_loss_um)], "",
              "sample-axes that stop percolating under coarsening, with the voxel at which they do")
    rel = 100 * (1 - loss.phi_x4 / loss.phi_native)
    for g in GROUP_ORDER:
        q = rel[loss.group == g]
        K.add(f"coarsen_phi_loss_pct_{g}", _range(q), "%", f"relative loss of porosity from native to x4 voxel, {g}")
    for sid in loss.id[loss.perc_native & (loss.group != "granular")]:
        d = ax_loss[ax_loss.id == sid]
        K.add(f"coarsen_conn_{sid}", [float(100 * d.conn_native.iloc[0]), float(100 * d.conn_x4.iloc[0])], "%",
              f"connected fraction of {sid} at the native and the x4 voxel")

    # ---- REV: spread of porosity across disjoint cubes of each VOI
    lp = main["l_p"].to_dict()
    rev = rev.copy()
    rev["overlapping"] = flag(rev, "overlapping")
    n_in_voi = rev.groupby(["id", "replicate", "edge"]).cube.transform("size")
    rev["basis"] = np.where(rev.overlapping, "overlapping corner cubes",
                            np.where(n_in_voi == 1, "one cube per VOI (between-VOI spread)", "disjoint tiles"))
    rev["l_p"] = rev.id.map(lp)
    rev["chi_l3"] = rev.chi_V * rev.l_p ** 3
    rev["chi_t"] = np.arcsinh(rev.chi_l3)
    cvs = []
    for (g, e, bs), d in rev.groupby(["group", "edge", "basis"]):
        per = d.groupby("id").agg(mean=("phi", "mean"), std=("phi", "std"), n=("phi", "size"),
                                  chi_mean=("chi_V", "mean"), chi_std=("chi_V", "std"))
        per = per[per.n >= 2]
        cvs.append({"group": g, "edge": e, "basis": bs, "in_cv_curve": bs == "disjoint tiles",
                    "edge_mm_mean": d.edge_mm.mean(),
                    "cv_phi_median": float((per["std"] / per["mean"]).median()) if len(per) else np.nan,
                    "sd_chi_over_abs_mean_median": float((per["chi_std"] / per["chi_mean"].abs()).median())
                    if len(per) else np.nan,
                    "n_samples": int(len(per)), "n_cubes": int(per.n.sum())})
    cv = pd.DataFrame(cvs)
    if len(cv):
        cv["_g"] = cv.group.map({g: i for i, g in enumerate(GROUP_ORDER)})
        cv = cv.sort_values(["_g", "edge"]).drop(columns="_g")
    cv.to_csv(tdir / "S_rev_cv.csv", index=False)
    notes.append("S_rev_cv: only rows with in_cv_curve True (disjoint tiles inside each VOI) form the CV curve; "
                 "the eight overlapping corner cubes and the single cube per VOI (spread between the VOIs of a "
                 "sample, a different sample set) are listed but not used.")
    for g in GROUP_ORDER:
        q = cv[(cv.group == g) & cv.in_cv_curve & np.isfinite(cv.cv_phi_median)] if len(cv) else cv
        if len(q):
            r = q.loc[q.edge.idxmax()]
            K.add(f"rev_cv_phi_{g}_largest", [int(r.edge), float(100 * r.cv_phi_median), int(r.n_samples)], "",
                  f"largest disjoint-tile edge (voxels), median CV of porosity (%), samples, {g}")

    # ---- percolation prediction
    models = {"phi": ["phi"], "chi": ["chi_t"], "phi+chi": ["phi", "chi_t"]}
    pred_rows = []
    for subset in ("all", "rocks", "coquina"):
        for e in sorted(rev.edge.unique()):
            d = rev[(rev.edge == e) & ~rev.overlapping]     # overlapping corner cubes are not independent cases
            if not len(d):
                continue
            if subset == "rocks":
                d = d[d.group != "granular"]
            elif subset == "coquina":           # within one structure type (a single lithology)
                d = d[d.group == "coquina"]
            long = pd.concat([d.assign(axis=a, y=flag(d, f"perc_{a}").astype(int)) for a in "zyx"], ignore_index=True)
            long = long[(long.phi > 0) & np.isfinite(long.chi_t)]
            if not len(long):
                continue
            r = {"subset": subset, "edge": int(e), "cases": len(long), "percolating": int(long.y.sum()),
                 "samples": long.id.nunique(), "samples_with_percolating_cases": int(long[long.y == 1].id.nunique()),
                 "samples_with_nonpercolating_cases": int(long[long.y == 0].id.nunique())}
            status = auc_design(long)
            if not status and long.id.nunique() < 4:
                status = "fewer than 4 samples"
            if not status and long.y.sum() < 10:
                status = "fewer than 10 percolating cases"
            for name in models:
                r[f"auc_{name}"] = r[f"auc_{name}_lo"] = r[f"auc_{name}_hi"] = np.nan
            r["boot_valid"] = 0
            if not status:                      # no bootstrap CI for a design that cannot be estimated
                fits = {name: loso_auc(long, feats) for name, feats in models.items()}
                skipped = max(f[2] for f in fits.values())
                if skipped:
                    status = f"{skipped} LOSO fold(s) with a single-class training set"
                else:
                    for name, (a, sc, _) in fits.items():
                        lo, hi, r["boot_valid"] = boot_auc(long, sc)
                        r[f"auc_{name}"], r[f"auc_{name}_lo"], r[f"auc_{name}_hi"] = a, lo, hi
            r["status"] = status or "ok"
            pred_rows.append(r)
    pred = pd.DataFrame(pred_rows)
    pred.to_csv(tdir / "T5_percolation_prediction.csv", index=False)
    for subset in ("all", "rocks", "coquina"):
        q = pred[(pred.subset == subset) & (pred.edge == 150)] if len(pred) else pred
        if len(q):
            best = q.iloc[0]
            for name in ("phi", "chi", "phi+chi"):
                K.add(f"auc_{subset}_{name.replace('+', '_')}_150",
                      [float(best[f"auc_{name}"]), float(best[f"auc_{name}_lo"]), float(best[f"auc_{name}_hi"])],
                      "", f"LOSO AUC (95% sample-bootstrap CI), percolation from {name}, 150-voxel cubes, {subset}")
            K.add(f"auc_cases_{subset}_150", [int(best["cases"]), int(best["percolating"]), int(best["samples"]),
                                              int(best["samples_with_percolating_cases"])], "",
                  f"cases, percolating cases, samples, samples with percolating cases (150-voxel cubes, {subset})")
            K.add(f"auc_status_{subset}_150", str(best["status"]), "",
                  f"'ok', or why the AUC is not estimable (NaN), 150-voxel cubes, {subset}")
    # porosity at which half of the 150-voxel cubes percolate (pooled axes), per group with a percolation transition
    rows = []
    d150 = rev[(rev.edge == 150) & ~rev.overlapping]
    for g in GROUP_ORDER:
        d = d150[d150.group == g]
        long = pd.concat([d.assign(axis=a, y=flag(d, f"perc_{a}").astype(int)) for a in "zyx"], ignore_index=True)
        if not len(long) or auc_design(long):      # both outcomes in at least 3 samples
            continue
        est, lo, hi = phi50(long)
        rows.append({"group": g, "edge": 150, "cases": len(long), "percolating": int(long.y.sum()),
                     "phi50": est, "phi50_lo": lo, "phi50_hi": hi})
        K.add(f"perc_phi50_{g}_150", [100 * est, 100 * lo, 100 * hi], "%",
              f"porosity at which a 150-voxel cube of {g} percolates along an axis with probability 0.5 "
              "(logistic fit, 95 % sample-bootstrap interval)")
    pd.DataFrame(rows).to_csv(tdir / "S_percolation_threshold.csv", index=False)
    bad = pred[pred.status != "ok"] if len(pred) else pred
    for _, r in bad.iterrows():
        notes.append(f"T5: AUC not estimable for subset {r.subset}, edge {r.edge}: {r.status}.")

    # ---- headline counts at every segmentation (main VOIs and all VOIs of the rocks)
    rk_topo = topo[topo.group != "granular"]
    for var in ("T40", "T50", "T60", "T50int", "otsu"):
        q = rk_topo[rk_topo.variant == var]
        if not len(q):
            continue
        qm = q[q.replicate == 0]
        K.add(f"rocks_perc_{var}", [int(qm.perc_any.sum()), int(len(qm)), int(q.perc_any.sum()), int(len(q)),
                                    int(q.groupby("id").perc_any.any().sum())], "",
              f"rocks at {var}: main VOIs percolating, main VOIs, VOIs percolating, VOIs, specimens with a "
              "percolating VOI")
        K.add(f"rocks_perc_ids_{var}", [f"{i} v{int(k)}" for i, k in zip(q.id[q.perc_any], q.replicate[q.perc_any])],
              "", f"rock VOIs percolating at {var}")
        K.add(f"rocks_perc_phi_{var}", {f"{i}_v{int(k)}": float(100 * p) for i, k, p in
                                        zip(q.id[q.perc_any], q.replicate[q.perc_any], q.phi[q.perc_any])}, "%",
              f"porosity of the rock VOIs percolating at {var}")
    for var in ("T40", "T60", "T50int", "otsu"):
        d = topo[topo.variant.isin(["T50", var])].pivot_table(index=["id", "replicate"], columns="variant",
                                                               values="perc_any", aggfunc="first")
        if var in d:
            ok = d[var].notna() & d["T50"].notna()
            ch = d[ok & (d[var].astype(bool) != d["T50"].astype(bool))]
            by = pd.Series([samples[i].group for i in ch.index.get_level_values(0)]).value_counts().to_dict()
            K.add(f"perc_changes_by_group_{var}", {g: int(by.get(g, 0)) for g in GROUP_ORDER}, "",
                  f"VOIs whose percolation differs from ISO-50 at {var}, by group")
    # sign of chi_V and percolation, every VOI at ISO-50
    neg, pc = allv.chi_V < 0, allv.perc_any
    K.add("chi_perc_crosstab", {"neg_perc": int((neg & pc).sum()), "neg_noperc": int((neg & ~pc).sum()),
                                "pos_perc": int((~neg & pc).sum()), "pos_noperc": int((~neg & ~pc).sum())}, "",
          "VOIs (ISO-50) by sign of chi_V and percolation")
    K.add("chi_perc_neg_perc_groups", sorted(set(allv.group[neg & pc])), "", "groups of the VOIs with chi_V < 0 that percolate")
    if "chi_V_26" in allv:
        same = np.sign(allv.chi_V) == np.sign(allv.chi_V_26)
        ratio = (allv.chi_V_26 / allv.chi_V).abs()
        K.add("chi26_sign_agree", [int(same.sum()), len(allv)], "", "VOIs whose chi_V has the same sign with 6- and 26-connected pores")
        K.add("chi26_over_chi6_range", _range(ratio[np.isfinite(ratio) & (allv.chi_V != 0)]), "",
              "|chi_V (26-connected pores)| / |chi_V (6-connected)|, every VOI at ISO-50")
    # isolated fraction of the main VOIs that do not percolate
    for g in GROUP_ORDER:
        d = main[(main.group == g) & ~main.perc_any]
        if len(d):
            K.add(f"iso_nonperc_{g}_range", _range(100 * d.frac_isolated), "%",
                  f"isolated share of the resolved porosity, main VOIs of {g} that do not percolate")
    # threshold span relative to porosity (main VOIs)
    sm = seg[seg.replicate == 0].set_index("id")
    rel = (sm.phi_T60 - sm.phi_T40) / sm.phi_T50
    for g in GROUP_ORDER:
        q = rel[[samples[i].group == g for i in rel.index]]
        K.add(f"thr_span_rel_{g}_range", _range(q), "", f"(phi_T60 - phi_T40) / phi_T50, main VOIs, {g}")
        q = 100 * (sm.phi_T60 - sm.phi_T40)[[samples[i].group == g for i in rel.index]]
        K.add(f"thr_span_main_{g}_range", _range(q), "pp", f"phi_T60 - phi_T40, main VOIs, {g}")
    # porosity at the internal-air reference relative to ISO-50 (main VOIs) and Otsu porosity of the limestones
    if "phi_T50int" in sm:
        r_int = sm.phi_T50int / sm.phi_T50
        for g in GROUP_ORDER:
            q = r_int[[samples[i].group == g for i in r_int.index]]
            if np.isfinite(q).any():
                K.add(f"phi_int_over_t50_{g}_range", _range(q), "", f"phi(T50int) / phi(T50), main VOIs, {g}")
        # trachytes with a helium porosity (plugs of Carvalho et al. 2022): resolved share at both references
        he_tq = lab[(lab.method == "helium") & lab.id.str.startswith("TQ")].set_index("id").porosity_pct
        if len(he_tq):
            K.add("tq_resolved_over_helium_main",
                  {sid: [float(100 * sm.loc[sid].phi_T50 / he), float(100 * sm.loc[sid].phi_T50int / he)]
                   for sid, he in he_tq.items() if sid in sm.index}, "",
                  "resolved porosity of the main VOI / helium porosity at T50 and at T50int, trachytes with a "
                  "helium porosity (Carvalho et al. 2022)")
    K.add("otsu_vn_phi", [float(100 * sm.loc[i, "phi_otsu"]) for i in ("VN01", "VN02", "VN04") if i in sm.index], "%",
          "porosity at Otsu's threshold, main VOIs of VN01, VN02, VN04")
    # first porosity minimum of the radial profile of the bead columns, in nominal bead diameters from the wall
    wp_f = cfg.outdir("wall") / "wall_profiles.csv"
    if wp_f.exists():
        wp = pd.read_csv(wp_f)
        mins = []
        for sid, d in wp.groupby("id"):
            x = d.distance_from_wall_mm.to_numpy(float) / samples[sid].grain_mm
            y = d.phi.rolling(3, center=True, min_periods=1).mean().to_numpy(float)
            sel = (x > 0.3) & (x < 1.0)
            if sel.any():
                mins.append(float(x[sel][np.argmin(y[sel])]))
        if mins:
            K.add("wall_first_min_d_range", _range(mins), "nominal bead diameter",
                  "distance of the first porosity minimum from the tube wall (0.5 for equal spheres)")
    # resolved / helium porosity of the limestones on both bases
    vmain = [100 * main.loc[i].phi for i in vnl.Sample]
    K.add("vn_resolved_over_helium_main", [round(a / b, 4) for a, b in zip(vmain, vnl["helium (%)"])], "",
          "resolved porosity of the main VOI / helium porosity, VN01 VN02 VN04")
    K.add("vn_phi_all_vois_max", float(100 * t50[t50.group == "limestone"].phi.max()), "%",
          "largest resolved porosity of any limestone VOI")
    # Archie exponent from the spanning porosity (main VOIs, whole VOI)
    if {"phi_span", "F"} <= set(perc.columns):
        perc["m_span"] = -np.log(perc.F.astype(float)) / np.log(perc.phi_span.astype(float))
        for g in GROUP_ORDER:
            d = perc[perc.group == g]
            if len(d):
                K.add(f"m_span_{g}_range", _range(d.m_span), "", f"Archie exponent with the spanning porosity, {g}")
    if "r_KT_dom" in lc:
        r = lc.loc[lc.used, "r_KT_dom"].to_numpy(float)
        if np.isfinite(r).any():
            K.add("kt_inv_prefactor_images_range", _range(226 * r), "",
                  "1/c with k_LBM = c d_c^2 / F on the image domains (Katz-Thompson: 226)")
    # Lambda sweep (rocks)
    lam_f = cfg.outdir("lbm") / "lbm_lambda.csv"
    if lam_f.exists():
        lam = pd.read_csv(lam_f)
        lam.to_csv(tdir / "S_lbm_lambda.csv", index=False)
        for (sid, a), d in lam.groupby(["id", "axis"]):
            ref = d[np.isclose(d.Lambda, 3 / 16)].k_lbm_mD
            if len(ref):
                K.add(f"lbm_lambda_{sid}_{a}", [float(d.Lambda.min()), float(d.Lambda.max()),
                                               float((d.k_lbm_mD / ref.iloc[0]).min()),
                                               float((d.k_lbm_mD / ref.iloc[0]).max()), bool(flag(d, "ok").all())], "",
                      f"{sid} {a}: Lambda range, k_LBM / k_LBM(3/16) min and max, all runs ok")
    # LBM at the internal-air reference (rocks)
    li_f = cfg.outdir("lbm") / "lbm_T50int.csv"
    if li_f.exists():
        li = pd.read_csv(li_f)
        ei_f = cfg.outdir("lbm") / "estimates_T50int.csv"
        ei = pd.read_csv(ei_f).set_index(["id", "axis"]) if ei_f.exists() else pd.DataFrame()
        li.to_csv(tdir / "S_lbm_T50int.csv", index=False)
        for _, rr in li.iterrows():
            ka = (rr.id, rr.axis)
            base_row = lc[(lc.id == rr.id) & (lc.axis == rr.axis) & lc.used] if len(lc) else pd.DataFrame()
            if len(base_row):
                K.add(f"lbm_T50int_over_T50_{rr.id}_{rr.axis}", float(rr.k_lbm_mD / base_row.k_lbm_mD.iloc[0]), "",
                      f"{rr.id} {rr.axis}: k_LBM at T50int / at T50 (same domain rule)")
            dc = float(ei.at[ka, "d_c_vox"]) if len(ei) and ka in ei.index else np.nan
            K.add(f"lbm_T50int_{rr.id}_{rr.axis}",
                  [float(rr.k_lbm_mD) / 1000, float(rr.k_pnm_mD / rr.k_lbm_mD), float(rr.k_pnmeq_mD / rr.k_lbm_mD),
                   dc, float(rr.F), bool(flag(pd.DataFrame([rr]), "ok").iloc[0]), str(rr.domain), float(rr.phi_span)],
                  "", f"{rr.id} {rr.axis} at T50int: k_LBM (D), k_PNM/k_LBM inscribed and equivalent, d_c (voxels), F, "
                  "LBM ok, domain, spanning porosity of the domain")
    # forced whole-VOI LBM runs (e.g. GT11 at the native voxel)
    lf_f = cfg.outdir("lbm") / "lbm_full.csv"
    if lf_f.exists():
        lf = pd.read_csv(lf_f)
        ef_f = cfg.outdir("lbm") / "estimates_full.csv"
        if ef_f.exists():
            ef = pd.read_csv(ef_f)
            lf = lf.merge(ef[["id", "axis", "k_KC_mD", "k_KC_span_mD", "k_KT_mD", "d_c_vox"]], on=["id", "axis"], how="left")
            for _, rr in lf.iterrows():
                K.add(f"lbm_full_est_{rr.id}_{rr.axis}",
                      [float(rr.k_KC_mD / rr.k_lbm_mD), float(rr.k_KC_span_mD / rr.k_lbm_mD),
                       float(rr.k_KT_mD / rr.k_lbm_mD), float(rr.d_c_vox)], "",
                      f"{rr.id} {rr.axis}, whole main VOI at the native voxel: k_KC/k_LBM, k_KC(spanning)/k_LBM, "
                      "k_KT/k_LBM, d_c (voxels)")
                if rr.k_KT_mD > 0 and np.isfinite(rr.k_KT_mD):
                    K.add(f"kt_inv_prefactor_{rr.id}_{rr.axis}_voi", float(226 * rr.k_KT_mD / rr.k_lbm_mD), "",
                          f"{rr.id} {rr.axis}, whole main VOI: inverse Katz-Thompson prefactor that would match k_LBM "
                          "(226 k_KT / k_LBM)")
        lf.to_csv(tdir / "S_lbm_full_voi.csv", index=False)
        for _, rr in lf.iterrows():
            crop = lc[(lc.id == rr.id) & (lc.axis == rr.axis)] if len(lc) else pd.DataFrame()
            kc = float(crop.k_lbm_mD.iloc[0]) if len(crop) else np.nan
            K.add(f"lbm_full_{rr.id}_{rr.axis}",
                  [float(rr.k_lbm_mD) / 1000, float(rr.k_pnm_mD / rr.k_lbm_mD), float(rr.k_pnmeq_mD / rr.k_lbm_mD),
                   float(rr.F), bool(flag(pd.DataFrame([rr]), "ok").iloc[0]), float(rr.k_lbm_mD) / kc,
                   float(rr.phi_span), int(rr.iterations)], "",
                  f"{rr.id} {rr.axis}, whole main VOI at the native voxel: k_LBM (D), k_PNM/k_LBM inscribed and "
                  "equivalent, F, LBM ok, k_LBM(whole VOI)/k_LBM(central sub-volume), spanning porosity, iterations")

    # ---- prediction statistics (stage predict)
    pdir = cfg.outdir("predict")
    pa_f = pdir / "predict_auc.csv"
    if pa_f.exists():
        pa = pd.read_csv(pa_f)
        pa.to_csv(tdir / "S_predict_auc.csv", index=False)
        # chi_main: chi_V times the cube of the specimen's main-VOI chord, a constant within a specimen, so the
        # within-specimen measures depend on chi_V alone; chi_cube (per-cube chord, which carries the porosity of
        # the cube) is kept as a sensitivity check
        labels = {"phi": "phi", "chi_main": "chi", "phi+chi_main": "both", "chi_cube": "chicube",
                  "phi+chi_cube": "bothcube", "chi_raw": "chiraw", "phi+chi_raw": "bothraw"}
        K.add("pred_chi_feature", "chi_main", "", "Euler-density feature of the prediction models")
        for _, r in pa.iterrows():
            lab_m = labels.get(r.model)
            if lab_m is None:
                continue
            base = f"pred_{r.variant}_{r.subset}_{int(r.edge)}"
            if lab_m == "phi":
                K.add(f"{base}_n", [int(r.n_cases), int(r.n_pos), int(r.n_specimens), int(r.n_specimens_both),
                                    int(r.n_empty)], "", f"{base}: cases, percolating, specimens, specimens with both "
                      "outcomes, empty cubes")
            if r.status != "ok":
                continue
            K.add(f"{base}_{lab_m}", [float(r[c]) for c in ("auc_pooled", "null_lo", "null_hi", "auc_within",
                                                          "auc_within_lo", "auc_within_hi", "d_auc_within",
                                                          "d_auc_within_lo", "d_auc_within_hi", "auc_within_expected")],
                  "", f"{base} model {r.model}: pooled LOSO AUC, its permutation null (2.5, 97.5 %), within-specimen "
                  "AUC with 95 % interval, change of within AUC vs phi alone with 95 % interval, within AUC of the "
                  "raw feature in the expected direction")
        ok = pa[pa.status == "ok"]
        for (v, sub, mod), d in ok.groupby(["variant", "subset", "model"]):
            lab_m = labels.get(mod)
            if lab_m is None:
                continue
            K.add(f"predrange_{v}_{sub}_{lab_m}", [*_range(d.auc_within), *_range(d.auc_pooled),
                                                   *_range(d.null_hi)], "",
                  f"{v} {sub} {mod}, over cube sizes: within-specimen AUC range, pooled AUC range, range of the "
                  "upper end of the pooled null")
            if d.d_auc_within.notna().any():
                K.add(f"predrange_{v}_{sub}_{lab_m}_dwithin", _range(d.d_auc_within), "",
                      f"{v} {sub} {mod}: change of within-specimen AUC vs phi alone, over cube sizes")
    pc_f = pdir / "predict_conditional.csv"
    if pc_f.exists():
        pcd = pd.read_csv(pc_f)
        pcd.to_csv(tdir / "S_predict_conditional.csv", index=False)
        prim = pcd[(pcd.chi_feature == "chi_main") & (pcd.status == "ok")]
        for _, r in prim.iterrows():
            K.add(f"predcond_{r.variant}_{r.subset}_{int(r.edge)}",
                  [float(r.beta_chi), float(r.beta_chi_lo), float(r.beta_chi_hi), float(r.share_chi_pos),
                   float(r.get("p_perm_cond", np.nan)), float(r.get("p_boot", np.nan))],
                  "", f"{r.variant} {r.subset} {int(r.edge)} voxels: coefficient of the Euler density given a flexible "
                  "porosity basis and specimen (per within-specimen SD), 95 % cluster-bootstrap interval, share of "
                  "resamples > 0, Freedman-Lane permutation p, two-sided bootstrap p")
        for (v, sub), d in prim.groupby(["variant", "subset"]):
            sig = d[d.beta_chi_hi < 0]
            K.add(f"predcond_sig_{v}_{sub}", [int(e) for e in sig.edge], "voxel",
                  f"{v} {sub}: cube sizes whose Euler-density coefficient interval lies below zero")
    p50_f = pdir / "predict_phi50.csv"
    if p50_f.exists():
        p50 = pd.read_csv(p50_f)
        p50.to_csv(tdir / "S_predict_phi50.csv", index=False)
        for (v, g), d in p50[p50.status == "ok"].groupby(["variant", "group"]):
            if "cal_max_abs_dev" in d:
                K.add(f"phi50_caldev_{v}_{g}_range", _range(d.cal_max_abs_dev), "",
                      f"largest |observed - fitted| share over calibration bins, logistic in phi, {g} ({v}), over cube sizes")
        for _, r in p50.iterrows():
            if np.isfinite(r.phi50):
                K.add(f"phi50_{r.variant}_{r.group}_{int(r.edge)}", [100 * r.phi50, 100 * r.phi50_lo, 100 * r.phi50_hi,
                                                                   float(r.edge_mm),
                                                                   bool(flag(pd.DataFrame([r]), "phi50_extrapolated").iloc[0]),
                                                                   float(r.get("cal_max_abs_dev", np.nan))], "%",
                      f"porosity at which a cube of {r.group} percolates along an axis with probability 0.5, "
                      f"{int(r.edge)} voxels ({r.variant}), 95 % interval, edge in mm, extrapolated flag, largest "
                      f"|observed - fitted| over calibration bins (status: {r.status})")

    # ---- imaging diagnostics (stage imaging)
    im_f = cfg.outdir("imaging") / "imaging_summary.json"
    if im_f.exists():
        S = read_json(im_f)
        for key, d in S.get("esf", {}).items():
            g, flt = key.rsplit("_", 1)
            st = d.get("sigma_cal_vox", {})
            if st.get("n", 0):
                K.add(f"esf_{g}_{flt}", [st["median"], st["min"], st["max"], d["n_fitted"], d["n_vois"],
                                         d["sigma_cal_um"]["median"]], "",
                      f"edge-spread sigma (calibrated, voxels) median, min, max; VOIs fitted, VOIs; median in um; "
                      f"{g}, filter {flt}")
                K.add(f"esf_{g}_{flt}_tails", [d["tail_pore_frac"].get("median", np.nan),
                                               d["tail_solid_frac"].get("median", np.nan),
                                               d["d0_vox"].get("median", np.nan)], "",
                      f"median slow tail on the pore and solid sides (fraction of contrast) and median offset d0 "
                      f"(voxels) of the local mid-grey from the T50 interface, {g}, filter {flt}")
                for q in ("d0_vox", "sigma_cal_um", "edge_10_90_cal_um", "noise_frac"):
                    s2 = d.get(q, {})
                    K.add(f"esf_{g}_{flt}_{q}", [s2.get("median", np.nan), s2.get("min", np.nan), s2.get("max", np.nan)],
                          "", f"{q} median, min, max over the fitted VOIs, {g}, filter {flt}")
        for key, d in S.get("filter", {}).items():
            st = d.get("phi_rel_change", {})
            K.add(f"filter_{key}", [d["n_vois"], d["n_perc_any_changed"], st.get("median", np.nan), st.get("min", np.nan),
                                    st.get("max", np.nan), d["chi_V_ratio"].get("median", np.nan),
                                    d["d_c_ratio"].get("median", np.nan)], "",
                  f"unfiltered vs median-filtered segmentation, {key}: VOIs, VOIs whose percolation changes, relative "
                  "porosity change median/min/max, median chi_V ratio, median d_c ratio")
            K.add(f"filter_{key}_ids", d["perc_changed_ids"], "", f"VOIs whose percolation changes without the filter, {key}")
        for sid, d in S.get("two_glass", {}).items():
            m = d["phi_mean"]
            K.add(f"two_glass_{sid}", [100 * m["phi_a_T50"], 100 * m["phi_b_T50_dense"], 100 * m["phi_c_beadwise"]], "%",
                  f"{sid} porosity (mean of VOIs): T50 on the light glass, ISO-50 of the dense glass, bead-wise local ISO-50")
        sm = S.get("solid_modes", {})
        K.add("solid_secondary_modes", {sid: [float(x) for x in d["contrast_frac"]] for sid, d in sm.items()}, "",
              "secondary solid modes of the bead packs, as fraction of the air-solid contrast")
        K.add("solid_secondary_beads", {sid: [int(x) for x in d["n_beads_this_mode"]] for sid, d in sm.items()}, "",
              "beads whose core grey lies on the side of the secondary mode, per bead pack")
        dep = S.get("enclosed_air_depth", {})
        for key, st in dep.items():
            g, src, dd = [x.strip() for x in key.split("|")]
            dd = dd.replace("depth ", "")
            if st.get("n", 0):
                K.add(f"encl_{g}_{src}_d{dd}".replace(" ", "_").replace("-", "_"), [st["median"], st["min"], st["max"], st["n"]], "",
                      f"air offset (fraction of contrast) at depth {dd} voxels, {g}, {src}, reference "
                      f"{st.get('air_reference', '?')}: median, min, max, n")
        for key, st in S.get("enclosed_air_difference", {}).items():
            g, kind, dd = [x.strip() for x in key.split("|")]
            dd = dd.replace("depth ", "")
            if st.get("n", 0):
                K.add(f"encldiff_{g}_d{dd}", [st["median"], st["min"], st["max"], st["n"]], "",
                      f"grey of enclosed minus open pore air at depth {dd} voxels (fraction of contrast), {g} ({kind}), "
                      f"reference {st.get('air_reference', '?')}: median, min, max, n (depth >= 3 sigma only)")
        es = cfg.outdir("imaging") / "enclosed_air_summary.csv"
        if es.exists():
            ea = pd.read_csv(es)
            ea = ea[ea.d_bin.astype(str) == "all"]
            K.add("encl_bubble_excess", {r.id: [int(r.n_bubbles), float(r.offset_deep_median),
                                                float(r.offset_deep_excess_median), float(r.r_in_median_vox)]
                                         for r in ea.itertuples()}, "",
                  "bubbles inside single beads with inscribed radius >= 3.5 vox, per bead pack: number, median offset of "
                  "their deepest voxels above the pore-air mode, median excess over the blur-only offset of synthetic "
                  "bubbles of the same size, median inscribed radius (voxels)")
        smf = cfg.outdir("imaging") / "solid_modes.csv"
        if smf.exists():
            sm2 = pd.read_csv(smf)
            sm2 = sm2[sm2.role == "solid_secondary"]
            K.add("solid_beads_total", {r.id: int(r.n_beads) for r in sm2.itertuples()}, "",
                  "beads (all VOIs) classified by core grey in the bead packs with a secondary solid mode")
        bub = S.get("enclosed_air_bubbles", {})
        if bub:
            K.add("encl_bubbles", {sid: [int(d["n_bubbles"]), float(d["offset_deep_median"]), float(d["d_eq_median_mm"])]
                                   for sid, d in bub.items()}, "",
                  "bubbles in the glass per bead pack: number, median offset of their deepest voxels, median diameter (mm)")
        for shape, d in S.get("synthetic", {}).items():
            K.add(f"syn_beads_{shape}", [float(d["aspect_true"]), int(d["n_beads"]), float(d["d_eq_ratio_median"]),
                                         float(d["aspect_median"]), float(d["aspect_p90"]), float(d["frac_aspect_gt_1_2"])],
                  "", f"synthetic pack ({shape}): true axis ratio, beads measured, median d_eq / true, median and P90 "
                  "measured axis ratio, share with ratio > 1.2")
        fp = S.get("footprint", {})
        if fp:
            K.add("footprint_pred_minus_measured", _range([100 * d["pred_minus_measured"] for d in fp.values()]), "pp",
                  "porosity of the VOI footprint from the radial profile minus the measured VOI porosity, bead packs")
    syn_f = cfg.outdir("imaging") / "synthetic_beads.csv"
    if syn_f.exists():
        sy = pd.read_csv(syn_f)
        sy = sy[sy.seed.astype(str) != "all"]
        if "phi_segmented" in sy and len(sy):
            K.add("syn_beads_phi_err_max_pp", float(100 * (sy.phi_segmented - sy.phi_true).abs().max()), "pp",
                  "largest |segmented - true| porosity of the synthetic bead packs (ISO-50 after blur, noise and median filter)")

    # ---- porosity budget against the laboratory porosity (stage budget)
    bud_f = cfg.outdir("budget") / "porosity_budget.csv"
    if bud_f.exists():
        bu = pd.read_csv(bud_f, dtype={"perc_main_at_lab": str})
        bu.to_csv(tdir / "S_porosity_budget.csv", index=False)
        for r in bu[np.isfinite(bu.phi_lab)].itertuples():
            K.add(f"budget_{r.id}", [100 * r.phi_lab, 100 * r.phi_T50, 100 * r.phi_T50int, 100 * r.phi_mb,
                                     100 * r.phi_mb_int, r.f_lab, 100 * r.phi_main_at_lab,
                                     100 * r.frac_connected_main_at_lab], "",
                  f"{r.id}: laboratory porosity (%), pooled image porosity at T50 and T50int (%), mass-balance porosity with "
                  "outside and internal air (%), threshold position f_lab that reproduces the laboratory porosity, "
                  "main-VOI porosity at that threshold (%) and its connected share (%)")
            if isinstance(r.perc_main_at_lab, str) and r.perc_main_at_lab:
                K.add(f"budget_perc_{r.id}", r.perc_main_at_lab.zfill(3), "",
                      f"{r.id}: percolation of the main VOI along z, y, x at the threshold that reproduces the laboratory porosity")
        for g in GROUP_ORDER:
            d = bu[(bu.group == g) & np.isfinite(bu.phi_lab)]
            if len(d):
                K.add(f"budget_f_lab_{g}_range", _range(d.f_lab), "",
                      f"threshold position f_lab that reproduces the laboratory porosity, {g}")
        beads = bu[(bu.group == "granular") & (bu.id != "CF10")]     # CF10: two glasses, the mass balance is not two-phase
        if len(beads):
            K.add("budget_mb_minus_t50_granular_range", _range(100 * (beads.phi_mb - beads.phi_T50)), "pp",
                  "mass-balance minus ISO-50 porosity (pooled VOIs), single-glass bead packs")
        for g in ("limestone", "coquina", "trachyte"):
            d = bu[bu.group == g]
            if len(d):
                K.add(f"budget_mb_over_t50_{g}_range", _range(d.phi_mb / d.phi_T50), "",
                      f"mass-balance / ISO-50 porosity (pooled VOIs, outside air), {g}")
                K.add(f"budget_mbint_over_t50_{g}_range", _range(d.phi_mb_int / d.phi_T50), "",
                      f"mass-balance / ISO-50 porosity (pooled VOIs, internal air), {g}")
    for _, r in meta[flag(meta, "window_fallback_no_margin")].iterrows() if "window_fallback_no_margin" in meta else []:
        end = (r.n_slices_stack - (r.slice_index_start + voi_edge)) * r.voxel_um / 1000
        K.add(f"voi_fallback_end_mm_{r.id}", float(end), "mm", f"{r.id} VOI {int(r.replicate)}: distance from the VOI to the end of the stack")

    # ---- write
    write_json(out / "key_numbers.json", K.d)
    digest = ["# Results digest (generated)", ""]
    for k, v in K.d.items():
        digest.append(f"- **{k}** = {v['value']} {v['unit']} - {v['description']}")
    if notes:
        digest += ["", "## Notes on the tables", ""] + [f"- {n}" for n in notes]
    (out / "RESULTS_DIGEST.md").write_text("\n".join(digest), encoding="utf-8")
    print(f"  report: {len(K.d)} key numbers, tables in {tdir}")
