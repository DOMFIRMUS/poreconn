"""Stage `figures`: figures of the article and of the supplementary material (PDF + 600-dpi PNG).

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Colour encodes the material group and is always paired with a marker shape (secondary encoding):
blue circles = glass-bead packs, aqua squares = reef limestones, orange triangles = coquinas,
dark grey diamonds = trachytes. The three hues are the first slots of a colour-blind-validated
categorical palette; the fourth group uses a neutral so that all pairs stay separable. Filled vs open
markers carry one binary attribute, named in each legend (percolates or not; mirror or wall faces).
Widths follow Springer's 84 mm (single column) and 174 mm (double column).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tifffile
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .config import Config, read_json  # noqa: E402
from .extract import voi_paths  # noqa: E402
from .morph import spanning_mask  # noqa: E402
from .report import PNM_TAGS, flag, pore_chord  # noqa: E402
from .segment import seg_paths  # noqa: E402

MM = 1 / 25.4
COL = {"granular": "#2a78d6", "limestone": "#1baf7a", "coquina": "#eb6834", "trachyte": "#52514e"}
MRK = {"granular": "o", "limestone": "s", "coquina": "^", "trachyte": "D"}
LAB = {"granular": "Glass-bead packs", "limestone": "Reef limestones", "coquina": "Coquinas", "trachyte": "Trachytes"}
ORDER = ["granular", "limestone", "coquina", "trachyte"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdcd8"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": False, "lines.linewidth": 1.4,
    "savefig.dpi": 600, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})


def _grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def _legend_groups(ax, groups=ORDER, loc="best", **kw):
    h = [Line2D([], [], ls="", marker=MRK[g], ms=5, mfc=COL[g], mec="white", mew=0.6, label=LAB[g]) for g in groups]
    return ax.legend(handles=h, loc=loc, frameon=False, handletextpad=0.3, borderaxespad=0.2, **kw)


def _plain_log(ax, which="y"):
    """Log axis labelled with plain numbers on a 1-1.5-2-3-5-7 grid (for ranges of about one decade)."""
    lo, hi = ax.get_ylim() if which == "y" else ax.get_xlim()
    t = [m * 10.0 ** e for e in range(-4, 5) for m in (1, 1.5, 2, 3, 5, 7)]
    t = [v for v in t if lo <= v <= hi]
    a = ax.yaxis if which == "y" else ax.xaxis
    if 2 <= len(t) <= 8:
        a.set_major_locator(matplotlib.ticker.FixedLocator(t))
        a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a.set_minor_formatter(matplotlib.ticker.NullFormatter())


def _save(fig, d, name):
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(d / f"{name}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def _halo(color):
    """Outline that keeps white labels readable on bright image regions."""
    return [pe.withStroke(linewidth=1.6, foreground="black")] if color == "white" else []


def _scalebar(ax, n_px, dx_mm, length_mm, color="white"):
    L = length_mm / dx_mm
    y = n_px * 0.94
    ax.plot([n_px * 0.05, n_px * 0.05 + L], [y, y], color=color, lw=2.2, solid_capstyle="butt",
            path_effects=[pe.Stroke(linewidth=3.4, foreground="black"), pe.Normal()] if color == "white" else None)
    ax.text(n_px * 0.05 + L / 2, y - n_px * 0.03, f"{length_mm:g} mm", color=color, ha="center", va="bottom",
            fontsize=6.5, path_effects=_halo(color))


# ------------------------------------------------------------------ Fig. 1: images and segmentation
def fig_overview(cfg, d, reps=("CF5", "VN02", "GT01", "TQ06")):
    fig, axs = plt.subplots(2, 4, figsize=(174 * MM, 92 * MM))
    for j, sid in enumerate(reps):
        s = cfg.sample(sid)
        grey = tifffile.imread(voi_paths(cfg, sid, 0)[0])
        pore = tifffile.imread(seg_paths(cfg, sid, 0)["pore"]).astype(bool)
        z = grey.shape[0] // 2
        g = grey[z].astype(float)
        lo, hi = np.percentile(g, [0.5, 99.5])
        ax = axs[0, j]
        ax.imshow(g, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"{sid} - {LAB[s.group][:-1] if LAB[s.group].endswith('s') else LAB[s.group]}", color=INK)
        _scalebar(ax, g.shape[0], s.voxel_mm, 2 if s.voxel_mm < 0.045 else 4)
        # segmentation: pore voxels of clusters spanning any axis vs the others
        span = spanning_mask(pore, 0) | spanning_mask(pore, 1) | spanning_mask(pore, 2)
        rgb = np.dstack([np.full(g.shape, 0.93)] * 3)
        rgb[pore[z]] = matplotlib.colors.to_rgb("#eb6834")
        rgb[span[z]] = matplotlib.colors.to_rgb("#2a78d6")
        ax = axs[1, j]
        ax.imshow(rgb, interpolation="nearest")
        _scalebar(ax, g.shape[0], s.voxel_mm, 2 if s.voxel_mm < 0.045 else 4, color=INK)
        for a in axs[:, j]:
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
    h = [matplotlib.patches.Patch(color="#2a78d6", label="pore, cluster spanning the VOI"),
         matplotlib.patches.Patch(color="#eb6834", label="pore, non-spanning cluster"),
         matplotlib.patches.Patch(color="#ededed", label="solid")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    for i, lab in enumerate("abcdefgh"):
        c = "white" if i < 4 else INK
        axs.flat[i].text(0.02, 0.98, f"({lab})", transform=axs.flat[i].transAxes, va="top", ha="left",
                         color=c, fontsize=8, fontweight="bold", path_effects=_halo(c))
    fig.subplots_adjust(wspace=0.04, hspace=0.12)
    _save(fig, d, "fig01_overview")


# ------------------------------------------------------------------ Fig. 2: histograms and thresholds
def fig_histograms(cfg, d, reps=("CF10", "VN02", "GT01", "TQ06")):
    fig, axs = plt.subplots(1, 4, figsize=(174 * MM, 52 * MM))
    lines = (("I_air", ":", INK, "air and solid references"), ("I_solid", ":", INK, None),
             ("T50", "-", INK, "$T_{50}$ (ISO-50)"), ("T50int", "-.", INK2, "$T_{50}^{int}$ (internal air)"),
             ("T_otsu", "--", INK2, "Otsu"))
    for i, (ax, sid) in enumerate(zip(axs, reps)):
        s = cfg.sample(sid)
        h = np.load(seg_paths(cfg, sid, 0)["pooled"])
        sg = read_json(seg_paths(cfg, sid, 0)["json"])
        c, n = h["centres"], h["hist"].astype(float)
        ax.fill_between(c, n / n.max(), color=COL[s.group], alpha=0.35, lw=0)
        ax.plot(c, n / n.max(), color=COL[s.group], lw=1.0)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.6)
        for key, ls, col, _ in lines:
            v = sg.get(key)
            if v is not None and np.isfinite(v):
                ax.axvline(v, color=col, lw=0.9, ls=ls)
        ax.axvspan(sg["T40"], sg["T60"], color="#c3c2b7", alpha=0.45, lw=0)
        short = {"granular": "bead pack", "limestone": "limestone", "coquina": "coquina", "trachyte": "trachyte"}
        ax.set_title(f"$\\bf{{({'abcd'[i]})}}$ {sid}, {short[s.group]}", color=INK)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.set_xlabel("Filtered grey value")
        if ax is axs[0]:
            ax.set_ylabel("Relative frequency")
    h = [Line2D([], [], color=col, lw=0.9, ls=ls, label=lab) for _, ls, col, lab in lines if lab]
    h.append(matplotlib.patches.Patch(color="#c3c2b7", alpha=0.45, label="$T_{40}$–$T_{60}$"))
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False, fontsize=6.5)
    fig.tight_layout()
    _save(fig, d, "fig02_histograms")


# ------------------------------------------------------------------ Fig. 3: porosity vs connectivity / topology
def fig_connectivity(cfg, d):
    topo = pd.read_csv(cfg.outdir("topology") / "topology.csv")
    t = topo[topo.variant == "T50"].copy()
    t["perc_any"] = flag(t, "perc_any")
    t["chi_l3"] = t.chi_V * pore_chord(t.phi, t.S_V) ** 3      # l_p = 4 phi / S_V of each VOI
    fig, axs = plt.subplots(1, 2, figsize=(174 * MM, 72 * MM))
    for g in ORDER:
        x = t[t.group == g]
        for perc, mfc, z in ((False, "white", 3), (True, COL[g], 4)):     # percolating VOIs drawn on top
            y = x[x.perc_any == perc]
            axs[0].scatter(100 * y.phi, 100 * y.frac_connected, marker=MRK[g], s=26, facecolor=mfc,
                           edgecolor=COL[g], linewidth=1.0, zorder=z)
            axs[1].scatter(100 * y.phi, y.chi_l3, marker=MRK[g], s=26, facecolor=mfc, edgecolor=COL[g],
                           linewidth=1.0, zorder=z)
    axs[0].set_xlabel("Resolved porosity $\\phi$ (%)")
    axs[0].set_ylabel("Pore volume in spanning clusters (%)")
    axs[1].set_xlabel("Resolved porosity $\\phi$ (%)")
    axs[1].set_ylabel("$\\chi_V\\,\\ell_p^3$ (Euler number per $\\ell_p^3$)")
    axs[1].set_yscale("symlog", linthresh=0.01)
    axs[1].axhline(0, color=INK2, lw=0.7)
    for ax in axs:
        ax.set_xscale("log")
        _grid(ax)
    _legend_groups(axs[0], loc="center left")
    h = [Line2D([], [], ls="", marker="o", ms=5, mfc=INK2, mec=INK2, label="percolates (any axis)"),
         Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=INK2, label="does not percolate")]
    axs[1].legend(handles=h, loc="upper left", frameon=False)
    for i, ax in enumerate(axs):
        ax.text(-0.14, 1.02, f"({'ab'[i]})", transform=ax.transAxes, fontsize=8, fontweight="bold")
    fig.tight_layout()
    _save(fig, d, "fig03_connectivity")


# ------------------------------------------------------------------ Fig. 4: percolation probability
def fig_percolation(cfg, d, edge=150, variant="T50"):
    """(a) coquinas: share of percolating cases (cube and axis) against porosity for every cube size, with the
    logistic fit and phi50 (stage predict); (b) every 150-voxel rock cube in the porosity - Euler plane, the
    Euler density normalised by the pore chord of the cube; (c) within-specimen AUC of the leave-one-specimen-
    out models against the cube edge, coquinas and all rocks (its null is 0.5)."""
    pdir = cfg.outdir("predict")
    cal = pd.read_csv(pdir / "predict_calibration.csv")
    p50 = pd.read_csv(pdir / "predict_phi50.csv")
    pa = pd.read_csv(pdir / "predict_auc.csv")
    rev = pd.read_csv(cfg.outdir("rev") / "rev.csv")
    if "variant" in rev:
        rev = rev[rev.variant == variant]
    fig, axs = plt.subplots(1, 3, figsize=(174 * MM, 58 * MM))
    # (a)
    ax = axs[0]
    edges = sorted(e for e in cal[(cal.variant == variant) & (cal.group == "coquina")].edge.unique() if e <= edge)
    shades = dict(zip(edges, ["#f6b48f", "#f08a55", "#eb6834", "#a8431c"][-len(edges):]))
    for e in edges:
        c = cal[(cal.variant == variant) & (cal.group == "coquina") & (cal.edge == e)]
        mm = float(c.edge_mm.iloc[0])
        ax.plot(100 * c.phi_mean, c.observed, ls="", marker=MRK["coquina"], ms=3.8, color=shades[e], mec="white",
                mew=0.4, zorder=3)
        f = p50[(p50.variant == variant) & (p50.group == "coquina") & (p50.edge == e)]
        if len(f) and np.isfinite(f.slope.iloc[0]):
            x = np.linspace(0, max(0.3, c.phi_mean.max()), 200)
            ax.plot(100 * x, 1 / (1 + np.exp(-(f.intercept.iloc[0] + f.slope.iloc[0] * x))), color=shades[e], lw=1.0,
                    label=f"{mm:g} mm")
            r = f.iloc[0]
            if np.isfinite(r.phi50):
                ax.errorbar(100 * r.phi50, 0.5, xerr=[[100 * (r.phi50 - r.phi50_lo)], [100 * (r.phi50_hi - r.phi50)]],
                            fmt="|", ms=6, mew=1.1, color=shades[e], elinewidth=0.8, capsize=1.5, zorder=4)
    ax.axhline(0.5, color=GRID, lw=0.6, zorder=0)
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Porosity of the cube (%)")
    ax.set_ylabel("Share percolating (coquinas)")
    ax.legend(frameon=False, loc="lower right", title="cube edge", title_fontsize=6.5, fontsize=6.5)
    _grid(ax)
    # (b)
    ax = axs[1]
    r = rev[(rev.edge == edge) & ~flag(rev, "overlapping") & (rev.group != "granular")].copy()
    topo = pd.read_csv(cfg.outdir("topology") / "topology.csv")
    m = topo[(topo.variant == "T50") & (topo.replicate == 0)].set_index("id")
    r["chi_l3"] = r.chi_V * r.id.map(pd.Series(pore_chord(m.phi, m.S_V), index=m.index)) ** 3
    ylab = "$\\chi_V\\,\\ell_p^3$ of the cube"
    r = r[r.phi > 0]
    for g in ORDER[1:]:
        x = r[r.group == g]
        pa_ = flag(x, "perc_any").to_numpy()
        for sel, mfc, z in ((~pa_, "white", 2), (pa_, COL[g], 3)):
            ax.scatter(100 * x.phi[sel], x.chi_l3[sel], marker=MRK[g], s=9, facecolor=mfc, edgecolor=COL[g],
                       linewidth=0.6, zorder=z)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xlabel("Porosity of the cube (%)")
    ax.set_ylabel(ylab)
    _grid(ax)
    h = [Line2D([], [], ls="", marker="o", ms=4, mfc=INK2, mec=INK2, label="percolates"),
         Line2D([], [], ls="", marker="o", ms=4, mfc="white", mec=INK2, label="does not")]
    ax.legend(handles=h, loc="upper left", frameon=False)
    # (c)
    ax = axs[2]
    chi = "chi_main"
    models = [("phi", "$\\phi$", "#c3c2b7"), (chi, "$\\chi_V$", "#2a78d6"), (f"phi+{chi}", "both", INK)]
    for sub, ls, mk, off in (("coquina", "-", "^", -0.06), ("rocks", "--", "s", 0.06)):
        for i, (mname, lab, c) in enumerate(models):
            q = pa[(pa.variant == variant) & (pa.subset == sub) & (pa.model == mname) & (pa.status == "ok")
                   & (pa.edge <= edge)].sort_values("edge")
            if not len(q):
                continue
            x = np.log2(q.edge.to_numpy(float)) + off + (i - 1) * 0.03
            ax.vlines(x, q.auc_within_lo, q.auc_within_hi, color=c, lw=0.7, zorder=2)
            ax.plot(x, q.auc_within, ls=ls, marker=mk, ms=3.6, color=c, mfc=c if sub == "coquina" else "white",
                    mec=c, lw=0.9, zorder=3)
    ax.axhline(0.5, color=INK2, lw=0.6, ls=":")
    es = sorted(e for e in pa.edge.unique() if e <= edge)
    ax.set_xticks(np.log2(es))
    ax.set_xticklabels([str(int(e)) for e in es])
    ax.set_xlabel("Cube edge (voxels)")
    ax.set_ylabel("Within-specimen AUC")
    ax.set_ylim(0.3, 1.0)
    _grid(ax, "y")
    h = [Line2D([], [], color=c, lw=1.2, label=lab) for _, lab, c in models]
    h += [Line2D([], [], color=INK2, ls="-", marker="^", ms=3.6, label="coquinas"),
          Line2D([], [], color=INK2, ls="--", marker="s", ms=3.6, mfc="white", label="all rocks")]
    ax.legend(handles=h, frameon=False, loc="lower left", fontsize=6, ncol=2, columnspacing=0.8, handlelength=1.6)
    for i, a in enumerate(axs):
        a.text(-0.24, 1.04, f"({'abc'[i]})", transform=a.transAxes, fontsize=8, fontweight="bold")
    fig.tight_layout()
    _save(fig, d, "fig04_percolation")


# ------------------------------------------------------------------ Fig. 5: coarsening
def fig_coarsening(cfg, d):
    """Top: resolved porosity relative to the native voxel; bottom: share of the pore volume in spanning
    clusters; one line per specimen (main VOI), filled markers where it percolates along some axis."""
    cs = pd.read_csv(cfg.outdir("coarsen") / "coarsen.csv")
    if "replicate" in cs:                                   # main VOIs at ISO-50
        cs = cs[(cs.replicate == 0) & (cs.variant == "T50")].copy()
    native = cs.sort_values("factor").groupby("id").phi.first()
    cs["phi_rel"] = 100 * cs.phi / cs.id.map(native)
    fig, axs = plt.subplots(2, 4, figsize=(174 * MM, 88 * MM), sharex=True, sharey="row")
    for j, g in enumerate(ORDER):
        for sid, x in cs[cs.group == g].groupby("id"):
            x = x.sort_values("voxel_mm")
            pa = flag(x, "perc_any").to_numpy()
            for ax, col in ((axs[0, j], "phi_rel"), (axs[1, j], "frac_connected")):
                y = x[col] if col == "phi_rel" else 100 * x[col]
                ax.plot(1000 * x.voxel_mm, y, color=COL[g], lw=0.9, alpha=0.9)
                ax.scatter(1000 * x.voxel_mm[pa], y[pa], marker=MRK[g], s=14, color=COL[g], edgecolor="white",
                           linewidth=0.4, zorder=3)
                ax.scatter(1000 * x.voxel_mm[~pa], y[~pa], marker=MRK[g], s=14, facecolor="white",
                           edgecolor=COL[g], linewidth=0.8, zorder=3)
        axs[0, j].set_title(LAB[g], color=INK)
        axs[1, j].set_xlabel("Voxel size ($\\mu$m)")
        for ax in axs[:, j]:
            ax.set_xscale("log")
            ax.set_xlim(20, 240)
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([25, 50, 100, 200]))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            _grid(ax)
    axs[0, 0].set_ylim(0, 108)
    axs[0, 0].set_ylabel("Porosity, % of native")
    axs[1, 0].set_ylabel("Pore volume in\nspanning clusters (%)")
    h = [Line2D([], [], ls="", marker="o", ms=4.5, mfc=INK2, mec=INK2, label="percolates"),
         Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec=INK2, label="does not")]
    axs[1, -1].legend(handles=h, loc="upper right", frameon=False)
    for i, a in enumerate(axs.flat):
        a.text(-0.1 if i % 4 else -0.32, 1.03, f"({'abcdefgh'[i]})", transform=a.transAxes, fontsize=8,
               fontweight="bold")
    fig.tight_layout()
    _save(fig, d, "fig05_coarsening")


# ------------------------------------------------------------------ Fig. 6: transport
def fig_transport(cfg, d):
    """(a) F vs the spanning porosity (whole main VOI; Archie m is defined with it); (b, c) PNM vs LBM on the same domain, LBM runs that are ok only.

    Filled markers: VOI faces as mirror planes (as in the LBM); open markers: faces as walls; a thin line
    joins the two values of the same case.
    """
    tt = pd.read_csv(cfg.outdir("report") / "tables" / "T4_transport.csv")
    if "fd_converged" in tt:
        tt = tt[flag(tt, "fd_converged", default=True)]
    tt = tt[np.isfinite(tt.F.astype(float))]
    lbm_f = cfg.outdir("report") / "tables" / "S_method_comparison.csv"
    fig, axs = plt.subplots(1, 3, figsize=(174 * MM, 62 * MM))
    ax = axs[0]
    for g in ORDER:
        x = tt[tt.group == g]
        ax.scatter(x.phi_span if "phi_span" in x else x.phi, x.F, marker=MRK[g], s=26, color=COL[g],
                   edgecolor="white", linewidth=0.5, zorder=3)
    p = np.logspace(np.log10(0.01), np.log10(0.6), 50)
    for m, ls in ((1.5, "-"), (2.0, "--"), (3.0, ":")):
        ax.plot(p, p ** -m, color=INK2, lw=0.8, ls=ls)
        ax.text(p[0] * 1.05, p[0] ** -m, f"m = {m:g}", fontsize=6, color=INK2, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Spanning porosity $\\phi_\\mathrm{span}$")
    ax.set_ylabel("Formation factor $F$")
    _grid(ax)
    _legend_groups(ax, groups=[g for g in ORDER if (tt.group == g).any()], loc="upper right")
    if lbm_f.exists():
        lc = pd.read_csv(lbm_f)
        lc = lc[flag(lc, "used")] if "used" in lc else lc[flag(lc, "ok") if "ok" in lc else flag(lc, "converged")]
        kcols = ["k_lbm_mD"] + [f"k_{t}_mD" for t in PNM_TAGS if f"k_{t}_mD" in lc]
        v = lc[kcols].to_numpy(float) / 1000
        v = v[np.isfinite(v) & (v > 0)]
        lo, hi = (v.min() / 3, v.max() * 3) if len(v) else (1e-3, 1e2)
        nan = np.full(len(lc), np.nan)
        w2ok = flag(lc, "whole2_ok").to_numpy() if "whole2_ok" in lc else np.zeros(len(lc), bool)
        for ax, (size, (tm, tw)) in zip(axs[1:], (("Inscribed sizes", ("pnm", "pnmw")),
                                                   ("Volume-/area-equivalent sizes", ("pnmeq", "pnmeqw")))):
            n_zero = 0
            for g in ORDER:
                sel = (lc.group == g).to_numpy()
                kl = lc.k_lbm_mD.to_numpy(float)[sel] / 1000
                km = (lc[f"k_{tm}_mD"].to_numpy(float) if f"k_{tm}_mD" in lc else nan)[sel] / 1000
                kw = (lc[f"k_{tw}_mD"].to_numpy(float) if f"k_{tw}_mD" in lc else nan)[sel] / 1000
                n_zero += int(np.sum(km == 0) + np.sum(kw == 0))
                both = (km > 0) & (kw > 0)
                ax.vlines(kl[both], np.minimum(km, kw)[both], np.maximum(km, kw)[both], color=INK2, lw=0.7, zorder=2)
                ax.scatter(kl, np.where(km > 0, km, np.nan), marker=MRK[g], s=22, facecolor=COL[g],
                           edgecolor="white", linewidth=0.5, zorder=4)
                ax.scatter(kl, np.where(kw > 0, kw, np.nan), marker=MRK[g], s=20, facecolor="white",
                           edgecolor=COL[g], linewidth=0.9, zorder=3)
                c2 = f"k_{tm}_whole2_mD"                # networks and LBM on the whole VOI binned by 2
                if c2 in lc:
                    k2 = lc[c2].to_numpy(float)[sel] / 1000
                    l2 = lc.k_lbm_whole2_mD.to_numpy(float)[sel] / 1000
                    ok2 = w2ok[sel] & (k2 > 0) & np.isfinite(l2)
                    ax.scatter(l2[ok2], k2[ok2], marker=MRK[g], s=26, facecolor="none", edgecolor=INK,
                               linewidth=0.8, zorder=5)
            lf_f = cfg.outdir("report") / "tables" / "S_lbm_full_voi.csv"   # whole VOI at the native voxel (GT11)
            if lf_f.exists():
                lf = pd.read_csv(lf_f)
                for g in ORDER:
                    q = lf[lf.group == g]
                    if len(q) and f"k_{tm}_mD" in q:
                        ax.scatter(q.k_lbm_mD / 1000, q[f"k_{tm}_mD"] / 1000, marker=MRK[g], s=26, facecolor="none",
                                   edgecolor=INK, linewidth=0.8, zorder=5)
            ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8)
            for f in (0.5, 2):
                ax.plot([lo, hi], [lo * f, hi * f], color=INK2, lw=0.6, ls=":")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(size, color=INK)
            if n_zero:
                ax.text(0.98, 0.02, f"$k_{{PNM}}$ = 0: {n_zero} (not shown)", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=6, color=INK2)
        h = [Line2D([], [], ls="", marker="o", ms=5, mfc=INK2, mec="white", label="sub-volume, mirror faces"),
             Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=INK2, label="sub-volume, wall faces"),
             Line2D([], [], ls="", marker="o", ms=5.5, mfc="none", mec=INK, label="whole VOI, mirror faces")]
        axs[1].legend(handles=h, loc="lower right", frameon=True, framealpha=1, facecolor="white", edgecolor="#cccccc", fontsize=6)
    for ax in axs[1:]:
        ax.set_xlabel("$k$ lattice Boltzmann (D)")
        _grid(ax)
    axs[1].set_ylabel("$k$ pore network, same domain (D)")
    for i, a in enumerate(axs):
        a.text(-0.16, 1.02, f"({'abc'[i]})", transform=a.transAxes, fontsize=8, fontweight="bold")
    fig.tight_layout()
    _save(fig, d, "fig06_transport")


# ------------------------------------------------------------------ Fig. 7: wall effect in the bead columns
def fig_wall(cfg, d):
    f = cfg.outdir("wall") / "wall_profiles.csv"
    if not f.exists():
        return
    prof = pd.read_csv(f)
    tl = pd.read_csv(cfg.outdir("report") / "tables" / "T3_lab_comparison.csv")
    fig, axs = plt.subplots(1, 2, figsize=(174 * MM, 62 * MM), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    shades = ["#2a78d6", "#5e9be3", "#1a4f91", "#8fb8ec"]
    for (sid, x), c in zip(prof.groupby("id", sort=False), shades):
        grain = cfg.sample(sid).grain_mm
        x = x[x.distance_from_wall_mm <= 0.92 * x.distance_from_wall_mm.max()]   # rings near the axis: few pixels
        ax.plot(x.distance_from_wall_mm / grain, 100 * x.phi.rolling(3, center=True, min_periods=1).mean(),
                color=c, lw=1.1, label=sid)
    ax.set_xlim(0, 3.5)
    ax.set_xlabel("Distance from the tube wall / grain diameter")
    ax.set_ylabel("Porosity (%)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _grid(ax)
    # (b) points, not bars on a truncated axis; the lab reference is the mean of the four lab methods (as in
    # the key numbers) and the individual methods are the small markers
    ax = axs[1]
    beads = tl[tl.Sample.str.startswith("CF")].reset_index(drop=True)
    xs = np.arange(len(beads))
    meth = [c for c in ("direct (%)", "gravimetric (%)", "indirect (%)", "helium (%)") if c in beads]
    ref = beads["Lab mean, 4 methods (%)"] if "Lab mean, 4 methods (%)" in beads else beads[meth].mean(axis=1)
    for j, c in enumerate(meth):
        ax.plot(xs - 0.22 + (j - 1.5) * 0.035, beads[c], ls="", marker="o", ms=2.6, mfc=INK2, mec="none", zorder=3)
    ax.plot(xs - 0.22, ref, ls="", marker="_", ms=11, mew=1.6, color=INK, zorder=4)
    ax.plot(xs, beads["VOI phi mean of replicates (%)"], ls="", marker=MRK["granular"], ms=5.5,
            mfc=COL["granular"], mec="white", mew=0.6, zorder=4)
    if "Whole section (%)" in beads:
        ax.plot(xs + 0.22, beads["Whole section (%)"], ls="", marker=MRK["granular"], ms=5, mfc="white",
                mec=COL["granular"], mew=1.0, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(beads.Sample)
    ax.set_xlim(-0.6, len(beads) - 0.4)
    ax.margins(y=0.3)
    ax.set_ylabel("Porosity (%)")
    h = [Line2D([], [], ls="", marker="_", ms=11, mew=1.6, color=INK, label="lab: mean of 4 methods"),
         Line2D([], [], ls="", marker="o", ms=2.6, mfc=INK2, mec="none", label="lab: individual methods"),
         Line2D([], [], ls="", marker=MRK["granular"], ms=5.5, mfc=COL["granular"], mec="white",
                label="central VOIs"),
         Line2D([], [], ls="", marker=MRK["granular"], ms=5, mfc="white", mec=COL["granular"], label="whole section")]
    ax.legend(handles=h, frameon=False, fontsize=6.5, loc="upper left", ncol=2)
    _grid(ax, "y")
    for i, a in enumerate(axs):
        a.text(-0.12, 1.02, f"({'ab'[i]})", transform=a.transAxes, fontsize=8, fontweight="bold")
    fig.tight_layout()
    _save(fig, d, "fig07_wall_effect")


# ------------------------------------------------------------------ Fig. 8: dissertation vs re-analysis
def fig_legacy(cfg, d):
    tl = pd.read_csv(cfg.outdir("report") / "tables" / "S_legacy_comparison.csv")
    inv = {v: k for k, v in LAB.items()}
    g_of = {"Glass-bead pack": "granular", "Reef limestone": "limestone", "Coquina": "coquina", "Trachyte": "trachyte"}
    fig, ax = plt.subplots(figsize=(84 * MM, 70 * MM))
    for _, r in tl.iterrows():
        g = g_of[r.Group]
        ax.scatter(r["phi dissertation (%)"], r["phi this study (%)"], marker=MRK[g], s=26, color=COL[g],
                   edgecolor="white", linewidth=0.5, zorder=3)
    m = max(tl["phi dissertation (%)"].max(), tl["phi this study (%)"].max()) * 1.05
    lime = tl[tl.Group == "Reef limestone"].sort_values("phi dissertation (%)")
    xl = lime["phi dissertation (%)"].to_numpy()
    for i, (_, r) in enumerate(lime.iterrows()):
        crowded = i + 1 < len(xl) and xl[i + 1] - xl[i] < 0.06 * m    # next label would overlap: go left
        ax.annotate(r.Sample, (r["phi dissertation (%)"], r["phi this study (%)"]), fontsize=6, color=INK2,
                    xytext=(-3 if crowded else 3, 3), textcoords="offset points", ha="right" if crowded else "left")
    ax.plot([0, m], [0, m], color=INK2, lw=0.7)
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.set_xlabel("Porosity in the dissertation (%)")
    ax.set_ylabel("Porosity, this re-analysis (%)")
    _legend_groups(ax, loc="upper left")
    _grid(ax)
    fig.tight_layout()
    _save(fig, d, "figS_legacy_porosity")
    del inv


# ------------------------------------------------------------------ supplementary: all central slices
def fig_all_slices(cfg, d):
    ids = [s.id for s in cfg.samples]
    n = len(ids)
    nr = int(np.ceil(n / 6))
    # grey and segmented slice of a specimen stacked without a gap; a spacer row above each pair holds the titles
    fig = plt.figure(figsize=(174 * MM, 232 * MM))
    gs = fig.add_gridspec(3 * nr, 6, height_ratios=[0.16, 1, 1] * nr, hspace=0.03, wspace=0.04)
    axs = np.array([[fig.add_subplot(gs[3 * (i // 2) + 1 + i % 2, c]) for c in range(6)] for i in range(2 * nr)])
    for k, sid in enumerate(ids):
        s = cfg.sample(sid)
        r, c = (k // 6) * 2, k % 6
        grey = tifffile.imread(voi_paths(cfg, sid, 0)[0])
        pore = tifffile.imread(seg_paths(cfg, sid, 0)["pore"]).astype(bool)
        z = grey.shape[0] // 2
        g = grey[z].astype(float)
        lo, hi = np.percentile(g, [0.5, 99.5])
        axs[r, c].imshow(g, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        axs[r, c].set_title(sid, fontsize=7, color=INK)
        span = spanning_mask(pore, 0) | spanning_mask(pore, 1) | spanning_mask(pore, 2)
        rgb = np.dstack([np.full(g.shape, 0.93)] * 3)
        rgb[pore[z]] = matplotlib.colors.to_rgb("#eb6834")
        rgb[span[z]] = matplotlib.colors.to_rgb("#2a78d6")
        axs[r + 1, c].imshow(rgb, interpolation="nearest")
        for a in (axs[r, c], axs[r + 1, c]):
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
        _scalebar(axs[r, c], g.shape[0], s.voxel_mm, 2 if s.voxel_mm < 0.045 else 4)
    for kk in range(k + 1, 6 * nr):
        for a in axs[(kk // 6) * 2:(kk // 6) * 2 + 2, kk % 6]:
            a.axis("off")
    _save(fig, d, "figS_all_slices")


def fig_rev(cfg, d):
    """Median CV of porosity across disjoint cubes of each VOI (overlapping and whole-VOI cubes left out).

    The number of samples behind each point goes in the legend when it is the same at every edge,
    next to the points otherwise.
    """
    cv = pd.read_csv(cfg.outdir("report") / "tables" / "S_rev_cv.csv")
    if "in_cv_curve" in cv:
        cv = cv[flag(cv, "in_cv_curve")]
    cv = cv[np.isfinite(cv.cv_phi_median.astype(float)) & (cv.cv_phi_median > 0)]
    fig, ax = plt.subplots(figsize=(84 * MM, 62 * MM))
    for g in ORDER:
        x = cv[cv.group == g].sort_values("edge_mm_mean")
        if not len(x):
            continue
        ns = x.n_samples.astype(int).unique()
        lab = f"{LAB[g]} (n = {ns[0]})" if len(ns) == 1 else LAB[g]
        ax.plot(x.edge_mm_mean, 100 * x.cv_phi_median, color=COL[g], marker=MRK[g], ms=4.5, mec="white", mew=0.5,
                label=lab)
        if len(ns) > 1:
            for _, r in x.iterrows():
                ax.annotate(f"{int(r.n_samples)}", (r.edge_mm_mean, 100 * r.cv_phi_median), fontsize=5.5,
                            color=INK2, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Sub-volume edge (mm)")
    ax.set_ylabel("Coefficient of variation of porosity (%)")
    _plain_log(ax, "x"); _plain_log(ax, "y")
    ax.legend(frameon=False, ncol=2, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    _grid(ax)
    fig.tight_layout()
    _save(fig, d, "figS_rev")


def fig_window(cfg, d):
    """Synthetic media: PNM/LBM ratio vs window size (mirror vs wall faces) and k_LBM vs window size.

    Colour = network size definition (black inscribed, grey volume-/area-equivalent); filled markers and
    solid lines = mirror faces, open markers and dashed lines = wall faces. Ratios are to the LBM of the
    same window (for the lattice, to the exact infinite-lattice value), LBM runs that are ok only.
    """
    f = cfg.outdir("report") / "tables" / "S_window_study.csv"
    if not f.exists():
        return
    ws = pd.read_csv(f)
    ws = ws[flag(ws, "lbm_ok", default=True)] if "lbm_ok" in ws else ws
    media = [("simple_cubic_spheres", "Sphere lattice", "o", "-"),
             ("overlapping_spheres", "Overlapping spheres", "x", "--")]
    style = {"pnm": (INK, True), "pnmeq": ("#8a8984", True), "pnmw": (INK, False), "pnmeqw": ("#8a8984", False)}
    fig, axs = plt.subplots(1, 3, figsize=(174 * MM, 62 * MM))
    ys = [1.0]
    for ax, (med, title, _, _) in zip(axs[:2], media):
        x = ws[ws.medium == med].sort_values("window_vox")
        for t, (c, mirror) in style.items():
            col = f"ratio_k_{t}_lbm"
            if col not in x or not len(x):
                continue
            ax.plot(x.window_vox, x[col], color=c, lw=1.0, ls="-" if mirror else "--", marker="o", ms=4,
                    mfc=c if mirror else "white", mec=c, mew=0.9, zorder=3)
            ys += [v for v in x[col].to_numpy(float) if np.isfinite(v) and v > 0]
        ax.axhline(1.0, color=INK2, lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_title(title, color=INK)
        ax.set_xlabel("Window edge (voxels)")
        _grid(ax)
    for ax in axs[:2]:                       # same scale for the two media
        ax.set_ylim(min(ys) / 1.15, max(ys) * 1.15)
        _plain_log(ax, "y")
    axs[0].set_ylabel("$k_{PNM}\\,/\\,k_{LBM}$")
    ax = axs[2]
    if "k_lbm_rel_largest" in ws:
        for med, title, mk, ls in media:
            x = ws[(ws.medium == med) & np.isfinite(ws.k_lbm_rel_largest.astype(float))].sort_values("window_vox")
            ax.plot(x.window_vox, x.k_lbm_rel_largest, color=INK, lw=1.0, ls=ls, marker=mk, ms=4.5, mew=1.0,
                    mfc=INK)
    ax.axhline(1.0, color=INK2, lw=0.6, ls=":")
    ax.set_title("LBM", color=INK)
    ax.set_xlabel("Window edge (voxels)")
    ax.set_ylabel("$k_{LBM}$ / $k_{LBM}$ of the largest window")
    _grid(ax)
    h = [Line2D([], [], color=c, lw=1.0, ls="-" if mirror else "--", marker="o", ms=4, mfc=c if mirror else "white",
                mec=c, label=f"PNM, {PNM_TAGS[t]}") for t, (c, mirror) in style.items()]
    h += [Line2D([], [], color=INK, lw=1.0, ls=ls, marker=mk, ms=4.5, mew=1.0, mfc=INK, label=f"LBM, {title.lower()}")
          for _, title, mk, ls in media]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False, fontsize=6.5)
    for i, a in enumerate(axs):
        a.text(-0.16, 1.02, f"({'abc'[i]})", transform=a.transAxes, fontsize=8, fontweight="bold")
    fig.tight_layout()
    _save(fig, d, "figS_window")


def run(cfg: Config) -> None:
    d = cfg.outdir("figures")
    for f in (fig_overview, fig_histograms, fig_connectivity, fig_percolation, fig_coarsening, fig_transport,
              fig_wall, fig_legacy, fig_all_slices, fig_rev, fig_window):
        try:
            f(cfg, d)
            print(f"  {f.__name__}: ok")
        except Exception as e:  # keep going; report the failure
            print(f"  {f.__name__}: FAILED {type(e).__name__}: {e}")
