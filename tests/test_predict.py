"""Percolation-prediction statistics on synthetic grouped data with known answers.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT
"""
import numpy as np
import pandas as pd
from scipy.special import expit

from poreconn import predict
from poreconn.config import Config
from poreconn.predict import (auc_cell, auc_weighted, case_table, cluster_counts, conditional_cell, cube_index, fe_logit,
                              fe_loso, loso_fast, loso_within_scores, perm_struct, phi50_cell, phi50_fit, within_auc,
                              within_parts, within_parts_w)
from poreconn.report import _fit_logit, auc, loso_auc, phi50

N_GROUPS, N_PER = 12, 60
QUIET = lambda *a, **k: None          # noqa: E731


def cubes(rng, n_spec, n_cube, x_fn, eta_fn):
    """Case table with the structure of stage rev: n_spec specimens x n_cube cubes x 3 axes. The cube feature
    x = x_fn(specimen mean, rng) and P(y) = expit(eta_fn(x, specimen mean) + cube effect + axis noise)."""
    g = np.repeat(np.arange(n_spec), n_cube)
    m = np.linspace(-2, 2, n_spec)[g]
    x = x_fn(m, rng)
    p = expit(eta_fn(x, m) + 0.5 * rng.normal(size=len(g)))
    y = rng.uniform(size=(len(g), 3)) < p[:, None]
    base = pd.DataFrame({"id": [f"S{k:02d}" for k in g], "replicate": 0, "cube": np.arange(len(g)) % n_cube,
                         "x": x, "phi": rng.uniform(0.05, 0.3, len(g))})
    return pd.concat([base.assign(axis=a, y=y[:, j].astype(int)) for j, a in enumerate("zyx")], ignore_index=True)


def grouped(rng, logit_fn, n_groups=N_GROUPS, n_per=N_PER):
    """12 groups x 60 cases; group intercepts spread so that the group prevalence varies (~0.1 to ~0.9)."""
    g = np.repeat(np.arange(n_groups), n_per)
    a = np.linspace(-2.2, 2.2, n_groups)[g]
    x = rng.normal(size=len(g))
    phi = rng.uniform(0.0, 0.3, size=len(g))
    y = (rng.uniform(size=len(g)) < expit(logit_fn(a, x, phi))).astype(float)
    return pd.DataFrame({"id": [f"S{k:02d}" for k in g], "g": g, "x": x, "phi": phi, "y": y})


def test_loso_fast_equals_report_loso():
    rng = np.random.default_rng(0)
    d = grouped(rng, lambda a, x, phi: a + 0.8 * x)
    d["x2"] = rng.normal(size=len(d))
    for feats in (["x"], ["x", "x2"]):
        a, s, _ = loso_auc(d, feats)
        s2, sk = loso_fast(d[feats].to_numpy(), d.y.to_numpy(), d.id.to_numpy())
        assert sk == 0 and np.max(np.abs(s - s2)) < 1e-4 and abs(auc(d.y, s2) - a) < 1e-6


def test_within_and_weighted_auc_against_brute_force():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 90).astype(float)
    s = np.round(rng.normal(size=90), 1)                # ties on purpose
    gi = rng.integers(0, 5, 90)
    num = den = 0.0
    for k in range(5):
        m = gi == k
        pos, neg = s[m & (y == 1)], s[m & (y == 0)]
        num += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        den += len(pos) * len(neg)
    assert np.isclose(within_auc(y, s, gi)[0], num / den)
    w = rng.integers(0, 4, 90)
    rep = np.repeat(np.arange(90), w)
    assert np.isclose(auc_weighted(y, s, w), auc(y[rep], s[rep]))
    c = cluster_counts(np.random.default_rng(2), 7, 50)
    assert c.shape == (50, 7) and np.all(c.sum(1) == 7)


def test_uninformative_feature_biases_pooled_loso_auc_but_not_within():
    """(i) pure noise: pooled LOSO AUC clearly below 0.5 (fold intercepts anti-correlate with the held-out
    prevalence), within-specimen AUC of the same scores about 0.5."""
    rng = np.random.default_rng(3)
    pooled, within = [], []
    for _ in range(30):
        d = grouped(rng, lambda a, x, phi: a)
        s, sk = loso_fast(d[["x"]].to_numpy(), d.y.to_numpy(), d.g.to_numpy())
        assert sk == 0
        pooled.append(auc(d.y, s))
        within.append(within_auc(d.y.to_numpy(), s, d.g.to_numpy())[0])
    assert np.mean(pooled) < 0.42
    assert abs(np.mean(within) - 0.5) < 0.03


def test_informative_feature_within_specimens():
    """(ii) a feature that acts within groups gives a within-specimen AUC > 0.7."""
    rng = np.random.default_rng(4)
    d = grouped(rng, lambda a, x, phi: a + 1.5 * x)
    s, _ = loso_fast(d[["x"]].to_numpy(), d.y.to_numpy(), d.g.to_numpy())
    wa, nboth = within_auc(d.y.to_numpy(), s, d.g.to_numpy())
    assert wa > 0.7 and nboth == N_GROUPS


def test_fixed_effects_recovers_signs():
    """(iii) y ~ phi + chi + group: the chi coefficient is negative, phi positive; bootstrap and LR agree."""
    rng = np.random.default_rng(5)
    d = grouped(rng, lambda a, x, phi: a + 1.0 * (phi - 0.15) / 0.0866 - 0.8 * x)
    d["chi"] = d.x
    X = d[["phi", "chi"]].to_numpy()
    X = (X - X.mean(0)) / X.std(0)
    b, a, ll, conv = fe_logit(X, d.y.to_numpy(), d.g.to_numpy())
    assert conv and b[0] > 0.6 and -1.1 < b[1] < -0.5
    r = conditional_cell(d, "chi", (0,))
    assert r["status"] == "ok" and r["beta_chi_hi"] < 0 and r["share_chi_pos"] < 0.01 and r["p_lr_naive"] < 1e-6
    assert r["p_boot"] < 0.01 and r["p_perm_cond"] < 0.01 and r["beta_phi"] > 0
    assert -1.1 < r["beta_chi"] < -0.5       # per within-specimen SD (x has SD 1 within groups)
    # with one group fe_logit is the plain logistic regression of report._fit_logit
    b1, a1, _, _ = fe_logit(X, d.y.to_numpy(), np.zeros(len(d), int))
    assert np.allclose(np.r_[a1, b1], _fit_logit(X, d.y.to_numpy(), l2=0.0), atol=1e-4)


def test_phi50_of_known_logistic():
    """(iv) P(y | phi) = logistic((phi - 0.15) / 0.02): phi50 = 0.15."""
    rng = np.random.default_rng(6)
    d = grouped(rng, lambda a, x, phi: (phi - 0.15) / 0.02)
    est, slope, icpt, conv = phi50_fit(d.phi, d.y)
    assert conv and abs(est - 0.15) < 0.006 and abs(slope - 50) < 15
    assert np.isclose(est, phi50(d, n=2)[0], atol=1e-4)
    r = phi50_cell(d, (0,))
    assert r["phi50_lo"] < 0.15 < r["phi50_hi"] and r["boot_used"] + r["boot_discarded"] == r["boot_n"]


def test_fe_loso_equals_one_fit_per_fold():
    """(v) the folds solved together equal fe_logit fitted fold by fold on the training specimens with both
    outcomes; for one column the within scores carry the sign of that fit (exact shortcut)."""
    rng = np.random.default_rng(7)
    d = grouped(rng, lambda a, x, phi: a + 0.8 * x - 4 * (phi - 0.15))
    d.loc[d.g == 0, "y"] = 0.0                                   # a specimen without both outcomes
    X, y, g = d[["x", "phi"]].to_numpy(), d.y.to_numpy(), d.g.to_numpy()
    B, conv = fe_loso(X, y, g)
    use = d.groupby("g").y.agg(lambda v: v.min() != v.max()).to_numpy()
    for f in range(N_GROUPS):
        tr = use[g] & (g != f)
        b = fe_logit(X[tr], y[tr], np.unique(g[tr], return_inverse=True)[1])[0]
        assert np.allclose(B[:, f], b, atol=1e-8)
    assert conv.all()
    s = loso_within_scores(X[:, :1], y, g)
    B1, _ = fe_loso(X[:, :1], y, g)
    assert np.array_equal(np.sign(s), np.sign(B1[0, g] * X[:, 0]))


def test_cube_level_equals_case_level():
    """(vi) fits on cubes (share of positive axes, weight 3) and cube-weighted AUCs equal those on the cases."""
    rng = np.random.default_rng(8)
    d = cubes(rng, 9, 40, lambda m, r: m + r.normal(size=len(m)), lambda x, m: m - 0.7 * x)
    key, first, w = cube_index(d)
    assert np.all(w == 3) and len(first) == 9 * 40
    y = d.y.to_numpy(float)
    n1 = np.bincount(key, weights=y)
    gi = np.unique(d.id, return_inverse=True)[1]
    X = d[["x", "phi"]].to_numpy()
    s_case, _, sl_case = loso_fast(X, y, gi, slope_scores=True)
    s_cube, _, sl_cube = loso_fast(X[first], n1 / 3, gi[first], w=w, slope_scores=True)
    assert np.allclose(s_case, s_cube[key]) and np.allclose(sl_case, sl_cube[key])
    assert np.allclose(loso_within_scores(X, y, gi), loso_within_scores(X[first], n1 / 3, gi[first], w)[key])
    s = np.round(s_cube, 2)                                      # ties on purpose
    U1, P1 = within_parts(y, s[key], gi)
    U2, P2 = within_parts_w(n1, 3 - n1, s, gi[first])
    assert np.allclose(U1, U2) and np.allclose(P1, P2)
    assert np.isclose(auc(y, s[key]), auc_weighted(n1 / 3, s, w))


def test_perm_struct_keeps_specimen_structure():
    """(vii) the structured permutation keeps every specimen's cube deviations (permuted among its cubes)
    and permutes the specimen means; without the mean swap it permutes the cubes within each specimen."""
    gc = np.repeat(np.arange(5), [3, 4, 5, 6, 7])
    F = np.random.default_rng(9).normal(size=(len(gc), 2)) + np.arange(5)[gc, None] * 10
    P0 = perm_struct(np.random.default_rng(1), F, gc, swap_means=False)
    for k in range(5):
        assert np.allclose(np.sort(P0[gc == k], 0), np.sort(F[gc == k], 0))
    P1 = perm_struct(np.random.default_rng(2), F, gc)
    m = np.stack([F[gc == k].mean(0) for k in range(5)])
    m1 = np.stack([P1[gc == k].mean(0) for k in range(5)])
    assert np.allclose(np.sort(m1[:, 0]), np.sort(m[:, 0])) and not np.allclose(m1, m)
    for k in range(5):
        dev, dev1 = F[gc == k] - m[k], P1[gc == k] - m1[k]
        assert np.allclose(np.sort(dev1, 0), np.sort(dev, 0))
        j = np.argmin(np.abs(m[:, 0] - m1[k, 0]))               # the columns of F move together
        assert np.allclose(m1[k], m[j])


def test_within_auc_follows_within_direction(monkeypatch):
    """(viii) Simpson's paradox: the feature rises with the percolating share between specimens but lowers
    it within them. Folds fitted on pooled data turn the within-specimen ranking around (within AUC < 0.5);
    the within-fitted folds of auc_cell report the within-specimen direction (> 0.6)."""
    monkeypatch.setattr(predict, "N_BOOT", 100)
    rng = np.random.default_rng(10)
    d = cubes(rng, 12, 40, lambda m, r: m + r.normal(size=len(m)), lambda x, m: 1.5 * m - 1.2 * (x - m))
    gi = np.unique(d.id, return_inverse=True)[1]
    y = d.y.to_numpy(float)
    _, s_pooled, _ = loso_auc(d, ["x"])
    assert within_auc(y, s_pooled, gi)[0] < 0.4                  # the former estimator: reversed
    r = auc_cell(d, {"x": ["x"]}, (0,), 9, log=QUIET)[0]
    assert r["auc_within"] > 0.6 and r["auc_within_lo"] > 0.55 and r["p_perm_within"] <= 0.1
    assert r["auc_within_specmean"] > 0.6
    assert abs(r["auc_within"] - (1 - r["auc_within_expected"])) < 1e-12   # 'x' expected up, acts down
    assert r["auc_pooled"] > 0.5                                 # between specimens the feature rises with y


def test_structured_null_keeps_nominal_level(monkeypatch):
    """(ix) a feature clustered by specimen but unrelated to y: the structured nulls of auc_cell reject at about
    the nominal 5 % (19 permutations: p <= 0.05 only when the observed AUC exceeds every null value), while
    the former null - rows permuted across all cases - rejects far more often."""
    monkeypatch.setattr(predict, "N_BOOT", 20)
    rng = np.random.default_rng(11)
    n_sim, n_perm, n_spec, n_cube = 40, 19, 10, 20
    spec = np.arange(n_spec * n_cube) // n_cube
    rej = {"p_perm": 0, "p_perm_slope": 0, "p_perm_within": 0}
    naive = 0
    for k in range(n_sim):
        u, v = rng.normal(size=n_spec), 1.5 * rng.normal(size=n_spec)
        d = cubes(rng, n_spec, n_cube, lambda m, r: u[spec] + 0.5 * r.normal(size=len(m)), lambda x, m: v[spec])
        r = auc_cell(d, {"x": ["x"]}, (k,), n_perm, log=QUIET)[0]
        for c in rej:
            rej[c] += r[c] <= 0.05
        y, x, g = d.y.to_numpy(float), d.x.to_numpy(), d.id.to_numpy()
        prng = np.random.default_rng(k)
        null = np.array([auc(y, loso_fast(x[prng.permutation(len(y))], y, g)[0]) for _ in range(n_perm)])
        naive += (1 + np.sum(null >= r["auc_pooled"])) / (1 + n_perm) <= 0.05
    for c, v in rej.items():
        assert v / n_sim <= 0.15, (c, v)
    assert naive / n_sim >= 0.3, naive


def test_flexible_phi_basis_does_not_credit_chi():
    """(x) P(y | phi) non-linear (in sqrt phi) and chi = sqrt(phi) + noise with no own effect: with phi
    linear the fixed-effects model credits chi; with the basis phi, sqrt(phi), phi^2 it does not."""
    rng = np.random.default_rng(12)
    d = grouped(rng, lambda a, x, phi: a + 14 * (np.sqrt(phi) - 0.3))
    d["chi"] = np.sqrt(d.phi) + 0.03 * d.x
    X = d[["phi", "chi"]].to_numpy()
    b_lin = fe_logit((X - X.mean(0)) / X.std(0), d.y.to_numpy(), d.g.to_numpy())[0]
    assert b_lin[1] > 1.0
    r = conditional_cell(d, "chi", (1,), n_perm=99)
    assert r["status"] == "ok" and r["beta_chi_lo"] < 0 < r["beta_chi_hi"] and r["p_perm_cond"] > 0.05
    assert r["p_boot"] > 0.05 and r["beta_phi"] > 0


def test_inference_guard_below_min_infer(monkeypatch):
    """(xi) fewer than MIN_INFER specimens with both outcomes: point estimates only, intervals and p NaN."""
    monkeypatch.setattr(predict, "N_BOOT_PHI50", 50)
    rng = np.random.default_rng(13)
    d = grouped(rng, lambda a, x, phi: a - 0.8 * x + 10 * (phi - 0.15), n_groups=5)
    d["chi"] = d.x
    r = conditional_cell(d, "chi", (0,))
    assert r["status"] == "descriptive only (5 specimens with both outcomes)" and r["n_distinct_resamples"] == 126
    assert np.isfinite(r["beta_chi"]) and r["beta_chi"] < 0
    assert all(np.isnan(r[c]) for c in ("beta_chi_lo", "beta_chi_hi", "share_chi_pos", "p_boot", "p_perm_cond",
                                        "p_lr_naive"))
    a = auc_cell(d, {"chi": ["chi"]}, (0,), 9, log=QUIET, infer=False)[0]
    assert np.isfinite(a["auc_within"]) and np.isnan(a["auc_within_lo"]) and np.isnan(a["p_perm_within"])
    p = phi50_cell(d, (0,))
    assert p["status"] == "descriptive only (5 specimens)" and np.isfinite(p["phi50"]) and np.isnan(p["phi50_lo"])


def _case_config(tmp_path, rev, topo, variants="[T50, T50int]"):
    (tmp_path / "config.yaml").write_text(
        "paths: {raw_root: raw, voi_dir: voi, lab_csv: lab.csv, legacy_csv: legacy.csv, outputs: outputs}\n"
        f"air_variants: {variants}\n"
        "samples:\n"
        "  - {id: A, group: coquina, label: A, raw: a, voxel_um: 40.0, bin: 1}\n"
        "  - {id: B, group: trachyte, label: B, raw: b, voxel_um: 20.0, bin: 2}\n", encoding="utf-8")
    cfg = Config(tmp_path / "config.yaml")
    rev.to_csv(cfg.outdir("rev") / "rev.csv", index=False)
    topo.to_csv(cfg.outdir("topology") / "topology.csv", index=False)
    return cfg


def test_case_table_features_and_units(tmp_path):
    """(xiii) case_table: 3 rows per non-overlapping cube, y from perc_<axis>, chi_V = 0 in empty cubes,
    chi_main with the main-VOI chord of the same variant (topology at T50; whole-VOI cube of rev at T50int,
    where topology has no S_V), chi_cube with the cube's own chord (4 phi / S_V, mm); stale inputs raise."""
    import pytest
    rows = []
    for sid, phi0 in (("A", 0.10), ("B", 0.20)):
        for v, f in (("T50", 1.0), ("T50int", 1.5)):
            for c in range(8):                                            # edge 50 in a 100^3 VOI
                phi = 0.0 if c == 0 else f * phi0 * (1 + 0.1 * c)
                rows.append({"id": sid, "group": "coquina" if sid == "A" else "trachyte", "replicate": 0,
                             "variant": v, "edge": 50, "edge_mm": 2.0, "cube": f"c{c}", "overlapping": False,
                             "phi": phi, "perc_z": c % 2 == 1, "perc_y": c % 3 == 0, "perc_x": False,
                             "chi_V": 5.0 - c, "S_V": 0.0 if c == 0 else 2.0 + 0.1 * c})
            rows.append({**rows[-1], "edge": 75, "cube": "x", "overlapping": True})          # dropped
            rows.append({**rows[-1], "edge": 100, "edge_mm": 4.0, "cube": "0-0-0", "overlapping": False,
                         "phi": f * phi0, "S_V": 2.0 * f ** 2, "chi_V": 1.0})              # whole VOI
    rev = pd.DataFrame(rows)
    topo = pd.DataFrame([{"id": sid, "replicate": 0, "variant": v, "phi": phi0 if v == "T50" else 1.5 * phi0,
                          "S_V": 2.0 if v == "T50" else np.nan, "chord_pore_mean": np.nan}
                         for sid, phi0 in (("A", 0.10), ("B", 0.20)) for v in ("T50", "T50int")])
    long = case_table(_case_config(tmp_path, rev, topo))
    assert len(long) == 3 * len(rev[~rev.overlapping]) and set(long.axis) == {"z", "y", "x"}
    r = long[(long.id == "A") & (long.variant == "T50") & (long.cube == "c3")].set_index("axis")
    assert list(r.y.loc[["z", "y", "x"]]) == [1, 1, 0]
    lp = {("A", "T50"): 4 * 0.1 / 2.0, ("A", "T50int"): 4 * 0.15 / 4.5}      # topology T50; rev whole VOI T50int
    for (sid, v), l in lp.items():
        q = long[(long.id == sid) & (long.variant == v) & (long.edge == 50)]
        assert np.allclose(q.l_p_main, l) and (q.l_p_main_variant == v).all()
        assert np.allclose(q.chi_main, np.arcsinh(q.chi_V * l ** 3 / predict.CHI_SCALE))
        full = q[q.phi > 0]
        assert np.allclose(full.chi_cube, np.arcsinh(full.chi_V * (4 * full.phi / full.S_V) ** 3 / predict.CHI_SCALE))
        empty = q[q.phi <= 0]
        assert len(empty) == 3 and (empty.chi_V == 0).all() and (empty.chi_cube == 0).all()
    assert np.isclose(long.loc[long.id == "B", "voxel_mm"].iloc[0], 0.04)
    with pytest.raises(ValueError, match="S_V"):
        case_table(_case_config(tmp_path, rev.drop(columns="S_V"), topo))
    with pytest.raises(ValueError, match="variant"):
        case_table(_case_config(tmp_path, rev[rev.variant == "T50"].drop(columns="variant"), topo))
    with pytest.raises(ValueError, match="chord"):
        case_table(_case_config(tmp_path, rev, topo[topo.id == "A"]))


def test_phi50_extrapolation_flags(monkeypatch):
    """(xii) phi50 inside the data is not flagged; beyond the observed porosities, or with a negative slope
    (status 'slope not positive'), it is."""
    monkeypatch.setattr(predict, "N_BOOT_PHI50", 200)
    rng = np.random.default_rng(14)
    d = grouped(rng, lambda a, x, phi: (phi - 0.15) / 0.02)
    r = phi50_cell(d, (0,))
    assert not r["phi50_extrapolated"] and r["status"] == "ok" and r["phi50_n_near"] >= 20
    assert 0 < r["phi50_n_near_pos"] < r["phi50_n_near"] and r["p_fit_phi0"] < 0.01
    assert r["cal_bins_gof"] >= 5 and r["cal_max_abs_dev"] < 0.15
    far = d.copy()                                       # porosities up to 0.1, P = 0.5 only at phi = 0.2
    far["phi"] = far.phi / 3
    far["y"] = (rng.uniform(size=len(far)) < expit((far.phi - 0.2) / 0.04)).astype(float)
    r = phi50_cell(far, (0,))
    assert r["phi50"] > r["phi_q975"] and r["phi50_extrapolated"] and r["phi50_n_near"] < 20
    neg = d.copy()
    neg["y"] = 1.0 - neg.y
    r = phi50_cell(neg, (0,))
    assert np.isnan(r["phi50"]) and r["phi50_extrapolated"] and r["status"] == "slope not positive"
