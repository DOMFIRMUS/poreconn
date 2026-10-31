"""Imaging diagnostics on synthetic data: edge-spread fit and step-like flag, bubbles and the enclosed-air
depth tables, the two-glass classification, VOI footprint integration and frame conversion, synthetic
bead packs.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from poreconn.grains import beads
from poreconn.imaging import (DEPTH_MIN_SIGMA, SYN_SUPER, blur_offset_at, blurred_pore, bubble_blur_reference,
                              bubble_summary, bubbles_voi, calibrate_sigma, dense_split, depth_histograms,
                              depth_rows, difference_table, digitise, digitised_overlap, esf_calibration,
                              esf_model, esf_profile, esf_selection, fit_esf, footprint_from, footprint_phi,
                              measure_synthetic, mode_split, relax_packing, spheroid_contact_distance, step_like,
                              summarise, synthetic_pack, synthetic_params, two_glass_voi, voi_section_origin)
from poreconn.segment import median3
from poreconn.wall import section_grid

NOISE = 0.03          # noise sd / contrast of the test volumes
AIR, GLASS = 8000.0, 13800.0


def blurred_solid_sphere(sigma, n=84, R=24.0, air=AIR, solid=GLASS, seed=7):
    """A glass sphere in air (the reverse of the calibration pore), point-sampled through a Gaussian edge of
    width sigma (voxels), centre off the voxel grid, white noise."""
    z, y, x = np.indices((n, n, n), dtype=np.float32) - (n - 1) / 2 + 0.17
    r = np.sqrt(z * z + y * y + x * x)
    g = air + (solid - air) * ndtr((R - r) / sigma)
    g += np.random.default_rng(seed).normal(0, NOISE * (solid - air), g.shape)
    return np.clip(np.rint(g), 0, 65535).astype(np.uint16), 0.5 * (air + solid)


def balls(shape, centres, radii, values, background, sigma=0.7, noise=0.02, seed=3):
    """uint16 volume of balls (point-sampled Gaussian edges of width sigma) on a uniform background."""
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    g = np.full(shape, background, np.float32)
    for c, R, v in zip(centres, radii, values):
        r = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
        g += (v - background) * ndtr((R - r) / sigma)
    g += np.random.default_rng(seed).normal(0, noise * 5800, shape)
    return np.clip(np.rint(g), 0, 65535).astype(np.uint16)


@pytest.fixture(scope="module")
def calibration():
    return esf_calibration(NOISE)


# ------------------------------------------------------------------ edge-spread function
@pytest.mark.parametrize("sigma", [0.9, 1.4, 2.2])
def test_esf_fit_recovers_sigma(sigma, calibration):
    g, t = blurred_solid_sphere(sigma)
    filt = median3(g)
    idx, d = esf_selection(filt < t)            # mask from the filtered volume, as for the VOIs
    f = fit_esf(esf_profile(d, g.ravel()[idx].astype(np.float32)))
    assert f["reason"] == ""
    assert "sigma_se_vox" not in f                                        # not an uncertainty: not exported
    assert abs(f["sigma_vox"] / sigma - 1) < 0.12                         # raw fit: 6-12 % low at 1-2 voxels
    assert abs(calibrate_sigma(f["sigma_vox"], calibration, "none") / sigma - 1) < 0.10
    assert abs(f["plateau_pore"] - 8000) < 100 and abs(f["plateau_solid"] - 13800) < 100
    assert abs(f["noise_pore_sd"] / (NOISE * 5800) - 1) < 0.15
    assert abs(f["contrast_fit"] / 5800 - 1) < 0.03
    assert step_like(f)


def test_esf_skips_small_features():
    rng = np.random.default_rng(1)
    pore = rng.random((60, 60, 60)) < 0.05          # isolated single voxels: no large feature
    idx, d = esf_selection(pore)
    f = fit_esf(esf_profile(d, np.zeros(len(d), np.float32)))
    assert np.isnan(f["sigma_vox"]) and f["reason"]
    assert not step_like(f)


def _profile(g_of_d, seed=0):
    d = np.random.default_rng(seed).uniform(-6, 6, 200000).astype(np.float32)
    return fit_esf(esf_profile(d, g_of_d(d).astype(np.float32)))


def test_step_like_wide_offset_erf():
    # a pure blurred step is step-like whatever its width and offset (the raw-profile tail of this erf was 0.25)
    f = _profile(lambda d: esf_model(d, GLASS, AIR - GLASS, -0.9, 1.6))
    assert f["sigma_vox"] == pytest.approx(1.6, rel=1e-3) and f["d0_vox"] == pytest.approx(-0.9, abs=1e-3)
    assert abs(f["tail_pore_frac"]) < 0.005 and abs(f["tail_solid_frac"]) < 0.005
    assert step_like(f)


def test_residual_tail_detects_a_rim_the_erf_misses():
    # a sharp step plus a rim of intermediate grey 3.5 voxels into the solid: the erf cannot follow it
    rim = _profile(lambda d: GLASS - 5800 * ndtr(d / 0.7) - 0.2 * 5800 * ndtr((d + 3.5) / 0.5) * (d < 0))
    assert rim["tail_solid_frac"] > 0.08
    assert not step_like(rim, tail_max=0.05)


def test_step_like_rule():
    assert step_like({"tail_pore_frac": 0.01, "tail_solid_frac": -0.02})
    assert not step_like({"tail_pore_frac": 0.01, "tail_solid_frac": 0.2})
    assert not step_like({"tail_pore_frac": -0.16, "tail_solid_frac": 0.0})
    assert step_like({"tail_pore_frac": np.nan, "tail_solid_frac": 0.03})
    assert not step_like({"tail_pore_frac": np.nan, "tail_solid_frac": np.nan})


# ------------------------------------------------------------------ bubbles and depth offsets
def test_bubbles_voi_large_bubble_is_at_the_air_level():
    R, t = 8.0, 0.5 * (AIR + GLASS)
    filt = median3(blurred_pore(0.7, NOISE, seed=2, n=33, R=R))
    b = bubbles_voi(filt, filt < t)
    assert len(b) == 1
    r = b.iloc[0]
    assert r.volume_vox == pytest.approx(4 / 3 * np.pi * R ** 3, rel=0.05)
    assert R - 1.0 <= r.r_in_vox <= R + 0.5
    assert abs((r.g_deep - AIR) / (GLASS - AIR)) < 0.015


def test_bubbles_voi_skips_open_and_small_clusters():
    filt = np.full((30, 30, 30), 13800, np.uint16)
    filt[10:14, 10:14, 10:14] = 8000            # enclosed, 64 voxels
    filt[20:22, 20:22, 20:22] = 8000            # enclosed, 8 voxels < BUBBLE_MIN_VOX
    filt[:, 3:7, 3:7] = 8000                    # open: touches two faces
    b = bubbles_voi(filt, filt < 10900)
    assert b.volume_vox.tolist() == [64]


def test_bubbles_voi_single_bead_check():
    filt = np.full((30, 30, 40), 13800, np.uint16)
    filt[10:16, 10:16, 5:11] = 8000             # inside bead 1
    filt[10:16, 10:16, 17:23] = 8000            # straddles the boundary of beads 1 and 2 (x = 20)
    reg = np.ones(filt.shape, np.int32)
    reg[:, :, 20:] = 2
    b = bubbles_voi(filt, filt < 10900, reg=reg).sort_values("label")
    assert b.in_single_bead.tolist() == [True, False]
    assert b.n_bead_labels.tolist() == [1, 2]


def test_bubble_in_a_real_bead_is_in_a_single_bead():
    # two glass beads in air (open pore space touching the faces), a bubble inside the first
    vol = balls((44, 44, 88), [(22, 22, 22), (22, 22, 65), (22, 22, 19)], [19.0, 19.0, 4.5],
                [GLASS, GLASS, AIR - GLASS + AIR], AIR)
    filt = median3(vol)
    pore = filt < 0.5 * (AIR + GLASS)
    reg = beads(pore)
    b = bubbles_voi(filt, pore, reg=reg)
    assert len(b) == 1
    assert bool(b.in_single_bead.iloc[0]) and int(b.n_bead_labels.iloc[0]) == 1


def test_blur_reference_small_bubbles_are_raised():
    ref = bubble_blur_reference(0.7, 0.05, radii=(2.0, 3.0, 6.0), reps=4)
    off = ref.offset_deep_median.to_numpy()
    assert off[0] > 0.2 and off[1] > 0.015 and abs(off[2]) < 0.015
    assert np.all(np.diff(ref.d_eq_vox_median) > 0)
    v = blur_offset_at([1.0, ref.d_eq_vox_median.iloc[1], 1000.0], ref)
    assert np.isnan(v[0]) and v[1] == pytest.approx(off[1]) and v[2] == pytest.approx(off[2])


def test_bubble_summary_uses_large_single_bead_bubbles():
    bub = pd.DataFrame({"r_in_vox": [1.7, 2.4, 4.4, 5.5, 6.0], "in_single_bead": [True, True, True, True, False],
                        "offset_deep": [0.12, 0.04, 0.01, 0.02, 0.30], "d_eq_vox": [4.3, 5.4, 9.7, 11.8, 12.0]})
    bub["d_eq_mm"] = bub.d_eq_vox * 0.05
    bub["offset_deep_blur_ref"] = [0.12, 0.04, 0.003, 0.003, 0.003]
    bub["offset_deep_excess"] = bub.offset_deep - bub.offset_deep_blur_ref
    s = bubble_summary(bub, "X", "granular", 0.05)
    allr = s[-1]
    assert allr["d_bin"] == "all" and allr["n_bubbles"] == 2
    assert allr["offset_deep_median"] == pytest.approx(0.015)
    assert allr["n_bubbles_recorded"] == 5 and allr["n_inter_bead"] == 1 and allr["n_below_r_min"] == 2
    assert sum(r["n_bubbles"] for r in s[:-1]) == 4                      # bins: single-bead bubbles only


def test_depth_offsets_open_and_enclosed():
    con = GLASS - AIR
    filt = np.full((48, 48, 48), GLASS, np.uint16)
    filt[4:24] = AIR                                                     # open slab (touches the y and x faces)
    zz, yy, xx = np.indices(filt.shape)
    filt[(zz - 35) ** 2 + (yy - 24) ** 2 + (xx - 24) ** 2 <= 10 ** 2] = AIR + 0.1 * con   # enclosed pore
    h = depth_histograms(filt, AIR, con, depths=(2, 3, 5))
    rows = depth_rows(h, "R1", "coquina", 0.04, AIR, con, ("rock open pore", "rock enclosed pore"), "outside air")
    d = pd.DataFrame(rows)
    assert set(d.air_reference) == {"outside air"}
    off = d.set_index(["source", "depth_vox"]).offset
    for dd in (2, 3, 5):
        assert off["rock open pore", dd] == pytest.approx(0.0, abs=2e-4)
        assert off["rock enclosed pore", dd] == pytest.approx(0.1, abs=2e-4)
    d["sigma_cal_vox"], d["sigma_cal_source"] = 1.25, "sample"
    d["depth_over_sigma"] = d.depth_vox / d.sigma_cal_vox
    diff = difference_table(d)
    assert diff.kind.unique().tolist() == ["rock"]
    assert np.allclose(diff.enclosed_minus_open, 0.1, atol=3e-4)
    assert diff.depth_over_sigma.tolist() == pytest.approx([1.6, 2.4, 4.0])


def test_summary_compares_depths_beyond_3_sigma_and_labels_the_air(tmp_path):
    rows = []
    for kind, grp, ref in (("bead pack", "granular", "pore air"), ("rock", "trachyte", "outside air")):
        for dd, sig in ((2, 1.0), (5, 1.0), (5, 1.8), (8, 1.8)):
            rows.append({"id": f"{grp}{sig}", "group": grp, "kind": kind, "depth_vox": dd, "depth_over_sigma": dd / sig,
                         "air_reference": ref, "enclosed_minus_open": 0.01 * dd})
    pd.DataFrame(rows).to_csv(tmp_path / "enclosed_air_difference.csv", index=False)
    S = summarise(None, tmp_path)["enclosed_air_difference"]
    assert S["granular | bead pack | depth 2"]["n"] == 0                  # 2 sigma: not compared
    assert S["trachyte | rock | depth 5"]["n"] == 1                       # 5 / 1.8 < 3 left out
    assert S["trachyte | rock | depth 5"]["n_below_min_depth_sigma"] == 1
    assert S["trachyte | rock | depth 8"]["air_reference"] == "outside air"
    assert S["granular | bead pack | depth 5"]["air_reference"] == "pore air"
    assert DEPTH_MIN_SIGMA == 3.0


# ------------------------------------------------------------------ two glasses
def test_mode_split_and_dense_split():
    c = np.linspace(0, 100, 101)
    hs = np.exp(-(c - 30) ** 2 / 50) + 0.5 * np.exp(-(c - 70) ** 2 / 50) + 0.01
    assert mode_split(c, hs, 30, 70) == pytest.approx(c[30 + int(np.argmin(hs[30:71]))])
    assert 45 < mode_split(c, hs, 70, 30) < 60
    rows = [{"role": "solid_main", "grey": 30.0, "I_solid": 30.0, "split_grey": np.nan},
            {"role": "solid_secondary", "grey": 70.0, "I_solid": 30.0, "split_grey": 52.0},
            {"role": "solid_secondary", "grey": 20.0, "I_solid": 30.0, "split_grey": 25.0}]
    assert dense_split(rows) == 52.0
    assert dense_split(rows[:1]) is None


def test_two_glass_classification_and_ordering():
    light, dense = 12000.0, 15000.0
    vol = balls((44, 44, 92), [(22, 22, 22), (22, 22, 68)], [19.0, 19.0], [light, dense], AIR)
    filt = median3(vol)
    T_light, T_dense = AIR + 0.5 * (light - AIR), AIR + 0.5 * (dense - AIR)
    pore = filt < T_light
    r, b = two_glass_voi(filt, pore, T_light, T_dense, split=13500.0)
    assert r["n_beads"] == 2 and r["n_dense"] == 1 and r["n_light"] == 1
    order = np.argsort(b["core_grey_median"])
    assert b["dense"][order].tolist() == [False, True]
    assert abs(r["core_grey_median_light"] - light) < 150 and abs(r["core_grey_median_dense"] - dense) < 150
    assert r["phi_a_T50"] < r["phi_c_beadwise"] < r["phi_b_T50_dense"]
    # only the voxels near the dense bead change class, and only from solid to pore
    assert r["vox_rethresholded_dense"] == pytest.approx((r["phi_c_beadwise"] - r["phi_a_T50"]) * pore.size, abs=0.5)
    assert 0 < r["vox_rethresholded_dense"] < r["vox_shell_dense"]


# ------------------------------------------------------------------ footprint
def test_footprint_constant_profile():
    coef = [254.5, 0.4, -0.3, -0.68, 0.48]
    slope = [0.002, 0.0005, 0.0005, -6e-5, 3e-4]
    f = footprint_phi(np.full(300, 0.37), coef, slope, (330.28, 325.08), (181.0, 174.0), 300, 300)
    assert f["phi_pred"] == pytest.approx(0.37, abs=1e-12)
    assert f["frac_outside_wall"] == 0


def test_footprint_geometry_of_a_centred_square():
    # circular wall of radius 250 about the centre of a 300-voxel square: the corners are 250 - 150 sqrt 2 away
    R, n = 250.0, 300
    c = (n - 1) / 2
    f = footprint_phi(np.arange(400, dtype=float), [R, 0, 0, 0, 0], np.zeros(5), (c, c), (0.0, 0.0), n, 10, sub=4)
    assert f["d_min_vox"] == pytest.approx(R - (n / 2 - 1 / 8) * np.sqrt(2), abs=1e-6)
    # profile phi(d) = floor(d): its footprint mean is the mean of floor(R - r) over the square
    u = (np.arange(n * 4) + 0.5) / 4 - 0.5 - c
    r = np.hypot(u[:, None], u[None, :])
    assert f["phi_pred"] == pytest.approx(np.floor(R - r).mean(), rel=1e-12)


def _bin(img, o, b, ny, nx):
    return img[o[0]:o[0] + ny * b, o[1]:o[1] + nx * b].reshape(ny, b, nx, b).mean(axis=(1, 3))


@pytest.mark.parametrize("b,crop0", [(2, (13, 6)), (2, (14, 7)), (3, (17, 10)), (1, (5, 9))])
def test_voi_section_origin_matches_the_binned_section(b, crop0):
    # the VOI (cut at crop0, binned by b) is the block of the wall stage's section grid at voi_section_origin
    raw = np.random.default_rng(0).random((90, 90))
    n = 12
    meta = {"bin": b, "crop_y0": crop0[0], "crop_x0": crop0[1], "centre_y": 45.0, "centre_x": 45.0}
    g = section_grid(meta, raw.shape)
    voi = _bin(raw, crop0, b, n, n)
    sec = _bin(raw, (g["origin_y"], g["origin_x"]), b, g["ny"], g["nx"])
    oy, ox = voi_section_origin(meta, (g["origin_y"], g["origin_x"]))
    assert oy == int(oy) and ox == int(ox)
    assert np.array_equal(voi, sec[int(oy):int(oy) + n, int(ox):int(ox) + n])


def test_footprint_from_matches_a_raw_pixel_computation():
    # circular wall about an axis 5 raw pixels off the VOI centre in y; binning 2 with phase o = (1, 0)
    b, n, vox, Rb = 2, 20, 0.05, 30.0
    crop0 = (13, 6)
    centre_raw = (crop0[0] + (n * b - 1) / 2 + 5, crop0[1] + (n * b - 1) / 2)
    meta = {"bin": b, "crop_y0": crop0[0], "crop_x0": crop0[1], "shape": [5, n, n],
            "centre_y": centre_raw[0], "centre_x": centre_raw[1]}
    g = section_grid(meta, (200, 200))
    w = {"voxel_mm": vox, "bin_origin_yx": [g["origin_y"], g["origin_x"]], "axis_yx_px": [g["cy"], g["cx"]],
         "wall_coef_px": [Rb, 0, 0, 0, 0], "wall_coef_slope_px": [0] * 5, "inner_radius_used_px": Rb,
         "phi_core": 0.4, "phi_section": 0.42}
    k = np.arange(40)
    prof = pd.DataFrame({"distance_from_wall_mm": (k + 0.5) * vox, "phi": k.astype(float)})   # phi(d) = floor(d)
    s = SimpleNamespace(id="T", group="granular", grain_mm=0.5)
    r = footprint_from(w, prof, meta, 0.4, s)
    assert r["voi_axis_offset_y_vox"] == pytest.approx(-2.5) and r["voi_axis_offset_x_vox"] == pytest.approx(0.0)
    # the same footprint in raw pixels: VOI pixel j (sub-point u) is at raw crop0 + (b - 1) / 2 + b (j + u)
    sub = r["sub_points_per_voxel"]
    u = (np.arange(n * sub) + 0.5) / sub - 0.5
    py, px = crop0[0] + (b - 1) / 2 + b * u, crop0[1] + (b - 1) / 2 + b * u
    dist = Rb - np.hypot(py[:, None] - centre_raw[0], px[None, :] - centre_raw[1]) / b    # binned voxels
    # (a frame error of half a binned voxel would move the mean by about 0.5, 2.5 %; 1e-4 allows for a
    # floor() flip at a distance that is an exact integer in one of the two float computations)
    assert r["phi_voi_pred_radial"] == pytest.approx(np.floor(dist).mean(), rel=1e-4)
    assert r["d_min_wall_mm"] == pytest.approx(dist.min() * vox, rel=1e-9)


# ------------------------------------------------------------------ synthetic packs
def test_relaxed_packing_has_no_overlap():
    x, it, worst = relax_packing(40, 150.0, 44.0 / 1.015, np.random.default_rng(3))
    assert worst <= 1e-4
    assert digitised_overlap(x, 150, 44.0 / 1.015 / 2, 44.0 / 1.015 / 2) == 0


def test_digitised_sphere_volume():
    occ = digitise(np.array([[30.2, 29.7, 30.4]]), 64, 20.0, 20.0)
    assert occ.sum() == pytest.approx(4 / 3 * np.pi * 20.0 ** 3, rel=0.01)


def test_spheroid_contact_distance():
    a, b = 1.25, 1.0
    ex, ey, ez = np.eye(3)
    assert spheroid_contact_distance(ex, ex, ex, a, b)[0] == pytest.approx(2 * a, rel=1e-6)     # end to end
    assert spheroid_contact_distance(ex, ey, ey, a, b)[0] == pytest.approx(2 * b, rel=1e-6)     # side by side
    assert spheroid_contact_distance(ex, ex, ey, a, b)[0] == pytest.approx(a + b, rel=1e-6)     # T
    assert spheroid_contact_distance(ex, ey, ez, a, b)[0] == pytest.approx(2 * b, rel=1e-6)     # crossed
    # random orientations: points of spheroid 1 enter spheroid 2 just inside the contact distance only
    rng = np.random.default_rng(5)

    def inside(p, c, e):
        q = p - c
        t = q @ e
        return t * t / a ** 2 + ((q * q).sum(1) - t * t) / b ** 2 <= 1

    for _ in range(5):
        u, e1, e2 = (v / np.linalg.norm(v) for v in rng.normal(size=(3, 3)))
        s = spheroid_contact_distance(u, e1, e2, a, b)[0]
        P = rng.uniform(-a, a, (400000, 3))
        P = P[inside(P, 0, e1)]
        assert inside(P, 0.98 * s * u, e2).any()
        assert not inside(P, 1.002 * s * u, e2).any()


def test_synthetic_spheres_measure_round_and_true_size():
    pore, info = synthetic_pack(150, 44.0, 1.0, seed=0, sigma=1.0, noise_sd=0.04 * 5800)
    rows, _ = measure_synthetic(pore, 0.05, "test")
    assert len(rows) >= 5
    assert info["relax_overlap_vox"] == 0 and info["solid_frac_true"] > 0.58
    aspect = np.median([r["aspect"] for r in rows])
    d_eq = np.median([r["d_eq_mm"] for r in rows]) / (44.0 * 0.05)
    assert aspect < 1.05
    assert abs(d_eq - 1) < 0.03


def test_synthetic_spheroids_are_detected_as_elongated_in_a_dense_pack():
    pore, info = synthetic_pack(150, 44.0, 1.25, seed=0, sigma=1.0, noise_sd=0.04 * 5800)
    assert info["relax_overlap_vox"] == 0 and info["relax_max_overlap_frac"] <= 1e-4
    assert info["solid_frac_true"] > 0.58                     # as dense as the sphere pack (real VOIs 0.60-0.70)
    rows, _ = measure_synthetic(pore, 0.05, "test")
    assert len(rows) >= 3
    assert np.median([r["aspect"] for r in rows]) > 1.15


def test_synthetic_params_aperture_and_two_glass_exclusion(tmp_path):
    e = pd.DataFrame({"id": ["CF4", "CF5", "CF10", "GT01"], "group": ["granular"] * 3 + ["coquina"],
                      "filter": "none", "sigma_cal_vox": [0.70, 0.70, 0.60, 1.1], "noise_frac": [0.04, 0.05, 0.07, 0.1],
                      "step_like": True})
    e.to_csv(tmp_path / "esf.csv", index=False)
    p = synthetic_params(tmp_path, exclude=("CF10",))
    assert p["sigma_esf_vox"] == pytest.approx(0.70) and p["noise_frac"] == pytest.approx(0.045)
    assert p["sigma_vox"] == pytest.approx(np.sqrt(0.49 - (1 - 1 / SYN_SUPER ** 2) / 12))
    assert synthetic_params(tmp_path)["sigma_esf_vox"] == pytest.approx(0.70)    # median of 0.7, 0.7, 0.6
