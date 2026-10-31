"""Wall-effect helpers on synthetic data: median filter, section grid, half-way edge, harmonic wall fit.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT
"""
import numpy as np
from scipy import ndimage as ndi

from poreconn.wall import (N_SECTORS, _hist_median, detect_wall, fit_wall, grey_sums, half_way_edge, harmonics,
                           median_inplace, section_grid)


def test_median_inplace_equals_whole_volume_filter():
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 60000, (53, 17, 19)).astype(np.uint16)
    ref = ndi.median_filter(vol, size=3, mode="nearest")
    for chunk in (1, 5, 24, 60):
        assert np.array_equal(median_inplace(vol.copy(), 3, chunk=chunk), ref)


def test_section_grid_phase_and_centre():
    meta = {"bin": 2, "crop_y0": 363, "crop_x0": 350, "centre_y": 663.0, "centre_x": 650.5}
    g = section_grid(meta, (1324, 1323))
    assert (g["origin_y"], g["origin_x"]) == (1, 0)
    # binned pixel j is centred at raw origin + 2 j + 0.5
    assert np.isclose(1 + 2 * g["cy"] + 0.5, 663.0) and np.isclose(2 * g["cx"] + 0.5, 650.5)
    # the VOI's first binned pixel falls on the section grid
    assert (363 - g["origin_y"]) % 2 == 0 and (350 - g["origin_x"]) % 2 == 0


def test_hist_median():
    for v in ([3], [1, 2], [5, 1, 9, 9], [0, 7, 7, 2, 65535]):
        h = np.bincount(np.array(v), minlength=65536)
        assert _hist_median(h) == np.median(v)


def test_half_way_edge_on_a_ramp():
    rc = (np.arange(400) + 0.5) * 0.25
    m = np.where(rc < 50.3, 12000.0, 15000.0)
    e, lo, hi = half_way_edge(m, rc, (45, 52), (55, 60))
    assert lo == 12000 and hi == 15000 and abs(e - 50.3) < 0.25


def test_fit_wall_recovers_offset_and_ovality():
    t = -np.pi + (np.arange(N_SECTORS) + 0.5) * 2 * np.pi / N_SECTORS
    c = np.array([254.0, 1.2, -0.7, 0.5, -0.3])
    e = harmonics(t) @ c
    e[3] += 4.0                                    # one outlier
    cf, rms, used = fit_wall(t, e)
    assert np.allclose(cf, c, atol=1e-6) and used == N_SECTORS - 1


def synthetic_column(ny=300, nx=300, cy=150.3, cx=149.6, R=110.0, wall=30.0, rng=None):
    """Bead pack inside an acrylic tube: glass 21000, air 12000, acrylic 15000, outside air 11000."""
    rng = rng or np.random.default_rng(1)
    nz = 40
    y, x = np.indices((ny, nx))
    r = np.hypot(y - cy, x - cx)
    vol = np.full((nz, ny, nx), 12000.0)
    zz, yy, xx = np.indices((25, 25, 25)) - 12
    ball = zz ** 2 + yy ** 2 + xx ** 2 < 144
    # random beads of radius 12 kept at least 1 voxel from the wall
    for _ in range(400):
        rb = rng.uniform(0, R - 13)
        a = rng.uniform(0, 2 * np.pi)
        bz, by, bx = int(rng.uniform(0, nz)), int(cy + rb * np.sin(a)), int(cx + rb * np.cos(a))
        bb = ball[max(12 - bz, 0):25 - max(bz + 13 - nz, 0)]
        vol[max(bz - 12, 0):bz + 13, by - 12:by + 13, bx - 12:bx + 13][bb] = 21000.0
    vol[:, (r >= R) & (r < R + wall)] = 15000.0
    vol[:, r >= R + wall] = 11000.0
    vol = ndi.gaussian_filter(vol, 0.8) + rng.normal(0, 150, vol.shape)
    return np.clip(vol, 0, 65535).astype(np.uint16)


def test_detect_wall_finds_the_tube_and_not_the_air_gap():
    vol = synthetic_column()
    vox = 0.05
    # VOI axis estimate off by (1.5, -1) voxels
    det = detect_wall(*grey_sums(vol, 151.8, 148.6, 1.3 * 110), 110.0, vox)
    assert det["ok"] and "rms" in det
    det2 = detect_wall(*grey_sums(vol, 151.8 + det["coef"][2], 148.6 + det["coef"][1], 1.3 * 110), 110.0, vox)
    assert abs(det2["coef"][0] - 110.0) < 0.3
    assert abs(151.8 + det["coef"][2] + det2["coef"][2] - 150.3) < 0.3
    assert abs(148.6 + det["coef"][1] + det2["coef"][1] - 149.6) < 0.3


def test_detect_wall_fails_without_a_wall():
    rng = np.random.default_rng(2)
    vol = rng.integers(10000, 22000, (10, 120, 120)).astype(np.uint16)
    det = detect_wall(*grey_sums(vol, 60, 60, 70), 50.0, 0.05)
    assert not det["ok"] and det["reason"]
