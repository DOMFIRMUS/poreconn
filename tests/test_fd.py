"""Tests of the diffusion solver against exact answers."""
import numpy as np
import pytest

from poreconn.fd import formation_factor


def test_empty_box_has_unit_formation_factor():
    im = np.ones((20, 12, 12), bool)
    out = formation_factor(im, 0)
    assert out["F"] == pytest.approx(1.0, rel=1e-5)
    assert out["flux_balance"] < 1e-5


def test_straight_channels_give_inverse_porosity():
    im = np.zeros((30, 20, 20), bool)
    im[:, 2:6, 2:6] = True
    im[:, 10:13, 12:18] = True
    out = formation_factor(im, 0)
    assert out["F"] == pytest.approx(1 / im.mean(), rel=1e-5)
    assert out["tau"] == pytest.approx(1.0, rel=1e-5)


def test_no_percolation_gives_infinite_F():
    im = np.zeros((20, 10, 10), bool)
    im[2:8, 3:6, 3:6] = True
    assert np.isinf(formation_factor(im, 0)["F"])


def test_axis_selection():
    im = np.zeros((10, 10, 30), bool)
    im[3:7, 3:7, :] = True                    # channel along x
    assert formation_factor(im, 2)["F"] == pytest.approx(1 / im.mean(), rel=1e-5)
    assert np.isinf(formation_factor(im, 0)["F"])


def _reference_F(im):
    """Same discretisation assembled with scipy and solved directly (axis 0, spanning cluster only)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    from poreconn.morph import spanning_mask
    m = spanning_mask(im, 0)
    nz = im.shape[0]
    fixed = np.zeros_like(m)
    fixed[0], fixed[-1] = m[0], m[-1]
    unk = m & ~fixed
    idx = -np.ones(m.shape, int)
    idx[unk] = np.arange(unk.sum())
    rows, cols, vals = [], [], []
    b = np.zeros(unk.sum())
    for (z, y, x) in zip(*np.nonzero(unk)):
        i = idx[z, y, x]
        for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            zz, yy, xx = z + dz, y + dy, x + dx
            if not (0 <= zz < nz and 0 <= yy < im.shape[1] and 0 <= xx < im.shape[2]) or not m[zz, yy, xx]:
                continue
            rows.append(i); cols.append(i); vals.append(1.0)
            if unk[zz, yy, xx]:
                rows.append(i); cols.append(idx[zz, yy, xx]); vals.append(-1.0)
            elif zz == 0:
                b[i] += 1.0
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(b), len(b)))
    c = spsolve(A, b) if len(b) else np.zeros(0)
    j = sum(1.0 - c[idx[1, y, x]] for (y, x) in zip(*np.nonzero(m[0])) if unk[1, y, x])
    j += np.count_nonzero(m[0] & m[1] & fixed[1]) if nz == 2 else 0
    return 1.0 / (j * (nz - 1) / (im.shape[1] * im.shape[2]))


def test_against_direct_sparse_solution_on_random_media():
    rng = np.random.default_rng(1)
    for _ in range(3):
        im = rng.random((16, 10, 10)) < 0.62
        out = formation_factor(im, 0, tol=1e-12)
        if np.isinf(out["F"]):
            continue
        assert out["F"] == pytest.approx(_reference_F(im), rel=1e-6)
        assert out["flux_balance"] < 1e-6


def test_two_slice_domain_uses_direct_contacts():
    assert formation_factor(np.ones((2, 12, 12), bool), 0)["F"] == pytest.approx(1.0)
    im = np.zeros((2, 20, 20), bool)
    im[:, 2:6, 2:6] = True
    im[:, 10:13, 12:18] = True
    im[0, 15:18, 2:5] = True                  # dead end on the inlet slice only
    out = formation_factor(im, 0)
    assert out["F"] == pytest.approx(20 * 20 / (16 + 18))
    assert out["flux_balance"] == 0.0
    rng = np.random.default_rng(2)
    for _ in range(3):
        im = rng.random((2, 10, 10)) < 0.5
        out = formation_factor(im, 0)
        assert np.isfinite(out["F"])
        assert out["F"] == pytest.approx(_reference_F(im), rel=1e-12)
