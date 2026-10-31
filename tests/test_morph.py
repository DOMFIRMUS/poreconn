"""Tests of the morphological and topological descriptors on shapes with known answers."""
import numpy as np
import pytest

from poreconn.morph import block_mean, connectivity, crofton_sv, euler, euler_density, spanning_mask


def ball(n, r, c=None):
    c = c or (n / 2 - 0.5,) * 3
    z, y, x = np.ogrid[:n, :n, :n]
    return (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2 <= r * r


def torus(n, R, r):
    z, y, x = np.ogrid[:n, :n, :n]
    c = n / 2 - 0.5
    q = np.sqrt((x - c) ** 2 + (y - c) ** 2) - R
    return q ** 2 + (z - c) ** 2 <= r * r


def test_euler_simple_shapes():
    assert euler(ball(40, 12)) == 1
    two = ball(60, 8, (15, 30, 30)) | ball(60, 8, (45, 30, 30))
    assert euler(two) == 2
    assert euler(torus(64, 18, 6)) == 0
    shell = ball(50, 20) & ~ball(50, 12)     # one component enclosing a cavity: 1 - 0 + 1
    assert euler(shell) == 2


def test_euler_of_tunnel_network_is_negative():
    n, ks = 41, range(5, 41, 10)             # a cubic lattice of channels, 4 x 4 x 4 nodes
    im = np.zeros((n, n, n), bool)
    for a in ks:
        for b in ks:
            im[:, a - 1:a + 2, b - 1:b + 2] = True
            im[a - 1:a + 2, :, b - 1:b + 2] = True
            im[a - 1:a + 2, b - 1:b + 2, :] = True
    # graph of 64 nodes and 3 * 16 * 3 = 144 internal edges: beta1 = 144 - 64 + 1 = 81, chi = 1 - 81
    assert euler(im) == -80


def test_percolation_of_a_straight_channel():
    im = np.zeros((30, 30, 30), bool)
    im[:, 14:17, 14:17] = True               # channel along z
    c = connectivity(im)
    assert c["perc_z"] and not c["perc_y"] and not c["perc_x"]
    assert c["n_clusters"] == 1
    assert c["phi_span_z"] == pytest.approx(im.mean())
    assert c["phi_isolated"] == 0.0


def test_isolated_pores_do_not_percolate():
    im = ball(40, 6, (10, 20, 20)) | ball(40, 6, (30, 20, 20))
    c = connectivity(im)
    assert not c["perc_any"]
    assert c["frac_isolated"] == pytest.approx(1.0)
    assert spanning_mask(im, 0).sum() == 0


def test_six_connectivity_does_not_join_diagonal_voxels():
    im = np.zeros((5, 5, 5), bool)
    im[0, 0, 0] = im[1, 1, 1] = True         # touch only at a corner
    assert connectivity(im)["n_clusters"] == 2


# voxel sets whose Euler characteristic depends on the adjacency: (voxels, chi 6-connected, chi 26-connected)
EDGE_PAIR = ([(0, 0, 0), (0, 1, 1)], 2, 1)                        # two voxels sharing only an edge
CORNER_PAIR = ([(0, 0, 0), (1, 1, 1)], 2, 1)                      # two voxels sharing only a corner
DIAMOND = ([(0, 0, 1), (0, 1, 0), (0, 1, 2), (0, 2, 1)], 4, 0)    # four voxels in a ring of edge contacts


@pytest.mark.parametrize("vox,chi6,chi26", [EDGE_PAIR, CORNER_PAIR, DIAMOND], ids=["edge", "corner", "diamond"])
def test_euler_depends_on_the_adjacency(vox, chi6, chi26):
    im = np.zeros((7, 7, 7), bool)                   # padded with solid: closed and interior counts agree
    for v in vox:
        im[tuple(np.add(v, 2))] = True
    assert euler(im) == euler(im, conn=6) == chi6
    assert euler(im, conn=26) == chi26
    assert euler_density(im) * 6 ** 3 == pytest.approx(chi6)
    assert euler_density(im, conn=26) * 6 ** 3 == pytest.approx(chi26)


@pytest.mark.parametrize("vox,chi6,chi26", [EDGE_PAIR, CORNER_PAIR, DIAMOND], ids=["edge", "corner", "diamond"])
def test_euler_density_depends_on_the_adjacency_in_a_periodic_window(vox, chi6, chi26):
    # one copy per 4^3 period; a window of 9 voxels holds exactly 2 periods of interior cells per axis,
    # while the copies on the far faces are cut by the window
    cell = np.zeros((4, 4, 4), bool)
    for v in vox:
        cell[v] = True
    win = np.tile(cell, (3, 3, 3))[:9, :9, :9]
    assert euler_density(win) == pytest.approx(chi6 / 64)
    assert euler_density(win, conn=26) == pytest.approx(chi26 / 64)


def test_crofton_surface_density_of_a_ball():
    r, n = 30, 80
    im = ball(n, r)
    sv = crofton_sv(im, dx=1.0)["S_V"]
    expected = 4 * np.pi * r ** 2 / n ** 3
    assert sv == pytest.approx(expected, rel=0.03)


def test_crofton_chords_of_slabs():
    im = np.zeros((40, 40, 40), bool)
    im[:, :, 10:20] = True                   # slab 10 voxels thick normal to x
    out = crofton_sv(im, dx=0.5)
    # Crofton chord 2 phi / P_L with the transition count over L - 1 links: 5 * 39 / 40
    assert out["chord_pore_x"] == pytest.approx(5.0 * 39 / 40)
    # no transitions along z and y: the mean chord is 2 phi / mean(P_L) = 4 phi / S_V, finite
    assert np.isinf(out["chord_pore_z"])
    assert out["chord_pore_mean"] == pytest.approx(3 * out["chord_pore_x"])
    assert out["chord_solid_mean"] == pytest.approx(3 * out["chord_solid_x"])


def test_crofton_mean_chords_match_the_surface_density():
    im = np.zeros((60, 60, 60), bool)
    for a in range(0, 60, 12):
        im[:, :, a:a + 4] = True             # slabs normal to x, period 12
    for a in range(0, 60, 20):
        im[a:a + 6] = True                   # slabs normal to z, period 20
    for a in range(0, 60, 15):
        im[:, a:a + 2] = True                # slabs normal to y, period 15
    out = crofton_sv(im, dx=0.1)
    phi = im.mean()
    assert out["chord_pore_mean"] == pytest.approx(4 * phi / out["S_V"])
    assert out["chord_solid_mean"] == pytest.approx(4 * (1 - phi) / out["S_V"])
    # harmonic, not arithmetic, mean of the per-axis chords
    hm = 3 / sum(1 / out[f"chord_pore_{a}"] for a in "zyx")
    assert out["chord_pore_mean"] == pytest.approx(hm)
    assert out["chord_pore_mean"] < 0.95 * np.mean([out[f"chord_pore_{a}"] for a in "zyx"])


def test_block_mean():
    v = np.arange(64, dtype=np.uint16).reshape(4, 4, 4)
    b = block_mean(v, 2)
    assert b.shape == (2, 2, 2)
    assert b[0, 0, 0] == pytest.approx(v[:2, :2, :2].mean())


def _periodic_field(n, seed, corr=3.0, q=0.0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, n, n))
    k = np.fft.fftfreq(n)
    kk = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
    f = np.real(np.fft.ifftn(np.fft.fftn(w) * np.exp(-kk * (2 * np.pi * corr) ** 2 / 2)))
    return f > np.quantile(f, 1 - q) if q else f > 0


def _wrap_density(u):
    """Exact Euler number density of the periodic structure u (count of all cells on the torus)."""
    from scipy import ndimage as ndi
    from poreconn.morph import _CONFIG
    from skimage.measure._regionprops_utils import EULER_COEFS3D_26
    p = np.pad(u.astype(np.int32), ((1, 0), (1, 0), (1, 0)), mode="wrap")
    XF = ndi.convolve(p, _CONFIG, mode="constant", cval=0)
    h = np.bincount(XF[1:, 1:, 1:].ravel(), minlength=256)
    return 0.125 * float(EULER_COEFS3D_26[::-1] @ h) / u.size


@pytest.mark.parametrize("q", [0.15, 0.5])
def test_euler_density_is_window_independent_on_a_periodic_structure(q):
    u = _periodic_field(40, seed=3, q=q)
    exact = _wrap_density(u)
    big = np.tile(u, (4, 4, 4))
    est = euler_density(big)
    assert est == pytest.approx(exact, rel=0.05, abs=2e-5)
    # in small windows the closed-window count chi/V is biased; the edge-corrected estimator is not
    big2 = np.tile(u, (2, 2, 2))
    wins = [big2[i:i + 20, j:j + 20, k:k + 20] for i in (0, 20, 40) for j in (0, 20, 40) for k in (0, 20, 40)]
    err_closed = abs(np.mean([euler(w) / w.size for w in wins]) - exact)
    err_corrected = abs(np.mean([euler_density(w) for w in wins]) - exact)
    assert err_corrected < err_closed


def test_euler_density_of_isolated_balls():
    # 8 balls fully inside: chi = 8 over the interior cells of the window
    n = 64
    im = np.zeros((n, n, n), bool)
    for c in ((16, 16, 16), (16, 16, 48), (16, 48, 16), (48, 16, 16), (48, 48, 16), (48, 16, 48), (16, 48, 48), (48, 48, 48)):
        im |= ball(n, 6, c)
    assert euler_density(im) * (n - 1) ** 3 == pytest.approx(8.0)


def test_critical_diameter_and_pore_size_of_ducts():
    from poreconn.topology import critical_diameter_vox, pore_size
    im = np.zeros((40, 30, 30), bool)
    im[:, 11:19, 11:19] = True                       # square duct of side 8 along z
    assert critical_diameter_vox(im, 0) == pytest.approx(8.0, rel=0.02)
    assert np.isnan(critical_diameter_vox(im, 1))  # no pore on the y faces
    z, y, x = np.ogrid[:40, :40, :40]
    cyl = (y - 19.5) ** 2 + (x - 19.5) ** 2 <= 7 ** 2  # cylinder of radius ~7 along z
    cyl = np.broadcast_to(cyl, (40, 40, 40)).copy()
    dc = critical_diameter_vox(cyl, 0)
    assert 12.5 <= dc <= 14.1
    ps_ = pore_size(cyl, dx=1.0)
    assert ps_["d90"] == pytest.approx(dc, rel=0.08)
