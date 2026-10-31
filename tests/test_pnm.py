"""Pore networks: conduit model and treatment of the window faces.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT
"""
import numpy as np
import pytest

openpnm = pytest.importorskip("openpnm")
porespy = pytest.importorskip("porespy")

from poreconn.pnm import conduit_lengths, hydraulic_size_factors, mirror_geometry, network_permeability  # noqa: E402
from poreconn.validation import lattice  # noqa: E402


class _Net(dict):
    pass


def _toy(coords, dp, dt):
    net = _Net()
    net["throat.conns"] = np.array([[0, 1]])
    net["pore.coords"] = np.array(coords, float)
    net["pore.diameter"] = np.array(dp, float)
    net["throat.diameter"] = np.array([dt], float)
    return net


def test_size_factors_match_openpnm_without_overlap():
    porespy.settings.tqdm["disable"] = True
    snow = porespy.networks.snow2(lattice(2), boundary_width=0, voxel_size=1.0, parallelization=None)
    pn = openpnm.io.network_from_porespy(snow.network)
    pn["pore.diameter"] = pn["pore.inscribed_diameter"]
    dmin = pn["pore.diameter"][pn.conns].min(axis=1)
    pn["throat.diameter"] = np.minimum(pn["throat.inscribed_diameter"], 0.99 * dmin)
    ref = openpnm.models.geometry.hydraulic_size_factors.spheres_and_cylinders(pn)
    assert np.allclose(hydraulic_size_factors(pn), ref, rtol=1e-12)


@pytest.mark.parametrize("coords, dp, dt", [
    ([[0, 0, 0], [10, 0, 0]], [8, 8], 2),       # separate spheres: cylinder of length 10 - 2 sqrt(15)
    ([[0, 0, 0], [10, 0, 0]], [14, 14], 2),     # intersecting spheres: split at the intersection plane
    ([[0, 0, 0], [3, 0, 0]], [20, 4], 1),       # one sphere contains the other's centre
])
def test_conduit_lengths_add_up_and_stay_inside_the_pores(coords, dp, dt):
    net = _toy(coords, dp, dt)
    L1, Lt, L2 = (x[0] for x in conduit_lengths(net))
    L = np.linalg.norm(np.diff(net["pore.coords"], axis=0))
    assert L1 + Lt + L2 == pytest.approx(L)
    assert 0 < L1 < dp[0] / 2 and 0 < L2 < dp[1] / 2 and Lt > 0
    assert np.all(np.isfinite(hydraulic_size_factors(net)))


def test_mirror_faces_make_the_network_independent_of_the_window():
    # every face of lattice(n) is a mirror plane of the lattice, so the exact k cannot depend on n
    k = {n: network_permeability(lattice(n), 1.0, axes=(0,)) for n in (2, 3)}
    for tag in ("pnm", "pnmeq"):
        assert k[3][f"k_{tag}_z_m2"] == pytest.approx(k[2][f"k_{tag}_z_m2"], rel=0.01)
    # the faces as walls are not: the share of cut pores changes with the window
    assert k[3]["k_pnmw_z_m2"] / k[2]["k_pnmw_z_m2"] > 1.1


def test_mirror_geometry_completes_a_cut_pore():
    porespy.settings.tqdm["disable"] = True
    pore = lattice(2)
    snow = porespy.networks.snow2(pore, boundary_width=0, voxel_size=1.0, parallelization=None)
    g = mirror_geometry(snow.regions, snow.network)
    cut = g["pore.inscribed_diameter"] > snow.network["pore.inscribed_diameter"] + 1e-9
    assert cut.any() and g["face_pore_frac"] > 0
    # pores touching no face keep their extracted geometry
    same = ~np.isclose(g["pore.equivalent_diameter"], snow.network["pore.equivalent_diameter"])
    assert np.all(g["pore.inscribed_diameter"][~same] == snow.network["pore.inscribed_diameter"][~same])
    assert np.all((g["throat.factor"] == 1) | (g["throat.factor"] == 0.5) | (g["throat.factor"] == 0.25)
                  | (g["throat.factor"] == 0.125))
