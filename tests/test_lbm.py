"""Tests of the TRT lattice Boltzmann permeability solver."""
import sys
import types

import numpy as np
import pandas as pd
import pytest
import tifffile

from poreconn import lbm
from poreconn.config import Config, read_json
from poreconn.lbm import permeability


def square_duct_k(a):
    n = np.arange(1, 400, 2)
    return a * a / 12.0 * (1 - 192 / np.pi ** 5 * np.sum(np.tanh(n * np.pi / 2) / n ** 5))


@pytest.mark.parametrize("a", [8, 16])
def test_square_duct(a):
    im = np.ones((6, a, a), bool)
    out = permeability(im, 0, mirror=False, lateral="wall", tol=1e-8, maxiter=200000, check=100)
    assert out["converged"]
    assert out["k_lu"] == pytest.approx(square_duct_k(a), rel=0.02 if a == 8 else 0.01)
    assert out["ok"] and out["flux_cv"] < 1e-12


def test_viscosity_independence():
    im = np.ones((4, 10, 10), bool)
    k1 = permeability(im, 0, tau=0.8, mirror=False, lateral="wall", tol=1e-8, maxiter=200000, check=100)["k_lu"]
    k2 = permeability(im, 0, tau=1.4, mirror=False, lateral="wall", tol=1e-8, maxiter=200000, check=100)["k_lu"]
    assert k1 == pytest.approx(k2, rel=1e-3)


def test_duct_in_a_solid_block_and_axis():
    im = np.zeros((10, 20, 20), bool)
    im[:, 5:15, 5:15] = True                    # 10 x 10 duct inside a 20 x 20 cross-section
    out = permeability(im, 0, tol=1e-8, maxiter=200000, check=100)
    expected = square_duct_k(10) * (100 / 400)  # superficial: duct area / total area
    assert out["k_lu"] == pytest.approx(expected, rel=0.02)
    imx = np.ascontiguousarray(np.moveaxis(im, 0, 2))   # same duct along x
    outx = permeability(imx, 2, tol=1e-8, maxiter=200000, check=100)
    assert outx["k_lu"] == pytest.approx(out["k_lu"], rel=1e-6)


def test_no_percolation_gives_zero():
    im = np.zeros((10, 8, 8), bool)
    im[2:5, 2:5, 2:5] = True
    assert permeability(im, 0)["k"] == 0.0


@pytest.mark.parametrize("a", [6, 12])
def test_slit_with_symmetric_lateral_faces(a):
    # plates normal to y, open (mirror-symmetric) in x: an infinite slit of aperture a. With TRT and
    # Lambda = 3/16 the nodal velocities are the exact parabola; summing them over the a nodes (the
    # voxel average) adds the midpoint-rule term 1/24, so k = (a^2/12 + 1/24) a / (a + 2) exactly.
    im = np.zeros((4, a + 2, 7), bool)
    im[:, 1:a + 1, :] = True
    out = permeability(im, 0, mirror=False, tol=1e-10, maxiter=400000, check=100)
    assert out["k_lu"] == pytest.approx((a * a / 12 + 1 / 24) * a / (a + 2), rel=1e-6)
    assert out["ok"] and out["flux_cv"] < 1e-12


def test_wall_and_mirror_agree_when_the_pore_is_enclosed():
    im = np.zeros((10, 20, 20), bool)
    im[:, 5:15, 5:15] = True
    k_w = permeability(im, 0, lateral="wall", tol=1e-8, maxiter=200000, check=100)["k_lu"]
    k_m = permeability(im, 0, lateral="mirror", tol=1e-8, maxiter=200000, check=100)["k_lu"]
    assert k_w == pytest.approx(k_m, rel=1e-9)


def two_ducts(a1, a2, part):
    """Square ducts of sides a1 and a2 along axis 0 sharing only the edge (y, x) = (a1 + 1/2, a1 + 1/2)."""
    im = np.zeros((4, a1 + a2 + 2, a1 + a2 + 2), bool)
    if part in (1, 3):
        im[:, 1:1 + a1, 1:1 + a1] = True
    if part in (2, 3):
        im[:, 1 + a1:1 + a1 + a2, 1 + a1:1 + a1 + a2] = True
    return im


@pytest.mark.parametrize("lateral", ["mirror", "wall"])
def test_ducts_touching_along_an_edge_add_up(lateral):
    # the pore phase is 6-connected: two ducts sharing an edge are two independent channels
    k = [permeability(two_ducts(6, 4, p), 0, mirror=False, lateral=lateral, tol=1e-11, maxiter=400000,
                      check=100)["k_lu"] for p in (1, 2, 3)]
    assert abs(k[2] - k[0] - k[1]) < 1e-6 * (k[0] + k[1])


def stepped_slits(a, n, part):
    """Two slits along axis 0 separated by a one-voxel wall that steps by one voxel at n // 2.

    The slits touch only along the two edges (normal to the flow) where the wall steps.
    """
    im = np.zeros((n, 2 * a + 2, 3), bool)
    for z in range(n):
        w = a + 1 if z < n // 2 else a
        if part in (1, 3):
            im[z, 1:w] = True
        if part in (2, 3):
            im[z, w + 1:2 * a + 1] = True
    return im


@pytest.mark.parametrize("mirror", [True, False])
def test_edge_contacts_normal_to_the_flow_carry_no_flow(mirror):
    # D3Q19 diagonal links joining the slits across those edges added 0.3-0.5 % to k before the fix
    k = [permeability(stepped_slits(6, 8, p), 0, mirror=mirror, tol=1e-11, maxiter=400000,
                      check=100)["k_lu"] for p in (1, 2, 3)]
    assert abs(k[2] - k[0] - k[1]) < 1e-6 * (k[0] + k[1])


def orifice():
    im = np.ones((16, 10, 10), bool)
    im[8] = False
    im[8, 4:6, 4:6] = True                      # plate with a 2 x 2 hole across a square channel
    return im


def test_flux_balance_flags_runs_stopped_too_early():
    loose = permeability(orifice(), 0, tol=0.2, check=20)
    assert loose["converged"] and loose["flux_cv"] > loose["flux_tol"] and not loose["ok"]
    tight = permeability(orifice(), 0, tol=1e-8, maxiter=200000, check=100)
    assert tight["ok"] and tight["flux_cv"] < 1e-6


# ---------------------------------------------------------------- stage
def _stage_case(tmp_path):
    """Config with two synthetic samples: A (central-cube domain) and B (whole-VOI domain)."""
    (tmp_path / "config.yaml").write_text(
        "paths: {raw_root: raw, voi_dir: voi, lab_csv: lab.csv, legacy_csv: legacy.csv, outputs: outputs}\n"
        "transport: {fd_tol: 1.0e-8, fd_maxiter: 20000, lbm_edge: 12, lbm_tau: 1.0, lbm_maxiter: 40000,"
        " lbm_tol: 1.0e-6}\n"
        "samples:\n"
        "  - {id: A, group: granular, label: A, raw: a, voxel_um: 25.0, bin: 2}\n"
        "  - {id: B, group: limestone, label: B, raw: b, voxel_um: 25.0, bin: 1}\n", encoding="utf-8")
    cfg = Config(tmp_path / "config.yaml")
    a = np.zeros((24, 24, 24), bool)
    a[:, 8:16, 8:16] = True                     # z duct through the central 12^3 cube
    a[2:4, :, 8:16] = True                      # y duct outside the cube: the cube does not span y
    b = np.zeros((12, 12, 12), bool)
    b[:, 5:7, 5:7] = True
    grey = {}
    for sid, pore in (("A", a), ("B", b)):
        seg = cfg.outdir("segment")
        tifffile.imwrite(seg / f"{sid}_v0_pore.tif", pore.astype(np.uint8))
        (seg / f"{sid}_v0_seg.json").write_text('{"T50": 500.0}', encoding="utf-8")
        grey[sid] = np.where(pore, 0, 1000).astype(np.uint16)     # stands for the filtered grey VOI
    return cfg, grey


def test_stage_rows_crop_axes_whole2_and_csv(tmp_path, monkeypatch):
    cfg, grey = _stage_case(tmp_path)
    monkeypatch.setattr("poreconn.segment.load_filtered", lambda cfg, sid, k: grey[sid])
    monkeypatch.setattr(lbm, "MAX_NODES", 100)   # A -> central cube; B (2 x 48 nodes) -> whole VOI

    def fake_pnm(pore, dx_m, axes=(0, 1, 2)):
        if pore.sum() > 48:                     # A's cube
            raise RuntimeError("extraction failed")
        return {"k_pnm_z_m2": 1e-12, "k_pnmeq_z_m2": 2e-12, "k_pnmw_z_m2": 3e-12, "k_pnmeqw_z_m2": np.nan,
                "pnmeqw_error": "singular", "pnm_n_pores": 3, "pnm_n_throats": 2}

    monkeypatch.setitem(sys.modules, "poreconn.pnm", types.SimpleNamespace(network_permeability=fake_pnm))
    out = cfg.outdir("lbm")
    lbm.run(cfg, only=["B"], log=lambda *_: None)
    assert pd.read_csv(out / "lbm.csv").id.tolist() == ["B"]
    lbm.run(cfg, only=["A"], log=lambda *_: None)
    assert pd.read_csv(out / "lbm.csv").id.tolist() == ["A", "A", "B"]   # config order, partial runs

    za, ya = read_json(out / "A_lbm.json")
    assert (za["axis"], ya["axis"]) == ("z", "y") and za["domain"] == "central 12^3"
    assert za["main_percolates"] and ya["main_percolates"]
    assert za["crop_percolates"] and za["ok"] and za["k_lbm_m2"] > 0 and za["fd_converged"]
    assert not ya["crop_percolates"] and not ya["ok"] and np.isnan(ya["k_lbm_m2"]) and np.isnan(ya["F"])
    for r in (za, ya):
        assert r["whole2_ok"] and r["k_lbm_whole2_m2"] > 0 and r["whole2_voxel_mm"] == pytest.approx(0.1)
        for t in lbm.TAGS:
            assert np.isnan(r[f"k_{t}_m2"]) and r[f"{t}_error"] == "RuntimeError: extraction failed"

    (zb,) = read_json(out / "B_lbm.json")
    assert zb["domain"] == "full VOI" and zb["crop_percolates"] and np.isnan(zb["k_lbm_whole2_m2"])
    assert zb["k_pnm_mD"] == pytest.approx(1e-12 / lbm.MD) and zb["k_pnmw_mD"] == pytest.approx(3e-12 / lbm.MD)
    assert np.isnan(zb["k_pnmeqw_m2"]) and zb["pnmeqw_error"] == "singular" and zb["pnm_error"] == ""
    assert zb["pnm_n_pores"] == 3
    assert {"fd_iterations", "fd_residual", "fd_flux_balance", "fd_converged", "flux_cv"} <= set(zb)
