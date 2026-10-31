"""Pore-network permeability: SNOW-2 extraction (PoreSpy) and Stokes flow (OpenPNM).

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

The network is extracted once, from the union of the clusters that span the VOI along any axis, without
boundary pores (SNOW-2 boundary pores are thin slabs whose small inscribed diameter would throttle
every inlet and outlet conduit). Conduit conductances follow the 'spheres_and_cylinders' model
(Hagen-Poiseuille through each half-pore and the throat). Two size definitions are evaluated on the
same network: inscribed diameters (k_pnm) and equivalent sizes, i.e. volume-equivalent sphere diameters
for the pores and area-equivalent circle diameters for the throats (k_pnmeq). A throat may not be wider
than the pores it joins (required by the conduit model), so throat diameters are clipped to 0.99 times
the smaller pore diameter; the clipped fraction is reported. Only the hydraulic size factors are modelled
(conduit_lengths, robust to overlapping pore spheres); the extracted sizes are never regenerated.

Faces. PoreSpy measures each pore inside its own region padded with solid, so a pore cut by a VOI face
gets a truncated inscribed diameter and volume, and its throats are then clipped to it: the faces act as
walls and k depends on how many pores the window cuts. With faces="mirror" the six faces are symmetry
planes, as in the lattice Boltzmann and diffusion solvers: a pore whose region touches a face is
completed by its mirror image (inscribed diameter of the mirrored region, volume doubled per face,
centroid on the face plane), a throat whose interface lies across a face is completed likewise (area
doubled per face) and, since only the half inside the VOI belongs to it, its conductance is halved per
face. faces="wall" keeps the geometry as extracted (tags pnmw, pnmeqw).

For each axis the pressure is fixed on the pores whose regions touch the two faces normal to it;
network components that do not connect the two faces carry no flow and get a fixed pressure so that
the linear system stays non-singular.

    k = Q mu L / (A dP),   L and A the length and cross-section of the VOI.
"""
from __future__ import annotations

import numpy as np
from edt import edt
from scipy import ndimage as ndi
from scipy.sparse.csgraph import connected_components

from .morph import spanning_mask

TAG = {("inscribed", "mirror"): "pnm", ("equivalent", "mirror"): "pnmeq",
       ("inscribed", "wall"): "pnmw", ("equivalent", "wall"): "pnmeqw"}
DROP = ("pore.seed", "pore.max_size", "pore.diameter", "pore.volume", "throat.max_size",
        "throat.diameter", "throat.cross_sectional_area", "throat.total_volume",
        "throat.lens_volume", "throat.volume", "throat.diffusive_size_factors")


def conduit_lengths(network, pore_diameter="pore.diameter", throat_diameter="throat.diameter"):
    """Pore-throat-pore lengths of OpenPNM's 'spheres_and_cylinders' conduits, made robust to overlap.

    Where the two pore spheres meet before the throat starts, the conduit is split at the plane of their
    intersection and the throat length is zero; where one sphere contains the centre of the other (OpenPNM
    raises 'The pores overlap too much'), it is split in proportion to the radii.
    """
    c = network["throat.conns"]
    L = np.maximum(np.linalg.norm(network["pore.coords"][c[:, 0]] - network["pore.coords"][c[:, 1]], axis=1), 1e-30)
    D1, D2 = network[pore_diameter][c[:, 0]], network[pore_diameter][c[:, 1]]
    Dt = network[throat_diameter]
    L1 = np.sqrt(D1 ** 2 - Dt ** 2) / 2
    L2 = np.sqrt(D2 ** 2 - Dt ** 2) / 2
    Lt = L - (L1 + L2)
    m = Lt < 0
    R1, R2 = D1 / 2, D2 / 2
    x = (L ** 2 - R2 ** 2 + R1 ** 2) / (2 * L)
    x = np.where((x <= 0) | (x >= L), L * R1 / (R1 + R2), x)
    return np.where(m, x, L1), np.where(m, 1e-15, Lt), np.where(m, L - x, L2)


def hydraulic_size_factors(network, pore_diameter="pore.diameter", throat_diameter="throat.diameter"):
    """OpenPNM's 'spheres_and_cylinders' hydraulic size factors with the lengths of conduit_lengths."""
    c = network["throat.conns"]
    D1, D2 = network[pore_diameter][c[:, 0]], network[pore_diameter][c[:, 1]]
    Dt = network[throat_diameter]
    L1, Lt, L2 = conduit_lengths(network, pore_diameter, throat_diameter)
    F1 = 4 / (D1 ** 3 * np.pi ** 2) * (2 * D1 * L1 / (D1 ** 2 - 4 * L1 ** 2) + np.arctanh(2 * L1 / D1))
    F2 = 4 / (D2 ** 3 * np.pi ** 2) * (2 * D2 * L2 / (D2 ** 2 - 4 * L2 ** 2) + np.arctanh(2 * L2 / D2))
    Ft = Lt / (np.pi / 4 * Dt ** 2) ** 2
    s = 1 / (16 * np.pi ** 2 / (2 * np.pi))          # 1 / (16 pi^2 I), I = 1 / (2 pi) for circles
    return np.vstack([s / F1, s / Ft, s / F2]).T


def _face_pores(regions: np.ndarray, axis: int, side: int) -> np.ndarray:
    sl = [slice(None)] * 3
    sl[axis] = 0 if side == 0 else -1
    lab = np.unique(regions[tuple(sl)])
    return lab[lab > 0] - 1        # SNOW-2 pore index = region label - 1


def _face_pairs(face: np.ndarray) -> np.ndarray:
    """Encoded label pairs (lo * 2^32 + hi) of different regions that are 4-adjacent on a face slice."""
    out = []
    for a, b in ((face[:-1], face[1:]), (face[:, :-1], face[:, 1:])):
        m = (a != b) & (a > 0) & (b > 0)
        lo = np.minimum(a[m], b[m]).astype(np.int64)
        hi = np.maximum(a[m], b[m]).astype(np.int64)
        out.append(lo * 2 ** 32 + hi)
    return np.unique(np.concatenate(out))


def mirror_geometry(regions: np.ndarray, net: dict) -> dict:
    """Geometry of a SNOW-2 network with the VOI faces treated as symmetry planes (see module docstring).

    Returns arrays aligned with the network: pore inscribed and volume-equivalent diameters, pore
    coordinates, throat area-equivalent diameters and the conductance factor of each throat, all in the
    units of `net` (voxel size taken from the ratio of its volumes to the region voxel counts).
    """
    shape = regions.shape
    lab = np.asarray(net["pore.region_label"])
    npore = len(lab)
    nvox = np.bincount(regions.ravel(), minlength=int(lab.max()) + 1)[lab].astype(float)
    dx = float(np.cbrt(np.median(net["pore.region_volume"] / nvox)))
    touch = np.zeros((npore, 3, 2), bool)          # pore region touches face (axis, side)
    pos = {int(v): i for i, v in enumerate(lab)}
    for ax in range(3):
        for side in (0, 1):
            sl = [slice(None)] * 3
            sl[ax] = 0 if side == 0 else -1
            for v in np.unique(regions[tuple(sl)]):
                if v > 0 and int(v) in pos:
                    touch[pos[int(v)], ax, side] = True
    d_ins = np.array(net["pore.inscribed_diameter"], float)
    d_eq = np.array(net["pore.equivalent_diameter"], float)
    coords = np.array(net["pore.coords"], float)
    objs = ndi.find_objects(regions)
    for i in np.where(touch.any(axis=(1, 2)))[0]:
        s = objs[lab[i] - 1]
        im = regions[s] == lab[i]
        mult = 1
        for ax in range(3):
            lo, hi = touch[i, ax]
            if lo:
                im = np.concatenate([np.flip(im, ax), im], axis=ax)
            if hi:
                im = np.concatenate([im, np.flip(im, ax)], axis=ax)
            if lo or hi:
                mult *= 2
            if lo != hi:                            # centroid of the completed pore lies on the face
                coords[i, ax] = (-0.5 if lo else shape[ax] - 0.5) * dx
        d_ins[i] = 2 * float(edt(np.pad(im, 1)).max()) * dx
        d_eq[i] = 2 * (3 * mult * nvox[i] * dx ** 3 / (4 * np.pi)) ** (1 / 3)
    conns = np.asarray(net["throat.conns"])
    key = np.minimum(lab[conns[:, 0]], lab[conns[:, 1]]).astype(np.int64) * 2 ** 32 + \
        np.maximum(lab[conns[:, 0]], lab[conns[:, 1]]).astype(np.int64)
    nface = np.zeros(len(conns), int)                # number of faces each throat interface crosses
    for ax in range(3):
        for side in (0, 1):
            sl = [slice(None)] * 3
            sl[ax] = 0 if side == 0 else -1
            nface += np.isin(key, _face_pairs(regions[tuple(sl)]))
    area = np.array(net["throat.cross_sectional_area"], float) * 2.0 ** nface
    return {"pore.inscribed_diameter": d_ins, "pore.equivalent_diameter": d_eq, "pore.coords": coords,
            "throat.equivalent_diameter": np.sqrt(4 * area / np.pi),
            "throat.inscribed_diameter": np.array(net["throat.inscribed_diameter"], float),
            "throat.factor": 0.5 ** nface, "face_pore_frac": float(touch.any(axis=(1, 2)).mean())}


def network_permeability(pore: np.ndarray, dx_m: float, axes=(0, 1, 2),
                         sizes=("inscribed", "equivalent"), faces=("mirror", "wall")) -> dict:
    """k (m^2) along each percolating axis: keys k_<tag>_<axis>_m2, tag as in TAG.

    0.0 when the axis does not percolate; NaN, with the message in <tag>_error, when the solve fails.
    """
    import openpnm as op
    import porespy as ps
    spans = {a: spanning_mask(pore, a) for a in range(3)}
    m_any = spans[0] | spans[1] | spans[2]
    out = {"pnm_n_pores": 0, "pnm_n_throats": 0}
    for a in range(3):
        for sz in sizes:
            for fc in faces:
                out[f"k_{TAG[sz, fc]}_{'zyx'[a]}_m2"] = 0.0 if not spans[a].any() else np.nan
    if not m_any.any():
        return out
    ps.settings.tqdm["disable"] = True
    snow = ps.networks.snow2(m_any, boundary_width=0, voxel_size=dx_m, parallelization=None)
    pn = op.io.network_from_porespy(snow.network)
    am = pn.create_adjacency_matrix(fmt="csr", triu=False)
    ncomp, comp = connected_components(am, directed=False)
    out.update(pnm_n_pores=int(pn.Np), pnm_n_throats=int(pn.Nt), pnm_components=int(ncomp))
    extracted = {k: np.array(pn[k]) for k in ("pore.coords", "pore.inscribed_diameter", "pore.equivalent_diameter",
                                              "throat.inscribed_diameter", "throat.equivalent_diameter")}
    extracted["throat.factor"] = np.ones(pn.Nt)
    mirrored = mirror_geometry(snow.regions, snow.network) if "mirror" in faces else None
    if mirrored is not None:
        out["pnm_face_pore_frac"] = mirrored["face_pore_frac"]
    shape = np.array(snow.regions.shape) * dx_m
    geo = {"throat.hydraulic_size_factors": {"model": hydraulic_size_factors, "pore_diameter": "pore.diameter",
                                             "throat_diameter": "throat.diameter"}}
    ports = {a: (_face_pores(snow.regions, a, 0), _face_pores(snow.regions, a, 1)) for a in axes if spans[a].any()}
    for fc in faces:
        g = mirrored if fc == "mirror" else extracted
        for sz in sizes:
            tag = TAG[sz, fc]
            try:
                pn["pore.coords"] = g["pore.coords"]
                pn["pore.diameter"] = g[f"pore.{sz}_diameter"]
                dmin = pn["pore.diameter"][pn.conns].min(axis=1)
                pn["throat.diameter"] = np.minimum(g[f"throat.{sz}_diameter"], 0.99 * dmin)
                out[f"{tag}_throats_clipped"] = float(np.mean(g[f"throat.{sz}_diameter"] > 0.99 * dmin))
                for k in list(pn.models.keys()):
                    pn.models.pop(k)
                pn.add_model_collection(geo)
                pn.regenerate_models()
                water = op.phase.Water(network=pn)
                water.add_model("throat.hydraulic_conductance",
                                op.models.physics.hydraulic_conductance.generic_hydraulic)
                water.regenerate_models()
                water["throat.hydraulic_conductance"] = water["throat.hydraulic_conductance"] * g["throat.factor"]
                mu = float(water["pore.viscosity"][0])
                for a, (pin, pout) in ports.items():
                    ok = np.intersect1d(np.unique(comp[pin]), np.unique(comp[pout]))
                    pin_ok = pin[np.isin(comp[pin], ok)]
                    pout_ok = pout[np.isin(comp[pout], ok)]
                    dead = np.setdiff1d(np.where(~np.isin(comp, ok))[0], np.concatenate([pin_ok, pout_ok]))
                    sf = op.algorithms.StokesFlow(network=pn, phase=water)
                    sf.set_value_BC(pores=pin_ok, values=1.0)
                    sf.set_value_BC(pores=pout_ok, values=0.0)
                    if len(dead):
                        sf.set_value_BC(pores=dead, values=0.0)
                    sf.run()
                    Q = float(abs(sf.rate(pores=pin_ok, mode="group")[0]))
                    out[f"k_{tag}_{'zyx'[a]}_m2"] = Q * mu * shape[a] / float(np.prod(np.delete(shape, a)))
            except Exception as e:                      # e.g. OpenPNM "pores overlap too much"
                out[f"{tag}_error"] = f"{type(e).__name__}: {e}"[:300]
    pn["pore.coords"] = extracted["pore.coords"]
    return out
