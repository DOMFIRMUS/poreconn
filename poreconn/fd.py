"""Steady diffusion (Laplace) in the resolved pore space: formation factor and tortuosity factor.

Author: Jean Firmino Cardoso (jean.firmino@ufpe.br) - poreconn
License: MIT

Finite-volume discretisation on the voxel grid: unknown concentrations on the pore voxels of the
clusters that span the VOI along the chosen axis, unit conductance between face-neighbouring pore
voxels, c = 1 on the pore voxels of the inlet slice and c = 0 on those of the outlet slice, no flux
through the solid and through the lateral faces. The linear system is solved by Jacobi-preconditioned
conjugate gradients (numba). In a domain of two slices there are no unknowns and the flux is carried
by the direct contacts, one unit per voxel that is pore in both slices. With J the diffusive flux
through the VOI cross-section A (in voxel units, solid included) and L = n - 1 the distance between the
Dirichlet slices,

    D_eff / D_0 = J L / A,   F = D_0 / D_eff,   tortuosity factor tau = phi F   (phi: total porosity),

so that F = 1 for an empty box and F = 1 / phi for straight channels. Archie's exponent is
m = -ln F / ln phi.
"""
from __future__ import annotations

import numba as nb
import numpy as np

from .morph import spanning_mask


@nb.njit(cache=True)
def _build(index, inlet_mask, outlet_mask):
    """Neighbour table of the unknown voxels: >=0 unknown, -1 none, -2 inlet (c=1), -3 outlet (c=0)."""
    nz, ny, nx = index.shape
    n = int(index.max()) + 1
    nbr = np.full((n, 6), -1, np.int32)
    dz = (-1, 1, 0, 0, 0, 0)
    dy = (0, 0, -1, 1, 0, 0)
    dx = (0, 0, 0, 0, -1, 1)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                i = index[z, y, x]
                if i < 0:
                    continue
                for d in range(6):
                    zz, yy, xx = z + dz[d], y + dy[d], x + dx[d]
                    if zz < 0 or zz >= nz or yy < 0 or yy >= ny or xx < 0 or xx >= nx:
                        continue
                    j = index[zz, yy, xx]
                    if j >= 0:
                        nbr[i, d] = j
                    elif inlet_mask[zz, yy, xx]:
                        nbr[i, d] = -2
                    elif outlet_mask[zz, yy, xx]:
                        nbr[i, d] = -3
    return nbr


@nb.njit(parallel=True, cache=True)
def _matvec(nbr, deg, x, out):
    n = x.shape[0]
    for i in nb.prange(n):
        s = deg[i] * x[i]
        for d in range(6):
            j = nbr[i, d]
            if j >= 0:
                s -= x[j]
        out[i] = s


@nb.njit(parallel=True, cache=True)
def _dot(a, b):
    s = 0.0
    for i in nb.prange(a.shape[0]):
        s += a[i] * b[i]
    return s


@nb.njit(cache=True)
def _cg(nbr, deg, b, x, tol, maxiter):
    n = b.shape[0]
    r = np.empty(n)
    Ap = np.empty(n)
    _matvec(nbr, deg, x, Ap)
    for i in range(n):
        r[i] = b[i] - Ap[i]
    z = r / deg
    p = z.copy()
    rz = _dot(r, z)
    bn = np.sqrt(_dot(b, b))
    if bn == 0.0:
        return x, 0, 0.0
    it = 0
    res = np.sqrt(_dot(r, r)) / bn
    while it < maxiter and res > tol:
        _matvec(nbr, deg, p, Ap)
        alpha = rz / _dot(p, Ap)
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
            z[i] = r[i] / deg[i]
        rz_new = _dot(r, z)
        beta = rz_new / rz
        rz = rz_new
        for i in range(n):
            p[i] = z[i] + beta * p[i]
        it += 1
        res = np.sqrt(_dot(r, r)) / bn
    return x, it, res


def formation_factor(pore: np.ndarray, axis: int, tol: float = 1e-6, maxiter: int = 20000) -> dict:
    """Formation factor of `pore` (True = pore) along `axis` (0 = z, 1 = y, 2 = x)."""
    p = np.moveaxis(pore, axis, 0)
    m = spanning_mask(np.ascontiguousarray(p), 0)
    nz = p.shape[0]
    A = p.shape[1] * p.shape[2]
    phi = float(pore.mean())
    out = {"axis": "zyx"[axis], "phi": phi, "phi_span": float(m.mean())}
    if not m.any():
        out.update(F=np.inf, tau=np.inf, m_archie=np.nan, iterations=0, residual=np.nan, flux_balance=np.nan)
        return out
    inlet = np.zeros_like(m)
    outlet = np.zeros_like(m)
    inlet[0] = m[0]
    outlet[-1] = m[-1]
    unknown = m & ~inlet & ~outlet
    index = np.full(m.shape, -1, np.int32)
    index[unknown] = np.arange(int(unknown.sum()), dtype=np.int32)
    nbr = _build(index, inlet, outlet)
    deg = (nbr != -1).sum(axis=1).astype(np.float64)
    b = (nbr == -2).sum(axis=1).astype(np.float64)
    x = np.zeros(len(deg))
    x, it, res = _cg(nbr, deg, b, x, tol, maxiter)
    # flux leaving the inlet and entering the outlet (should agree)
    j_in = float(((nbr == -2).sum(axis=1) * (1.0 - x)).sum())
    j_out = float(((nbr == -3).sum(axis=1) * x).sum())
    # direct inlet-to-outlet contacts (unit conductance, unit drop; only in a 2-slice domain)
    if nz == 2:
        j_dir = float(np.count_nonzero(inlet[0] & outlet[1]))
        j_in += j_dir
        j_out += j_dir
    D = 0.5 * (j_in + j_out) * (nz - 1) / A
    F = 1.0 / D if D > 0 else np.inf
    out.update(F=F, tau=phi * F, m_archie=float(-np.log(F) / np.log(phi)) if 0 < phi < 1 else np.nan,
               iterations=int(it), residual=float(res),
               flux_balance=float(abs(j_in - j_out) / max(j_in, 1e-300)))
    return out
