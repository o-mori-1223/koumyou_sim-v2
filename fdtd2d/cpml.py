"""Convolutional PML (CPML) profile generation for the z boundaries.

x is periodic (the grating period), so only z needs absorbing boundaries. This follows
the standard polynomial-graded CPML of Gedney / Taflove & Hagness, ch. 7: the spatial
derivative d/dz appearing in the curl equations is replaced by

    (1/kappa) d/dz + psi

with psi a per-cell auxiliary variable updated each timestep by a recursive convolution
psi <- b*psi + a*(dF/dz). b and a are precomputed here for both the E-node (integer z,
positions 0..n_e-1) and the staggered H-node grid (z + dz/2, positions 0..n_e-2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6


@dataclass
class CpmlProfile1D:
    kappa: np.ndarray
    b: np.ndarray
    a: np.ndarray


def _grading_shapes(z, n_total, pml_cells, m, m_alpha):
    """z: node positions in units of the E-grid spacing (0..n_total-1, half-integers OK).
    Returns (sigma_shape, kappa_shape, alpha_shape) in [0, 1], zero outside the PML slabs."""
    sig = np.zeros_like(z)
    kap = np.zeros_like(z)
    alp = np.zeros_like(z)
    for i, zi in enumerate(z):
        d_left = pml_cells - zi        # depth into the left PML, >0 while inside it
        d_right = zi - (n_total - 1 - pml_cells)  # depth into the right PML
        d = max(d_left, d_right, 0.0)
        if d <= 0.0:
            continue
        rho = min(d / pml_cells, 1.0)
        sig[i] = rho ** m
        kap[i] = rho ** m
        alp[i] = (1.0 - rho) ** m_alpha  # peaks at the PML/physical interface, 0 at the outer wall
    return sig, kap, alp


def make_cpml(n_e, dz, pml_cells, dt, eta_boundary=377.0,
              m=3, m_alpha=1, kappa_max=7.0, alpha_max=0.05, R0=1e-6):
    """Build CPML coefficient profiles for the E-node grid (length n_e) and the
    staggered H-node grid (length n_e - 1, located at z_i + dz/2)."""
    sigma_max = -(m + 1) * np.log(R0) / (2.0 * eta_boundary * pml_cells * dz)

    z_e = np.arange(n_e, dtype=float)
    z_h = np.arange(n_e - 1, dtype=float) + 0.5

    def build(z):
        sig_s, kap_s, alp_s = _grading_shapes(z, n_e, pml_cells, m, m_alpha)
        sigma = sigma_max * sig_s
        kappa = 1.0 + (kappa_max - 1.0) * kap_s
        alpha = alpha_max * alp_s
        b = np.exp(-(sigma / kappa + alpha) * dt / EPS0)
        denom = kappa * (sigma + kappa * alpha)
        a = np.zeros_like(sigma)
        nz = denom > 0
        a[nz] = sigma[nz] * (b[nz] - 1.0) / denom[nz]
        return CpmlProfile1D(kappa, b, a)

    return build(z_e), build(z_h)
