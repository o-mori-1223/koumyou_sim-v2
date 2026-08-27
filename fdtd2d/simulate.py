"""Top-level broadband reflectance/transmittance spectrum via 2D dispersive FDTD.

Two time-domain runs are needed per structure (standard scattered-field technique):
  1. "reference": the whole grid filled with the (non-dispersive) ambient medium, to
     record the incident pulse's own spectrum at the monitor plane.
  2. "structure": the real geometry, to record the *total* field at the same plane.
Reflected field = total - incident (both are frequency-domain phasors, so this is exact
even though the two pulses overlap in time). Reflectance/transmittance are then the
ratio of time-averaged Poynting flux, S = -0.5*Re(Ey * conj(Hx)) integrated over x.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import FDTD2D, gaussian_pulse_source, C0
from .geometry import build_grid, GridResult


@dataclass
class SpectrumResult:
    wl_nm: np.ndarray
    R: np.ndarray
    T: np.ndarray
    A: np.ndarray
    grid: GridResult


def _flux(Ey_dft, Hx_dft, dx):
    S = -0.5 * np.real(Ey_dft * np.conj(Hx_dft))
    return S.sum(axis=0) * dx


def run_spectrum(columns, nk_subst_name, n_env, wl_min_nm, wl_max_nm, n_wl=41,
                  dx_nm=8.0, n_poles=6, pml_cells=15, nk_dir='data/nk',
                  courant_pad_periods=4.0, verbose=False) -> SpectrumResult:
    grid = build_grid(columns, nk_subst_name, n_env, wl_min_nm, wl_max_nm, dx_nm,
                       n_poles=n_poles, pml_cells=pml_cells, nk_dir=nk_dir)
    wl_nm = np.linspace(wl_min_nm, wl_max_nm, n_wl)
    omegas = 2 * np.pi * C0 / (wl_nm * 1e-9)

    def make_sim(structure: bool):
        sim = FDTD2D(grid.nx, grid.nz, grid.dx, grid.dz, pml_cells=grid.pml_cells)
        if structure:
            sim.set_material(grid.eps_inf, grid.strength, grid.w0, grid.gamma)
        else:
            eps_inf = np.full((grid.nx, grid.nz), float(n_env) ** 2)
            zeros = np.zeros((0, grid.nx, grid.nz))
            sim.set_material(eps_inf, zeros, zeros, zeros)
        waveform, t0, sigma_t = gaussian_pulse_source(wl_min_nm, wl_max_nm)
        sim.add_source(grid.k_src, waveform)
        mon_r = sim.add_monitor(grid.k_mon_r, omegas)
        mon_t = sim.add_monitor(grid.k_mon_t, omegas)
        return sim, mon_r, mon_t, t0, sigma_t

    sim_ref, monr_ref, mont_ref, t0, sigma_t = make_sim(False)
    n_steps = int((t0 + 8 * sigma_t + (grid.nz * grid.dz) / C0 * courant_pad_periods) / sim_ref.dt)
    if verbose:
        print(f'grid: nx={grid.nx} nz={grid.nz} dx={grid.dx*1e9:.2f}nm dt={sim_ref.dt*1e18:.3f}as '
              f'n_steps={n_steps}')
        print('running reference (no structure) pass...')
    sim_ref.run(n_steps, progress_every=(n_steps // 4 if verbose else 0))

    if verbose:
        print('running structure pass...')
    sim_str, monr_str, mont_str, _, _ = make_sim(True)
    sim_str.run(n_steps, progress_every=(n_steps // 4 if verbose else 0))

    S_inc = _flux(monr_ref.Ey_dft, monr_ref.Hx_dft, grid.dx)
    Ey_refl = monr_str.Ey_dft - monr_ref.Ey_dft
    Hx_refl = monr_str.Hx_dft - monr_ref.Hx_dft
    S_refl = _flux(Ey_refl, Hx_refl, grid.dx)
    S_trans = _flux(mont_str.Ey_dft, mont_str.Hx_dft, grid.dx)

    R = -S_refl / S_inc
    T = S_trans / S_inc
    A = 1.0 - R - T
    return SpectrumResult(wl_nm, R, T, A, grid)
