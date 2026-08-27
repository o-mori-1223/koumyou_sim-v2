"""Rasterize the project's `columns` structure definition onto a 2D dispersive-FDTD grid.

`columns` is the same list-of-dict format used by RCWA2D/structure_builder.py /
R8_koumyou_v3.py (width, subst_raise, film_layers per column) so a structure can be
cross-checked between the RCWA backend and this FDTD backend without re-entering the
geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from RCWA2D.structure_builder import columns_to_layers, ENV_MATERIAL, SUBST_MATERIAL
from .materials import get_lorentz_model, LorentzModel


@dataclass
class GridResult:
    dx: float
    dz: float
    nx: int
    nz: int
    eps_inf: np.ndarray          # (nx, nz)
    strength: np.ndarray         # (n_poles, nx, nz)
    w0: np.ndarray                # (n_poles, nx, nz)
    gamma: np.ndarray            # (n_poles, nx, nz)
    pitch_nm: float
    k_top_struct: int
    k_bottom_struct: int
    k_src: int
    k_mon_r: int
    k_mon_t: int
    pml_cells: int
    n_env: float
    materials: dict               # name -> LorentzModel used


def _material_id_grid(columns, pitch_nm, dx_nm, nx, n_poles_dummy=None):
    """Top-first (material_name, thickness_nm) rows -> (nx, nz_struct) int id grid,
    plus the list of unique material names encountered (index 0 reserved for ENV)."""
    layers = columns_to_layers(columns, flat_align=False)
    names = [ENV_MATERIAL]
    name_to_id = {ENV_MATERIAL: 0}

    def get_id(name):
        if name not in name_to_id:
            name_to_id[name] = len(names)
            names.append(name)
        return name_to_id[name]

    total_h = sum(L['thickness'] for L in layers)
    nz_struct = max(1, round(total_h / dx_nm))
    grid = np.zeros((nx, nz_struct), dtype=int)  # default ENV (id 0)

    z0 = 0.0
    k0 = 0
    for L in layers:  # top-first: row 0 of the grid is the top of the stack
        z1 = z0 + L['thickness']
        k1 = round(z1 / dx_nm)
        k1 = min(k1, nz_struct)
        if k1 > k0:
            x0_frac = 0.0
            for mat, wfrac in L['segments']:
                x1_frac = x0_frac + wfrac
                i0 = round(x0_frac * nx)
                i1 = round(x1_frac * nx)
                i1 = min(i1, nx)
                if i1 > i0:
                    grid[i0:i1, k0:k1] = get_id(mat)
                x0_frac = x1_frac
        z0, k0 = z1, k1
    return grid, names, nz_struct


def build_grid(columns, nk_subst_name, n_env, wl_min_nm, wl_max_nm, dx_nm,
                n_poles=6, pml_cells=15, pad_wl_factor=1.3, nk_dir='data/nk'):
    pitch_nm = sum(c['width'] for c in columns)
    nx = max(4, round(pitch_nm / dx_nm))
    dx = dx_nm * 1e-9
    dz = dx  # isotropic grid

    struct_grid, names, nz_struct = _material_id_grid(columns, pitch_nm, dx_nm, nx)

    pad_nm = pad_wl_factor * wl_max_nm
    n_env_pad = max(20, round(pad_nm / dx_nm))
    n_sub_pad = max(20, round(pad_nm / dx_nm))
    nz = pml_cells + n_env_pad + nz_struct + n_sub_pad + pml_cells

    k_top_struct = pml_cells + n_env_pad
    k_bottom_struct = k_top_struct + nz_struct
    k_src = pml_cells + max(5, n_env_pad // 6)
    k_mon_r = (k_src + k_top_struct) // 2
    k_mon_t = (k_bottom_struct + (nz - pml_cells)) // 2

    material_id = np.zeros((nx, nz), dtype=int)
    material_id[:, k_top_struct:k_bottom_struct] = struct_grid
    subst_id = len(names)
    names_full = names + [SUBST_MATERIAL]
    material_id[:, k_bottom_struct:] = subst_id

    # Fit / look up a dispersion model per unique material name.
    models: dict[str, LorentzModel] = {}
    for name in names_full:
        if name == ENV_MATERIAL:
            models[name] = LorentzModel(ENV_MATERIAL, float(n_env) ** 2,
                                         np.zeros(n_poles), np.full(n_poles, 1e15),
                                         np.full(n_poles, 1e14))
        else:
            real_name = nk_subst_name if name == SUBST_MATERIAL else name
            models[name] = get_lorentz_model(real_name, wl_min_nm, wl_max_nm, n_poles, nk_dir)

    n_mat = len(names_full)
    eps_inf_by_mat = np.zeros(n_mat)
    strength_by_mat = np.zeros((n_mat, n_poles))
    w0_by_mat = np.full((n_mat, n_poles), 1e15)
    gamma_by_mat = np.full((n_mat, n_poles), 1e14)
    for idx, name in enumerate(names_full):
        m = models[name]
        eps_inf_by_mat[idx] = m.eps_inf
        p = m.n_poles
        strength_by_mat[idx, :p] = m.strength
        w0_by_mat[idx, :p] = m.w0
        gamma_by_mat[idx, :p] = m.gamma

    eps_inf_grid = eps_inf_by_mat[material_id]
    strength_grid = np.transpose(strength_by_mat[material_id], (2, 0, 1))
    w0_grid = np.transpose(w0_by_mat[material_id], (2, 0, 1))
    gamma_grid = np.transpose(gamma_by_mat[material_id], (2, 0, 1))

    # CPML's stretched-coordinate derivation assumes a locally simple (non-dispersive)
    # backing medium. Every fitted pole here is passive (strength >= 0, see materials.py),
    # so this is a belt-and-suspenders precaution rather than a stability requirement: zero
    # the pole terms in the outermost `pml_cells` at each z end, leaving only the
    # (non-dispersive) eps_inf background there. By the time a wave reaches the PML it is
    # already absorbed/attenuated, so this has negligible effect on the physics.
    strength_grid[:, :, :pml_cells] = 0.0
    strength_grid[:, :, -pml_cells:] = 0.0

    return GridResult(dx, dz, nx, nz, eps_inf_grid, strength_grid, w0_grid, gamma_grid,
                       pitch_nm, k_top_struct, k_bottom_struct, k_src, k_mon_r, k_mon_t,
                       pml_cells, float(n_env), models)
