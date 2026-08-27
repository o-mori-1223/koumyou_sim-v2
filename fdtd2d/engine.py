"""2D dispersive FDTD engine (TE / s-polarization: fields Ey, Hx, Hz).

Grid layout (standard Yee, x periodic, z terminated by CPML):
  Ey, Dy      : (nx, nz)   at integer (i, k)              -- i*dx, k*dz
  Hx          : (nx, nz-1) at (i, k+1/2)
  Hz          : (nx, nz)   at (i+1/2, k)                   -- periodic roll in x

Governing equations (non-magnetic, mu = mu0 everywhere):
  dDy/dt =  dHx/dz - dHz/dx
  dHx/dt =  (1/mu0) dEy/dz
  dHz/dt = -(1/mu0) dEy/dx

Material dispersion is a sum of Lorentz poles per cell, integrated with the standard
auxiliary differential equation (ADE) method (Taflove & Hagness ch. 9):
  D = eps0*eps_inf*E + sum_p P_p
  P_p^{n+1} = C1_p P_p^n - C2_p P_p^{n-1} + C3_p E^n

CPML absorbing boundaries are applied only in z (x is periodic -- the grating period).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from .cpml import make_cpml, EPS0, MU0

C0 = 299792458.0


@dataclass
class Monitor:
    k_index: int
    omegas: np.ndarray
    Ey_dft: np.ndarray = dc_field(default=None)
    Hx_dft: np.ndarray = dc_field(default=None)

    def init(self, nx):
        self.Ey_dft = np.zeros((nx, len(self.omegas)), dtype=complex)
        self.Hx_dft = np.zeros((nx, len(self.omegas)), dtype=complex)

    def accumulate(self, Ey_line, Hx_line, t, dt):
        phase = np.exp(1j * np.outer(np.ones(1), self.omegas) * t) * dt  # (1, n_omega)
        self.Ey_dft += Ey_line[:, None] * phase
        self.Hx_dft += Hx_line[:, None] * phase


class FDTD2D:
    def __init__(self, nx, nz, dx, dz, pml_cells=15, courant=0.98, eta_boundary=377.0):
        self.nx, self.nz = nx, nz
        self.dx, self.dz = dx, dz
        self.pml_cells = pml_cells
        self.dt = courant / (C0 * np.sqrt(1.0 / dx ** 2 + 1.0 / dz ** 2))

        self.Ey = np.zeros((nx, nz))
        self.Dy = np.zeros((nx, nz))
        self.Hx = np.zeros((nx, nz - 1))
        self.Hz = np.zeros((nx, nz))

        self.psi_Hx_z = np.zeros((nx, nz - 1))
        self.psi_Dy_z = np.zeros((nx, nz))

        self.cpml_e, self.cpml_h = make_cpml(nz, dz, pml_cells, self.dt, eta_boundary=eta_boundary)

        self.eps_inf = np.ones((nx, nz))
        self.strength = np.zeros((0, nx, nz))
        self.w0 = np.zeros((0, nx, nz))
        self.gamma = np.zeros((0, nx, nz))
        self._ade_ready = False

        self.sources = []   # list of (k_index, waveform_fn)
        self.monitors: list[Monitor] = []
        self.t = 0.0
        self.step_n = 0

    def set_material(self, eps_inf, strength, w0, gamma):
        """eps_inf: (nx,nz). strength, w0, gamma: (n_poles, nx, nz).

        Pole term contributes strength / (w0**2 - i*omega*gamma - omega**2) to eps(omega);
        w0 == 0 is a valid Drude pole. strength must be >= 0 everywhere for stability
        (see fdtd2d/materials.py docstring)."""
        self.eps_inf = eps_inf
        self.strength, self.w0, self.gamma = strength, w0, gamma
        n_poles = strength.shape[0]
        dt = self.dt
        denom = 1.0 + gamma * dt / 2.0
        self.ade_C1 = (2.0 - w0 ** 2 * dt ** 2) / denom
        self.ade_C2 = (1.0 - gamma * dt / 2.0) / denom
        self.ade_C3 = (EPS0 * strength * dt ** 2) / denom
        self.P = np.zeros((n_poles, self.nx, self.nz))
        self.P_prev = np.zeros((n_poles, self.nx, self.nz))
        self._ade_ready = True

    def add_source(self, k_index, waveform_fn):
        self.sources.append((k_index, waveform_fn))

    def add_monitor(self, k_index, omegas):
        mon = Monitor(k_index, np.asarray(omegas))
        mon.init(self.nx)
        self.monitors.append(mon)
        return mon

    def step(self):
        dx, dz, dt = self.dx, self.dz, self.dt

        dEy_dz = (self.Ey[:, 1:] - self.Ey[:, :-1]) / dz
        self.psi_Hx_z = self.cpml_h.b * self.psi_Hx_z + self.cpml_h.a * dEy_dz
        self.Hx += (dt / MU0) * ((1.0 / self.cpml_h.kappa) * dEy_dz + self.psi_Hx_z)

        dEy_dx = (np.roll(self.Ey, -1, axis=0) - self.Ey) / dx
        self.Hz += -(dt / MU0) * dEy_dx

        Hx_ext = np.zeros((self.nx, self.nz + 1))
        Hx_ext[:, 1:self.nz] = self.Hx
        dHx_dz = (Hx_ext[:, 1:] - Hx_ext[:, :-1]) / dz
        self.psi_Dy_z = self.cpml_e.b * self.psi_Dy_z + self.cpml_e.a * dHx_dz
        term_z = (1.0 / self.cpml_e.kappa) * dHx_dz + self.psi_Dy_z
        dHz_dx = (self.Hz - np.roll(self.Hz, 1, axis=0)) / dx
        self.Dy += dt * (term_z - dHz_dx)

        for k_index, waveform_fn in self.sources:
            self.Dy[:, k_index] += dt * waveform_fn(self.t + dt)

        Ey_old = self.Ey
        if self._ade_ready and self.strength.shape[0] > 0:
            P_new = (self.ade_C1 * self.P - self.ade_C2 * self.P_prev
                     + self.ade_C3 * Ey_old[None, :, :])
            self.P_prev = self.P
            self.P = P_new
            self.Ey = (self.Dy - np.sum(self.P, axis=0)) / (EPS0 * self.eps_inf)
        else:
            self.Ey = self.Dy / (EPS0 * self.eps_inf)

        self.t += dt
        self.step_n += 1

        for mon in self.monitors:
            mon.accumulate(self.Ey[:, mon.k_index], self.Hx[:, min(mon.k_index, self.nz - 2)],
                            self.t, dt)

    def run(self, n_steps, progress_every=0):
        for i in range(n_steps):
            self.step()
            if progress_every and (i % progress_every == 0):
                print(f'  step {i}/{n_steps}  t={self.t*1e15:.1f} fs')


def gaussian_pulse_source(wl_min_nm, wl_max_nm, amplitude=1.0):
    """Modulated-Gaussian current source whose spectrum covers [wl_min_nm, wl_max_nm]."""
    wl_c = 0.5 * (wl_min_nm + wl_max_nm) * 1e-9
    omega0 = 2 * np.pi * C0 / wl_c
    omega_min = 2 * np.pi * C0 / (wl_max_nm * 1e-9)
    omega_max = 2 * np.pi * C0 / (wl_min_nm * 1e-9)
    sigma_omega = (omega_max - omega_min) / 4.0
    sigma_t = 1.0 / sigma_omega
    t0 = 5.0 * sigma_t

    def waveform(t):
        env = np.exp(-((t - t0) ** 2) / (2 * sigma_t ** 2))
        return amplitude * env * np.cos(omega0 * (t - t0))

    return waveform, t0, sigma_t
