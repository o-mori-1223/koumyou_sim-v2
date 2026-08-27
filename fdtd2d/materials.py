"""nk data loading + broadband Drude-Lorentz dispersion fitting for dispersive FDTD.

The RCWA side of this project (ppap.py / R8_koumyou_v3.py) evaluates each material's
n,k at one wavelength at a time, so a simple lookup/interpolation table is enough.
A time-domain broadband FDTD run instead needs a closed-form model of epsilon(omega)
so the auxiliary-differential-equation (ADE) update can be marched in time. This module
fits

    eps(omega) = eps_inf + sum_p  S_p / (w0_p**2 - 1j*omega*gamma_p - omega**2)

to the tabulated (wavelength, n, k) data over the wavelength band actually used by the
simulation, via nonlinear least squares. This is the same technique used by commercial
FDTD tools ("multi-coefficient material fitting"). S_p >= 0 is enforced for every pole:
for gamma_p > 0, a single term's Im(eps(w)) has the same sign as S_p at every real w, so
S_p >= 0 makes each pole individually passive (non-negative loss) at *all* frequencies,
not just inside the fit window. A broadband time-domain run excites frequencies outside
that window too (including near the numerical Nyquist limit), so a pole that is only
"passive in-band" but gains energy elsewhere makes the ADE time-marching blow up --
this bit the first version of this fitter (allowing S_p < 0 gave a better in-band curve
fit but was unconditionally unstable in the actual FDTD run). Allowing w0_p = 0 recovers
the free-electron/Drude term (eps = eps_inf - wp**2/(w**2 + i*w*gamma)) as a special case
of the same formula, which is what actually lets a lossy metal like SUS be fit well
under the S_p >= 0 restriction -- forcing S_p = d_eps*w0_p**2 with a resonant (w0_p > 0)
pole is a very inefficient way to build up the large negative real permittivity a metal
needs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

C0 = 299792458.0  # m/s


def _nk_file_path(nk_dir: str, name: str) -> str:
    for ext in ('.nk', '.txt', ''):
        p = os.path.join(nk_dir, name + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f'nk data for material "{name}" not found under {nk_dir}')


def load_nk_table(name: str, nk_dir: str = 'data/nk'):
    """Load (wavelength_nm, n, k) arrays for a material name, honoring .nk or .txt files.

    Also accepts a bare number (e.g. "1.5") as a constant, dispersionless n.
    """
    try:
        const_n = float(name)
        wl = np.array([200.0, 1000.0])
        return wl, np.full(2, const_n), np.zeros(2)
    except (TypeError, ValueError):
        pass
    path = _nk_file_path(nk_dir, name)
    mat = np.loadtxt(path, comments=';', encoding='utf-8-sig')
    wl, n, k = mat[:, 0], mat[:, 1], mat[:, 2]
    order = np.argsort(wl)
    return wl[order], n[order], k[order]


@dataclass
class LorentzModel:
    """eps(omega) = eps_inf + sum_p strength_p / (w0_p**2 - i*omega*gamma_p - omega**2).

    w0_p == 0 is a valid (Drude) pole: the term reduces to -strength_p/(omega**2 + i*omega*gamma_p).
    """

    name: str
    eps_inf: float
    strength: np.ndarray   # (P,) rad^2/s^2, >= 0
    w0: np.ndarray         # (P,) rad/s, >= 0
    gamma: np.ndarray      # (P,) rad/s, > 0
    fit_rms_n: float = 0.0
    fit_rms_k: float = 0.0

    @property
    def n_poles(self) -> int:
        return len(self.w0)

    def eps(self, omega):
        omega = np.asarray(omega, dtype=complex)
        out = np.full(omega.shape, complex(self.eps_inf))
        for s, w0, g in zip(self.strength, self.w0, self.gamma):
            out = out + s / (w0 ** 2 - 1j * omega * g - omega ** 2)
        return out

    def n_k(self, wl_nm):
        omega = 2 * np.pi * C0 / (np.asarray(wl_nm) * 1e-9)
        eps = self.eps(omega)
        nk = np.sqrt(eps)
        return nk.real, nk.imag


def fit_lorentz_model(name: str, wl_min_nm: float, wl_max_nm: float, n_poles: int = 5,
                       nk_dir: str = 'data/nk', pad_frac: float = 0.25, n_fit_pts: int = 120,
                       n_restarts: int = 8, seed: int = 0) -> LorentzModel:
    """Fit a passive sum-of-poles (Drude+Lorentz) model to the material's tabulated nk data.

    The fit window is padded beyond [wl_min_nm, wl_max_nm] so the model stays well
    behaved (and the broadband pulse spectral tails stay accurate) slightly outside
    the requested band.
    """
    wl_tab, n_tab, k_tab = load_nk_table(name, nk_dir)

    pad = (wl_max_nm - wl_min_nm) * pad_frac
    lo = max(wl_tab.min(), wl_min_nm - pad)
    hi = min(wl_tab.max(), wl_max_nm + pad)
    wl_fit = np.linspace(lo, hi, n_fit_pts)
    n_fit = np.interp(wl_fit, wl_tab, n_tab)
    k_fit = np.interp(wl_fit, wl_tab, k_tab)
    eps_data = (n_fit + 1j * k_fit) ** 2
    omega_fit = 2 * np.pi * C0 / (wl_fit * 1e-9)

    eps_inf_guess = max(1.0, float(np.min(n_fit) ** 2 - np.max(np.abs(k_fit)) ** 2))
    # Non-dispersive (or nearly so) material: skip the nonlinear fit entirely.
    if np.allclose(k_fit, 0.0, atol=1e-6) and np.ptp(n_fit) < 1e-6 and n_poles == 0:
        return LorentzModel(name, float(n_fit[0] ** 2), np.array([]), np.array([]), np.array([]))

    rng = np.random.default_rng(seed)
    w_span = omega_fit.max() - omega_fit.min()
    # Half the poles start as Drude-like (w0 ~ 0), half as in-band Lorentz resonances, so the
    # optimizer has both mechanisms available from the start instead of having to discover Drude
    # poles (w0 -> 0) purely through gradient descent from a resonant initial guess.
    w0_guess = np.concatenate([
        np.zeros(n_poles - n_poles // 2),
        np.linspace(omega_fit.min(), omega_fit.max() + 0.3 * w_span, n_poles // 2),
    ])

    def pack(eps_inf, strength, w0, gamma):
        return np.concatenate(([eps_inf], strength / w_span ** 2, w0 / w_span, gamma / w_span))

    def unpack(x):
        eps_inf = x[0]
        strength = x[1:1 + n_poles] * w_span ** 2
        w0 = x[1 + n_poles:1 + 2 * n_poles] * w_span
        gamma = x[1 + 2 * n_poles:1 + 3 * n_poles] * w_span
        return eps_inf, strength, w0, gamma

    def residuals(x):
        eps_inf, strength, w0, gamma = unpack(x)
        model = LorentzModel(name, eps_inf, strength, w0, gamma)
        eps_model = model.eps(omega_fit)
        return np.concatenate([(eps_model.real - eps_data.real), (eps_model.imag - eps_data.imag)])

    x0 = pack(eps_inf_guess, rng.uniform(0.1, 3.0, n_poles) * w_span ** 2, w0_guess,
              np.full(n_poles, 0.15 * w_span))
    # eps_inf must stay positive: it is the *instantaneous* response (D = eps0*eps_inf*E + P), and a
    # negative value makes the explicit ADE update locally non-hyperbolic (unconditionally unstable).
    # strength (S) and gamma must stay >= 0 for passivity everywhere (see module docstring); w0 may
    # be 0 (a Drude pole) up to comfortably above the fit band.
    lb = np.concatenate(([1.0], np.zeros(n_poles), np.zeros(n_poles), np.full(n_poles, 1e-4)))
    ub = np.concatenate(([50.0], np.full(n_poles, 30.0), np.full(n_poles, 3.0), np.full(n_poles, 5.0)))

    best, best_cost = None, np.inf
    for trial in range(n_restarts):
        if trial > 0:
            w0_trial = np.abs(w0_guess + rng.normal(0, 0.25 * w_span, n_poles))
            x0 = pack(eps_inf_guess, rng.uniform(0.1, 5.0, n_poles) * w_span ** 2, w0_trial,
                      rng.uniform(0.02, 0.4, n_poles) * w_span)
        res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=5000)
        if res.cost < best_cost:
            best_cost, best = res.cost, res

    eps_inf, strength, w0, gamma = unpack(best.x)
    model = LorentzModel(name, float(eps_inf), strength, w0, gamma)
    n_model, k_model = model.n_k(wl_fit)
    model.fit_rms_n = float(np.sqrt(np.mean((n_model - n_fit) ** 2)))
    model.fit_rms_k = float(np.sqrt(np.mean((k_model - k_fit) ** 2)))
    return model


_MODEL_CACHE: dict = {}


def get_lorentz_model(name: str, wl_min_nm: float, wl_max_nm: float, n_poles: int = 5,
                       nk_dir: str = 'data/nk') -> LorentzModel:
    key = (name, round(wl_min_nm, 3), round(wl_max_nm, 3), n_poles, nk_dir)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = fit_lorentz_model(name, wl_min_nm, wl_max_nm, n_poles, nk_dir)
    return _MODEL_CACHE[key]
