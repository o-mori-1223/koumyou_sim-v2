"""2D dispersive-FDTD normal-incidence spectrum for the NEW 4200nm/8-column/st235
structure, compared against COMSOL (eta_P=0, theta=0) and rcwa_mh (theta=0, s-pol).

IMPORTANT LIMITATION: fdtd2d/engine.py only implements the TE (Ey/Hx/Hz) field set with
a plane-wave current source injected uniformly in x -- i.e. NORMAL INCIDENCE, S-POLARIZATION
ONLY (no oblique-incidence source injection, no TM/p-polarization fields). So this script
can only be compared against the theta=0, eta_P=0 (assumed s-pol) COMSOL rows -- it cannot
reproduce the angle sweep or p-polarization data.

Run (native Windows venv, no WSL needed): .venv/Scripts/python.exe diag_st235_fdtd.py
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fdtd2d.simulate import run_spectrum
from comsol_reference_st235 import build_columns, load_comsol_data

nk_subst_name = 'SUS'      # data/nk/SUS.nk: constant n=1.71, k=2.88 (matches N_SUS)
n_env = 1.0
wl_min_nm, wl_max_nm = 380.0, 780.0
n_wl = 41
dx_nm = 10.0                # 4200nm pitch / 10nm = 420 cells in x; coarser than fdtd_demo's 6nm
n_poles = 6                 # fdtd_demo.py's known-stable default; a flat *lossy* (SUS, k=2.88)
                             # table still needs a real nonlinear pole fit (eps_inf alone is real-only
                             # and can't carry the constant imaginary part) -- n_poles=2 tried first
                             # produced a degenerate/ill-conditioned fit that blew up the ADE time-march
                             # (inf/nan in R/T, see git history of this file) -- not a physics finding,
                             # just an under-fit dispersion model for a lossy but "flat" nk table.

if __name__ == '__main__':
    columns = build_columns()
    result = run_spectrum(columns, nk_subst_name, n_env, wl_min_nm, wl_max_nm,
                           n_wl=n_wl, dx_nm=dx_nm, n_poles=n_poles, verbose=True)

    data = load_comsol_data()
    print()
    print(f'pitch = {result.grid.pitch_nm:.1f} nm   grid = {result.grid.nx} x {result.grid.nz} cells')
    print(f'{"wl(nm)":>8} {"FDTD R":>8} {"FDTD T":>8} {"FDTD A":>8} | {"COMSOL Rs(0deg)":>16} {"diff":>8}')
    out = []
    for wl, R, T, A in zip(result.wl_nm, result.R, result.T, result.A):
        wl_k = round(float(wl), 3)
        # nearest COMSOL wavelength key (grid is linspace, may not land exactly on 10nm steps)
        near_wl = min(data['s'].keys(), key=lambda w: abs(w - wl_k))
        cref = data['s'][near_wl][0.0]
        out.append((wl, R, T, A, cref))
        print(f'{wl:8.1f} {R:8.4f} {T:8.4f} {A:8.4f} | {cref:16.4f} {R-cref:8.4f}')

    diffs = np.array([o[1] - o[4] for o in out])
    print()
    print(f'mean|diff|={np.mean(np.abs(diffs)):.4f}  max|diff|={np.max(np.abs(diffs)):.4f}  '
          f'RMS={np.sqrt(np.mean(diffs**2)):.4f}')

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fdtd_st235_result.json')
    with open(out_path, 'w') as f:
        json.dump({'wl_nm': [float(w) for w in result.wl_nm], 'R': [float(r) for r in result.R],
                    'T': [float(t) for t in result.T]}, f)
    print(f'saved: {out_path}')
