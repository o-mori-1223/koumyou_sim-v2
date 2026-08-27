"""Cross-check rcwa_mh (custom C++) against S4 (independent, third-party solver) on the
EXACT SAME multi-column reference structure -- rules out a bug specific to how either
solver's Fourier factorization handles a period decomposed into several lateral columns.
The two agree to within ~0.01 across all tested wavelengths/angles, so a shared "column
decomposition" bug is not a viable explanation for the COMSOL mismatch (see
comsol_reference.py for the fuller picture).

Requires rcwa_mh built for the same Python (Linux .so under WSL, or the Windows .pyd
natively) -- run inside the same environment S4 lives in (WSL s4env).
Run: cd koumyou_sim && python diag_mh_vs_s4.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh
from RCWA2D.rcwa_s4_backend import calc_rcwa1d_s4
from RCWA2D.structure_builder import columns_to_layers, get_layer_tuple
from comsol_reference import build_columns, PITCH_NM, NK_FN_MAP, NK_SUBST_FN

NORDER = 43
p0 = NORDER // 2

layers = columns_to_layers(build_columns(), flat_align=False)
print(f"n layers: {len(layers)}")

print(f"{'wl':>5} {'aoi':>4} | {'rcwa_mh Rp':>11} {'S4 Rp':>9} | {'diff':>9}")
for wl in [400, 470, 540, 600, 700]:
    for aoi in [0, 30, 60]:
        layer_tuple = get_layer_tuple(float(wl), layers, NK_FN_MAP, 1.0, NK_SUBST_FN)
        wl_um = wl / 1000.0
        coef = 2 * np.pi * np.sin(aoi * np.pi / 180) / wl_um
        ir_mh, it_mh = rcwa_mh.Rcwa1d('p', wl_um, coef, PITCH_NM / 1000.0, layer_tuple, NORDER)
        r_mh = float(ir_mh[p0])

        irp, itp, irs, its = calc_rcwa1d_s4(np.array([float(wl)]), aoi * np.pi / 180,
                                             PITCH_NM / 1000.0, NORDER, layers,
                                             NK_FN_MAP, 1.0, NK_SUBST_FN)
        r_s4 = float(irp[0, p0])

        print(f"{wl:>5} {aoi:>4} | {r_mh:>11.5f} {r_s4:>9.5f} | {r_mh-r_s4:>9.5f}")
