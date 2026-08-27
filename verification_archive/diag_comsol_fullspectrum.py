"""Full 380-780nm (10nm step) Rp spectrum at AOI=0, computed vs COMSOL -- shows the
resonance-position-matches / peak-amplitude-doesn't pattern described in comsol_reference.py.
Run: cd koumyou_sim && python diag_comsol_fullspectrum.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D.rcwa_s4_backend import calc_rcwa1d_s4
from RCWA2D.structure_builder import columns_to_layers
from comsol_reference import build_columns, PITCH_NM, NK_FN_MAP, NK_SUBST_FN, COMSOL_RP_AOI0_SPECTRUM

NORDER = 43
p0 = NORDER // 2

layers = columns_to_layers(build_columns(), flat_align=False)
wls = sorted(COMSOL_RP_AOI0_SPECTRUM.keys())

irp, itp, irs, its = calc_rcwa1d_s4(np.array(wls, dtype=float), 0.0, PITCH_NM / 1000.0,
                                     NORDER, layers, NK_FN_MAP, 1.0, NK_SUBST_FN)

print(f"{'wl':>5} | {'mine':>8} {'COMSOL':>8} | {'diff':>7}")
for i, wl in enumerate(wls):
    mine = float(irp[i, p0])
    ref = COMSOL_RP_AOI0_SPECTRUM[wl]
    print(f"{wl:>5} | {mine:>8.4f} {ref:>8.4f} | {mine-ref:>7.4f}")
