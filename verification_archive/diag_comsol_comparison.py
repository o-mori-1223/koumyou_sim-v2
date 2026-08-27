"""Compare S4-computed Rp against the COMSOL reference at 5 wavelengths x 9 angles.
Run: cd koumyou_sim && python diag_comsol_comparison.py
See comsol_reference.py for the structure/data provenance and current investigation status.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D.rcwa_s4_backend import calc_rcwa1d_s4
from RCWA2D.structure_builder import columns_to_layers
from comsol_reference import (
    build_columns, PITCH_NM, NK_FN_MAP, NK_SUBST_FN, COMSOL_RP_BY_ANGLE,
)

NORDER = 43
p0 = NORDER // 2

layers = columns_to_layers(build_columns(), flat_align=False)
print(f"n layers: {len(layers)}")

print(f"{'wl':>5} {'aoi':>4} | {'mine':>8} {'COMSOL':>8} | {'diff':>7}")
for wl, by_angle in COMSOL_RP_BY_ANGLE.items():
    for aoi, ref in by_angle.items():
        irp, itp, irs, its = calc_rcwa1d_s4(np.array([float(wl)]), aoi * np.pi / 180,
                                             PITCH_NM / 1000.0, NORDER, layers,
                                             NK_FN_MAP, 1.0, NK_SUBST_FN)
        mine = float(irp[0, p0])
        print(f"{wl:>5} {aoi:>4} | {mine:>8.4f} {ref:>8.4f} | {mine-ref:>7.4f}")
