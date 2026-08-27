"""nOrder convergence check for the reference structure -- rules out Fourier-order
truncation as the cause of the COMSOL mismatch (see comsol_reference.py). At nOrder=43
values are already within ~1% of nOrder=201, so 43 is not the problem.
Run: cd koumyou_sim && python diag_convergence.py
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D.rcwa_s4_backend import calc_rcwa1d_s4
from RCWA2D.structure_builder import columns_to_layers
from comsol_reference import build_columns, PITCH_NM, NK_FN_MAP, NK_SUBST_FN, COMSOL_RP_BY_ANGLE

layers = columns_to_layers(build_columns(), flat_align=False)
print(f"n layers: {len(layers)}")

print(f"{'norder':>7} | {'470nm@0':>9} {'470nm@30':>9} | {'540nm@0':>9} {'540nm@30':>9} | elapsed")
for norder in [43, 81, 121, 161, 201]:
    p0 = norder // 2
    t0 = time.time()
    vals = []
    for wl in [470.0, 540.0]:
        for aoi in [0, 30]:
            irp, itp, irs, its = calc_rcwa1d_s4(np.array([wl]), aoi * np.pi / 180,
                                                 PITCH_NM / 1000.0, norder, layers,
                                                 NK_FN_MAP, 1.0, NK_SUBST_FN)
            vals.append(float(irp[0, p0]))
    print(f"{norder:>7} | {vals[0]:>9.4f} {vals[1]:>9.4f} | {vals[2]:>9.4f} {vals[3]:>9.4f} | {time.time()-t0:.0f}s")

print()
print(f"Reference (COMSOL): 470nm@0={COMSOL_RP_BY_ANGLE[470][0]:.4f} "
      f"470nm@30={COMSOL_RP_BY_ANGLE[470][30]:.4f} | "
      f"540nm@0={COMSOL_RP_BY_ANGLE[540][0]:.4f} 540nm@30={COMSOL_RP_BY_ANGLE[540][30]:.4f}")
