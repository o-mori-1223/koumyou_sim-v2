"""Sanity check: the SAME 15-layer TiO2/SiO2 recipe with NO lateral column patterning
(a plain flat DBR mirror) should reflect close to 1.0 in its stopband if the recipe
(thicknesses/indices) is a properly-functioning high-contrast Bragg stack. Confirms the
multilayer recipe itself is correct -- R~0.99 at 470-540nm, matching COMSOL's peak
magnitude and location. The gap only appears once the lateral column/raise pattern is
added (see diag_raise_sensitivity.py and comsol_reference.py).
Run: cd koumyou_sim && python diag_flat_stack_sanity.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh
from comsol_reference import flat_stack_layer_tuple

layer_tuple = flat_stack_layer_tuple()
print(f"n layers: {len(layer_tuple)}")

NORDER = 5  # no lateral pattern -> order doesn't matter, keep it fast
p0 = NORDER // 2

print("wl,Rp")
for wl in range(380, 781, 10):
    wl_um = wl / 1000.0
    ir, it = rcwa_mh.Rcwa1d('p', wl_um, 0.0, 1.0, layer_tuple, NORDER)  # normal incidence
    print(f"{wl},{ir[p0]:.6f}")
