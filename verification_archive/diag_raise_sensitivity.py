"""How sensitive is Rp@470nm to the substrate-raise height? Sweeps raise from 0 (flat,
matches diag_flat_stack_sanity.py's R~0.99) to 100nm (the reference value, R~0.20) --
smooth, monotonic, physically-expected decay as more lossy substrate gets laterally
exposed near the bottom of the Bragg stack. Demonstrates that the ~0.5 gap to COMSOL
(which shows R~0.71 at the same wavelength/angle) could plausibly come from a fairly
small remaining error in the assumed raise pattern/height, given how steep this curve is.
Run: cd koumyou_sim && python diag_raise_sensitivity.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh
from RCWA2D.structure_builder import columns_to_layers, get_layer_tuple
from comsol_reference import build_columns, PITCH_NM, NK_FN_MAP, NK_SUBST_FN, COL_WS

NORDER = 43
p0 = NORDER // 2

print(f"{'raise_nm':>9} | {'Rp@470,AOI0':>12} | {'n_layers':>8}")
for raise_h in [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]:
    raises = [raise_h, 0.0, raise_h, 0.0, raise_h, 0.0]
    layers = columns_to_layers(build_columns(raises=raises), flat_align=False)
    layer_tuple = get_layer_tuple(470.0, layers, NK_FN_MAP, 1.0, NK_SUBST_FN)
    ir, it = rcwa_mh.Rcwa1d('p', 0.470, 0.0, PITCH_NM / 1000.0, layer_tuple, NORDER)
    print(f"{raise_h:>9.0f} | {ir[p0]:>12.5f} | {len(layers):>8}")
