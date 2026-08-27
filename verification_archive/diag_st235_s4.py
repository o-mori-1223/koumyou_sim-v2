"""S4 cross-check for the NEW 4200nm/8-column/st235 structure: a representative subset
(S4 is far slower per-point than rcwa_mh -- see RCWA2D/rcwa_s4_backend.py docstring) of
wavelengths x all 9 angles x both polarizations, compared against both COMSOL and the
full-grid rcwa_mh result (data/rcwa_mh_st235_result.json, produced by diag_st235_rcwa_mh.py).
Run in WSL: source ~/s4env/bin/activate && cd .../koumyou_sim && python3 diag_st235_s4.py
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D.rcwa_s4_backend import calc_rcwa1d_s4
from RCWA2D.structure_builder import columns_to_layers
from comsol_reference_st235 import build_columns, PITCH_NM, NK_FN_MAP, NK_SUBST_FN, load_comsol_data

NORDER = 61
p0 = NORDER // 2
PITCH_UM = PITCH_NM / 1000.0

layers = columns_to_layers(build_columns(), flat_align=False)
print(f"n layers: {len(layers)}, pitch: {PITCH_NM}nm, NumBasis: {NORDER}")

data = load_comsol_data()
all_wls = sorted(data['s'].keys())
angles = sorted(next(iter(data['s'].values())).keys())
SUBSET_WLS = all_wls[::8]   # ~5 wavelengths spanning 380-780nm (S4 is slow -- start small)
SUBSET_ANGLES = angles[::2]  # 0,20,40,60,80 -- halve the angle count too for a first pass

mh_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'rcwa_mh_st235_result.json')
mh = None
if os.path.isfile(mh_path):
    with open(mh_path) as f:
        mh = json.load(f)

print(f"{'wl':>5} {'th':>4} | {'S4(s)':>8} {'mh(s)':>8} {'COMSOL(s)':>9} | "
      f"{'S4(p)':>8} {'mh(p)':>8} {'COMSOL(p)':>9}")
for wl in SUBSET_WLS:
    for th in SUBSET_ANGLES:
        irp, itp, irs, its = calc_rcwa1d_s4(np.array([float(wl)]), th * np.pi / 180.0,
                                             PITCH_UM, NORDER, layers, NK_FN_MAP, 1.0, NK_SUBST_FN)
        rs_s4, rp_s4 = float(irs[0, p0]), float(irp[0, p0])
        cs, cp = data['s'][wl][th], data['p'][wl][th]
        # mh json keys are the python float repr (json.dump serializes float dict keys via repr)
        rs_mh = mh['s'].get(str(wl)) if mh else None
        rp_mh = mh['p'].get(str(wl)) if mh else None
        rs_mh_v = rs_mh[str(th)] if rs_mh else float('nan')
        rp_mh_v = rp_mh[str(th)] if rp_mh else float('nan')
        print(f"{wl:>5.0f} {th:>4.0f} | {rs_s4:>8.4f} {rs_mh_v:>8.4f} {cs:>9.4f} | "
              f"{rp_s4:>8.4f} {rp_mh_v:>8.4f} {cp:>9.4f}")
