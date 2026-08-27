"""gsolver_reference_oldstruct_235_rounded.py の構造について、法線入射だけでなく
全角度(-80〜80deg, 10deg刻み)でのs偏光スペクトルを計算し、koumyouアプリの
angle_Rs_*.csv と比較する。
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh
from RCWA2D.structure_builder import get_layer_tuple
from gsolver_reference_oldstruct_235_rounded import build_layers, PITCH_NM, NK_FN_MAP, NK_SUBST_FN

NORDER = 43
p0 = NORDER // 2
PITCH_UM = PITCH_NM / 1000.0

layers = build_layers()
wls = list(range(380, 781, 10))
angles = list(range(-80, 81, 10))

results = {}
for ang in angles:
    ang_rad = ang * np.pi / 180.0
    row = {}
    for wl in wls:
        wl_um = wl / 1000.0
        lt = get_layer_tuple(wl, layers, NK_FN_MAP, 1.0, NK_SUBST_FN)
        coef = float(2 * np.pi * np.sin(ang_rad) / wl_um)
        ir, it = rcwa_mh.Rcwa1d('s', wl_um, coef, PITCH_UM, lt, NORDER)
        row[wl] = float(ir[p0])
    results[ang] = row
    print(f"done angle={ang}", flush=True)

out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'rcwa_mh_oldstruct_235_rounded_angles_result.json')
with open(out_json, 'w') as f:
    json.dump(results, f)
print(f"wrote: {out_json}")
