"""Full-grid rcwa_mh vs COMSOL comparison for the NEW 4200nm/8-column/st235 structure.
Run (native Windows venv, no WSL needed): .venv/Scripts/python.exe diag_st235_rcwa_mh.py
See comsol_reference_st235.py for structure/data provenance.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh
from RCWA2D.structure_builder import columns_to_layers, get_layer_tuple
from comsol_reference_st235 import build_columns, PITCH_NM, NK_FN_MAP, NK_SUBST_FN, load_comsol_data

NORDER = 61   # higher than the 43 used for the old (3000nm) structure since this pitch is larger
p0 = NORDER // 2
PITCH_UM = PITCH_NM / 1000.0

layers = columns_to_layers(build_columns(), flat_align=False)
print(f"n layers: {len(layers)}, pitch: {PITCH_NM}nm, NumBasis: {NORDER}")

data = load_comsol_data()
wls = sorted(data['s'].keys())
angles = sorted(next(iter(data['s'].values())).keys())

results = {'s': {}, 'p': {}}
for pol in ('s', 'p'):
    for wl in wls:
        wl_um = wl / 1000.0
        lt = get_layer_tuple(wl, layers, NK_FN_MAP, 1.0, NK_SUBST_FN)
        row = {}
        for th in angles:
            coef = float(2 * np.pi * np.sin(th * np.pi / 180.0) / wl_um)
            ir, it = rcwa_mh.Rcwa1d(pol, wl_um, coef, PITCH_UM, lt, NORDER)
            row[th] = float(ir[p0])
        results[pol][wl] = row

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'rcwa_mh_st235_result.json')
with open(out_path, 'w') as f:
    json.dump(results, f)
print(f"saved: {out_path}")

for pol in ('s', 'p'):
    diffs = []
    for wl in wls:
        for th in angles:
            diffs.append(results[pol][wl][th] - data[pol][wl][th])
    diffs = np.array(diffs)
    print(f"pol={pol}: n={len(diffs)}  mean|diff|={np.mean(np.abs(diffs)):.4f}  "
          f"max|diff|={np.max(np.abs(diffs)):.4f}  RMS={np.sqrt(np.mean(diffs**2)):.4f}")

print()
print(f"{'wl':>5} {'th':>4} | {'mine(s)':>8} {'COMSOL(s)':>9} {'diff':>7} | "
      f"{'mine(p)':>8} {'COMSOL(p)':>9} {'diff':>7}")
for wl in wls[::4]:
    for th in angles:
        ms, cs = results['s'][wl][th], data['s'][wl][th]
        mp, cp = results['p'][wl][th], data['p'][wl][th]
        print(f"{wl:>5.0f} {th:>4.0f} | {ms:>8.4f} {cs:>9.4f} {ms-cs:>7.4f} | "
              f"{mp:>8.4f} {cp:>9.4f} {mp-cp:>7.4f}")
