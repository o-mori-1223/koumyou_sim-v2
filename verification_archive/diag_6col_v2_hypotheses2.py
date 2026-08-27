"""diag_6col_v2_hypotheses.py の続き。最有力候補「上げ列反転(幅300nm列を上げ)」を軸に、
(1) flat_align=Trueとの組み合わせ、(2) この反転パターンでの上げ高さスイープ、を追加検証する。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh
from RCWA2D.structure_builder import columns_to_layers, get_layer_tuple
from gsolver_reference_6col_v2 import COL_WS, PITCH_NM, NK_FN_MAP, NK_SUBST_FN, GLOBAL_FILM, load_gsolver_data

PITCH_UM = PITCH_NM / 1000.0
gsolver = load_gsolver_data()
wls = sorted(gsolver.keys())
N_SUS = 1.71 + 2.88j


def make_columns(raises):
    return [{'width': w, 'subst_raise': r, 'film_layers': GLOBAL_FILM} for w, r in zip(COL_WS, raises)]


def run(raises, norder=43, flat_align=False):
    layers = columns_to_layers(make_columns(raises), flat_align=flat_align)
    p0 = norder // 2
    out = {}
    for wl in wls:
        wl_um = wl / 1000.0
        lt = get_layer_tuple(wl, layers, NK_FN_MAP, 1.0, lambda w: N_SUS)
        ir, it = rcwa_mh.Rcwa1d('s', wl_um, 0.0, PITCH_UM, lt, norder)
        out[wl] = float(ir[p0])
    return out


def stats(res):
    diffs = np.array([res[wl] - gsolver[wl] for wl in wls])
    return np.mean(np.abs(diffs)), np.sqrt(np.mean(diffs ** 2)), np.max(np.abs(diffs))


results = []

print("=== 上げ列反転(300nm列) + flat_align=True ===", flush=True)
r = stats(run([0.0, 235.0, 0.0, 235.0, 0.0, 235.0], flat_align=True))
results.append(("raise 1,3,5 + flat_align=True", r))
print(f"  {r}", flush=True)

print("=== 上げ列反転(300nm列) 上げ高さスイープ ===", flush=True)
for h in (100.0, 150.0, 180.0, 200.0, 220.0, 235.0, 250.0, 270.0, 300.0):
    raises = [0.0, h, 0.0, h, 0.0, h]
    r = stats(run(raises))
    results.append((f"raise 1,3,5, h={h:.0f}nm", r))
    print(f"  h={h:>5.0f}nm: mean|diff|={r[0]:.4f} RMS={r[1]:.4f} max={r[2]:.4f}", flush=True)

print()
print(f"{'variant':<38} {'mean|diff|':>10} {'RMS':>8} {'max':>8}")
results.sort(key=lambda x: x[1][0])
for name, r in results:
    print(f"{name:<38} {r[0]:>10.4f} {r[1]:>8.4f} {r[2]:>8.4f}")
