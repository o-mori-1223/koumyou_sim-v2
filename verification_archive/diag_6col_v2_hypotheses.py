"""gsolver_reference_6col_v2.py (周期3000nm/6列/幅900,600,600の列を235nm上げ) で
rcwa_mh がGsolver(data/gsolver_6col_v2.csv, s偏光)から系統的に外れている件について、
考えられる原因を一通り機械的にテストする。baseline: NORDER=43, conformal
(flat_align=False), raise=235nm@列0,2,4, n_sus=1.71+2.88j, s偏光, 法線入射。

テストする仮説:
  A. NORDER収束(43/61/81) -- 次数不足の可能性
  B. flat_align=True (列の薄膜開始高さを最大raiseに揃える) -- 平坦界面寄りの近似
  C. 基板屈折率の虚部符号反転 (n=1.71-2.88j)
  D. 上げ列パターン反転 (幅300nmの列3本を上げる -- st235と同じ側を上げる)
  E. 上げ高さスイープ (100〜350nm) -- 実際のGsolver設定値とのズレを探る
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


def make_columns(raises, film=GLOBAL_FILM):
    return [{'width': w, 'subst_raise': r, 'film_layers': film} for w, r in zip(COL_WS, raises)]


def run(raises, n_sus, norder, flat_align=False):
    nk_subst_fn = lambda wl: n_sus
    layers = columns_to_layers(make_columns(raises), flat_align=flat_align)
    p0 = norder // 2
    out = {}
    for wl in wls:
        wl_um = wl / 1000.0
        lt = get_layer_tuple(wl, layers, NK_FN_MAP, 1.0, nk_subst_fn)
        ir, it = rcwa_mh.Rcwa1d('s', wl_um, 0.0, PITCH_UM, lt, norder)
        out[wl] = float(ir[p0])
    return out


def stats(res):
    diffs = np.array([res[wl] - gsolver[wl] for wl in wls])
    return np.mean(np.abs(diffs)), np.sqrt(np.mean(diffs ** 2)), np.max(np.abs(diffs))


N_SUS_POS = 1.71 + 2.88j
N_SUS_NEG = 1.71 - 2.88j
RAISE_H = 235.0
RAISES_BASE = [RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H, 0.0]     # 列0,2,4 (900,600,600)
RAISES_INV = [0.0, RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H]      # 列1,3,5 (300,300,300)

results = []

print("=== A. NORDER収束 ===", flush=True)
for n in (43, 61, 81):
    r = stats(run(RAISES_BASE, N_SUS_POS, n))
    results.append((f"NORDER={n} (baseline pattern)", r))
    print(f"  NORDER={n}: mean|diff|={r[0]:.4f} RMS={r[1]:.4f} max={r[2]:.4f}", flush=True)

print("=== B. flat_align=True ===", flush=True)
r = stats(run(RAISES_BASE, N_SUS_POS, 43, flat_align=True))
results.append(("flat_align=True", r))
print(f"  {r}", flush=True)

print("=== C. 基板屈折率 符号反転 ===", flush=True)
r = stats(run(RAISES_BASE, N_SUS_NEG, 43))
results.append(("n_sus = 1.71-2.88j", r))
print(f"  {r}", flush=True)

print("=== D. 上げ列パターン反転 (幅300nm列を上げ) ===", flush=True)
r = stats(run(RAISES_INV, N_SUS_POS, 43))
results.append(("raise cols 1,3,5 (300nm) instead", r))
print(f"  {r}", flush=True)

print("=== E. 上げ高さスイープ ===", flush=True)
for h in (100.0, 150.0, 200.0, 235.0, 270.0, 300.0, 350.0):
    raises = [h, 0.0, h, 0.0, h, 0.0]
    r = stats(run(raises, N_SUS_POS, 43))
    results.append((f"raise_h={h:.0f}nm", r))
    print(f"  raise_h={h:>5.0f}nm: mean|diff|={r[0]:.4f} RMS={r[1]:.4f} max={r[2]:.4f}", flush=True)

print()
print(f"{'variant':<38} {'mean|diff|':>10} {'RMS':>8} {'max':>8}")
results.sort(key=lambda x: x[1][0])
for name, r in results:
    print(f"{name:<38} {r[0]:>10.4f} {r[1]:>8.4f} {r[2]:>8.4f}")
