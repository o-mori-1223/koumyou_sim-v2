"""gsolver_reference_8col_altraise.py の構造について:
  1. Gsolverインポート用テキストを data/gsolver_import_8col_altraise.txt に書き出す
  2. rcwa_mh でのスペクトル(法線入射, s/p, 380-780nm 10nm刻み)を計算し
     data/rcwa_mh_8col_altraise_result.json に保存する（Gsolverの結果が揃ったら比較用）
Gsolver側の手順: GSolverV61.exe を起動 → File > Import Text で
data/gsolver_import_8col_altraise.txt を読み込み → 断面プレビューで
幅[900,300,900,300,600,300,600,300]nmのうち幅900/600nmの列だけが235nm上げられている
（st235とは上げ列が逆）ことを目視確認 → RUN → 反射スペクトルをテキスト書き出し。
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh
from RCWA2D.structure_builder import get_layer_tuple
from RCWA2D.gsolver_export import write_gsolver_import
from gsolver_reference_8col_altraise import build_layers, PITCH_NM, NK_FN_MAP, NK_SUBST_FN

NORDER = 61   # st235と同じ大きいピッチ(4200nm)なので同じNumBasisを使う
p0 = NORDER // 2
PITCH_UM = PITCH_NM / 1000.0

layers = build_layers()
print(f"pitch: {PITCH_NM}nm, RCWA layers: {len(layers)}")

out_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gsolver_import_8col_altraise.txt')
write_gsolver_import(layers, out_txt)
print(f"wrote: {out_txt}")

results = {'s': {}, 'p': {}}
for pol in ('s', 'p'):
    for wl in range(380, 781, 10):
        wl_um = wl / 1000.0
        lt = get_layer_tuple(wl, layers, NK_FN_MAP, 1.0, NK_SUBST_FN)
        ir, it = rcwa_mh.Rcwa1d(pol, wl_um, 0.0, PITCH_UM, lt, NORDER)  # normal incidence
        results[pol][wl] = float(ir[p0])

out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'rcwa_mh_8col_altraise_result.json')
with open(out_json, 'w') as f:
    json.dump(results, f)
print(f"wrote: {out_json}")

print()
print(f"{'wl':>5} {'Rs':>8} {'Rp':>8}")
for wl in range(380, 781, 20):
    print(f"{wl:>5} {results['s'][wl]:>8.4f} {results['p'][wl]:>8.4f}")
