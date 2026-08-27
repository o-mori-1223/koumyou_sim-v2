"""旧構造（comsol_reference.py: 周期3000nm・6列・幅[900,600,300,300,600,300]nm、
列0,2,4(900,300,600)を100nm上げ、TiO2/SiO2 7.5ペア、基板n=1.71+2.88j）について:
  1. Gsolverインポート用テキストを data/gsolver_import_oldstruct.txt に書き出す
  2. rcwa_mh でのスペクトル(法線入射, s/p, 380-780nm 10nm刻み)を計算し
     data/rcwa_mh_oldstruct_result.json に保存する（Gsolverの結果が揃ったら比較用）

この構造は2026-08-04のセッションでCOMSOLとの間に470/540nm付近の不一致が残ったまま
未解決だったもの（HANDOVER.md 4節参照）。2026-08-24に判明した「GSolver側のORDERS
不足」が真の原因だった可能性を検証する目的で、Gsolverを4つ目の独立ツールとして使う。

Gsolver側の手順: ORDERSを41以上（rcwa_mhのNORDER=43に合わせる）に設定し、
File > Import Text で data/gsolver_import_oldstruct.txt を読み込み（Approximateは
押さない）、Listing/RUNで直接RUN。
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh
from RCWA2D.structure_builder import columns_to_layers, get_layer_tuple
from RCWA2D.gsolver_export import write_gsolver_import
from comsol_reference import build_columns, PITCH_NM, NK_FN_MAP, NK_SUBST_FN, COMSOL_RP_AOI0_SPECTRUM

NORDER = 43   # diag_convergence.pyでこの旧構造(3000nm)向けに収束確認済みの次数
p0 = NORDER // 2
PITCH_UM = PITCH_NM / 1000.0

layers = columns_to_layers(build_columns(), flat_align=False)
print(f"pitch: {PITCH_NM}nm, RCWA layers: {len(layers)}")

out_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gsolver_import_oldstruct.txt')
write_gsolver_import(layers, out_txt)
print(f"wrote: {out_txt}")

results = {'s': {}, 'p': {}}
for pol in ('s', 'p'):
    for wl in range(380, 781, 10):
        wl_um = wl / 1000.0
        lt = get_layer_tuple(wl, layers, NK_FN_MAP, 1.0, NK_SUBST_FN)
        ir, it = rcwa_mh.Rcwa1d(pol, wl_um, 0.0, PITCH_UM, lt, NORDER)  # normal incidence
        results[pol][wl] = float(ir[p0])

out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'rcwa_mh_oldstruct_result.json')
with open(out_json, 'w') as f:
    json.dump(results, f)
print(f"wrote: {out_json}")

print()
print(f"{'wl':>5} {'Rs':>8} {'Rp':>8} {'COMSOL(Rp,旧データ)':>20}")
for wl in range(380, 781, 20):
    cref = COMSOL_RP_AOI0_SPECTRUM.get(wl)
    cref_s = f"{cref:.4f}" if cref is not None else "  n/a"
    print(f"{wl:>5} {results['s'][wl]:>8.4f} {results['p'][wl]:>8.4f} {cref_s:>20}")
