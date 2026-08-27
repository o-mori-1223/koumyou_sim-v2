"""gsolver_reference_6col_v2.py の構造 (周期3000nm/6列、幅[900,300,600,300,600,300]nm、
幅900/600/600の列だけ235nm上げ) を rcwa_mh で計算し、ユーザー提供のGsolver結果
(data/gsolver_6col_v2.csv, 2026-08-24, Gsolver側はs偏光(TE)で計算とユーザー確認済み)
と比較する。

注意: 当初「法線入射ならs/p偏光は縮退するはず」という前提でs/p両方を計算し一致するか
チェックするコードを入れていたが、これは誤り -- 平坦な多層膜スタックなら法線入射で
s/p縮退するが、列(グレーティング)構造は周期方向の面内回転対称性が破れているため
法線入射でもs/pは一致しない（実際 max|Rs-Rp|~0.30 の差があった）。以下ではGsolverと
同じs偏光の結果のみを主に使う。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh
from RCWA2D.structure_builder import get_layer_tuple
from gsolver_reference_6col_v2 import build_layers, PITCH_NM, NK_FN_MAP, NK_SUBST_FN, load_gsolver_data

NORDER = 43   # 旧構造(3000nm)と同じピッチなのでdiag_convergence.pyで収束確認済みの次数を流用
p0 = NORDER // 2
PITCH_UM = PITCH_NM / 1000.0

layers = build_layers()
print(f"pitch: {PITCH_NM}nm, RCWA layers: {len(layers)}")

gsolver = load_gsolver_data()
wls = sorted(gsolver.keys())

rcwa_s, rcwa_p = {}, {}
for wl in wls:
    wl_um = wl / 1000.0
    lt = get_layer_tuple(wl, layers, NK_FN_MAP, 1.0, NK_SUBST_FN)
    ir_s, _ = rcwa_mh.Rcwa1d('s', wl_um, 0.0, PITCH_UM, lt, NORDER)
    ir_p, _ = rcwa_mh.Rcwa1d('p', wl_um, 0.0, PITCH_UM, lt, NORDER)
    rcwa_s[wl] = float(ir_s[p0])
    rcwa_p[wl] = float(ir_p[p0])

sp_diff = np.array([rcwa_s[wl] - rcwa_p[wl] for wl in wls])
print(f"s/p degeneracy check (should be ~0 at normal incidence): "
      f"max|Rs-Rp|={np.max(np.abs(sp_diff)):.5f}")

print()
print("=== rcwa_mh (s-pol, matches Gsolver's confirmed TE setting) vs Gsolver ===")
print(f"{'wl':>5} {'rcwa_mh(s)':>10} {'Gsolver':>8} {'diff':>8}")
diffs = []
for wl in wls:
    d = rcwa_s[wl] - gsolver[wl]
    diffs.append(d)
    flag = " <<<" if abs(d) > 0.1 else ""
    print(f"{wl:>5} {rcwa_s[wl]:>10.4f} {gsolver[wl]:>8.4f} {d:>+8.4f}{flag}")

diffs = np.array(diffs)
print()
print(f"mean|diff|={np.mean(np.abs(diffs)):.4f}  max|diff|={np.max(np.abs(diffs)):.4f}  "
      f"RMS={np.sqrt(np.mean(diffs**2)):.4f}")
