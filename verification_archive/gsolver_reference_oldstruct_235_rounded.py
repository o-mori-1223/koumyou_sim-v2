"""旧構造(comsol_reference.py)の列幅[900,600,300,300,600,300]nm・列0,2,4(900,300,600)を
上げるパターンはそのままに、次の2点だけ変えたバリエーション（HANDOVER.md 5.6参照）:
  - 上げ高さ: 100nm → 235nm（今セッションの他の構造と揃える）
  - 膜厚: TiO2=77.6nm/SiO2=54.4nm(厳密値) → TiO2=77nm/SiO2=54nm(丸めた値、
    ユーザーがCanvas手描き版で実際に入力していた値)

目的: 「膜厚パラメータの丸め（0.6nm+0.4nm×7.5ペア=累積約7.5nm相当）」が数値に
どれだけ影響するかを、Canvas量子化とは完全に切り離してテキストインポートのみで
直接検証する。基板n=1.71+2.88j・TiO2/SiO2は数値安定化用の微小損失
(n=2.30+1e-4j, n=1.45+1e-4j)込みで、他の現行構造と揃えてある。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D.structure_builder import columns_to_layers

TH_A, TH_B = 54.0, 77.0         # 丸めた膜厚（TiO2=54nm, SiO2=77nm。ユーザー指定に合わせて入替済み）
N_PAIRS = 7                      # + bottom TiO2 half-pair = 7.5 pairs total
COL_WS = [900.0, 600.0, 300.0, 300.0, 600.0, 300.0]
RAISE_H = 235.0                  # 旧構造の100nmではなく235nm
RAISES = [RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H, 0.0]   # 列0,2,4(900,300,600)を上げ
PITCH_NM = sum(COL_WS)           # 3000.0

N_TIO2 = 2.30 + 1e-4j
N_SIO2 = 1.45 + 1e-4j
N_SUS = 1.71 + 2.88j

NK_FN_MAP = {'TiO2': lambda wl: N_TIO2, 'SiO2': lambda wl: N_SIO2}
NK_SUBST_FN = lambda wl: N_SUS

GLOBAL_FILM = [
    {'type': 'simple', 'material': 'TiO2', 'thickness': TH_A},
    {'type': 'pair', 'mat_a': 'TiO2', 'th_a': TH_A, 'mat_b': 'SiO2', 'th_b': TH_B, 'n_pairs': N_PAIRS},
]


def build_columns():
    return [{'width': w, 'subst_raise': r, 'film_layers': GLOBAL_FILM}
            for w, r in zip(COL_WS, RAISES)]


def build_layers():
    return columns_to_layers(build_columns(), flat_align=False)


if __name__ == '__main__':
    layers = build_layers()
    print(f"pitch = {PITCH_NM} nm, {len(COL_WS)} columns, {len(layers)} RCWA layers")
    for w, r in zip(COL_WS, RAISES):
        print(f"  width={w:>6.1f}nm  raise={r:>6.1f}nm")
