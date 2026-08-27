"""新しい「段差＋列」ジオメトリでの追加比較の第1弾（HANDOVER.md 5.6参照）。

st235構造（comsol_reference_st235.py）と全く同じ多層膜レシピ・上げ高さ・基板光学定数を
流用しつつ、列数を8→2に減らした最小構成。列分解＋段差の核心ロジックだけを切り出して
検証できる。COMSOLではなくGsolver（テキストインポート、RCWA2D/gsolver_export.py）を
参照ツールとして使う（HANDOVER.md 5.6の方針）。

構造:
  - TiO2/SiO2 7.5ペア（st235と同一）: 膜厚 TiO2=77.6nm, SiO2=54.4nm, n(TiO2)=2.30,
    n(SiO2)=1.45（定数、吸収なし）
  - 基板屈折率: n = 1.71 + 2.88j（st235と同一）
  - 基板上げ高さ: 235nm（st235と同一）
  - 列構成（2列）: 幅900nm（上げなし）+ 幅300nm（上げ235nm）、周期1200nm
  - 堆積モデル: 等形成膜（conformal, flat_align=False）のみ
    （2026-08-20、平坦界面モデルはst235でむしろ不一致を悪化させると判明したため
    廃止済み — HANDOVER.md 4.の8番参照）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D.structure_builder import columns_to_layers

TH_A, TH_B = 77.6, 54.4         # TiO2 (top+bottom), SiO2
N_PAIRS = 7                      # + bottom TiO2 half-pair = 7.5 pairs total
COL_WS = [900.0, 300.0]
RAISE_H = 235.0
RAISES = [0.0, RAISE_H]
PITCH_NM = sum(COL_WS)           # 1200.0

N_TIO2 = 2.30 + 1e-4j   # 数値安定化用の微小損失（GSolverのNaN対策、GSolver.iniと揃える）
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
    for c, w, r in zip(build_columns(), COL_WS, RAISES):
        print(f"  width={w:>6.1f}nm  raise={r:>6.1f}nm")
