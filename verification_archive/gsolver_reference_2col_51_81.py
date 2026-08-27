"""gsolver_reference_2col.py と同じ2列ジオメトリ（幅900nm＋幅300nm、周期1200nm、
幅300nmの列だけ235nm上げ）に、新しい膜厚レシピ（TiO2=51nm, SiO2=81nm）を組み合わせた
バリエーション（HANDOVER.md 5.6参照）。基板n=1.71+2.88j・7.5ペア・TiO2/SiO2の
数値安定化用微小損失(k=1e-4)は他の構造と同一。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D.structure_builder import columns_to_layers

TH_A, TH_B = 51.0, 81.0         # TiO2, SiO2
N_PAIRS = 7                      # + bottom TiO2 half-pair = 7.5 pairs total
COL_WS = [900.0, 300.0]
RAISE_H = 235.0
RAISES = [0.0, RAISE_H]          # 幅300nmの列（列1）のみ上げ
PITCH_NM = sum(COL_WS)           # 1200.0

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
