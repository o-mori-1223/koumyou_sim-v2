"""新しい「段差＋列」ジオメトリでの追加比較・8列版（HANDOVER.md 5.6参照）。

st235構造（comsol_reference_st235.py）と列幅・多層膜レシピ・基板光学定数は完全に同じ
だが、**上げる列を入れ替えた**バージョン: st235では幅300nmの列（C, 奇数インデックス）
だけが235nm上げられているのに対し、こちらは幅900/600nmの列（A/B, 偶数インデックス）
だけを上げる。列幅の並び自体は変えず「どの列を上げるか」だけを反転させることで、
st235が「あえて段をずらしている」非対称配置そのものが不一致の原因かどうかを
切り分ける。COMSOLの新規モデリングは行わず、Gsolverのテキストインポート
（RCWA2D/gsolver_export.py）を4つ目の独立参照ツールとして使う。

RAISES_INVERT は元々 diag_st235_hypotheses.py / diag_st235_hypotheses2.py で
「上げパターン反転」仮説として rcwa_mh のみでテスト済み（結果はbaselineより悪化、
ただし比較対象はst235用のCOMSOLデータのままだったので「反転構造としての物理的な
妥当性」自体は未検証）。今回はGsolverで独立に計算し、この反転構造そのものの
振る舞いを確認する。

構造:
  - TiO2/SiO2 7.5ペア・基板n=1.71+2.88j・上げ高さ235nm（st235と同一）
  - 列幅（8列）: [900, 300, 900, 300, 600, 300, 600, 300] nm（st235と同一、周期4200nm）
  - 上げ列: 幅900/600nmの列（インデックス0,2,4,6）のみ。幅300nmの列は上げなし
    （st235の逆）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D.structure_builder import columns_to_layers
from comsol_reference_st235 import (
    COL_WS, RAISE_H, PITCH_NM, TH_A, TH_B, N_PAIRS,
    N_TIO2, N_SIO2, N_SUS, NK_FN_MAP, NK_SUBST_FN,
)

RAISES_INVERT = [RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H, 0.0]

GLOBAL_FILM = [
    {'type': 'simple', 'material': 'TiO2', 'thickness': TH_A},
    {'type': 'pair', 'mat_a': 'TiO2', 'th_a': TH_A, 'mat_b': 'SiO2', 'th_b': TH_B, 'n_pairs': N_PAIRS},
]


def build_columns():
    return [{'width': w, 'subst_raise': r, 'film_layers': GLOBAL_FILM}
            for w, r in zip(COL_WS, RAISES_INVERT)]


def build_layers():
    return columns_to_layers(build_columns(), flat_align=False)


if __name__ == '__main__':
    layers = build_layers()
    print(f"pitch = {PITCH_NM} nm, {len(COL_WS)} columns, {len(layers)} RCWA layers")
    for w, r in zip(COL_WS, RAISES_INVERT):
        print(f"  width={w:>6.1f}nm  raise={r:>6.1f}nm")
