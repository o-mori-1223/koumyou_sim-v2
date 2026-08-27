"""新しい「段差＋列」ジオメトリでの追加比較・6列版（HANDOVER.md 5.6参照）。

ユーザーがGsolverで直接構築・計算した構造（2026-08-24、data/gsolver_6col_v2.csv）:
  - 周期3000nm、6列: 幅 [900, 300, 600, 300, 600, 300] nm
  - 上げ列: 幅900・600・600の列（インデックス0,2,4）のみ235nm上げ。幅300nmの列
    （インデックス1,3,5）は上げなし
  - 上げ高さ235nm・TiO2/SiO2 7.5ペアレシピ・基板n=1.71+2.88jはst235と同一
    （ユーザーに確認済み、2026-08-24）

旧構造（comsol_reference.py: 幅[900,600,300,300,600,300]nm・上げ高さ100nm・
列1,3,5(900,300,600)上げ）とは列の並び順・上げ高さの両方が異なる、別のジオメトリ。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D.structure_builder import columns_to_layers

TH_A, TH_B = 77.6, 54.4         # TiO2 (top+bottom), SiO2
N_PAIRS = 7                      # + bottom TiO2 half-pair = 7.5 pairs total
COL_WS = [900.0, 300.0, 600.0, 300.0, 600.0, 300.0]
RAISE_H = 235.0
RAISES = [RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H, 0.0]   # 列0,2,4(900,600,600)を上げ
PITCH_NM = sum(COL_WS)           # 3000.0

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


def load_gsolver_data(path=None):
    """data/gsolver_6col_v2.csv -> {wl_nm(int): R}"""
    import csv
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'gsolver_6col_v2.csv')
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            wl, r = row
            out[int(float(wl))] = float(r)
    return out


if __name__ == '__main__':
    layers = build_layers()
    print(f"pitch = {PITCH_NM} nm, {len(COL_WS)} columns, {len(layers)} RCWA layers")
    for w, r in zip(COL_WS, RAISES):
        print(f"  width={w:>6.1f}nm  raise={r:>6.1f}nm")
