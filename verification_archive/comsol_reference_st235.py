"""Reference structure + COMSOL ground-truth data for the NEW "4200nm / 8列 / 段差235nm"
structure (distinct from the 3000nm/6列 structure in comsol_reference.py).

Structure, as given by the user 2026-08-19 (image + COMSOL file name
`A0_C1_A0_C1_B0_C1_B0_C1_st235_new.mph`):
  - TiO2/SiO2 周期多層膜 7.5ペア（最上層・最下層ともTiO2）: bottom-to-top =
    TiO2, (SiO2,TiO2) x7 = 8xTiO2 + 7xSiO2 = 15層
  - TiO2: 77.6nm, n=2.30 (実数、k=0)。SiO2: 54.4nm, n=1.45 (実数、k=0)
  - 基盤屈折率: n = 1.71 + 2.88j
  - 基盤上げ高さ: 235nm (= λ0/2, 画像ラベルより)
  - 周期(ピッチ): 4200nm
  - 列構成 (左→右, 8列): ファイル名 `A0_C1_A0_C1_B0_C1_B0_C1` を
    A=900nm, B=600nm, C=300nm、フラグ 0=上げなし/1=上げ、と解読
    (画像の断面図 [900,300,900,300,600,300,600,300]nm・上げ位置の網掛け、および
    「基盤上げの割合: 28.5%」= 1200/4200 と全て整合 -- 3つの独立情報源で確認済み):
        column:  A0    C1    A0    C1    B0    C1    B0    C1
        width :  900   300   900   300   600   300   600   300   (nm)
        raise :  0     235   0     235   0     235   0     235   (nm)
    つまり幅300nmの列(4列とも)だけが235nm上げられており、900nm/600nm幅の列は
    上げられていない。ユーザー強調点: **この構造はあえて段をずらしている**
    （対称・周期整列ではなく、上げ列と非上げ列の幅が列ごとに異なる非対称配置）。

COMSOL データ: `data/comsol_st235_new.csv` (COMSOL 6.4.0, 同モデル名, Table 4 -
Global Evaluation 4)。列は eta_P, lamda(nm), theta(deg), freq(THz), Reflectance order0。
波長 380-780nm を10nm刻み(41点)、角度 0-80degを10deg刻み(9点)、eta_P in {0,1}。
eta_P=0 -> s偏光, eta_P=1 -> p偏光と推定（旧データセットの命名規則を踏襲したと
仮定 -- comsol_reference.py 参照。このマッピングは未確認なので、s/pの取り違えが
疑わしい場合はまずここを反転して再検証すること）。
"""
import os
import csv

TH_A, TH_B = 77.6, 54.4        # TiO2 (top+bottom), SiO2
N_PAIRS = 7                     # + bottom TiO2 half-pair = 7.5 pairs total
COL_WS = [900.0, 300.0, 900.0, 300.0, 600.0, 300.0, 600.0, 300.0]
RAISE_H = 235.0
RAISES = [0.0, RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H]
PITCH_NM = sum(COL_WS)   # 4200.0

N_TIO2 = 2.30 + 1e-4j   # 数値安定化用の微小損失（GSolverのNaN対策、GSolver.iniと揃える）
N_SIO2 = 1.45 + 1e-4j
N_SUS = 1.71 + 2.88j

NK_FN_MAP = {'TiO2': lambda wl: N_TIO2, 'SiO2': lambda wl: N_SIO2}
NK_SUBST_FN = lambda wl: N_SUS

_DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'comsol_st235_new.csv')


def build_columns(raises=None):
    """R9-style `columns` list (see structure_builder.py) for this structure.
    Pass a custom `raises` list (len 8) to test alternative raise patterns/heights
    (e.g. for a boundary-condition / registration sensitivity sweep)."""
    if raises is None:
        raises = RAISES

    def make_column(width, raise_h):
        return {
            'width': width, 'subst_raise': raise_h,
            'film_layers': [
                {'type': 'simple', 'material': 'TiO2', 'thickness': TH_A},
                {'type': 'pair', 'mat_a': 'TiO2', 'th_a': TH_A,
                 'mat_b': 'SiO2', 'th_b': TH_B, 'n_pairs': N_PAIRS},
            ],
        }
    return [make_column(w, r) for w, r in zip(COL_WS, raises)]


def load_comsol_data(path=_DATA_CSV):
    """Parse data/comsol_st235_new.csv -> {pol: {wl_nm(float): {theta_deg(float): R}}}
    where pol in ('s', 'p') per the eta_P=0->s / eta_P=1->p mapping (see module docstring).
    Wavelength/angle keys are rounded to avoid float-repr duplicates (e.g. 379.99999999994).
    """
    out = {'s': {}, 'p': {}}
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('%'):
                continue
            eta_p, wl, theta, freq, r = row
            pol = 'p' if eta_p.strip() == '1' else 's'
            wl_k = round(float(wl), 3)
            th_k = round(float(theta), 3)
            out[pol].setdefault(wl_k, {})[th_k] = float(r)
    return out


if __name__ == '__main__':
    cols = build_columns()
    print(f"pitch = {PITCH_NM} nm, {len(cols)} columns")
    for c in cols:
        print(f"  width={c['width']:>6.1f}nm  raise={c['subst_raise']:>6.1f}nm")
    data = load_comsol_data()
    for pol in ('s', 'p'):
        wls = sorted(data[pol].keys())
        n_th = len(next(iter(data[pol].values())))
        print(f"pol={pol}: {len(wls)} wavelengths x {n_th} angles "
              f"(wl range {wls[0]}-{wls[-1]}nm)")
