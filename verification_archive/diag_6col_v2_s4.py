"""gsolver_reference_6col_v2.py の構造 (周期3000nm/6列、幅[900,300,600,300,600,300]nm)
をS4で計算し、rcwa_mh・Gsolverと3者比較する。両方の上げパターンを試す:
  baseline: 幅900/600/600の列(index 0,2,4)を235nm上げ
  inverted: 幅300nmの列3本(index 1,3,5)を235nm上げ (diag_6col_v2_hypotheses.pyで
            mean|diff|が改善した方 -- st235と同じ側)
目的: rcwa_mh–Gsolver間の不一致が、rcwa_mh固有のバグなのか、両ソルバーが共有する
structure_builder.pyの列分解ロジックの問題なのかを切り分ける。
S4はrcwa_mhより1点あたり大幅に遅いので、法線入射・s偏光のみ(41波長)に絞る。

Run in WSL: source ~/s4env/bin/activate && cd .../koumyou_sim && python3 diag_6col_v2_s4.py
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh
from RCWA2D.rcwa_s4_backend import calc_rcwa1d_s4
from RCWA2D.structure_builder import columns_to_layers, get_layer_tuple
from gsolver_reference_6col_v2 import COL_WS, PITCH_NM, NK_FN_MAP, NK_SUBST_FN, GLOBAL_FILM, load_gsolver_data

NORDER = 43
p0 = NORDER // 2
PITCH_UM = PITCH_NM / 1000.0
N_SUS = 1.71 + 2.88j

gsolver = load_gsolver_data()
wls = sorted(gsolver.keys())
wl_ar = np.array(wls, dtype=float)


def make_columns(raises):
    return [{'width': w, 'subst_raise': r, 'film_layers': GLOBAL_FILM} for w, r in zip(COL_WS, raises)]


def rcwa_mh_spectrum(layers):
    out = {}
    for wl in wls:
        wl_um = wl / 1000.0
        lt = get_layer_tuple(wl, layers, NK_FN_MAP, 1.0, lambda w: N_SUS)
        ir, it = rcwa_mh.Rcwa1d('s', wl_um, 0.0, PITCH_UM, lt, NORDER)
        out[wl] = float(ir[p0])
    return out


def s4_spectrum(layers):
    irp, itp, irs, its = calc_rcwa1d_s4(wl_ar, 0.0, PITCH_UM, NORDER, layers, NK_FN_MAP, 1.0, lambda w: N_SUS)
    return {wl: float(irs[i, p0]) for i, wl in enumerate(wls)}


def stats(res, ref):
    diffs = np.array([res[wl] - ref[wl] for wl in wls])
    return np.mean(np.abs(diffs)), np.sqrt(np.mean(diffs ** 2)), np.max(np.abs(diffs))


RAISE_H = 235.0
PATTERNS = {
    'baseline (900/600/600 raised)': [RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H, 0.0],
    'inverted (300nm x3 raised)':     [0.0, RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H],
}

for name, raises in PATTERNS.items():
    print(f"=== {name} ===", flush=True)
    layers = columns_to_layers(make_columns(raises), flat_align=False)
    print("  running rcwa_mh...", flush=True)
    mh = rcwa_mh_spectrum(layers)
    print("  running S4 (slow)...", flush=True)
    s4 = s4_spectrum(layers)

    mh_vs_gs = stats(mh, gsolver)
    s4_vs_gs = stats(s4, gsolver)
    s4_vs_mh = stats(s4, mh)
    print(f"  rcwa_mh vs Gsolver: mean|diff|={mh_vs_gs[0]:.4f} RMS={mh_vs_gs[1]:.4f} max={mh_vs_gs[2]:.4f}")
    print(f"  S4      vs Gsolver: mean|diff|={s4_vs_gs[0]:.4f} RMS={s4_vs_gs[1]:.4f} max={s4_vs_gs[2]:.4f}")
    print(f"  S4      vs rcwa_mh: mean|diff|={s4_vs_mh[0]:.4f} RMS={s4_vs_mh[1]:.4f} max={s4_vs_mh[2]:.4f}")
    print()
    print(f"  {'wl':>5} {'rcwa_mh':>9} {'S4':>9} {'Gsolver':>9}")
    for wl in wls[::4]:
        print(f"  {wl:>5.0f} {mh[wl]:>9.4f} {s4[wl]:>9.4f} {gsolver[wl]:>9.4f}")
    print()
