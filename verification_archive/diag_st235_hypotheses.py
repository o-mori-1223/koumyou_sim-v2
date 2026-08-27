"""Quick hypothesis tests for the RCWA-vs-COMSOL mismatch on the st235 structure:
  1. raise pattern inverted (A/B raised instead of C)
  2. substrate index imaginary-part sign flipped (n=1.71-2.88j instead of +2.88j)
on a wavelength subset (every 5th of 41) x all 9 angles x both polarizations.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh
from RCWA2D.structure_builder import columns_to_layers, get_layer_tuple
from comsol_reference_st235 import COL_WS, RAISE_H, PITCH_NM, load_comsol_data

NORDER = 61
p0 = NORDER // 2
PITCH_UM = PITCH_NM / 1000.0
TH_A, TH_B = 77.6, 54.4
N_PAIRS = 7
data = load_comsol_data()
wls_all = sorted(data['s'].keys())
subset = wls_all[::5]
angles = sorted(next(iter(data['s'].values())).keys())
print(f"subset: {len(subset)} wavelengths x {len(angles)} angles x 2 pol = "
      f"{len(subset)*len(angles)*2} rcwa calls per variant", flush=True)


def make_columns(raises):
    def mk(w, r):
        return {'width': w, 'subst_raise': r, 'film_layers': [
            {'type': 'simple', 'material': 'TiO2', 'thickness': TH_A},
            {'type': 'pair', 'mat_a': 'TiO2', 'th_a': TH_A, 'mat_b': 'SiO2', 'th_b': TH_B,
             'n_pairs': N_PAIRS}]}
    return [mk(w, r) for w, r in zip(COL_WS, raises)]


def run(raises, n_sus):
    nk_fn_map = {'TiO2': lambda wl: 2.30 + 0j, 'SiO2': lambda wl: 1.45 + 0j}
    nk_subst_fn = lambda wl: n_sus
    layers = columns_to_layers(make_columns(raises), flat_align=False)
    out = {'s': {}, 'p': {}}
    for wl in subset:
        wl_um = wl / 1000.0
        lt = get_layer_tuple(wl, layers, nk_fn_map, 1.0, nk_subst_fn)
        for pol in ('s', 'p'):
            row = {}
            for th in angles:
                coef = float(2 * np.pi * np.sin(th * np.pi / 180.0) / wl_um)
                ir, it = rcwa_mh.Rcwa1d(pol, wl_um, coef, PITCH_UM, lt, NORDER)
                row[th] = float(ir[p0])
            out[pol][wl] = row
        print(f"  ...done wl={wl}", flush=True)
    return out


def stats(res):
    diffs = []
    for pol in ('s', 'p'):
        for wl in subset:
            for th in angles:
                diffs.append(res[pol][wl][th] - data[pol][wl][th])
    diffs = np.array(diffs)
    return np.mean(np.abs(diffs)), np.sqrt(np.mean(diffs ** 2))


RAISES_ORIG = [0.0, RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H]
RAISES_INVERT = [RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H, 0.0, RAISE_H, 0.0]
N_SUS_POS = 1.71 + 2.88j
N_SUS_NEG = 1.71 - 2.88j

print("=== baseline (C-raised, +2.88i) ===", flush=True)
r1 = stats(run(RAISES_ORIG, N_SUS_POS))
print("=== inverted-raise (A/B-raised, +2.88i) ===", flush=True)
r2 = stats(run(RAISES_INVERT, N_SUS_POS))
print("=== sign-flip subst (C-raised, -2.88i) ===", flush=True)
r3 = stats(run(RAISES_ORIG, N_SUS_NEG))
print("=== both alt (A/B-raised, -2.88i) ===", flush=True)
r4 = stats(run(RAISES_INVERT, N_SUS_NEG))

print()
print(f"baseline  (C-raised, +2.88i):      mean|diff|={r1[0]:.4f} RMS={r1[1]:.4f}")
print(f"inverted-raise (A/B-raised,+2.88i): mean|diff|={r2[0]:.4f} RMS={r2[1]:.4f}")
print(f"sign-flip subst (C-raised,-2.88i):  mean|diff|={r3[0]:.4f} RMS={r3[1]:.4f}")
print(f"both alt (A/B-raised, -2.88i):      mean|diff|={r4[0]:.4f} RMS={r4[1]:.4f}")
