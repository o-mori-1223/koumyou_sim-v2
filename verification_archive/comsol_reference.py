"""Shared reference structure + COMSOL ground-truth data for the diag_comsol_*.py scripts.

Structure confirmed with the user on 2026-08-04 while investigating why R9's computed
reflectance didn't match a real measured/simulated mirror-stack (COMSOL FEM + Gsolver):
  - pitch 3000nm, 6 columns of width 900/600/300/300/600/300 nm (left to right)
  - columns 1, 3, 5 (the 900/300/600nm-wide ones) have a 100nm substrate raise;
    columns 2, 4, 6 sit flat. (Confirmed from a COMSOL cross-section screenshot showing
    4 raised "towers" with sharp, non-tapered vertical edges -- i.e. no sidewall coating,
    each column stacks its own film vertically only. An earlier "sidewall coating" model
    was built and then removed after this was confirmed -- see git history.)
  - each column: TiO2/SiO2 x 7.5 pairs, TiO2 touching BOTH the substrate side and the air
    side (i.e. bottom-to-top: TiO2, (SiO2,TiO2) x7 = 8x TiO2 + 7x SiO2 = 15 layers/column)
  - TiO2: 77.6nm, n=2.30 (constant, k=0). SiO2: 54.4nm, n=1.45 (constant, k=0)
  - substrate: SUS, n=1.71+2.88j. Physically 300nm thick but confirmed by the user to be
    effectively semi-infinite (k is large enough that backside reflection is negligible),
    so it's modeled as a semi-infinite substrate.

Investigation summary (as of 2026-08-04): with this structure, computed Rp matches COMSOL's
resonance POSITION well (peaks/dips align in wavelength -- see diag_comsol_fullspectrum.py)
but under-predicts the PEAK AMPLITUDE at 470/540nm specifically (~0.2 computed vs ~0.7
COMSOL). Ruled out as causes: nOrder truncation (converged already at 43, see
diag_convergence.py), a bug in rcwa_mh's or S4's handling of multi-column lateral patterns
(the two independently-implemented solvers agree to within ~0.01, see diag_mh_vs_s4.py),
and the multilayer recipe itself (a flat/unpatterned version of the same 15-layer stack
gives R~0.99 at the same wavelengths, matching COMSOL's peak magnitude -- see
diag_flat_stack_sanity.py). Reflectance is extremely sensitive to the substrate-raise
height (0.99 at 0nm raise down to 0.20 at 100nm raise, smoothly -- see
diag_raise_sensitivity.py), so the remaining gap is most likely a still-unconfirmed detail
of exactly which columns/how much they're raised in the real COMSOL model, not a solver bug.
"""

TH_A, TH_B = 77.6, 54.4       # TiO2 thickness (air + substrate side), SiO2 thickness
N_PAIRS = 7                    # + the extra bottom TiO2 half-pair = 7.5 pairs total
COL_WS = [900.0, 600.0, 300.0, 300.0, 600.0, 300.0]
RAISES = [100.0, 0.0, 100.0, 0.0, 100.0, 0.0]   # columns 1, 3, 5 raised
PITCH_NM = sum(COL_WS)

N_TIO2 = 2.30 + 1e-4j   # 数値安定化用の微小損失（GSolverのNaN対策、GSolver.iniと揃える）
N_SIO2 = 1.45 + 1e-4j
N_SUS = 1.71 + 2.88j

NK_FN_MAP = {'TiO2': lambda wl: N_TIO2, 'SiO2': lambda wl: N_SIO2}
NK_SUBST_FN = lambda wl: N_SUS


def build_columns(raises=None):
    """R9-style `columns` list (see structure_builder.py) for the reference structure.
    Pass a custom `raises` list (same length as COL_WS) to sweep the raise pattern/height,
    e.g. for diag_raise_sensitivity.py.
    """
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


def flat_stack_layer_tuple():
    """rcwa_mh substrate-first layer tuple for the SAME 15-layer recipe with no lateral
    patterning at all (used by diag_flat_stack_sanity.py as a ground-truth sanity check --
    a well-designed 7.5-pair TiO2/SiO2 DBR should reflect close to 1.0 in its stopband).
    """
    seq = [('TiO2', TH_A)]
    for _ in range(N_PAIRS):
        seq.append(('SiO2', TH_B))
        seq.append(('TiO2', TH_A))
    layer_tuple = [(0, N_SUS, 0)]
    for mat, th in seq:
        n = N_TIO2 if mat == 'TiO2' else N_SIO2
        layer_tuple.append((th / 1000.0, n, 1.0))
    layer_tuple.append((0, 1.0 + 0j, 0))
    return tuple(layer_tuple)


# ----------------------------------------------------------------------------------------
# COMSOL reference data (source: user-provided "900基盤上げ.csv", COMSOL 6.4.0, model
# A1_B0_C1_C0_B1_C0_riron_st235.mph, Table 3 - Global Evaluation 3, eta_P=1 => p-polarization
# per the user; eta_P=0 rows (s-polarization) were not used here).
# ----------------------------------------------------------------------------------------

# Rp(theta=0), every 10nm from 380 to 780nm -- used by diag_comsol_fullspectrum.py.
COMSOL_RP_AOI0_SPECTRUM = {
    380: 0.091177534, 390: 0.143942175, 400: 0.088049873, 410: 0.109625769,
    420: 0.244652001, 430: 0.224336368, 440: 0.038808105, 450: 0.567478407,
    460: 0.702987097, 470: 0.707093606, 480: 0.678175981, 490: 0.624570845,
    500: 0.591683254, 510: 0.612334253, 520: 0.653145815, 530: 0.592087924,
    540: 0.451591830, 550: 0.402061588, 560: 0.385331885, 570: 0.337329093,
    580: 0.263006316, 590: 0.194829294, 600: 0.151422650, 610: 0.153841451,
    620: 0.139964980, 630: 0.111681322, 640: 0.069771875, 650: 0.038795209,
    660: 0.076959774, 670: 0.123289050, 680: 0.163900837, 690: 0.154862888,
    700: 0.144379151, 710: 0.153543017, 720: 0.136727054, 730: 0.097902188,
    740: 0.047764114, 750: 0.006630478, 760: 0.001504566, 770: 0.030247166,
    780: 0.072267794,
}

# Rp at 5 wavelengths x 9 angles (0-80deg, step 10) -- used by diag_comsol_comparison.py.
COMSOL_RP_BY_ANGLE = {
    400: {0: 0.088049873, 10: 0.036880083, 20: 0.039676345, 30: 0.010492013,
          40: 0.005094524, 50: 0.016330603, 60: 0.043430225, 70: 0.073731619,
          80: 0.194251302},
    470: {0: 0.707093606, 10: 0.560528493, 20: 0.367587788, 30: 0.385245414,
          40: 0.162559475, 50: 0.169885824, 60: 0.055149235, 70: 0.027692242,
          80: 0.043381898},
    540: {0: 0.451591830, 10: 0.548742182, 20: 0.404114129, 30: 0.379202559,
          40: 0.220909280, 50: 0.251000829, 60: 0.195307868, 70: 0.174976172,
          80: 0.287502254},
    600: {0: 0.151422650, 10: 0.197495057, 20: 0.158610208, 30: 0.121988185,
          40: 0.086406323, 50: 0.045434439, 60: 0.053461011, 70: 0.066708587,
          80: 0.273265178},
    700: {0: 0.144379151, 10: 0.172362970, 20: 0.110095711, 30: 0.049003254,
          40: 0.013164355, 50: 0.000850000, 60: 0.003441576, 70: 0.033147762,
          80: 0.256948583},
}
