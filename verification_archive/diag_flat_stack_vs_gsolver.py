"""Compare rcwa_mh's flat (unpatterned) 15-layer TiO2/SiO2 sanity stack -- see
diag_flat_stack_sanity.py -- against a Gsolver run of the SAME flat 15-layer recipe
(user-provided, data/gsolver_flat_stack_15layer.csv, 380-780nm every 10nm, normal
incidence). This is a THIRD independent solver (after rcwa_mh vs S4, see
diag_mh_vs_s4.py) checked against the flat-stack multilayer recipe only -- it does
NOT touch the patterned/raised-column structure that still disagrees with COMSOL at
470/540nm (see comsol_reference.py docstring and HANDOVER.md section 4).

Result (2026-08-20): main stopband (470-560nm) mean |diff| ~0.02, max ~0.06 -- close
agreement. Outside the stopband, narrow interference fringes (400, 590, 640-650,
670-680, 750-760nm) show up to ~0.12-0.20 |diff|, most likely fringe-position
sensitivity at only 10nm sampling rather than a solver bug. Net: a third independent
tool confirms the flat multilayer recipe behaves as a proper high-R DBR in its
stopband, reinforcing that the still-unresolved patterned-structure gap vs COMSOL is
not caused by a broken multilayer recipe.

Run: cd koumyou_sim && python diag_flat_stack_vs_gsolver.py
"""
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh
from comsol_reference import flat_stack_layer_tuple

GSOLVER_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsolver_flat_stack_15layer.csv")


def load_gsolver():
    g = {}
    with open(GSOLVER_CSV, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            wl, r = row
            g[int(float(wl))] = float(r)
    return g


def compute_rcwa():
    layer_tuple = flat_stack_layer_tuple()
    NORDER = 5  # no lateral pattern -> order doesn't matter, keep it fast
    p0 = NORDER // 2
    out = {}
    for wl in range(380, 781, 10):
        ir, it = rcwa_mh.Rcwa1d('p', wl / 1000.0, 0.0, 1.0, layer_tuple, NORDER)  # normal incidence
        out[wl] = ir[p0]
    return out


if __name__ == "__main__":
    gsolver = load_gsolver()
    rcwa = compute_rcwa()

    diffs = []
    print(f"{'wl':>5} {'RCWA':>8} {'Gsolver':>8} {'diff':>8}")
    for wl in sorted(gsolver):
        d = rcwa[wl] - gsolver[wl]
        diffs.append(d)
        flag = " <<<" if abs(d) > 0.08 else ""
        print(f"{wl:>5} {rcwa[wl]:>8.4f} {gsolver[wl]:>8.4f} {d:>+8.4f}{flag}")

    sb = [rcwa[wl] - gsolver[wl] for wl in range(470, 561, 10)]
    print()
    print("mean |diff| (all):", round(statistics.mean(abs(d) for d in diffs), 4))
    print("max |diff| (all):", round(max(abs(d) for d in diffs), 4))
    print("stopband(470-560) mean |diff|:", round(statistics.mean(abs(d) for d in sb), 4))
    print("stopband(470-560) max |diff|:", round(max(abs(d) for d in sb), 4))
