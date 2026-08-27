"""gsolver_reference_oldstruct_235_rounded.py の構造 (周期3000nm/6列、幅
[900,600,300,300,600,300]nm、列0,2,4を235nm上げ、TiO2=54nm/SiO2=77nm)をS4で計算し、
rcwa_mh・Gsolverと3者比較する。s/p両偏光、法線入射、全41波長。
Run in WSL: source ~/s4env/bin/activate && cd .../koumyou_sim && python3 diag_oldstruct_235_rounded_s4.py
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D.rcwa_s4_backend import calc_rcwa1d_s4
from gsolver_reference_oldstruct_235_rounded import build_layers, PITCH_NM, NK_FN_MAP, NK_SUBST_FN

NORDER = 43
p0 = NORDER // 2
PITCH_UM = PITCH_NM / 1000.0

layers = build_layers()
wls = list(range(380, 781, 10))
wl_ar = np.array(wls, dtype=float)

irp, itp, irs, its = calc_rcwa1d_s4(wl_ar, 0.0, PITCH_UM, NORDER, layers, NK_FN_MAP, 1.0, NK_SUBST_FN)

results = {'s': {wl: float(irs[i, p0]) for i, wl in enumerate(wls)},
           'p': {wl: float(irp[i, p0]) for i, wl in enumerate(wls)}}

out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 's4_oldstruct_235_rounded_result.json')
with open(out_json, 'w') as f:
    json.dump(results, f)
print(f"wrote: {out_json}")

print()
print(f"{'wl':>5} {'S4(s)':>8} {'S4(p)':>8}")
for wl in wls[::4]:
    print(f"{wl:>5} {results['s'][wl]:>8.4f} {results['p'][wl]:>8.4f}")
