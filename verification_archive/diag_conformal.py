"""
診断スクリプト: 等形成膜 vs 平坦界面 比較
実行: cd koumyou_sim && python diag_conformal.py
"""
import sys, os, numpy as np
from scipy.interpolate import interp1d
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh

# ------ 材料関数 ------
def load_nk(name):
    p = os.path.join(os.path.dirname(__file__), "data", "nk", name + ".nk")
    d = np.loadtxt(p, comments=";", encoding="utf-8_sig")
    return interp1d(d[:,0], d[:,1]+d[:,2]*1j, kind="linear", fill_value="extrapolate")

nk_tio2 = load_nk("TiO2")
nk_sio2 = load_nk("SiO2")
nk_sus  = load_nk("SUS")

N_PAIRS = 7
TH_A    = 77.6   # TiO2 [nm]
TH_B    = 54.4   # SiO2 [nm]
RAISE   = 100.0
COL_WS  = [900, 600, 300, 300, 600, 300]
PITCH   = sum(COL_WS)
NORDER  = 43

MAT_ENV  = -1
MAT_SUS  = 0
MAT_SIO2 = 1
MAT_TIO2 = 2

def make_bands(subst_raise):
    bands, h = [], 0.0
    if subst_raise > 1e-9:
        bands.append((h, h+subst_raise, MAT_SUS)); h += subst_raise
    for _ in range(N_PAIRS):
        bands.append((h, h+TH_B, MAT_SIO2)); h += TH_B
        bands.append((h, h+TH_A, MAT_TIO2)); h += TH_A
    return bands

def mat_at(bands, h_mid):
    for lo, hi, mc in bands:
        if lo <= h_mid < hi:
            return mc
    return MAT_ENV

def resolve_n(mc, wl, n_sus_val):
    if mc == MAT_ENV:  return 1.0+0j
    if mc == MAT_SUS:  return n_sus_val
    if mc == MAT_SIO2: return complex(nk_sio2(wl))
    return complex(nk_tio2(wl))

def build_conformal(col_ws, raises, wl, n_sus_val):
    total_w = sum(col_ws)
    all_bands = [make_bands(r) for r in raises]
    heights = sorted({lo for b in all_bands for lo,hi,_ in b} |
                     {hi for b in all_bands for lo,hi,_ in b} | {0.0})
    layers_bf = []
    for i in range(len(heights)-1):
        h_lo, h_hi = heights[i], heights[i+1]
        h_mid = (h_lo+h_hi)/2
        segs = []
        for ci, bands in enumerate(all_bands):
            mc = mat_at(bands, h_mid)
            n  = resolve_n(mc, wl, n_sus_val)
            cw = col_ws[ci]/total_w
            if segs and segs[-1][0] == n:
                segs[-1] = (n, segs[-1][1]+cw)
            else:
                segs.append((n, cw))
        flat = [(h_hi-h_lo)/1000.0]
        for n,w in segs: flat += [n, w]
        layers_bf.append(tuple(flat))
    n_env = 1.0+0j
    return tuple([(0, n_sus_val, 0)] + list(reversed(layers_bf)) + [(0, n_env, 0)])

def build_flat(col_ws, raises, wl, n_sus_val):
    total_w = sum(col_ws)
    max_raise = max(raises)
    n_sio2 = complex(nk_sio2(wl))
    n_tio2 = complex(nk_tio2(wl))
    n_env  = 1.0+0j
    layers_bf = []
    if max_raise > 1e-9:
        segs = []
        for ci, r in enumerate(raises):
            n  = n_sus_val if r > 1e-9 else n_env
            cw = col_ws[ci]/total_w
            if segs and segs[-1][0] == n:
                segs[-1] = (n, segs[-1][1]+cw)
            else:
                segs.append((n, cw))
        flat = [max_raise/1000.0]
        for n,w in segs: flat += [n, w]
        layers_bf.append(tuple(flat))
    for _ in range(N_PAIRS):
        layers_bf.append((TH_B/1000.0, n_sio2, 1.0))
        layers_bf.append((TH_A/1000.0, n_tio2, 1.0))
    return tuple([(0, n_sus_val, 0)] + list(reversed(layers_bf)) + [(0, n_env, 0)])

def R0(layer, pol, wl):
    p = NORDER//2
    ir, _ = rcwa_mh.Rcwa1d(pol, wl/1000.0, 0.0, PITCH/1000.0, layer, NORDER)
    return float(ir[p])

wls = [420, 470, 520, 570, 620, 670]
RAISES_ALT  = [0, RAISE, 0, RAISE, 0, RAISE]   # 交互 (実際の段差)
RAISES_ALL  = [RAISE]*len(COL_WS)                # 全列同じ高さ (段差なし相当)
RAISES_NONE = [0]*len(COL_WS)                    # 全列raise=0

print("="*72)
print("TEST 1: 全列 raise=0 → 等形成膜 ≡ 平坦界面 (一致するはず)")
print("="*72)
print(f"{'wl':>5} | {'等形成膜 Rp':>11} {'平坦界面 Rp':>11} | {'差':>10}")
for wl in wls:
    ns = complex(nk_sus(wl))
    lc = build_conformal(COL_WS, RAISES_NONE, wl, ns)
    lf = build_flat(COL_WS, RAISES_NONE, wl, ns)
    rc = R0(lc, 'p', wl)
    rf = R0(lf, 'p', wl)
    print(f"{wl:>5} | {rc:>11.5f} {rf:>11.5f} | {abs(rc-rf):>10.2e}")

print()
print("="*72)
print("TEST 2: 全列 raise=100nm → 等形成膜 ≡ 平坦界面 (一致するはず)")
print("="*72)
print(f"{'wl':>5} | {'等形成膜 Rp':>11} {'平坦界面 Rp':>11} | {'差':>10}")
for wl in wls:
    ns = complex(nk_sus(wl))
    lc = build_conformal(COL_WS, RAISES_ALL, wl, ns)
    lf = build_flat(COL_WS, RAISES_ALL, wl, ns)
    rc = R0(lc, 'p', wl)
    rf = R0(lf, 'p', wl)
    print(f"{wl:>5} | {rc:>11.5f} {rf:>11.5f} | {abs(rc-rf):>10.2e}")

print()
print("="*72)
print("TEST 3: 交互 raise (実際の段差構造) → 等形成膜 vs 平坦界面")
print("="*72)
print(f"{'wl':>5} | {'等形成膜 Rp':>11} {'等形成膜 Rs':>11} | {'平坦界面 Rp':>11} {'平坦界面 Rs':>11}")
for wl in wls:
    ns = complex(nk_sus(wl))
    lc = build_conformal(COL_WS, RAISES_ALT, wl, ns)
    lf = build_flat(COL_WS, RAISES_ALT, wl, ns)
    rcp = R0(lc, 'p', wl); rcs = R0(lc, 's', wl)
    rfp = R0(lf, 'p', wl); rfs = R0(lf, 's', wl)
    print(f"{wl:>5} | {rcp:>11.5f} {rcs:>11.5f} | {rfp:>11.5f} {rfs:>11.5f}")

print()
print("="*72)
print("TEST 4: SUS光学定数の影響 (等形成膜・交互段差)")
print("  → 常に n=3.0+4.0j (Gsolver仮定) vs SUS.nk (文献値)")
print("="*72)
print(f"{'wl':>5} | {'SUS.nk Rp':>11} {'n=3+4j Rp':>11} | {'差':>10}")
for wl in wls:
    ns_file = complex(nk_sus(wl))
    ns_guess = 3.0+4.0j
    lc_file  = build_conformal(COL_WS, RAISES_ALT, wl, ns_file)
    lc_guess = build_conformal(COL_WS, RAISES_ALT, wl, ns_guess)
    r_file  = R0(lc_file,  'p', wl)
    r_guess = R0(lc_guess, 'p', wl)
    print(f"{wl:>5} | {r_file:>11.5f} {r_guess:>11.5f} | {abs(r_file-r_guess):>10.4f}")

print()
print("--- 使用中の SUS n,k (SUS.nk) ---")
for wl in wls:
    nk = nk_sus(wl)
    print(f"  {wl} nm: n={nk.real:.3f} k={nk.imag:.3f}")
