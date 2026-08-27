"""
数値比較診断: ppap.py vs R8_koumyou_v3.py
対象: SUS304基板 + 7ペア TiO2/SiO2 多層膜（均一膜 fill=1.0）

実行: cd koumyou_sim && python diag_ppap_vs_r9.py
"""
import sys, os, numpy as np
from scipy.interpolate import interp1d
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RCWA2D import rcwa_mh

# ── 光学定数ロード ──────────────────────────────────────────────
def load_nk(name):
    p = os.path.join(os.path.dirname(__file__), "data", "nk", name + ".nk")
    d = np.loadtxt(p, comments=";", encoding="utf-8_sig")
    return interp1d(d[:,0], d[:,1] + d[:,2]*1j, kind="linear", fill_value="extrapolate")

nk_tio2 = load_nk("TiO2")
nk_sio2 = load_nk("SiO2")
nk_sus  = load_nk("SUS")

# ── 構造パラメータ ─────────────────────────────────────────────
N_PAIRS = 7
TH_A    = 77.6   # TiO2 [nm]  (Layer A, top/air 側)
TH_B    = 54.4   # SiO2 [nm]  (Layer B, substrate 側)
PITCH   = 500.0  # nm  (単純均一膜なので任意。ppap と同じデフォルト値)
NORDER  = 11
N_ENV   = 1.0 + 0j

WLS = [420, 470, 520, 570, 620, 670, 720]

# ──────────────────────────────────────────────────────────────────
# ppap.py 方式の layer tuple 生成
# nk_name_list = [nk_a, nk_b] * n_pairs   (index 0 = TiO2, 1 = SiO2, …)
# loop k from (2*n_pairs-1) downto 0
#   k=2n-1 (odd) → nk_b → 基板側   k=0 (even) → nk_a → 空気側
# 各層フォーマット: (d[nm]/1000, nk, w/2, n_env, 1-w, nk, w/2)  w=1.0
# ──────────────────────────────────────────────────────────────────
def make_layer_ppap(wl):
    n_env  = N_ENV
    ns     = complex(nk_sus(wl))
    n      = N_PAIRS * 2                   # 14 layers
    # nk_name_list: [TiO2, SiO2] * 7  (index 0=TiO2, 1=SiO2, …)
    d_list = [TH_A, TH_B] * N_PAIRS       # index 0=TiO2, 1=SiO2, …
    nk_fns = [nk_tio2, nk_sio2] * N_PAIRS

    layer_list = [(0, ns, 0)]             # 基板
    for k in range(n-1, -1, -1):
        nk = complex(nk_fns[k](wl))
        w  = 1.0
        layer = (d_list[k]/1000.0, nk, w/2.0, n_env, 1-w, nk, w/2.0)
        layer_list.append(layer)
    layer_list.append((0, n_env, 0))      # 媒質
    return tuple(layer_list)


# ──────────────────────────────────────────────────────────────────
# R8_koumyou_v3.py 方式の layer tuple 生成
# 等形成膜・単一列(fill=1.0) → pair ftype: B(SiO2)→A(TiO2) per pair
# 各層フォーマット: (d[nm]/1000, nk, 1.0)
# 層リストは Top-first → reversed() してから layer_list に追加
# ──────────────────────────────────────────────────────────────────
def make_layer_r9(wl):
    n_env = N_ENV
    ns    = complex(nk_sus(wl))

    # Top-first リストを手動構築 (columns_to_layers の結果と同等)
    # pair 展開: B(SiO2) が substrate 側 → Top-first では末尾に来る
    # 7ペア Top-first: [TiO2, SiO2, TiO2, SiO2, ... × 7]
    top_first = []
    for _ in range(N_PAIRS):
        top_first.append({'thickness': TH_A, 'segments': [('TiO2', 1.0)]})  # A: TiO2, 空気側
        top_first.append({'thickness': TH_B, 'segments': [('SiO2', 1.0)]})  # B: SiO2, 基板側

    nk_map = {'TiO2': nk_tio2, 'SiO2': nk_sio2}

    layer_list = [(0, ns, 0)]                  # 基板
    for L in reversed(top_first):              # Top-first → substrate-first
        mat, w = L['segments'][0]
        n = complex(nk_map[mat](wl))
        layer_list.append((L['thickness']/1000.0, n, w))
    layer_list.append((0, n_env, 0))           # 媒質
    return tuple(layer_list)


def R0(layer, pol, wl):
    p = NORDER // 2
    ir, _ = rcwa_mh.Rcwa1d(pol, wl/1000.0, 0.0, PITCH/1000.0, layer, NORDER)
    return float(ir[p])


print("="*72)
print("ppap.py vs R8_koumyou_v3.py: 均一膜 (fill=1.0), 7ペア TiO2/SiO2")
print("基板: SUS304  ピッチ: {} nm  nOrder: {}".format(PITCH, NORDER))
print("="*72)
print(f"\n【1】Layer tuple 先頭・末尾比較 (wl=550nm)")
lp = make_layer_ppap(550)
lr = make_layer_r9(550)
print(f"  ppap  layers count : {len(lp)}")
print(f"  R9    layers count : {len(lr)}")
print(f"  ppap  [0]  (subst) : {lp[0]}")
print(f"  R9    [0]  (subst) : {lr[0]}")
print(f"  ppap  [-1] (env)   : {lp[-1]}")
print(f"  R9    [-1] (env)   : {lr[-1]}")
print(f"  ppap  [1]  (1st film): {lp[1]}")
print(f"  R9    [1]  (1st film): {lr[1]}")
print(f"  ppap  [-2] (last film): {lp[-2]}")
print(f"  R9    [-2] (last film): {lr[-2]}")

print(f"\n【2】順序確認: substrate 直上の層 (ppap k=最後 → 基板側)")
print(f"  ppap [1] (substrate 直上) → SiO2 のはず: n={lp[1][1]:.4f}")
n_sio2_550 = complex(nk_sio2(550))
n_tio2_550 = complex(nk_tio2(550))
print(f"  SiO2(550nm) n = {n_sio2_550:.4f}")
print(f"  TiO2(550nm) n = {n_tio2_550:.4f}")
print(f"  R9   [1] (substrate 直上) → SiO2 のはず: n={lr[1][1]:.4f}")

print(f"\n【3】スペクトル数値比較")
print(f"{'wl':>5} | {'ppap Rp':>10} {'R9 Rp':>10} | {'差 Rp':>10} | {'ppap Rs':>10} {'R9 Rs':>10} | {'差 Rs':>10}")
max_dp, max_ds = 0.0, 0.0
for wl in WLS:
    lp = make_layer_ppap(wl)
    lr = make_layer_r9(wl)
    rp_ppap = R0(lp, 'p', wl)
    rp_r9   = R0(lr, 'p', wl)
    rs_ppap = R0(lp, 's', wl)
    rs_r9   = R0(lr, 's', wl)
    dp = abs(rp_ppap - rp_r9)
    ds = abs(rs_ppap - rs_r9)
    max_dp = max(max_dp, dp)
    max_ds = max(max_ds, ds)
    print(f"{wl:>5} | {rp_ppap:>10.6f} {rp_r9:>10.6f} | {dp:>10.2e} | {rs_ppap:>10.6f} {rs_r9:>10.6f} | {ds:>10.2e}")
print(f"\n  最大差: Rp={max_dp:.2e}  Rs={max_ds:.2e}")
if max_dp < 1e-12 and max_ds < 1e-12:
    print("  → 機械精度で一致 ✅")
elif max_dp < 1e-6 and max_ds < 1e-6:
    print("  → 数値誤差範囲内で一致 ✅")
else:
    print("  → 有意な差あり ⚠")

print(f"\n【4】layer tuple 中間層の n 比較 (ppap 3段 vs R9 1段)")
lp = make_layer_ppap(550)
lr = make_layer_r9(550)
print("  ppap [1]:", lp[1])
print("  R9   [1]:", lr[1])
print("  ppap [2]:", lp[2])
print("  R9   [2]:", lr[2])

print("\n" + "="*72)
print("診断完了")
