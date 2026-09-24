"""3-run減算法（vacuum / bare_substrate / disk）を実行し、孤立ナノ円盤の
散乱効率スペクトル Q_scatter,up を求めるランナー。

【2026-09-24更新】Q_scatter,upは、円盤という摂動が全方向（上・側方・
基板透過分の変化）に再配分した総電力／入射電力から求めた量。今回2つの
独立なバグを修正した:
  (1) geometry.pyのflux boxを球のMie検証と同じ「対象物を完全に包む閉じた箱」
      方式に修正（以前の「上方向のみ・基板透過分は除外」という定義は、円盤
      高さがbox_marginを超えると底面モニタが円盤内部を貫通するバグを内包
      していたため撤回した）。
  (2) 本関数呼び出し時のi0（下記`q = ...`の行）で、`-fluxA['top']`と符号を
      反転させるように修正——光源が-z(下向き)伝搬なのに対し、'top'面はz法線
      でMEEPの符号規約(+zが正)と伝搬方向が逆なため、fluxA['top']は常に負
      だった。
両方を直したことで、無損失材質なら球のMie散乱効率と同様に**常に0以上**に
なるはずというサニティチェックが実際に通ることを確認済み（TiO2円盤/SiO2
基板、height=100nm・300nmの両方でQ>0、詳細はdiag_meep_disk_boxmargin.py）。
**厳密な4π全方向のMie散乱断面積そのもの（真空中の孤立散乱体の値）ではなく**、
基板が半無限に存在する系での等価量である点は変わらず注意。「上から見た色」
用途では基板透過分の変化も含む分だけ過大評価になりうる——将来「上方向のみ」
に厳密化したい場合は、geometry.pyのdocstring末尾に記載した代替設計（円盤
直上だけを覆う薄い箱）を検討すること。

flux 6面の符号付き結合方法・幾何断面積での正規化・最後の符号反転は、
MEEP公式のsphere-in-vacuum Mie散乱チュートリアルの式をそのまま踏襲した
_scattering_efficiency_from_fluxes() に切り出してある。この関数は
diag_meep_sphere_mie.py（公式チュートリアルの再現による検証）からも
同じものを呼ぶことで、「MEEP自体」ではなく「ここで実際に使うコード」
を検証する。
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import build_cell

try:
    import meep as mp
    HAVE_MEEP = True
except ImportError:
    mp = None
    HAVE_MEEP = False


def _require_meep():
    if not HAVE_MEEP:
        raise RuntimeError("meep をインポートできません。conda環境 'mp' を有効化してください。")


@dataclass
class ScatteringResult:
    wl_nm: np.ndarray
    q_scatter_up: np.ndarray          # クリップ前の生値。1.0を超えることがある。
    meta: dict = field(default_factory=dict)


def _scattering_efficiency_from_fluxes(face_flux, r_box, r_geom, i0_flux):
    """MEEP公式Mie散乱チュートリアルの式をそのまま一般化した係数計算（純粋な算術、
    MEEP呼び出しなし）。face_flux: dict名->np.ndarray(周波数ごとの値)、
    キーは 'top','bottom','x+','x-','y+','y-'。r_box: flux boxの半幅(um)。
    r_geom: 幾何断面積の正規化に使う半径(um、円盤/球の実半径)。
    i0_flux: 真空run(Run A)で得た、同じboxの'top'面を通る入射光量(周波数ごと)。

    戻り値: Q = 散乱効率（幾何断面積 pi*r_geom**2 で正規化、無次元、1超あり得る）。
    """
    raw = ((face_flux['x-'] - face_flux['x+'])
           + (face_flux['y-'] - face_flux['y+'])
           + (face_flux['bottom'] - face_flux['top']))
    intensity = i0_flux / (2.0 * r_box) ** 2
    scatt_cross_section = raw / intensity
    q = scatt_cross_section * (-1.0) / (math.pi * r_geom ** 2)
    return q


def _add_flux_all_faces(sim, cell_cfg, fcen_mon, df_mon):
    return {name: sim.add_flux(fcen_mon, df_mon, cell_cfg.n_freq,
                                mp.FluxRegion(center=c, size=s))
            for name, c, s in cell_cfg.flux_box.faces}


def run_scattering_spectrum(radius_nm, height_nm, disk_medium, substrate_medium,
                             wl_min_nm=400.0, wl_max_nm=800.0, n_freq=81,
                             resolution=60, dpml=0.4, pad=0.4, box_margin=0.1,
                             decay_by=1e-6, use_symmetry=True, verbose=True):
    """法線入射・広帯域パルス3回（vacuum/bare_substrate/disk）で
    Q_scatter,up(波長)を求める。

    ソースパルスの帯域(fcen/df)は要求波長範囲より広めのマージンを持たせる
    （帯域端でのソース強度低下によるノイズを避けるため）が、flux監視は
    要求されたwl_min_nm〜wl_max_nmぴったりの範囲でサンプリングする
    （この2つを混同すると、報告される波長範囲が意図よりずれる）。
    """
    _require_meep()

    r_um, h_um = radius_nm / 1000.0, height_nm / 1000.0
    fmin_req = 1.0 / (wl_max_nm / 1000.0)
    fmax_req = 1.0 / (wl_min_nm / 1000.0)
    fcen_mon = 0.5 * (fmin_req + fmax_req)
    df_mon = fmax_req - fmin_req
    fcen_src = fcen_mon
    df_src = df_mon * 1.5   # ソース帯域は監視範囲より広めに取り、帯域端のノイズを避ける

    def _build(mode):
        cfg, cz = build_cell(mode, r_um, h_um, disk_medium, substrate_medium,
                              dpml=dpml, pad=pad, box_margin=box_margin,
                              fcen=fcen_src, df=df_src, n_freq=n_freq, use_symmetry=use_symmetry)
        sim = mp.Simulation(
            cell_size=cfg.cell_size, boundary_layers=cfg.boundary_layers,
            geometry=cfg.geometry, sources=cfg.sources, symmetries=cfg.symmetries,
            resolution=resolution, geometry_center=mp.Vector3(0, 0, cz),
            default_material=mp.Medium(index=1.0),
        )
        return sim, cfg

    def _run_and_get_fluxes(sim, cfg, load_from=None):
        fluxes = _add_flux_all_faces(sim, cfg, fcen_mon, df_mon)
        if load_from is not None:
            for name, obj in fluxes.items():
                sim.load_minus_flux_data(obj, load_from[name])
        pt = mp.Vector3(0, 0, cfg.flux_box.z_top)
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, pt, decay_by))
        wl_nm = 1000.0 / np.array(mp.get_flux_freqs(fluxes['top']))
        values = {name: np.array(mp.get_fluxes(obj)) for name, obj in fluxes.items()}
        data = {name: sim.get_flux_data(obj) for name, obj in fluxes.items()}
        return wl_nm, values, data

    if verbose:
        print('MEEP3D: Run A (vacuum) ...')
    simA, cfgA = _build('vacuum')
    wl_nm, fluxA, _ = _run_and_get_fluxes(simA, cfgA)

    if verbose:
        print('MEEP3D: Run B (bare substrate) ...')
    simB, cfgB = _build('bare_substrate')
    _, fluxB, dataB = _run_and_get_fluxes(simB, cfgB)

    if verbose:
        print('MEEP3D: Run C (substrate + disk) ...')
    simC, cfgC = _build('disk')
    _, fluxC, _ = _run_and_get_fluxes(simC, cfgC, load_from=dataB)

    # 光源は-z(下向き)伝搬だが、'top'面はz法線でMEEPの符号規約は+zが正 --
    # 伝搬方向と規約の+方向が逆なので、fluxA['top'](真空runでの入射光量)は
    # 常に負の値になる(2026-09-24に実測で確認: 全周波数で負)。
    # _scattering_efficiency_from_fluxes()はsphere-in-vacuum検証(伝搬+x、
    # 入射光量に規約と伝搬方向が一致する'x-'面を使用=正値)を前提に書かれて
    # おり、正の入射光量を期待している。符号を反転して渡す必要がある
    # (でないとQ_scatter,upの符号が常に反転する -- 無損失媒質なら本来
    # 0以上のはずが常に負になっていた実際のバグがこれ)。
    q = _scattering_efficiency_from_fluxes(fluxC, cfgC.flux_box.r_box, r_um, -fluxA['top'])

    meta = dict(radius_nm=radius_nm, height_nm=height_nm, resolution=resolution,
                dpml=dpml, pad=pad, box_margin=box_margin, decay_by=decay_by,
                use_symmetry=use_symmetry, n_freq=n_freq)
    return ScatteringResult(wl_nm=wl_nm, q_scatter_up=q, meta=meta)
