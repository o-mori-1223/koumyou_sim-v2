"""孤立ナノ円盤3D FDTD(MEEP)用の材質定義。

data/nk/*.nk の読み込みは fdtd2d.materials.load_nk_table をそのまま再利用する
（パーサを二重管理しない）。ただし本モジュールが返すのはMEEP用の
mp.Medium であって、fdtd2d/materials.py のLorentz極フィッティング
（fit_lorentz_model/LorentzModel）はPhase 1では使わない。

Phase 1で使うTiO2/SiO2はこのプロジェクトのdata/nk上では波長非依存の
定数（k=0）なので、mp.Medium(index=n)で厳密かつ広帯域パルス全体で
正確に表現できる。これに対しSUS(n=1.71+2.88j)のような損失媒質は、
MEEPの通常のMediumでは「特定の1周波数でのみ厳密」なD_conductivity
近似(single_frequency_lossy_medium)でしか表現できない
（広帯域パルス全体で正確に表現するには fdtd2d.materials.fit_lorentz_model
の出力をMEEPのLorentzianSusceptibility形式に変換する必要があり、
その変換式 sigma = strength / w0**2 は確認済みだが実装はPhase 1.5以降）。
"""
import cmath
import math

import numpy as np

from fdtd2d.materials import load_nk_table

try:
    import meep as mp
    HAVE_MEEP = True
except ImportError:
    mp = None
    HAVE_MEEP = False


def _require_meep():
    if not HAVE_MEEP:
        raise RuntimeError(
            "meep をインポートできません。WSL上でconda環境 'mp' を"
            "作成・有効化してから実行してください（HANDOVER.md参照）。"
        )


def dielectric_medium(name, nk_dir='data/nk', k_tol=1e-3):
    """波長非依存とみなせる誘電体材質をmp.Medium(index=n)として返す。

    tabulated data中に|k| > k_tolの点が1つでもあれば例外を出す
    （損失媒質を無損失として黙って扱わないため）。
    """
    _require_meep()
    wl, n, k = load_nk_table(name, nk_dir)
    if np.any(np.abs(k) > k_tol):
        raise ValueError(
            f'{name} は無視できない損失(|k|max={np.abs(k).max():.4g})を持つため'
            f'dielectric_medium()では扱えません。single_frequency_lossy_medium()'
            f'を使うか、Phase 1.5のLorentz極フィッティングを検討してください。'
        )
    n_mean = float(np.mean(n))
    return mp.Medium(index=n_mean)


def single_frequency_lossy_medium(n_complex, meep_freq):
    """複素屈折率n_complexを、MEEP周波数meep_freq（1/a単位、a=1umの場合1/um）
    **1点でのみ厳密**なmp.Mediumに変換する。

    D_conductivity = 2*pi*f*Im(eps)/Re(eps) という変換式で、指定した
    meep_freq以外の周波数では近似精度が急速に悪化する。広帯域パルスの
    一部としてではなく、単一周波数(狭帯域/CW的)runでのみ使うこと。
    """
    _require_meep()
    eps = complex(n_complex) ** 2
    eps_re, eps_im = eps.real, eps.imag
    conductivity = 2.0 * math.pi * meep_freq * eps_im / eps_re
    return mp.Medium(epsilon=eps_re, D_conductivity=conductivity)
