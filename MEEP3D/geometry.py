"""孤立ナノ円盤(基板上)のMEEPセル・PML・flux box・対称性を組み立てるモジュール。

3-run減算法（vacuum / bare_substrate / disk）で使う3つのセルは、
**flux boxの位置・PML・ソース配置が完全に同一**でなければならない
（幾何以外の条件が少しでも違うと減算が成立しない）。そのため
build_cell()を1つの関数にまとめ、modeパラメータでgeometryリストだけを
切り替える設計にしている（3つの似て非なるスクリプトを別々に書かない）。

flux boxは6面の閉じた箱（底面を含む）だが、底面はz=0（基板表面）ぴったり
ではなく、その少し上（真空側）に置く。理由は2つ:
  1. 箱の内部を完全に真空だけにするため（内部に異なる媒質の境界が
     あると、閉曲面での流束の意味が「単一媒質中のPoynting流」でなく
     なり、散乱断面積の正規化が崩れる）。
  2. 材質境界ちょうどにモニタ面を置くと、MEEPのYee格子上のsubpixel
     averagingによる数値ノイズを拾いやすいため。

対称性: 入射光をz方向下向き伝搬・x偏光(Ex)とすると、伝搬軸(z)に垂直な
2つの軸(x, y)が鏡映面の候補になる。基板はx,y方向に一様(無限平板)なので、
基板の有無はこの2つの鏡映対称性を壊さない
（MEEP公式のMie散乱チュートリアルでも、伝搬軸自身は鏡映面として
使われていない — 伝搬方向を反転する鏡映はどのみち使えないため）。
  symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.X, phase=-1)]
Yはただの「伝搬にもソース偏光にも関与しない」横方向なのでphase=+1
（デフォルト）、Xはソース偏光(Ex)の向きなのでphase=-1。
**これは理論的な導出であり、必ずsymmetries=[]との数値一致を実行時に
確認してから信用すること**（位相の選び間違いは黙って誤った答えを返す）。
"""
from dataclasses import dataclass

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
class FluxBox:
    """flux box(6面, 底面はz=0よりbox_marginだけ上)の各面のFluxRegion仕様。
    faces: [(center, size, weight), ...] -- weightは外向き=+1になるよう
    MEEPの各面法線の符号規約に合わせて調整済み（build_cell内で構築）。
    """
    faces: list
    z_bottom: float
    z_top: float
    r_box: float


@dataclass
class CellConfig:
    cell_size: 'mp.Vector3'
    boundary_layers: list
    sources: list
    symmetries: list
    geometry: list
    flux_box: FluxBox
    fcen: float
    df: float
    n_freq: int = 1


def build_cell(mode, radius_um, height_um, disk_medium=None, substrate_medium=None,
                dpml=0.4, pad=0.4, box_margin=0.1, fcen=1.875, df=1.7,
                n_freq=1, use_symmetry=True):
    """mode in {'vacuum', 'bare_substrate', 'disk'} に応じたCellConfigを返す。
    'vacuum'    : 基板も円盤もなし（Run A、入射光量I0の基準値取得用）
    'bare_substrate': 基板のみ（Run B、減算のベースライン）
    'disk'      : 基板+円盤（Run C、これからRun Bを差し引いて散乱成分を得る）

    3つのmodeで、cell_size/boundary_layers/sources/flux_boxは常に同一
    （geometryリストだけが異なる）。これが3-run減算法の前提条件。
    """
    _require_meep()
    if mode not in ('vacuum', 'bare_substrate', 'disk'):
        raise ValueError(f'unknown mode: {mode}')

    r = radius_um
    sxy = 2.0 * (dpml + pad + r)
    # z方向: 下からPML -> 基板の余白 -> 基板表面(z=0基準) -> 円盤(mode=='disk'のみ) ->
    #        上側の余白 -> PML。基板ブロックは常に置いた上でmodeで材質を切り替える
    # （'vacuum'モードだけは基板ブロック自体を入れない＝真の真空）。
    z_subst_margin = pad
    z_top_margin = pad
    sz = dpml + z_subst_margin + height_um + z_top_margin + dpml
    # z=0を基板表面(円盤の底)に固定するため、セル中心をずらす。
    z_bottom_of_cell = -(dpml + z_subst_margin)
    cell_size = mp.Vector3(sxy, sxy, sz)
    cell_center_z = z_bottom_of_cell + sz / 2.0

    boundary_layers = [mp.PML(dpml)]

    geometry = []
    if mode in ('bare_substrate', 'disk'):
        geometry.append(mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, dpml + z_subst_margin),
            center=mp.Vector3(0, 0, -(dpml + z_subst_margin) / 2.0),
            material=substrate_medium,
        ))
    if mode == 'disk':
        geometry.append(mp.Cylinder(
            radius=r, height=height_um, axis=mp.Vector3(0, 0, 1),
            center=mp.Vector3(0, 0, height_um / 2.0),
            material=disk_medium,
        ))

    # ソース: 上側PMLのすぐ内側に置き、xy断面全体を覆う平面波近似、下向き伝搬。
    src_z = cell_center_z + sz / 2.0 - dpml
    sources = [mp.Source(
        mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
        center=mp.Vector3(0, 0, src_z),
        size=mp.Vector3(sxy, sxy, 0),
        component=mp.Ex,
    )]

    symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.X, phase=-1)] if use_symmetry else []

    # flux box: 6面、底面はz=0の box_margin上、上面は円盘頂部のbox_margin上。
    z_bottom = box_margin
    z_top = height_um + box_margin
    r_box = r + box_margin
    faces = [
        # (name, center, size)
        ('top',    mp.Vector3(0, 0, z_top),    mp.Vector3(2 * r_box, 2 * r_box, 0)),
        ('bottom', mp.Vector3(0, 0, z_bottom), mp.Vector3(2 * r_box, 2 * r_box, 0)),
        ('x+',     mp.Vector3(r_box, 0, (z_bottom + z_top) / 2.0), mp.Vector3(0, 2 * r_box, z_top - z_bottom)),
        ('x-',     mp.Vector3(-r_box, 0, (z_bottom + z_top) / 2.0), mp.Vector3(0, 2 * r_box, z_top - z_bottom)),
        ('y+',     mp.Vector3(0, r_box, (z_bottom + z_top) / 2.0), mp.Vector3(2 * r_box, 0, z_top - z_bottom)),
        ('y-',     mp.Vector3(0, -r_box, (z_bottom + z_top) / 2.0), mp.Vector3(2 * r_box, 0, z_top - z_bottom)),
    ]
    flux_box = FluxBox(faces=faces, z_bottom=z_bottom, z_top=z_top, r_box=r_box)

    return CellConfig(
        cell_size=cell_size, boundary_layers=boundary_layers, sources=sources,
        symmetries=symmetries, geometry=geometry, flux_box=flux_box, fcen=fcen, df=df,
        n_freq=n_freq,
    ), cell_center_z
