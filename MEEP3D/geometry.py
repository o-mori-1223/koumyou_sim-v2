"""孤立ナノ円盤(基板上)のMEEPセル・PML・flux box・対称性を組み立てるモジュール。

3-run減算法（vacuum / bare_substrate / disk）で使う3つのセルは、
**flux boxの位置・PML・ソース配置が完全に同一**でなければならない
（幾何以外の条件が少しでも違うと減算が成立しない）。そのため
build_cell()を1つの関数にまとめ、modeパラメータでgeometryリストだけを
切り替える設計にしている（3つの似て非なるスクリプトを別々に書かない）。

flux boxは6面の閉じた箱で、MEEP公式のsphere-in-vacuum Mie散乱チュートリアルと
同じ発想（対象物を全方向にbox_margin分の余白付きで完全に包む）を円盤+基板の
ジオメトリに適用する: z∈[-box_margin, height_um+box_margin]、r∈[0, r+box_margin]。
つまり底面は基板表面(z=0)ではなく**円盤の下端(z=0)からbox_margin分沈めた、
基板内部**に置く（上面は円盤の上端からbox_margin分浮かせる、というのは変えず）。

【2026-09-24修正】以前は底面を「z=0からbox_margin分浮かせる（真空側）」と
していたが、これはheight_um（円盤高さ）を全く考慮しない固定オフセットだった
ため、height_um > box_margin の場合（現実的な膜厚では普通に起こる）に底面
モニタ面が**円盤内部を貫通**するバグがあった。3-run減算法（vacuum/
bare_substrate/disk）が成立するには、モニタ面上の背景材質がbare_substrate run
とdisk runで完全に一致していなければならない——円盤内部を貫通する面は、
bare_substrate runではその位置が真空、disk runではその位置が円盤材質になり、
この前提が崩れていた。実際にheight_um=300nm・box_margin=100nmで検証したところ、
底面のflux値が他の面の10〜100倍かつ符号反転し、Q_scatter,upが物理的にあり得ない
-4〜-6という値になることを確認した（診断スクリプト`diag_meep_disk_boxmargin.py`
に再現手順が残っている。詳細はHANDOVER.md参照）。

修正後は円盤がどんな高さでも常に閉じた箱として成立する（球のMie検証と同じ
根拠）。ただし底面が基板内部にあるため、この閉曲面の流束は「円盤の存在によって
基板側へ透過する量がどう変化したか」も含むようになった——**これにより
Q_scatter,upは元々意図していた「上方向のみ（基板透過分を除く）」という定義
からは外れ、「円盤という摂動が全方向（上・側方・基板透過の変化分）に再配分した
総電力」に近い量になっている**。上から見た色・反射率の予測用途では基板透過分は
本来見えない量なので過大評価になりうる点に注意——「上方向だけ」の定義に戻したい
場合は、円盤直上だけを覆う薄い箱に設計変更する必要があるが、その場合は側面が
円盤の高さ全体(z=0〜height_um、浅い角度での側方散乱が主に起きる範囲)をカバー
できなくなるため別のチューニングが必要になる（検討過程はHANDOVER.md参照）。

このzbottom修正だけを入れて再検証した時点では、まだQ_scatter,upが常に負
（height=box_margin=100nmで-1.6〜-2.7、height=300nm>box_marginで-4〜-6）
だった。円盤ではなく基板を真空(n=1)にして「実質的に円盤だけが真空に浮いている」
状態でも同じ負の値が出たため、犯人は基板ではなくもっと基本的な部分にあると
判明——**2つ目の独立なバグ**が`simulate.py`側の入射光量(i0)の符号にあった
（光源が-z下向き伝搬なのに、i0に使う'top'面はMEEPの符号規約(+zが正)と
向きが逆で、fluxA['top']が常に負になっていた。詳細はsimulate.pyのdocstring
参照）。この2つ目のバグも修正した結果、無損失材質なら球のMie検証と同様に
**Q_scatter,upが常に0以上**になるというサニティチェックが実際に通ることを
確認済み（TiO2円盤/SiO2基板でheight=100nm・300nmの両方、全波長でQ>0。
`diag_meep_disk_boxmargin.py`参照）。

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
    """flux box(6面, 円盤をbox_margin分の余白で完全に包む — 底面は基板内部)の
    各面のFluxRegion仕様。
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

    # flux box: 6面。円盤(z=[0,height_um], r=[0,r])を全方向にbox_margin分の
    # 余白で完全に包む（モジュールdocstring2026-09-24追記参照）。底面は
    # 円盤の下端(z=0)からbox_margin分**基板内部に**沈む — box_marginが
    # 基板の余白(z_subst_margin=pad)を超えるとPML領域に達してしまうので、
    # box_margin < pad であることが前提（デフォルトはbox_margin=0.1 << pad=0.4）。
    assert box_margin < z_subst_margin, (
        f'box_margin({box_margin})がsubst margin({z_subst_margin})以上だと'
        'flux boxの底面が基板のPML領域に入ってしまいます。')
    z_bottom = -box_margin
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
