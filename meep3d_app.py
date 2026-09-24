"""孤立ナノ円盤(基板上)の3D FDTD散乱シミュレータ（Phase 1: 法線入射のみ）。

R8_koumyou_v4.py（列ベースの周期構造モデル）とは別の、孤立3D粒子専用の
新規アプリ。MEEPを使うためWSL上のconda環境 'mp' でのみ動作する
（run_meep3d.sh参照。rcwa_mhのようなWindowsネイティブ対応はしていない）。

Phase 1のスコープ:
- 円盤材質: 無損失誘電体（TiO2/SiO2のようにdata/nk上でk≈0のもの）
- 基板: 無損失誘電体のみ（SUS等の損失基板はPhase 1.5、今回未対応）
- 入射角: 法線(0°)のみ。角度掃引・0°/30°/60°スウォッチはPhase 2で対応

表示している色・スペクトルは「反射率」ではなく、円盤が上半球方向・
周囲へ再配分した散乱効率 Q_scatter,up（1.0でクリップ）である
（詳細はMEEP3D/geometry.py, HANDOVER.md参照）。
"""
import json
import os

import numpy as np
import streamlit as st

from MEEP3D.materials import dielectric_medium, HAVE_MEEP
from MEEP3D.simulate import run_scattering_spectrum
from MEEP3D.color_glue import render_normal_incidence_result

st.set_page_config(page_title="Nanodisk 3D FDTD (MEEP)", page_icon="🔵", layout="wide")

st.title("孤立ナノ円盤 3D FDTD 散乱シミュレータ（Phase 1: 法線入射）")
st.caption(
    "基板上の孤立ナノ円盤1個をMEEP(3D FDTD)で解析します。表示する色・スペクトルは"
    "周期構造の反射率ではなく、円盤が上半球方向へ散乱する効率 Q_scatter,up"
    "（1.0でクリップ）です。角度掃引(30°/60°)はPhase 2で対応予定、未実装です。"
)

if not HAVE_MEEP:
    st.error(
        "meep をインポートできません。WSL上でconda環境 'mp' を作成・有効化してから"
        "`bash run_meep3d.sh` で起動してください。"
    )
    st.stop()

with st.sidebar:
    st.subheader("円盤")
    disk_material = st.selectbox("材質", ["TiO2", "SiO2"], index=0, key="disk_material")
    disk_radius_nm = st.number_input("半径 [nm]", 10.0, 2000.0, 150.0, 10.0, key="disk_radius_nm")
    disk_height_nm = st.number_input("高さ [nm]", 10.0, 2000.0, 100.0, 10.0, key="disk_height_nm")

    st.subheader("基板")
    subst_material = st.selectbox(
        "材質（無損失のみ、Phase 1）", ["SiO2", "TiO2"], index=0, key="subst_material",
        help="SUSのような損失基板はPhase 1.5未対応です。")

    st.subheader("波長・計算設定")
    wl_min_nm = st.number_input("波長 min [nm]", 200.0, 1200.0, 400.0, 10.0, key="wl_min_nm")
    wl_max_nm = st.number_input("波長 max [nm]", 200.0, 1200.0, 800.0, 10.0, key="wl_max_nm")
    n_freq = st.number_input("波長点数", 11, 201, 81, 2, key="n_freq")

    with st.expander("詳細設定（解像度・PML等）", expanded=False):
        resolution = st.number_input("解像度 [px/um]", 10, 200, 60, 5, key="resolution")
        dpml = st.number_input("PML厚さ [um]", 0.1, 2.0, 0.4, 0.1, key="dpml")
        pad = st.number_input("余白 [um]", 0.1, 2.0, 0.4, 0.1, key="pad")
        decay_by = st.number_input("減衰しきい値 (log10)", -12, -3, -6, 1, key="decay_log")
        use_symmetry = st.checkbox("鏡映対称性を使う（約4倍高速化）", value=True, key="use_symmetry")

    run_btn = st.button("計算を実行", type="primary")

if run_btn:
    disk_medium = dielectric_medium(disk_material)
    subst_medium = dielectric_medium(subst_material)

    with st.spinner("MEEPで3回のFDTD計算を実行中（真空 → 基板のみ → 基板+円盤）..."):
        result = run_scattering_spectrum(
            radius_nm=disk_radius_nm, height_nm=disk_height_nm,
            disk_medium=disk_medium, substrate_medium=subst_medium,
            wl_min_nm=wl_min_nm, wl_max_nm=wl_max_nm, n_freq=int(n_freq),
            resolution=int(resolution), dpml=dpml, pad=pad,
            decay_by=10.0 ** decay_by, use_symmetry=use_symmetry,
        )
    st.session_state["meep3d_result"] = result

    out_path = os.path.join("data", "meep3d_disk_%s_on_%s_normal_result.json" % (disk_material, subst_material))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "wl_nm": result.wl_nm.tolist(),
            "q_scatter_up": result.q_scatter_up.tolist(),
            "meta": result.meta,
        }, f, indent=2)
    st.caption(f"生データを保存しました: {out_path}")

result = st.session_state.get("meep3d_result")
if result is not None:
    st.subheader("散乱効率スペクトル Q_scatter,up（生値、クリップなし）")
    st.line_chart(dict(zip(result.wl_nm.tolist(), result.q_scatter_up.tolist())))
    st.caption(
        f"半径={result.meta['radius_nm']:.0f}nm, 高さ={result.meta['height_nm']:.0f}nm, "
        f"解像度={result.meta['resolution']}px/um"
    )

    st.subheader("色表示（Gsolver'S COLOR.html と同じ表示コンポーネント）")
    st.caption(
        "ここで使っている「色」はQ_scatter,upを0〜1にクリップしたものをそのまま"
        "反射率相当としてCIE色計算に流用したものです。周期構造の反射率とは"
        "異なる物理量である点に注意してください。"
    )
    render_normal_incidence_result(result)
else:
    st.info("左のサイドバーでパラメータを設定し、「計算を実行」を押してください。")
