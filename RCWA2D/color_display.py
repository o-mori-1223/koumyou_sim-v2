"""スウォッチ・角度チャート・スペクトルチャートを、参考ツール
`Gsolver'S COLOR.html`（プロジェクトルート直下）と見た目・機能を完全に統一した
生HTML/JSで描画する。ロジックはそのHTMLファイルからほぼ検証済みの移植
（`color_display_template.html`参照。差分はそのファイル内のDEVIATIONコメント3箇所のみ）。

Streamlitのst.color_picker/Plotlyは寸法・スタイルが固定でHTML版と揃えられない
ため、st.components.v1.html()で完全に独立したiframeとして埋め込む。
"""
import json
from pathlib import Path

import streamlit.components.v1 as components

_TEMPLATE_PATH = Path(__file__).parent / 'color_display_template.html'
_PLACEHOLDER = '__KOUMYOU_INJECTED_JSON__'

_HEIGHT_WITH_SWATCHES = 2050
_HEIGHT_WITHOUT_SWATCHES = 1800


def render_gsolver_style_display(lambdas, thetas, values,
                                  show_swatches, swatch_thetas, swatch_values,
                                  target_wavelengths=None, height=None):
    """スウォッチ(0°/30°/60°)＋角度チャート＋スペクトルチャートを1つの
    埋め込みHTMLブロックとして描画する。

    lambdas, thetas: 1次元配列(nm, deg)。
    values: [len(thetas)][len(lambdas)] の2次元配列（角度掃引の反射率）。
    show_swatches: Falseならスウォッチカードごと非表示にする
        （既存コードの `wl_option == spMenu[0]` ゲートに対応）。
    swatch_thetas, swatch_values: 掃引とは独立に0°/30°/60°で個別計算した
        反射率（掃引の角度刻みが30°・60°ちょうどに当たらない場合でも
        スウォッチが「未入力」にならないようにするため。掃引matrixを
        そのまま流用しないこと）。
    target_wavelengths: 角度チャートに表示する波長(nm)のリスト。
        Noneなら`Gsolver'S COLOR.html`と同じ固定5本(400/470/540/600/700)。
    height: iframeの高さ(px)。st.components.v1.htmlには自動リサイズが
        無いため固定値が必要（Streamlit側に自動リサイズ用ブリッジは
        通常のcomponents.v1.htmlには存在しないことを確認済み）。
        Noneならshow_swatchesの有無に応じたデフォルト値を使う。
    """
    template = _TEMPLATE_PATH.read_text(encoding='utf-8')

    payload = {
        'lambdas': [float(v) for v in lambdas],
        'thetas': [float(v) for v in thetas],
        'values': [[float(v) for v in row] for row in values],
        'showSwatches': bool(show_swatches),
        'swatchThetas': [float(v) for v in swatch_thetas],
        'swatchValues': [[float(v) for v in row] for row in swatch_values],
    }
    if target_wavelengths:
        payload['targetWavelengthsAngleChart'] = [float(v) for v in target_wavelengths]

    html = template.replace(_PLACEHOLDER, json.dumps(payload))

    if height is None:
        height = _HEIGHT_WITH_SWATCHES if show_swatches else _HEIGHT_WITHOUT_SWATCHES

    components.html(html, height=height, scrolling=True)
