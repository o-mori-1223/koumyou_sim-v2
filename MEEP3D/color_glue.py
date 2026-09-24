"""ScatteringResult(Q_scatter,up スペクトル) を、既存の
RCWA2D.color_display.render_gsolver_style_display() が期待する
values[theta][lambda] 形式に変換して呼び出す。

注意: Q_scatter,up は反射率ではない。render_gsolver_style_display自体
（および共有テンプレートcolor_display_template.html）には一切手を
入れず、色計算用に0〜1へクリップした値を渡すだけにとどめる
（クリップ前の生データは呼び出し側でJSONに保存すること）。
Phase 1は法線入射のみなので、thetas=[0.0]の1点だけを渡す
（30°/60°はPhase 2で角度掃引が実装されてから追加する）。
"""
import numpy as np

from RCWA2D.color_display import render_gsolver_style_display


def render_normal_incidence_result(result, height=None):
    """result: MEEP3D.simulate.ScatteringResult。
    色・チャート表示に、クリップ後のQ_scatter,upを渡す
    （生データはresult.q_scatter_up側に別途保持されている）。
    """
    q_clipped = np.clip(result.q_scatter_up, 0.0, 1.0)
    values = [q_clipped.tolist()]        # thetas=[0.0]の1行だけ
    render_gsolver_style_display(
        lambdas=result.wl_nm, thetas=[0.0], values=values,
        show_swatches=True, swatch_thetas=[0.0], swatch_values=values,
        height=height,
    )
