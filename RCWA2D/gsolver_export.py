"""structure_builder.columns_to_layers() の出力（top-first、各層
{'thickness': nm, 'segments': [(material_name, width_fraction), ...]}）を、
Gsolver の「File > Import Text」構造インポート形式に変換する。

フォーマット（HANDOVER.md 6.節、GSolver UserManual61.pdf 記載のものをユーザーが
書き起こし。このプロジェクトで実際にインポートを試したことはまだ無いので未検証）:
    [LAYER]
    <厚さ(um)>
    <幅(周期比)> <材質タイプ> <材質名> <true/false>
    ...
    [LAYER]
    ...
    [END]
層は基板側から順（layer[0]が基板直上）= bottom-first。columns_to_layers() の出力は
top-first なので反転が必要。

<true/false> フラグの意味はマニュアル未確認のため、常に false を出力する。初回インポート
時は必ず Gsolver 側の断面プレビューで意図した構造になっているか目視確認してから RUN する
こと（特にこのフラグが実は重要な意味を持っていた場合、見た目で気付けるはず）。

材質名は GSolver.ini の [CONSTANTS] カタログに登録済みの名前と一致させる必要がある。
ENV_MATERIAL（空気）と SUBST_MATERIAL（基板）は、このプロジェクトのカタログに合わせて
既定で 'Ones'（n=1,0）と 'SUS'（n=1.71,2.88）にマップする。
"""
from .structure_builder import ENV_MATERIAL, SUBST_MATERIAL

DEFAULT_NAME_MAP = {ENV_MATERIAL: 'Ones', SUBST_MATERIAL: 'SUS'}


def layers_to_gsolver_text(layers, name_map=None, mat_type='CONSTANT'):
    """RCWA2D の top-first `layers` を Gsolver インポート用テキスト（文字列）に変換する。

    layers: columns_to_layers() の出力
    name_map: 内部材質名 -> GSolver.ini カタログ名。ENV_MATERIAL/SUBST_MATERIAL 以外の
              材質（例: 'TiO2', 'SiO2'）はデフォルトでそのままの名前を使う
              （カタログに同名で登録されている前提）。
    """
    nm = dict(DEFAULT_NAME_MAP)
    if name_map:
        nm.update(name_map)

    lines = []
    for layer in reversed(layers):  # bottom-first に反転
        th_um = layer['thickness'] / 1000.0
        segs = layer['segments']
        frac_sum = sum(w for _, w in segs)
        if abs(frac_sum - 1.0) > 1e-6:
            raise ValueError(f"segments の幅比合計が1.0でない: {frac_sum} (layer={layer})")
        lines.append('[LAYER]')
        lines.append(f'{th_um:.6f}')
        for mat, frac in segs:
            gname = nm.get(mat, mat)
            lines.append(f'{frac:.6f} {mat_type} {gname} false')
    lines.append('[END]')
    return '\n'.join(lines) + '\n'


def write_gsolver_import(layers, out_path, name_map=None, mat_type='CONSTANT'):
    text = layers_to_gsolver_text(layers, name_map=name_map, mat_type=mat_type)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return out_path
