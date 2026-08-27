"""Column-based structure -> RCWA horizontal-layer conversion.

Extracted verbatim from R8_koumyou_v3.py so that both the existing rcwa_mh_cpp backend and
the new S4 backend (rcwa_s4_backend.py) consume the EXACT SAME structure-building code. This
matters specifically because the user has flagged the "column-periodic" (列周期, i.e.
multiple side-by-side columns each with an independent film stack -- the "等形成膜/conformal"
deposition model) results as suspect: if the two backends ever disagreed, having them share
this module means the disagreement can only come from the RCWA solvers themselves, not from
two independently-transcribed (and possibly divergently-buggy) versions of the geometry
conversion.
"""

ENV_MATERIAL = '__ENV__'
SUBST_MATERIAL = '__SUBSTRATE__'


def _add_col_segs(segs, mat, fill_frac, align, col_frac):
    """列幅 col_frac の中で material を fill_frac 分だけ配置してsegsに追加する"""
    if mat == ENV_MATERIAL or fill_frac < 1e-9:
        segs.append((ENV_MATERIAL, col_frac))
        return
    if fill_frac >= 1.0 - 1e-9:
        segs.append((mat, col_frac))
        return
    mat_w = fill_frac * col_frac
    env_w = (1.0 - fill_frac) * col_frac
    if align == 'Right':
        if env_w > 1e-9: segs.append((ENV_MATERIAL, env_w))
        segs.append((mat, mat_w))
    elif align == 'Center':
        hw = env_w / 2.0
        if hw > 1e-9: segs.append((ENV_MATERIAL, hw))
        segs.append((mat, mat_w))
        if hw > 1e-9: segs.append((ENV_MATERIAL, hw))
    else:  # Left
        segs.append((mat, mat_w))
        if env_w > 1e-9: segs.append((ENV_MATERIAL, env_w))


def _expand_to_bands(col, flat_align_height=0.0):
    """Column の film_layers を高さバンドリストに展開する。
    flat_align_height > 0 のとき、基板上げ高さ〜flat_align_height を ENV で埋めてから薄膜を積む。
    これにより等形成膜モードでも平坦界面モードと同等の計算が可能になる。
    Returns: [(h_lo, h_hi, material, fill_fraction, alignment), ...]
    """
    bands, h = [], 0.0
    if col['subst_raise'] > 1e-9:
        bands.append((h, h + col['subst_raise'], SUBST_MATERIAL, 1.0, 'Left'))
        h += col['subst_raise']
    # flat_align: 非隆起列に ENV ギャップを追加して全列の薄膜開始高さを揃える
    if flat_align_height > h + 1e-9:
        bands.append((h, flat_align_height, ENV_MATERIAL, 1.0, 'Left'))
        h = flat_align_height
    for fl in col['film_layers']:
        ftype = fl.get('type', 'simple')
        if ftype == 'simple':
            th = fl.get('thickness', 0.0)
            if th > 1e-9:
                bands.append((h, h + th, fl['material'], 1.0, 'Left'))
                h += th
        elif ftype == 'pair':
            n    = max(1, fl.get('n_pairs', 1))
            th_a = fl.get('th_a', 0.0)
            th_b = fl.get('th_b', 0.0)
            ma   = fl.get('mat_a', '')
            mb   = fl.get('mat_b', '')
            for _ in range(n):
                # B層を基板側（先）、A層を空気側（後）に積む（ppap.py の "Layer A (top)" と同一規約）
                if th_b > 1e-9 and mb:
                    bands.append((h, h + th_b, mb, 1.0, 'Left')); h += th_b
                if th_a > 1e-9 and ma:
                    bands.append((h, h + th_a, ma, 1.0, 'Left')); h += th_a
        elif ftype == 'taper':
            th_total = fl.get('thickness', 0.0)
            n_sl  = max(1, fl.get('n_slices', 10))
            align = fl.get('alignment', 'Center')
            mat   = fl.get('material', '')
            fb    = max(0.0, min(1.0, fl.get('frac_bottom', 1.0)))
            ft    = max(0.0, min(1.0, fl.get('frac_top',    0.5)))
            th_sl = th_total / n_sl
            if th_sl > 1e-9 and mat:
                for i in range(n_sl):
                    frac = fb + (ft - fb) * i / n_sl   # i=0 が下端
                    bands.append((h, h + th_sl, mat, frac, align))
                    h += th_sl
    return bands


def _col_total_height(col):
    bands = _expand_to_bands(col, flat_align_height=0.0)
    return bands[-1][1] if bands else 0.0


def columns_to_layers(columns, flat_align=False):
    """列定義（幅・基板上げ・フィルム積層）を RCWA 水平層リスト（Top-first）に変換する。
    flat_align=True のとき、全列の薄膜開始高さを最大 subst_raise に揃える（平坦界面と同等）。
    """
    total_width = sum(c['width'] for c in columns)
    if total_width <= 1e-9:
        return []

    align_h = max((c.get('subst_raise', 0.0) for c in columns), default=0.0) if flat_align else 0.0
    col_bands = [_expand_to_bands(col, flat_align_height=align_h) for col in columns]

    heights = set([0.0])
    for bands in col_bands:
        for h_lo, h_hi, *_ in bands:
            heights.add(h_lo); heights.add(h_hi)
    heights = sorted(heights)

    result = []
    for i in range(len(heights) - 1):
        h_lo, h_hi = heights[i], heights[i + 1]
        h_mid = (h_lo + h_hi) / 2.0
        segs = []
        for col, bands in zip(columns, col_bands):
            col_frac = col['width'] / total_width
            mat, fill_frac, align = ENV_MATERIAL, 1.0, 'Left'
            for (blo, bhi, bmat, bff, bal) in bands:
                if blo <= h_mid < bhi:
                    mat, fill_frac, align = bmat, bff, bal; break
            _add_col_segs(segs, mat, fill_frac, align, col_frac)
        result.append({'thickness': h_hi - h_lo, 'segments': segs})

    result.reverse()   # Top-first
    return result


def get_layer_tuple(wl, layers, nk_fn_map, n_env_val, nk_subst_fn):
    """R9's top-first `layers` dict list -> rcwa_mh_cpp's substrate-first tuple format."""
    n_env = complex(n_env_val)
    layer_list = [(0, complex(nk_subst_fn(wl)), 0)]
    for L in reversed(layers):
        flat = []
        for mat, w in L['segments']:
            if w <= 1e-9: continue
            if   mat == ENV_MATERIAL:   n = n_env
            elif mat == SUBST_MATERIAL: n = complex(nk_subst_fn(wl))
            else:                       n = complex(nk_fn_map[mat](wl))
            flat.append(n); flat.append(w)
        if not flat: continue
        layer_list.append(tuple([L['thickness']/1000.0] + flat))
    layer_list.append((0, n_env, 0))
    return tuple(layer_list)
