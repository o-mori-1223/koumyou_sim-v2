#from wsgiref.headers import tspecials
import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
import time
import datetime
import functools
import plotly.graph_objects as go
import rcwa_mh
import colour


st.set_page_config(
     page_title="Structural color simulator",
     page_icon="🦋",
     layout="wide",
     initial_sidebar_state="auto",
     menu_items={
         'Get Help': 'https://koumyou.org/',
         'Report a bug': "https://koumyou.org/",
         'About': "# This is a structural color simulator app!"
     }
 )

wl_min=400.0
wl_max=800.0
wl_n=81

n_env=1.0
inc_angle=0.0
nlayers=1

nk_idx_subst=0
nk_idx_film=0

pitch_nm = 500. # 周期（nm）
norder=11

ENV_MATERIAL = '__ENV__' # 媒質（環境屈折率）を表す特別なセグメント材質名




def tictoc(func):
    def _wrapper(*args,**keywargs):
        start_time=time.time()
        result=func(*args,**keywargs)
        print('time: {:.9f} [sec]'.format(time.time()-start_time))
        return result
    return _wrapper


@functools.cache
def Rcwa1d_cached( pol, lambda0, kx0, period, layer, norder):
    ir, it = rcwa_mh.Rcwa1d( pol, lambda0, kx0, period, layer, norder)
    return (ir,it)


def order_n(i): return {1:"1st (Top)", 2:"2nd", 3:"3rd"}.get(i) or "%dth"%i



@st.cache_data
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

@st.cache_data
def get_nk_list():
    """
    フォルダ内のnkファイル名一覧の取得
    Parameters
    ----------
    nk_path : str
        nkファイルのパス.
    Returns
    -------
    name_list : list of str
        ファイル名のリスト

    """

    nk_list=[]
    nk_dirs="data//nk"
    files=os.listdir(nk_dirs)
    nk_files=[f for f in files if os.path.isfile(os.path.join(nk_dirs, f))]

    for nk_file in nk_files:
        basename = os.path.splitext(os.path.basename(nk_file))[0]
        nk_list.append(basename)

    if len(nk_list)<1:
        st.error('not find nk data in '+nk_dirs)
        files_data=glob.glob("data")
        st.error('dir of data =',files_data)
        files_nk=glob.glob("data\\nk")
        st.error('dir of data_nk =',files_nk)

    nk_list.sort()
    return nk_list

@st.cache_data
def calc_nk_list(nk_fn_list,wl):
    """
    各層の光学定数の関数リストと与えられた波長から、薄膜の光学定数リストを返す

    Parameters
    ----------
    nk_fn_list : list of fn(wl)
        光学定数の関数リスト.
    wl : float
        波長(nm).
    Returns
    -------
    nk_list : array of complex
        各層の光学定数.

    """
    nk_list=[]
    for nk in nk_fn_list:
        nk_list.append(nk(wl))
    return nk_list


def make_nk_fn(nk_name_list=[]):
    """
    各層の光学定数の関数を返す
    Parameters
    ----------
    nk_name_list : list of string
        光学定数名のリスト.

    Returns
    -------
    nk_fn_list : list of fn(wl)
        各層の光学定数の関数リスト.

    """
    nk_path="data//nk//" # nkファイルのパス
    nk_fn_list=[]
    for idx,nk_name in enumerate(nk_name_list):
        if isinstance(nk_name,complex) or isinstance(nk_name,float) or isinstance(nk_name,int):
            nk=complex(nk_name)
            nk_fn = lambda wavelength: nk
            #print(f'Idx={idx},Instance==numeric, val={nk}')
        elif isinstance(nk_name,str) and str(nk_name).isnumeric():
            nk=float(nk_name)
            nk_fn = lambda wavelength: nk
            #print(f'Idx={idx},Instance==str, val={nk}')
        else:
            fname_path=nk_path+nk_name+'.nk'
            if os.path.isfile(fname_path):
                nk_mat=np.loadtxt(fname_path,comments=';',encoding="utf-8_sig")
                #st.write(nk_mat)
                w_mat=nk_mat[:,0]
                n_mat=np.array(nk_mat[:,1]+nk_mat[:,2]*1j)
                #nk_fn= interp1d(w_mat,n_mat, kind='quadratic', fill_value='extrapolate')
                nk_fn= interp1d(w_mat,n_mat, kind='linear', fill_value='extrapolate')
                #print(f'Idx={idx},Instance=={fname_path} exist')
            else:
                try:
                    nk=complex(nk_name)
                except ValueError:
                    nk=complex(1.0)
                #print(f'Idx={idx},Instance=={fname_path} not exist, nk={nk}')
                nk_fn = lambda wavelength: nk

        nk_fn_list.append(nk_fn)
    #print(nk_fn_list)
    return nk_fn_list


def build_nk_fn_map(material_names):
    """
    材質名のリスト（重複あり）から、重複を除いた {材質名: fn(wl)} の辞書を返す。
    多リッジ/テーパーブロックで同じ材質が何度も参照されても、nkファイルのロードは1回だけになる。
    """
    unique_names=[]
    seen=set()
    for name in material_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
    fn_list=make_nk_fn(unique_names)
    return dict(zip(unique_names,fn_list))


def build_manual_segments(features):
    """
    1層分のセグメント列を構築する。
    features: [{'material':str,'width':float,'gap_before':float}, ...]
        gap_before: 直前のセグメント右端（最初は x=0）からの媒質ギャップ幅（ピッチ規格化）
    戻り値: (segments, was_scaled)
        segments: [(material_or_ENV, width), ...]  幅の合計は1.0
        was_scaled: 合計が1.0を超えていたため比例縮小したかどうか
    """
    total=sum(f['gap_before']+f['width'] for f in features)
    scale=1.0
    was_scaled=False
    if total>1.0+1e-9:
        scale=1.0/total
        was_scaled=True

    segments=[]
    cursor=0.0
    for f in features:
        gap=max(0.0,f['gap_before'])*scale
        width=max(0.0,f['width'])*scale
        if gap>1e-9:
            segments.append((ENV_MATERIAL,gap))
        if width>1e-9:
            segments.append((f['material'],width))
        cursor+=gap+width

    trailing=1.0-cursor
    if trailing>1e-9:
        segments.append((ENV_MATERIAL,trailing))

    return segments, was_scaled


def resolve_manual_block(block):
    segments,was_scaled=build_manual_segments(block['features'])
    return [{'thickness':block['thickness'],'segments':segments}], was_scaled


def resolve_pair_block(block):
    layers=[]
    any_scaled=False
    for _ in range(int(block['n_pairs'])):
        for side in ('a','b'):
            material=block[f'mat_{side}']
            width=block[f'w_{side}']
            thickness=block[f'd_{side}']
            gap=(1.0-width)/2.0
            segments,scaled=build_manual_segments([{'material':material,'width':width,'gap_before':gap}])
            any_scaled=any_scaled or scaled
            layers.append({'thickness':thickness,'segments':segments})
    return layers, any_scaled


def resolve_taper_block(block):
    n=max(1,int(block['n_slices']))
    layers=[]
    any_scaled=False
    for i in range(n):
        t=i/(n-1) if n>1 else 0.0
        width=block['top_width']+(block['bottom_width']-block['top_width'])*t
        width=max(0.0,min(1.0,width))
        if block['alignment']=='Left':
            gap=0.0
        elif block['alignment']=='Right':
            gap=1.0-width
        else:
            gap=(1.0-width)/2.0
        segments,scaled=build_manual_segments([{'material':block['material'],'width':width,'gap_before':gap}])
        any_scaled=any_scaled or scaled
        layers.append({'thickness':block['thickness']/n,'segments':segments})
    return layers, any_scaled


def resolve_all_blocks(blocks):
    layers=[]
    any_scaled=False
    for block in blocks:
        if block['type']=='manual':
            block_layers,scaled=resolve_manual_block(block)
        elif block['type']=='pair':
            block_layers,scaled=resolve_pair_block(block)
        elif block['type']=='taper':
            block_layers,scaled=resolve_taper_block(block)
        else:
            continue
        layers.extend(block_layers)
        any_scaled=any_scaled or scaled
    return layers, any_scaled


# 可視化用の関数（材料名で色分け、層内の任意個数のセグメントに対応）
def create_structure_figure(layers, pitch_nm, nk_subst_name):
    """設定された多層膜構造（ブロックから解決された層リスト）を可視化するPlotly Figureを生成する"""

    shapes = []
    annotations = []

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    unique_materials = sorted({mat for L in layers for mat, w in L['segments'] if mat != ENV_MATERIAL})
    color_map = {material: colors[i % len(colors)] for i, material in enumerate(unique_materials)}
    env_color = 'rgba(240, 240, 240, 0.95)'

    total_thickness = sum(L['thickness'] for L in layers)

    # 1. 基板を描画
    substrate_thickness = total_thickness * 0.2 if total_thickness > 0 else 50
    shapes.append(go.layout.Shape(
        type="rect", xref="x", yref="y",
        x0=0, y0=-substrate_thickness, x1=pitch_nm, y1=0,
        fillcolor="lightgrey", line_width=0
    ))
    annotations.append(dict(
        x=pitch_nm / 2, y=-substrate_thickness / 2,
        text=nk_subst_name, showarrow=False, font=dict(color="black")
    ))

    current_y = 0
    # 2. 各層を基板の上から順番に描画（layersはTopが先頭なので、逆順にして基板側から描く）
    for L in reversed(layers):
        d = L['thickness']
        cursor = 0.0
        for mat, w in L['segments']:
            x0 = pitch_nm * cursor
            x1 = pitch_nm * (cursor + w)
            color = env_color if mat == ENV_MATERIAL else color_map.get(mat, '#cccccc')
            shapes.append(go.layout.Shape(
                type="rect", xref="x", yref="y",
                x0=x0, y0=current_y, x1=x1, y1=current_y + d,
                fillcolor=color, line=dict(color="black", width=1)
            ))
            if mat != ENV_MATERIAL:
                annotations.append(dict(
                    x=(x0 + x1) / 2, y=current_y + d / 2,
                    text=f"{mat}<br>d={d:g}nm", showarrow=False, font=dict(color="white", size=10)
                ))
            cursor += w

        current_y += d

    fig = go.Figure()

    fig.update_layout(
        title="Film Stack Visualization",
        xaxis_title="Position (nm)",
        yaxis_title="Height (nm)",
        xaxis=dict(range=[0, pitch_nm], showgrid=False, zeroline=False),
        yaxis=dict(range=[-substrate_thickness, current_y * 1.1 if current_y > 0 else 100], scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False),
        shapes=shapes,
        annotations=annotations,
        plot_bgcolor='white'
    )

    return fig


def get_layer_tuple(wl, layers, nk_fn_map, n_env_val, nk_subst_fn):
    """
    指定波長wl[nm]での layerを返す（layersは任意個数のセグメントを持つ層のリスト、Topが先頭）
    """
    n_env = complex(n_env_val)
    layer_env = (0, n_env, 0)                                   # 媒質層
    layer_subst = (0, complex(nk_subst_fn(wl)), 0)               # 基板層

    layer_list = [layer_subst]
    for L in reversed(layers):
        flat = []
        for mat, w in L['segments']:
            if w <= 1e-9:
                continue
            n = n_env if mat == ENV_MATERIAL else complex(nk_fn_map[mat](wl))
            flat.append(n)
            flat.append(w)
        if not flat:
            continue
        layer_list.append(tuple([L['thickness'] / 1000.0] + flat))

    layer_list.append(layer_env)
    return tuple(layer_list)


@tictoc
def calc_rcwa1d(wl_nm_ar, inc_angle_rad, pitch_um, norder, layers, nk_fn_map, n_env_val, nk_subst_fn):
    nwl = len(wl_nm_ar)
    irp = np.empty([nwl, norder],dtype=float) # 反射回折効率(p)の格納用
    itp = np.empty([nwl, norder],dtype=float) # 透過回折効率(p)の格納用
    irs = np.empty([nwl, norder],dtype=float) # 反射回折効率(s)の格納用
    its = np.empty([nwl, norder],dtype=float) # 透過回折効率(s)の格納用

    for idx,wl_nm in enumerate(wl_nm_ar):
        wl_um=float(wl_nm/1000.0)
        layer=get_layer_tuple(float(wl_nm), layers, nk_fn_map, n_env_val, nk_subst_fn)
        coef=float(2*np.pi*np.sin(inc_angle_rad)/wl_um)
        irp[idx,:], itp[idx,:] = Rcwa1d_cached('p', wl_um, coef, pitch_um, layer, norder)    # RCWAの呼び出し
        irs[idx,:], its[idx,:] = Rcwa1d_cached('s', wl_um, coef, pitch_um, layer, norder)    # RCWAの呼び出し
    return (irp,itp,irs,its)




st.title('Structural color simulator (v2: 複合構造ビルダー)')

st.sidebar.header('Light parameters')

nk_namelist=get_nk_list()
if len(nk_namelist)<1:
    st.error('nk list not find')

nk_idx_subst=nk_namelist.index('Air')
nk_idx_subst=nk_namelist.index('Silicon')
nk_idx_film=nk_namelist.index('SiO2')


angle_mode=st.sidebar.radio('Angle mode',('Single angle (2D)','Angle sweep (3D)'),key='angle_mode')
if angle_mode=='Single angle (2D)':
    inc_angle=st.sidebar.number_input('Incident angle [deg]',min_value=0.0,max_value=89.0,value=0.0,step=0.1,format='%3.1f')
else:
    angle_range=st.sidebar.slider('Angle range [deg]',min_value=0.0,max_value=89.0,value=(0.0,80.0),step=1.0,format='%.0f')
    angle_step=st.sidebar.number_input('Angle step [deg]',min_value=0.5,max_value=45.0,value=2.0,step=0.5,format='%.1f')
    inc_angle=angle_range[0]   # 色計算・CSV用の代表角度（スイープ先頭）
spMenu=('Visible[380-780nm]','UV[200-400nm]','NIR[700-1000nm]','All[200-1000nm]','Any')
wl_option=st.sidebar.selectbox('Spetrum range',spMenu)
if wl_option==spMenu[0]:
    wl_min=380.0
    wl_max=780.0
    wl_n=81
elif wl_option==spMenu[1]:
    wl_min=200.0
    wl_max=400.0
    wl_n=101
elif wl_option==spMenu[2]:
    wl_min=700.0
    wl_max=1000.0
    wl_n=61
elif wl_option==spMenu[3]:
    wl_min=200.0
    wl_max=1000.0
    wl_n=81

if wl_option==spMenu[4]:
    wl_range=st.sidebar.slider('Wavelength range [nm]',min_value=200.0,max_value=1000.0,value=(wl_min,wl_max),step=20.0,format='%.0f')
    if wl_range:
        wl_min=wl_range[0]
        wl_max=wl_range[1]

    wl_n=st.sidebar.number_input('Number of Wavelength',min_value=11,max_value=101,value=int(wl_n),step=10,format='%d',key='wln')


st.sidebar.header('Atmosphere')
n_env=st.sidebar.number_input('Refractive index (air:1.00)',min_value=1.0,max_value=3.0,value=n_env,step=0.01,format='%3.2f',key='nenv')

st.sidebar.header('Substrate')
nk_subst_name=st.sidebar.selectbox('Substrate',nk_namelist,index=nk_idx_subst,key='substrate')




st.header('Structural color of patterned film using 1D RCWA/FMM')

st.subheader('Film parameters (Number of layers, Period(pitch), Order)')

col1,col2=st.columns(2)
with col1:
    pitch_nm=st.number_input('Pitch[nm]',min_value=1.0,max_value=1e6,value=pitch_nm,step=1.0,format='%g',key='Pitch')
with col2:
    norder=st.number_input('nOrder',min_value=5,max_value=101,value=norder,step=2,format='%d',key='nOrder')


# --- 複合構造ビルダー ---
# 「ブロック」を任意の順序・個数で追加して積み重ねる。先に追加したブロックほどTop（光が最初に当たる側）。
# - Manual layer: 1層分。セグメント(材質・幅・直前からのギャップ)を複数指定できるので多リッジ構造を表現できる。
#   幅の異なるManual layerを複数積めば段差(非対称な幅変化)も表現できる。
# - Periodic pair: 既存のA/B交互積層機能。
# - Tapered block: top/bottom幅を線形補間したN枚の薄層に自動分割し、台形(斜め側壁)を近似する。

if 'blocks' not in st.session_state:
    st.session_state.blocks = []
    st.session_state.next_block_id = 0

def _new_block_id():
    st.session_state.next_block_id += 1
    return st.session_state.next_block_id

st.subheader('Patterned film stack (blocks)')
st.caption('ブロックを追加して任意の順序で積み重ねられます。先に追加したブロックほど光が最初に当たる側（Top）になります。')

bcol1, bcol2, bcol3 = st.columns(3)
with bcol1:
    if st.button('+ Manual layer (多リッジ・段差用)'):
        st.session_state.blocks.append({
            'id': _new_block_id(), 'type': 'manual',
            'thickness': 100.0,
            'features': [{'material': nk_namelist[nk_idx_film], 'width': 0.5, 'gap_before': 0.25}],
        })
with bcol2:
    if st.button('+ Periodic pair (A+B) x N'):
        idx_b = 1 if len(nk_namelist) > 1 else 0
        st.session_state.blocks.append({
            'id': _new_block_id(), 'type': 'pair',
            'n_pairs': 5,
            'mat_a': nk_namelist[nk_idx_film], 'w_a': 0.5, 'd_a': 100.0,
            'mat_b': nk_namelist[idx_b], 'w_b': 0.5, 'd_b': 100.0,
        })
with bcol3:
    if st.button('+ Tapered block (台形用)'):
        st.session_state.blocks.append({
            'id': _new_block_id(), 'type': 'taper',
            'material': nk_namelist[nk_idx_film],
            'top_width': 0.3, 'bottom_width': 0.7,
            'thickness': 200.0, 'n_slices': 10, 'alignment': 'Center',
        })

remove_idx = None
move = None

for i, block in enumerate(st.session_state.blocks):
    bid = block['id']
    with st.expander(f"Block {i + 1} [{block['type']}]", expanded=True):
        if block['type'] == 'manual':
            block['thickness'] = st.number_input('Thickness[nm]', min_value=0.0, max_value=1e6, value=block['thickness'], step=0.1, format='%g', key=f'th_{bid}')
            n_feat = st.number_input('Number of ridges/segments', min_value=1, max_value=6, value=len(block['features']), step=1, format='%d', key=f'nf_{bid}')
            while len(block['features']) < n_feat:
                block['features'].append({'material': nk_namelist[nk_idx_film], 'width': 0.2, 'gap_before': 0.1})
            while len(block['features']) > n_feat:
                block['features'].pop()

            for fi, feat in enumerate(block['features']):
                c1, c2, c3 = st.columns(3)
                with c1:
                    mat_index = nk_namelist.index(feat['material']) if feat['material'] in nk_namelist else nk_idx_film
                    feat['material'] = st.selectbox(f'Material #{fi + 1}', nk_namelist, index=mat_index, key=f'mat_{bid}_{fi}')
                with c2:
                    feat['width'] = st.number_input(f'Width #{fi + 1} (fraction of pitch)', min_value=0.0, max_value=1.0, value=feat['width'], step=0.001, format='%.3f', key=f'w_{bid}_{fi}')
                with c3:
                    feat['gap_before'] = st.number_input(f'Gap before #{fi + 1} (fraction of pitch)', min_value=0.0, max_value=1.0, value=feat['gap_before'], step=0.001, format='%.3f', key=f'gap_{bid}_{fi}')

        elif block['type'] == 'pair':
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('##### Layer A (top side of each pair)')
                block['mat_a'] = st.selectbox('Material A', nk_namelist, index=nk_namelist.index(block['mat_a']), key=f'mata_{bid}')
                block['w_a'] = st.number_input('Ratio A', 0.0, 1.0, block['w_a'], 0.001, key=f'wa_{bid}')
                block['d_a'] = st.number_input('Thickness A[nm]', 0.0, 1e6, block['d_a'], 0.1, key=f'da_{bid}')
            with c2:
                st.markdown('##### Layer B (bottom side of each pair)')
                block['mat_b'] = st.selectbox('Material B', nk_namelist, index=nk_namelist.index(block['mat_b']), key=f'matb_{bid}')
                block['w_b'] = st.number_input('Ratio B', 0.0, 1.0, block['w_b'], 0.001, key=f'wb_{bid}')
                block['d_b'] = st.number_input('Thickness B[nm]', 0.0, 1e6, block['d_b'], 0.1, key=f'db_{bid}')
            block['n_pairs'] = st.number_input('Number of pairs', min_value=1, max_value=50, value=block['n_pairs'], step=1, format='%d', key=f'np_{bid}')

        elif block['type'] == 'taper':
            block['material'] = st.selectbox('Material', nk_namelist, index=nk_namelist.index(block['material']), key=f'tmat_{bid}')
            c1, c2, c3 = st.columns(3)
            with c1:
                block['top_width'] = st.number_input('Top width (fraction)', 0.0, 1.0, block['top_width'], 0.001, key=f'tw_{bid}')
            with c2:
                block['bottom_width'] = st.number_input('Bottom width (fraction)', 0.0, 1.0, block['bottom_width'], 0.001, key=f'bw_{bid}')
            with c3:
                block['n_slices'] = st.number_input('Number of slices', min_value=1, max_value=100, value=block['n_slices'], step=1, format='%d', key=f'ns_{bid}')
            block['thickness'] = st.number_input('Total thickness[nm]', min_value=0.0, max_value=1e6, value=block['thickness'], step=0.1, format='%g', key=f'tt_{bid}')
            block['alignment'] = st.radio('Alignment', ('Left', 'Center', 'Right'), index=('Left', 'Center', 'Right').index(block['alignment']), key=f'al_{bid}', horizontal=True)

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            if st.button('▲ Move up', key=f'up_{bid}', disabled=(i == 0)):
                move = (i, -1)
        with mcol2:
            if st.button('▼ Move down', key=f'down_{bid}', disabled=(i == len(st.session_state.blocks) - 1)):
                move = (i, 1)
        with mcol3:
            if st.button('✕ Remove block', key=f'rm_{bid}'):
                remove_idx = i

if remove_idx is not None:
    st.session_state.blocks.pop(remove_idx)
    st.rerun()
if move is not None:
    i, d = move
    j = i + d
    st.session_state.blocks[i], st.session_state.blocks[j] = st.session_state.blocks[j], st.session_state.blocks[i]
    st.rerun()

layers, was_scaled = resolve_all_blocks(st.session_state.blocks)
if was_scaled:
    st.warning('一部のブロックでセグメント幅の合計が1.0(ピッチ)を超えていたため、自動的に縮小しました。')

nlayers = len(layers)
st.caption(f'Resolved into {nlayers} physical layer(s) for RCWA calculation.')


st.subheader('Film Stack Visualization')
fig_structure = create_structure_figure(layers, pitch_nm, nk_subst_name)
st.plotly_chart(fig_structure, use_container_width=True)


st.subheader('Spectrum')

material_names = [mat for L in layers for mat, w in L['segments'] if mat != ENV_MATERIAL]
nk_fn_map = build_nk_fn_map(material_names)
nk_subst_fn = make_nk_fn([nk_subst_name])[0]

pitch_um=pitch_nm/1000.0
wl_nm_ar=np.linspace(wl_min,wl_max,wl_n,dtype=float)
idex_0=norder // 2
nmax_order=idex_0
order_list=list(range(-nmax_order, nmax_order+1))

def _order_label(m):
    return '0 (specular)' if m==0 else f'{m:+d}'

def _idx_for_order(m):
    return idex_0 + m

st.markdown('##### Diffraction order')
ocol1, ocol2 = st.columns(2)
with ocol1:
    plot_orders=st.multiselect('Orders to show in spectrum plot', order_list, default=[0], format_func=_order_label, key='plot_orders')
    if not plot_orders:
        plot_orders=[0]
with ocol2:
    repr_order=st.selectbox('Order for colorimetry / CSV / 3D surface', order_list, index=order_list.index(0), format_func=_order_label, key='repr_order')

if angle_mode=='Single angle (2D)':
    inc_angle_rad=inc_angle*np.pi/180.0
    (irp, itp,irs,its)=calc_rcwa1d(wl_nm_ar, inc_angle_rad, pitch_um, norder, layers, nk_fn_map, n_env, nk_subst_fn)

    Rp=irp[:,_idx_for_order(repr_order)]
    Rs=irs[:,_idx_for_order(repr_order)]
    Tp=itp[:,_idx_for_order(repr_order)]
    Ts=its[:,_idx_for_order(repr_order)]

    fig = go.Figure()

    for m in sorted(plot_orders):
        idx_m=_idx_for_order(m)
        fig.add_trace(go.Scatter(
            x=wl_nm_ar, y=irp[:,idx_m],
            name=f'Rp (m={_order_label(m)})',
            mode='lines',
        ))
        fig.add_trace(go.Scatter(
            x=wl_nm_ar, y=irs[:,idx_m],
            name=f'Rs (m={_order_label(m)})',
            mode='lines',
        ))

    title_msg=f'AOI at {round(inc_angle,1)}[deg]'
    fig.update_layout(title=title_msg,
                    yaxis_zeroline=True, xaxis_zeroline=True)
    fig.update_xaxes(title_text='Wavelength(nm)')
    fig.update_yaxes(title_text='R/T',range=[0, 1])

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f'※ 下の色計算・CSVは回折次数 m={_order_label(repr_order)} のスペクトルに基づきます。')

else:
    # 角度配列を作成（範囲＋刻み）
    angle_ar=np.arange(angle_range[0], angle_range[1]+angle_step/2.0, angle_step, dtype=float)
    nang=len(angle_ar)

    Rp2d=np.empty([nang, wl_n],dtype=float)
    Rs2d=np.empty([nang, wl_n],dtype=float)
    Tp2d=np.empty([nang, wl_n],dtype=float)
    Ts2d=np.empty([nang, wl_n],dtype=float)

    idx_repr=_idx_for_order(repr_order)
    prog=st.progress(0.0, text='Calculating angle sweep...')
    for ia, ang in enumerate(angle_ar):
        ang_rad=ang*np.pi/180.0
        (irp, itp, irs, its)=calc_rcwa1d(wl_nm_ar, ang_rad, pitch_um, norder, layers, nk_fn_map, n_env, nk_subst_fn)
        Rp2d[ia,:]=irp[:,idx_repr]
        Rs2d[ia,:]=irs[:,idx_repr]
        Tp2d[ia,:]=itp[:,idx_repr]
        Ts2d[ia,:]=its[:,idx_repr]
        prog.progress((ia+1)/nang, text=f'Calculating angle sweep... {ia+1}/{nang}')
    prog.empty()

    pol_sel=st.radio('Polarization (3D surface)',('Rp','Rs'),horizontal=True,key='pol3d')
    Z = Rp2d if pol_sel=='Rp' else Rs2d
    fig = go.Figure(data=[go.Surface(
        x=wl_nm_ar, y=angle_ar, z=Z,
        colorscale='Viridis', cmin=0.0, cmax=1.0,
        colorbar=dict(title='R')
    )])
    fig.update_layout(
        title=f'{pol_sel} (m={_order_label(repr_order)}): Reflectance vs Wavelength & Angle',
        scene=dict(
            xaxis_title='Wavelength(nm)',
            yaxis_title='Angle(deg)',
            zaxis=dict(title='R', range=[0,1]),
        ),
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 角度依存グラフ（波長ごとの反射率 vs 角度）---
    st.subheader('Reflectance vs Angle (per wavelength)')

    # 表示波長の選択肢：wl_nm_ar から整数丸めで重複除去
    wl_choices = sorted(list(dict.fromkeys([int(round(w)) for w in wl_nm_ar])))

    # デフォルト波長と対応色
    _PRESET_WLS    = [400, 470, 540, 600, 700]
    _PRESET_COLORS = ['#8B00FF', '#0000FF', '#008000', '#FFD700', '#FF0000']
    _WL_COLOR_MAP  = dict(zip(_PRESET_WLS, _PRESET_COLORS))

    import numpy as _np
    default_wls = [w for w in _PRESET_WLS if w in wl_choices]

    acol1, acol2 = st.columns(2)
    with acol1:
        sel_wls = st.multiselect(
            'Wavelengths to plot [nm]', wl_choices,
            default=default_wls, key='sel_wls_angle'
        )
    with acol2:
        pol_angle = st.radio(
            'Polarization', ('Rp', 'Rs'),
            horizontal=True, key='pol_angle'
        )

    if sel_wls:
        Z_ang = Rp2d if pol_angle == 'Rp' else Rs2d

        # 角度軸を対称に折り返す（計算なし、データのミラー）
        # angle_ar が 0 始まりの場合は 0 を重複させない
        if float(angle_ar[0]) == 0.0:
            _x_sym = _np.concatenate([-angle_ar[1:][::-1], angle_ar])
            def _mirror(col): return _np.concatenate([col[1:][::-1], col])
        else:
            _x_sym = _np.concatenate([-angle_ar[::-1], angle_ar])
            def _mirror(col): return _np.concatenate([col[::-1], col])

        fig_ang = go.Figure()
        _auto_colors = ['#17becf','#e377c2','#7f7f7f','#bcbd22','#8c564b']
        _auto_idx = 0
        for wl_sel in sorted(sel_wls):
            idx_wl = int(_np.argmin(_np.abs(wl_nm_ar - wl_sel)))
            actual_wl = int(round(wl_nm_ar[idx_wl]))
            color = _WL_COLOR_MAP.get(wl_sel)
            if color is None:
                color = _auto_colors[_auto_idx % len(_auto_colors)]
                _auto_idx += 1
            fig_ang.add_trace(go.Scatter(
                x=_x_sym, y=_mirror(Z_ang[:, idx_wl]),
                name=f'{actual_wl} nm',
                mode='lines',
                line=dict(color=color, width=2),
            ))
        fig_ang.update_layout(
            title=f'{pol_angle} (m={_order_label(repr_order)}): Reflectance vs Angle',
            xaxis_title='Angle (deg)',
            yaxis_title='Reflectance',
        )
        fig_ang.update_xaxes(range=[float(-angle_ar[-1]), float(angle_ar[-1])])
        fig_ang.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_ang, use_container_width=True)

    # 下流の色計算・CSV用に代表角度（先頭）のスペクトルを1Dとして用意
    inc_angle=float(angle_ar[0])
    Rp=Rp2d[0,:]; Rs=Rs2d[0,:]; Tp=Tp2d[0,:]; Ts=Ts2d[0,:]
    st.caption(f'※ 下の色計算・CSVは代表角度 {round(inc_angle,1)}°・回折次数 m={_order_label(repr_order)} のスペクトルに基づきます。')

if wl_option==spMenu[0]:
    st.subheader('Colorimetry')


    colour_Rp,colour_Rs,colour_Tp,colour_Ts,colour_wl_ar=Rp,Rs,Tp,Ts,wl_nm_ar

    sd_Rp = colour.SpectralDistribution(colour_Rp, name='Rp')
    sd_Rs = colour.SpectralDistribution(colour_Rs, name='Rs')
    # sd_Tp = colour.SpectralDistribution(colour_Tp, name='Tp')
    # sd_Ts = colour.SpectralDistribution(colour_Ts, name='Ts')

    sd_Rp.wavelengths=wl_nm_ar
    sd_Rs.wavelengths=wl_nm_ar
    # sd_Tp.wavelengths=wl_nm_ar
    # sd_Ts.wavelengths=wl_nm_ar


    # Convert to Tristimulus Values
    cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER['CIE 1931 2 Degree Standard Observer']
    illuminant = colour.SDS_ILLUMINANTS['D65']

    # Calculating the sample spectral distribution *CIE XYZ* tristimulus values.
    XYZ_Rp = colour.sd_to_XYZ(sd_Rp, cmfs, illuminant)
    XYZ_Rs = colour.sd_to_XYZ(sd_Rs, cmfs, illuminant)
    # XYZ_Tp = colour.sd_to_XYZ(sd_Tp, cmfs, illuminant)
    # XYZ_Ts = colour.sd_to_XYZ(sd_Ts, cmfs, illuminant)
    RGB_Rp = colour.XYZ_to_sRGB(XYZ_Rp / 100)
    RGB_Rs = colour.XYZ_to_sRGB(XYZ_Rs / 100)
    # RGB_Tp = colour.XYZ_to_sRGB(XYZ_Tp / 100)
    # RGB_Ts = colour.XYZ_to_sRGB(XYZ_Ts / 100)
    b_Rp=[]
    for v in RGB_Rp:
        b_Rp.append(np.clip(round(v*255),0,255))
    b_Rs=[]
    for v in RGB_Rs:
        b_Rs.append(np.clip(round(v*255),0,255))
    # b_Tp=[]
    # for v in RGB_Tp:
    #     b_Tp.append(np.clip(round(v*255),0,255))
    # b_Ts=[]
    # for v in RGB_Ts:
    #     b_Ts.append(np.clip(round(v*255),0,255))


    strRGB_Rp='#'+format(b_Rp[0], '02x')+format(b_Rp[1], '02x')+format(b_Rp[2], '02x')
    strRGB_Rs='#'+format(b_Rs[0], '02x')+format(b_Rs[1], '02x')+format(b_Rs[2], '02x')
    # strRGB_Tp='#'+format(b_Tp[0], '02x')+format(b_Tp[1], '02x')+format(b_Tp[2], '02x')
    # strRGB_Ts='#'+format(b_Ts[0], '02x')+format(b_Ts[1], '02x')+format(b_Ts[2], '02x')

    col1,col2,col3,col4=st.columns(4)
    with col1:
        color_Rp = st.color_picker('Rp', strRGB_Rp,key='cp_Rp'+ strRGB_Rp)
        st.write('XYZ chromaticity',XYZ_Rp)
    with col2:
        color_Rs = st.color_picker('Rs', strRGB_Rs,key='cp_Rs'+ strRGB_Rs)
        st.write('XYZ chromaticity',XYZ_Rs)
    # with col3:
    #     color_Tp = st.color_picker('Tp', strRGB_Tp,key='cp_Tp')
    #     st.write('XYZ chromaticity',XYZ_Tp)
    # with col4:
    #     color_Ts = st.color_picker('Ts', strRGB_Ts,key='cp_Ts')
    #     st.write('XYZ chromaticity',XYZ_Ts)




nwl=len(wl_nm_ar)
data=np.concatenate([wl_nm_ar.reshape([nwl,1]),Rp.reshape([nwl,1]),Rs.reshape([nwl,1]),Tp.reshape([nwl,1]),Ts.reshape([nwl,1])],1)
df=pd.DataFrame(data,columns=['Wavelength(nm)', 'Rp', 'Rs', 'Tp', 'Ts'])
#df=df.reset_index(drop=True)
df=df.set_index('Wavelength(nm)')

csv = convert_df(df)

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
now = datetime.datetime.now(JST)
# YYYYMMDDhhmmss形式に書式化
d = now.strftime('%Y%m%d%H%M%S')
fname='data_'+d+'.csv'

st.subheader('Download spectrum and color data')

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name=fname,
    mime='text/csv',
)

print(Rp)
#print(Rp0)
