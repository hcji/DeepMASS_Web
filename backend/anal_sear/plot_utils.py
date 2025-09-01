# import gradio as gr
import numpy as np
from matchms import Spectrum
from matchms import filtering as msfilters
from matplotlib.figure import Figure
from rdkit import Chem
from rdkit.Chem import Draw
import plotly.graph_objects as go
from io import BytesIO
import base64
from typing import Optional

def show_default_mol_search(res_state):
    """
    为 res_state 中的每个谱图生成分子结构图，返回一个包含 Base64 编码 PNG 图像的列表。
    
    Args:
        res_state (pd.DataFrame): 包含 'Identified Spectrum' 列的 DataFrame。
    
    Returns:
        list: 包含每个谱图的分子结构图（Base64 编码的 PNG 字符串）的列表。
              如果某谱图无法生成图像，则对应位置为 None。
    """
    try:
        if res_state is None or "Identified Spectrum" not in res_state:
            print("Error: res_state is None or missing 'Identified Spectrum' column")
            return []
        
        ref_img_list = []
        
        # 遍历 res_state 的每一行
        for idx in range(len(res_state)):
            try:
                ref_spec = res_state["Identified Spectrum"][idx]
                if ref_spec is None or ref_spec.metadata is None or "smiles" not in ref_spec.metadata:
                    ref_img_list.append(None)
                    continue
                
                ref_smi = ref_spec.metadata["smiles"]
                mol_ref = Chem.MolFromSmiles(ref_smi)
                if mol_ref is None:
                    ref_img_list.append(None)
                    continue
                
                # 生成分子结构图
                ref_img = Draw.MolToImage(mol_ref, wedgeBonds=False, size=(300, 300))
                
                # 转换为 Base64 编码的 PNG
                buffered = BytesIO()
                ref_img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                ref_img_list.append(f"data:image/png;base64,{img_base64}")
                
            except Exception as e:
                print(f"Error processing molecule at index {idx}: {e}")
                ref_img_list.append(None)  # 失败时添加 None 占位
        
        return ref_img_list
    
    except Exception as e:
        print(f"show_default_mol_search error: {e}")
        return []
def plot_ref_mol(smi_ref):
    try:
        if smi_ref is None:
            return None
        mol_ref = Chem.MolFromSmiles(smi_ref)
        ref_img = Draw.MolToImage(mol_ref, wedgeBonds=False)
        return ref_img
    except Exception as e:
        print("plot_ref_mol error: {e}")
        return None


def _mk_vertical_trace(mz, inten, color, name, reverse=False,
                       n_seg: int = 15):
    """
    把 [0, intensity] 这一段细分成 n_seg 份，
    每份再添加一个数据点 ⇒ 整根竖线处处可 Hover
    """
    sign = -1 if reverse else 1
    x, y, cd = [], [], []

    for m, i in zip(mz, inten):
        # 从 0 → i，等距插值  n_seg + 1  个点（含 端点）
        ys = np.linspace(0, sign * i, n_seg + 1)
        xs = np.full_like(ys, m)

        x  += xs.tolist() + [np.nan]
        y  += ys.tolist() + [np.nan]
        cd += [i] * (n_seg + 1) + [np.nan]

    return go.Scatter(
        x=x, y=y, customdata=cd,
        mode="lines",
        line=dict(color=color, width=1.6),
        hovertemplate="m/z=%{x:.4f}<br>Intensity=%{customdata:.4f}",
        name=name, showlegend=False
    )

def plot_2_spectrum(spectrum: Spectrum,
                    reference: Spectrum,
                    loss: bool = False):
    if spectrum is None or reference is None:
        return None

    # ---------- 1) peak / loss ----------
    mz_q, int_q = spectrum.peaks.mz, spectrum.peaks.intensities
    mz_r, int_r = reference.peaks.mz, reference.peaks.intensities

    if loss:
        try:
            spectrum  = msfilters.add_parent_mass(spectrum)
            reference = msfilters.add_parent_mass(reference)
            spectrum  = msfilters.add_losses(spectrum, 10.0, 2000.0)
            reference = msfilters.add_losses(reference, 10.0, 2000.0)

            mz_q, int_q = spectrum.losses.mz,  spectrum.losses.intensities
            mz_r, int_r = reference.losses.mz, reference.losses.intensities
        except Exception as e:
            print("Cannot plot losses:", e)

    # ---------- 2) 归一化 ----------
    int_r = int_r / np.max(int_r) if np.max(int_r) else int_r
    int_q = int_q / np.max(int_q) if np.max(int_q) else int_q

    # ---------- 3) 图形 ----------
    fig = go.Figure()
    fig.add_trace(_mk_vertical_trace(mz_q, int_q, "red",  "Query"))
    fig.add_trace(_mk_vertical_trace(mz_r, int_r, "blue", "Reference", reverse=True))

    # ---------- 4) top-k 注释 ----------
    # def annotate(mz, inten, reverse=False):
    #     idx = np.argsort(inten)[-top_k:]
    #     for i in idx:
    #         fig.add_annotation(
    #             x=float(mz[i]),
    #             y=float((-1 if reverse else 1) * inten[i]),
    #             text=f"{mz[i]:.4f}",
    #             showarrow=False,
    #             font=dict(size=8),
    #             yshift=6 if reverse else -6
    #         )
    # annotate(mz_q, int_q, False)
    # annotate(mz_r, int_r, True)

    # ---------- 5) 外观 ----------
    fig.update_layout(
        autosize=True,
        barmode="overlay",
        bargap=0,
        bargroupgap=0,
        template="simple_white",
        font=dict(family="Times New Roman", size=10, color="black"),
        xaxis=dict(domain=[0.12, 0.95]),
        yaxis=dict(domain=[0.12, 0.90]),
        yaxis_range=[-1.1, 1.1],
        margin=dict(l=60, r=25, t=40, b=55)
    )

    # --- 轴标题---
    fig.update_xaxes(
        title_text="m/z",
        title_font=dict(size=18, family="Times New Roman"),
        title_standoff=8,
        showline=True, 
        linewidth=2, 
        linecolor="black"
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=True,
        zerolinewidth=1.3,
        zerolinecolor="black",
        showline=True, linewidth=2, linecolor="black",
        title_text="abundance",
        title_font=dict(size=18, family="Times New Roman"),
        title_standoff=8
    )
    return fig



def show_default_ref_spectrum_search(res_state):
    """
    为 res_state 中的每个谱图生成 spectrum 和 loss 图表，返回两个列表。
    
    Args:
        res_state (pd.DataFrame): 包含 'spectrum' 和 'Identified Spectrum' 列的 DataFrame。
    
    Returns:
        tuple: (fig_loss_list, fig_list)
            - fig_loss_list: 包含每个谱图的 loss 图表（JSON 格式）的列表。
            - fig_list: 包含每个谱图的 spectrum 图表（JSON 格式）的列表。
    """
    try:
        if res_state is None or "spectrum" not in res_state or "Identified Spectrum" not in res_state:
            print("Error: res_state is None or missing required columns")
            return [], []
        
        fig_loss_list = []
        fig_list = []
        
        # 遍历 res_state 的每一行
        for idx in range(len(res_state)):
            try:
                spectrum = res_state["spectrum"][idx]
                identified_spectrum = res_state["Identified Spectrum"][idx]
                
                # 生成 loss 图表
                fig_loss = plot_2_spectrum(spectrum, identified_spectrum, loss=True)
                # 生成 spectrum 图表
                fig = plot_2_spectrum(spectrum, identified_spectrum, loss=False)
                
                # 转换为 JSON 格式（供前端使用）
                fig_loss_json = fig_loss.to_json() if fig_loss else None
                fig_json = fig.to_json() if fig else None
                
                fig_loss_list.append(fig_loss_json)
                fig_list.append(fig_json)
                
            except Exception as e:
                print(f"Error processing spectrum at index {idx}: {e}")
                # 失败时添加 None 占位，保持列表长度一致
                fig_loss_list.append(None)
                fig_list.append(None)
        
        return fig_loss_list, fig_list
    
    except Exception as e:
        print(f"show_default_ref_spectrum_search error: {e}")
        return [], []



# === 把 SMILES 画成 PNG base64 ===
def smiles_to_png_base64(smiles: str, size=(300, 300)) -> Optional[str]:

    try:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, wedgeBonds=False, size=size)
        buf = BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"smiles_to_png_base64 error: {e}")
        return None

# === 按“(query_spec, ref_spec)”即时生成三种图（JSON/Base64）===
def build_plots_for_pair(query_spec: Spectrum, ref_spec: Spectrum):
    """
    返回 (spectrum_plot_json, loss_plot_json, structure_png_base64)
    任一失败则对应返回 None
    """
    try:
        fig_spec = plot_2_spectrum(query_spec, ref_spec, loss=False)
        spec_json = fig_spec.to_json() if fig_spec else None
    except Exception as e:
        print(f"build_plots_for_pair spectrum error: {e}")
        spec_json = None

    try:
        fig_loss = plot_2_spectrum(query_spec, ref_spec, loss=True)
        loss_json = fig_loss.to_json() if fig_loss else None
    except Exception as e:
        print(f"build_plots_for_pair loss error: {e}")
        loss_json = None

    try:
        smiles = None
        if hasattr(ref_spec, "metadata") and isinstance(ref_spec.metadata, dict):
            smiles = ref_spec.metadata.get("smiles")
        struct_png = smiles_to_png_base64(smiles) if smiles else None
    except Exception as e:
        print(f"build_plots_for_pair structure error: {e}")
        struct_png = None

    return spec_json, loss_json, struct_png

