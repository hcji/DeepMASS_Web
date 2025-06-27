import gradio as gr
import numpy as np
from matchms import Spectrum
from matchms import filtering as msfilters
from matplotlib.figure import Figure
from rdkit import Chem
from rdkit.Chem import Draw
import plotly.graph_objects as go


# 质谱图比例
dpi_config = 900
width_config = 2
length_config = 1


_annotation_kws = {
    "horizontalalignment": "left",  # if not mirror_intensity else "right",
    "verticalalignment": "center",
    "fontsize": 2,
    "rotation": 90,
    "rotation_mode": "anchor",
    "zorder": 5,
}
_annotation_reverse_kws = {
    "horizontalalignment": "left",  # if not mirror_intensity else "right",
    "verticalalignment": "center",
    "fontsize": 2,
    "rotation": 270,
    "rotation_mode": "anchor",
    "zorder": 5,
}


def show_mol_search(res_state, evt: gr.SelectData):
    try:
        line_num = evt.index[0]
        if res_state is None or "Identified Spectrum" not in res_state.columns or res_state["Identified Spectrum"] is None:
            return None
        return show_default_mol_search(res_state, line_num)
    except Exception as e:
        print("show_mol_search error: {e}")
        return None

def show_default_mol_search(res_state, idx=0):
    try:
        ref_spec = res_state["Identified Spectrum"][idx]
        if ref_spec is None or ref_spec.metadata is None or "smiles" not in ref_spec.metadata:
            return None
        ref_smi = ref_spec.metadata["smiles"]
        ref_img = plot_ref_mol(ref_smi)
        return ref_img
    except Exception as e:
        print("show_default_mol_search error: {e}")
        return None
    

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


def show_ref_spectrum_search(res_state, evt: gr.SelectData):
    try:
        line_num = evt.index[0]
        if res_state is None or "spectrum" not in res_state or "Identified Spectrum" not in res_state:
            return None, None
        return show_default_ref_spectrum_search(res_state, line_num)
    except Exception as e:
        print(f"show_ref_spectrum_search error: {e}")

        return None, None
def show_default_ref_spectrum_search(res_state, idx=0):
    try:
        if res_state is None or "spectrum" not in res_state or "Identified Spectrum" not in res_state:
            return None, None
        fig_loss = plot_2_spectrum(
            res_state["spectrum"][idx], res_state["Identified Spectrum"][idx], loss=True
        )
        fig = plot_2_spectrum(
            res_state["spectrum"][idx], res_state["Identified Spectrum"][idx], loss=False
        )
        return fig_loss, fig
    except Exception as e:
        print(f"show_default_ref_spectrum_search error: {e}")
        return None, None

if __name__ == "__main__":
    print(dpi_config, width_config, length_config)
