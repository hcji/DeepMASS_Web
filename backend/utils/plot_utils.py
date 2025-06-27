import heapq
import logging
from itertools import chain

import gradio as gr
import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms import filtering as msfilters
from matplotlib.figure import Figure
from molmass import Formula
from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw

from backend.load_config import GLOBAL_CONFIG
import plotly.graph_objects as go
# 从配置文件获取质谱图DPI和比例
dpi_config = GLOBAL_CONFIG["identification"]["plot"]["dpi"]
width_config = GLOBAL_CONFIG["identification"]["plot"]["width"]
length_config = GLOBAL_CONFIG["identification"]["plot"]["length"]


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


def show_mol(structure_state, cur_spectrum, evt: gr.SelectData):
    # ① 空值保护
    if cur_spectrum is None or structure_state is None:
        return None, None
    
    try:
        if evt is None or evt.index is None:
            return None, None

        line_num = evt.index[0]
        return show_default_mol(structure_state, cur_spectrum, line_num)
    except Exception as e:
        logging.error(f"Error in show_mol: {e}")
        return None, None

def show_default_mol(structure_state, cur_spectrum, idx=0):
    # ① 空值保护
    if cur_spectrum is None or not hasattr(cur_spectrum, "metadata"):
        return None, None
    
    try:
        ref_smi = cur_spectrum.metadata["reference"][idx].metadata["smiles"]
        anno_img, ref_img = plot_2_mol(structure_state, ref_smi)
        return anno_img, ref_img
    except Exception as e:
        logging.error(f"Error in show_default_mol: {e}")
        return None, None

def get_formula_mass(formula: str):
    f = Formula(formula.replace("-", ""))
    try:
        mass = f.isotope.mass
    except Exception as e:
        logging.warn(f"{e}")
        mass = 0
    return mass


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


def plot_2_mol(smi_anno, smi_ref, hightlight=True):
    mol_anno = Chem.MolFromSmiles(smi_anno)
    mol_ref = Chem.MolFromSmiles(smi_ref)
    if hightlight:
        mcs = rdFMCS.FindMCS(
            [mol_anno, mol_ref],
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            matchValences=True,
            ringMatchesRingOnly=True,
        )
        mcs_str = mcs.smartsString
        mcs_mol = Chem.MolFromSmarts(mcs_str)
        all_subs_anno = tuple(
            chain.from_iterable(mol_anno.GetSubstructMatches(mcs_mol))
        )
        all_subs_ref = tuple(chain.from_iterable(mol_ref.GetSubstructMatches(mcs_mol)))
    else:
        all_subs_anno = ()
        all_subs_ref = ()

    ref_img = Draw.MolToImage(mol_ref, highlightAtoms=all_subs_ref, wedgeBonds=False)
    anno_img = Draw.MolToImage(mol_anno, highlightAtoms=all_subs_anno, wedgeBonds=False)
    return anno_img, ref_img


def show_ref_spectrums(spectrum_state, structure_obj, evt: gr.SelectData):
    try:
        line_num = evt.index[0]
        return get_reference_table(spectrum_state, structure_obj, line_num)
    except Exception as e:
        logging.error(f"Error in show_ref_spectrum: {e}")
        return None, None

def get_reference_table(spectrum_state, structure_obj, idx=0):
    try:                
        smi_anno = structure_obj["CanonicalSMILES"][idx]
        current_reference = spectrum_state.metadata["reference"]
        annotation = spectrum_state.metadata["annotation"]
        i = np.where(annotation["CanonicalSMILES"].values == smi_anno)[0][0]
        reference_table = []
        for s in current_reference:
            if "smiles" in s.metadata.keys():
                smiles = s.metadata["smiles"]
            else:
                smiles = ""
            if "compound_name" in s.metadata.keys():
                name = s.metadata["compound_name"]
            else:
                name = smiles
            if "adduct" in s.metadata.keys():
                adduct = s.metadata["adduct"]
            else:
                adduct = ""
            if "parent_mass" in s.metadata.keys():
                parent_mass = s.metadata["parent_mass"]
            else:
                parent_mass = ""
            if "database" in s.metadata.keys():
                ref_database = s.metadata["database"]
            else:
                ref_database = ""
            reference_table.append([name, adduct, smiles, parent_mass, ref_database])
        reference_table = pd.DataFrame(
            reference_table, columns=["name", "adduct", "smiles", "parent_mass", "database"]
        )  # 创建一个DataFrame对象，用于存储参考表格的数据

        return reference_table, smi_anno
    except Exception as e:
        logging.error(f"Error in get_reference_table: {e}")
        return pd.DataFrame(columns=["name", "adduct", "smiles", "parent_mass", "database"]), ""

def show_structure_select_all(spectrum_state, structure_obj, evt: gr.SelectData):
    try:
        # 先更新 Reference Spectrums 表格 & structure_state
        ref_spectrums, structure_state = show_ref_spectrums(spectrum_state, structure_obj, evt)
        # 再用刚更新的 structure_state 和 spectrum_state 直接画分子图
        anno_img, ref_img = show_default_mol(structure_state, spectrum_state)
        return ref_spectrums, structure_state, anno_img, ref_img
    except Exception as e:
        logging.error(f"Error in on_structure_select_all: {e}")
        return pd.DataFrame(columns=["name", "adduct", "smiles", "parent_mass", "database"]), "", None, None
def get_default_structure_select_all(spectrum_state, structure_obj):
    try:
        # 先更新 Reference Spectrums 表格 & structure_state
        ref_spectrums, structure_state = get_reference_table(spectrum_state, structure_obj)
        # 再用刚更新的 structure_state 和 spectrum_state 直接画分子图
        anno_img, ref_img = show_default_mol(structure_state, spectrum_state)
        return ref_spectrums, structure_state, anno_img, ref_img
    except Exception as e:
        logging.error(f"Error in on_structure_select_all: {e}")
        return pd.DataFrame(columns=["name", "adduct", "smiles", "parent_mass", "database"]), "", None, None
    
if __name__ == "__main__":
    print(dpi_config, width_config, length_config)
