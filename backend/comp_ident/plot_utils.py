import numpy as np
from matchms import Spectrum
from matchms import filtering as msfilters
from rdkit import Chem
from rdkit.Chem import Draw, rdFMCS
import plotly.graph_objects as go
from io import BytesIO
import base64
from typing import Optional
from itertools import chain
from molmass import Formula

def _mk_vertical_trace(mz, inten, color, name, reverse=False, n_seg: int = 15):
    """把 [0, intensity] 细分为 n_seg 段，让整根峰柱都能 Hover。"""
    sign = -1 if reverse else 1
    x, y, cd = [], [], []
    for m, i in zip(mz, inten):
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

def plot_2_spectrum(spectrum: Spectrum, reference: Spectrum, loss: bool = False):
    """Query vs Reference 双向镜像质谱；返回 Plotly Figure（由上层 .to_json()）。"""
    if spectrum is None or reference is None:
        return None

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

    # 归一化
    int_r = int_r / np.max(int_r) if np.max(int_r) else int_r
    int_q = int_q / np.max(int_q) if np.max(int_q) else int_q

    fig = go.Figure()
    fig.add_trace(_mk_vertical_trace(mz_q, int_q, "red",  "Query"))
    fig.add_trace(_mk_vertical_trace(mz_r, int_r, "blue", "Reference", reverse=True))

    fig.update_layout(
        autosize=True, template="simple_white",
        font=dict(family="Times New Roman", size=10, color="black"),
        xaxis=dict(domain=[0.12, 0.95]),
        yaxis=dict(domain=[0.12, 0.90], range=[-1.1, 1.1]),
        margin=dict(l=60, r=25, t=40, b=55)
    )
    fig.update_xaxes(
        title_text="m/z", title_font=dict(size=18, family="Times New Roman"),
        title_standoff=8, showline=True, linewidth=2, linecolor="black"
    )
    fig.update_yaxes(
        showgrid=False, zeroline=True, zerolinewidth=1.3, zerolinecolor="black",
        showline=True, linewidth=2, linecolor="black",
        title_text="abundance", title_font=dict(size=18, family="Times New Roman"), title_standoff=8
    )
    return fig

def plot_2_mol(smi_anno: str, smi_ref: str, highlight: bool = True, size=(300, 300)):
    """返回（注释结构、参考结构）PNG Base64；失败返回 (None, None)。"""
    try:
        mol_anno = Chem.MolFromSmiles(smi_anno)
        mol_ref  = Chem.MolFromSmiles(smi_ref)
        if mol_anno is None or mol_ref is None:
            return None, None

        if highlight:
            mcs = rdFMCS.FindMCS(
                [mol_anno, mol_ref],
                bondCompare=rdFMCS.BondCompare.CompareOrderExact,
                matchValences=True,
                ringMatchesRingOnly=True,
            )
            mcs_str = mcs.smartsString
            mcs_mol = Chem.MolFromSmarts(mcs_str)
            all_subs_anno = tuple(chain.from_iterable(mol_anno.GetSubstructMatches(mcs_mol)))
            all_subs_ref  = tuple(chain.from_iterable(mol_ref.GetSubstructMatches(mcs_mol)))
        else:
            all_subs_anno = ()
            all_subs_ref  = ()

        def mol_to_base64(mol, highlight_atoms):
            img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, wedgeBonds=False, size=size)
            buf = BytesIO(); img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        return mol_to_base64(mol_anno, all_subs_anno), mol_to_base64(mol_ref, all_subs_ref)
    except Exception as e:
        print(f"plot_2_mol error: {e}")
        return None, None

def get_formula_mass(formula: str):
    f = Formula(formula.replace("-", ""))
    try:
        mass = f.isotope.mass
    except Exception as e:
        print(f"{e}")
        mass = 0
    return mass
