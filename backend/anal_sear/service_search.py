from fastapi import HTTPException
from matchms import Spectrum
import pandas as pd
from gensim.models import Word2Vec
from hnswlib import Index
import shutil
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
import os, uuid, pickle
from tqdm import tqdm
import numpy as np
import hnswlib, gensim

from backend.anal_sear.identify_unkown import id_spectrum_list
from backend.anal_sear.plot_utils import show_default_mol_search, show_default_ref_spectrum_search
from analogSearch.spectrum_process import load_spectrum_file

# 预加载模型路径常量
MODEL_POS_PATH = 'model/Ms2Vec_allGNPSpositive.hdf5'
MODEL_NEG_PATH = 'model/Ms2Vec_allGNPSnegative.hdf5'
model_pos = Word2Vec.load(MODEL_POS_PATH)
model_neg = Word2Vec.load(MODEL_NEG_PATH)

def df_to_clean_records(df: pd.DataFrame):
    """
    将 DataFrame 转换为字典列表，处理 NaN/inf，保留 Plotly JSON 和 Base64 字符串。
    
    Args:
        df (pd.DataFrame): 输入 DataFrame。
    
    Returns:
        list: 字典列表，包含处理后的数据。
    """
    def convert_value(x):
        if x is None or pd.isna(x):
            return None
        if isinstance(x, (str, dict)):  # 保留 JSON 和 Base64 字符串
            return x
        return str(x)

    return (
        df.replace([np.inf, -np.inf], np.nan)
          .astype(object)
          .where(pd.notnull(df), None)
          .applymap(convert_value)
          .to_dict(orient="records")
    )


def load_files(file_list):
    """
    从文件列表中读取质谱，并输出到窗口
    Args:
        file_list: 文件名列表，是保存在服务器临时路径中
    Returns:
        所有读取的质谱
        所有质谱的显示名称
    """

    # 目标保存压缩文件名称
    target_zip_file_name = "result.zip"
    if len(file_list) == 1:
        target_zip_file_name = f"{os.path.basename(file_list[0])}.zip"
    # 读取每个质谱文件
    spectrum_list = []
    for file_name in file_list:  # 遍历每一个文件名
        try:
            loaded_spectra_list = load_spectrum_file(file_name)
        except Exception as e:
            print(f"Error when loading {file_name}: {e}")
            # raise gr.Error("Please upload standard file")
        spectrum_list.extend(loaded_spectra_list)

    # 获取所有的质谱名称，若无，则使用编号代替s
    titles = [
        s.metadata["compound_name"]
        if "compound_name" in list(s.metadata.keys())
        else f"spectrum {i}"
        for i, s in enumerate(spectrum_list)
    ]
    spectrums_df = pd.DataFrame({"title": titles, "spectrum": spectrum_list})
    # 用于返回nav的质谱名列表
    name_list = spectrums_df[["title"]]

    return spectrums_df, name_list, target_zip_file_name


def matchms_click_fn(
        threshold: float,
        res_state: pd.DataFrame,
        refs_pos_state, refs_neg_state,
        hnsw_pos_state, hnsw_neg_state,
        progress_cb=None,
):
    spectra = res_state["spectrum"].tolist()

    if hnsw_pos_state is not None:
        refs_pos, refs_neg = refs_pos_state, refs_neg_state
        hnsw_pos, hnsw_neg = hnsw_pos_state, hnsw_neg_state
    else:
        with open("analogSearch_data/references_spectrums_positive.pickle", "rb") as f:
            refs_pos = pickle.load(f)
        with open("analogSearch_data/references_spectrums_negative.pickle", "rb") as f:
            refs_neg = pickle.load(f)
        hnsw_pos = Index(space="l2", dim=300)
        hnsw_pos.load_index("analogSearch_data/references_index_positive_spec2vec.bin")
        hnsw_pos.set_ef(300)
        hnsw_neg = Index(space="l2", dim=300)
        hnsw_neg.load_index("analogSearch_data/references_index_negative_spec2vec.bin")
        hnsw_neg.set_ef(300)

    # === 关键：拿到“每个查询的 TopK 列表”，每项是 (ref_spec, score, dbidx) ===
    all_topk = id_spectrum_list(
        spectra,
        hnsw_pos, model_pos, refs_pos,
        hnsw_neg, model_neg, refs_neg,
        on_progress=progress_cb
    )

    # 1) 为每个查询构建 annotation（多行：TopK 全部），并确定 Top1 是否过阈
    annotations = []
    identified_list = []
    for q_idx, topk_list in enumerate(all_topk, 1):
        per_query_rows = []
        identified_spectrum = None

        for rank, (ref_spec, score, dbidx) in enumerate(topk_list, 1):
            meta = {k: v for k, v in getattr(ref_spec, "metadata", {}).items()
                    if k not in ("mz", "intensities")}
            smiles = meta.get("smiles")
            per_query_rows.append({
                "query_index": q_idx,
                "rank": rank,
                "StructSimScore": score,
                "database_index": dbidx,
                "smiles": smiles,
                **meta
            })

        if topk_list:
            top1_ref, top1_score, _ = topk_list[0]
            if top1_score >= threshold:
                identified_spectrum = top1_ref

        identified_list.append(identified_spectrum)
        # 不做阈值过滤：CSV 会包含全 TopK；若需过滤，在这里筛掉 < threshold 的行
        annotations.append(pd.DataFrame(per_query_rows) if per_query_rows else pd.DataFrame(
            [{"StructSimScore": None, "database_index": None, "smiles": None}]
        ))

    # 2) 回填到 res_state，生成 Top1 的图（与原流程兼容）
    res_state["Identified Spectrum"] = identified_list
    res_state["annotation"] = annotations

    # from backend.anal_sear.plot_utils import show_default_mol_search, show_default_ref_spectrum_search
    # spectrum_loss_plot_list, spectrum_plot_list = show_default_ref_spectrum_search(res_state)
    # ref_structure_plot_list = show_default_mol_search(res_state)

    # 3) 组装“扁平表”：Top1 带图，其它先置空（前端点击再懒加载）
    flat_rows = []
    for q_idx, topk_list in enumerate(all_topk, 1):
        for rank, (ref_spec, score, dbidx) in enumerate(topk_list, 1):
            meta = {k: v for k, v in getattr(ref_spec, "metadata", {}).items()
                    if k not in ("mz", "intensities")}
            smiles = meta.get("smiles")
            # is_top1 = (rank == 1)
            # flat_rows.append({
            #     "query_index": q_idx,
            #     "rank": rank,
            #     "StructSimScore": score,
            #     "database_index": dbidx,
            #     "smiles": smiles,
            #     **meta,
            #     "spectrum_plot": spectrum_plot_list[q_idx - 1] if is_top1 else None,
            #     "spectrum_loss_plot": spectrum_loss_plot_list[q_idx - 1] if is_top1 else None,
            #     "ref_structure_plot": ref_structure_plot_list[q_idx - 1] if is_top1 else None,
            # })
            flat_rows.append({
                "query_index": q_idx,
                "rank": rank,
                "StructSimScore": score,
                "database_index": dbidx,
                "smiles": smiles,
                **meta,
                "spectrum_plot": None,
                "spectrum_loss_plot": None,
                "ref_structure_plot": None,
            })
    result_df = pd.DataFrame(flat_rows)
    
    return res_state, result_df, all_topk # spectrum_loss_plot_list, spectrum_plot_list, ref_structure_plot_list


def get_title_from_spectrum(spectrum: Spectrum, idx=None):
    if "compound_name" in spectrum.metadata.keys():
        return spectrum.metadata["compound_name"]
    else:
        if idx is not None:
            return f"Spectrum {idx}"
        else:
            return f"Spectrum {uuid.uuid4()}"
        

def init_custom_database(mgf_path):
    # try:

    # 调用上面那个脚本，生成 pickle + bin
    pos_pkl, neg_pkl, pos_idx_bin, neg_idx_bin, out_dir = process_spectrums(
        mgf_path,
        pos_model_path="model/Ms2Vec_allGNPSpositive.hdf5",
        neg_model_path="model/Ms2Vec_allGNPSnegative.hdf5"
    )
    # 把 pickle list 加载到内存，HNSW 索引对象也 load 到内存
    with open(pos_pkl,"rb") as f: refs_pos = pickle.load(f)
    with open(neg_pkl,"rb") as f: refs_neg = pickle.load(f)

    hnsw_pos = hnswlib.Index(space="l2", dim=300)
    hnsw_pos.load_index(pos_idx_bin)
    hnsw_pos.set_ef(300)

    hnsw_neg = hnswlib.Index(space="l2", dim=300)
    hnsw_neg.load_index(neg_idx_bin)
    hnsw_neg.set_ef(300)


    return refs_pos, refs_neg, hnsw_pos, hnsw_neg, out_dir


def process_spectrums(file_path, pos_model_path, neg_model_path, out_dir=None):
    """
    读取 file_path （.mgf），用正/负模型生成：
     - positive_spectrums.pickle
     - negative_spectrums.pickle
     - positive_index.bin
     - negative_index.bin
    并返回这四个文件的绝对路径。
    """
    if out_dir is None:
        out_dir = os.path.join("tmp/database_tmp", uuid.uuid4().hex)
    os.makedirs(out_dir, exist_ok=True)

    # 1）读 mgf 并分类
    specs = load_spectrum_file(file_path)
    pos_specs = []
    neg_specs = []
    for index, s in enumerate(specs):
        s.set("database_index", index)
        mode = s.metadata.get("ionmode", "positive")
        if mode == "negative":
            neg_specs.append(s)
        else:
            pos_specs.append(s)
    # print(f"Detected {len(pos_specs)} positive and {len(neg_specs)} negative spectra.")
    # —— 在这里检查有没有读到正谱或负谱 —— 
    if not pos_specs:
        raise "No positive-mode spectra detected. Please check the ionmode annotation."
    if not neg_specs:
        raise "No negative-mode spectra detected. Please check the ionmode annotation."
    
    # 2）保存 pickle
    pos_pkl = os.path.join(out_dir, "positive_spectrums.pickle")
    neg_pkl = os.path.join(out_dir, "negative_spectrums.pickle")
    with open(pos_pkl,"wb") as f: pickle.dump(pos_specs, f)
    with open(neg_pkl,"wb") as f: pickle.dump(neg_specs, f)
    print(f"Saved positive spectra to {pos_pkl}, negative spectra to {neg_pkl}.")
    # 内部函数：从 pickle + model 生成 index
    def build_index(pkl_path, model_path, dim=300, prefix="pos"):
        with open(pkl_path,"rb") as f:
            refs = pickle.load(f)
        model = gensim.models.Word2Vec.load(model_path)
        vecs = []
        for s in tqdm(refs, desc=f"Vectorize {prefix}"):
            v = calc_vector(model, SpectrumDocument(s,n_decimals=2), allowed_missing_percentage=100)
            vecs.append(v)
        xb = np.array(vecs,dtype="float32")
        xb /= np.linalg.norm(xb,axis=1,keepdims=True)
        idxs = np.arange(len(xb))
        idx = hnswlib.Index(space="l2", dim=dim)
        idx.init_index(max_elements=len(xb), ef_construction=800, M=64)
        idx.add_items(xb, idxs)
        idx.set_ef(300)
        bin_path = os.path.join(out_dir, f"{prefix}_index.bin")
        idx.save_index(bin_path)
        return bin_path

    pos_idx = build_index(pos_pkl, pos_model_path, prefix="pos")
    neg_idx = build_index(neg_pkl, neg_model_path, prefix="neg")

    return pos_pkl, neg_pkl, pos_idx, neg_idx, out_dir
