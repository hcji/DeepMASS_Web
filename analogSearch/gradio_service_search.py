import zipfile
from zipfile import ZipFile
from matchms import Spectrum
import gradio as gr
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

from analogSearch.identify_unkown import id_spectrum_list
from analogSearch.plot_utils import show_default_mol_search, show_default_ref_spectrum_search
from analogSearch.spectrum_process import load_spectrum_file

def load_files(file_list, request: gr.Request):
    """
    从文件列表中读取质谱，并输出到窗口
    Args:
        file_list: 文件名列表，是保存在服务器临时路径中
        request: Gradio自带Request对象，可用于获取信息

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
            raise gr.Error("Please upload standard file")
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



# 预加载模型路径常量（若你在 utils 里已经定义，也可以直接 import）
MODEL_POS_PATH = 'model/Ms2Vec_allGNPSpositive.hdf5'
MODEL_NEG_PATH = 'model/Ms2Vec_allGNPSnegative.hdf5'
model_pos = Word2Vec.load(MODEL_POS_PATH)
model_neg = Word2Vec.load(MODEL_NEG_PATH)

def matchms_click_fn(
        threshold:float,
        res_state: pd.DataFrame, 
        refs_pos_state, refs_neg_state,
        hnsw_pos_state, hnsw_neg_state,
        request: gr.Request,
        progress=gr.Progress()):
    # --- 验证和提取 ---
    if res_state is None or "spectrum" not in res_state.columns:
        raise gr.Error("No spectra loaded. Please upload your MGF first.")
    spectra = res_state["spectrum"].tolist()

    # --- 判断用 Default 还是 Custom ---
    if hnsw_pos_state is not None:
        refs_pos, refs_neg = refs_pos_state, refs_neg_state
        hnsw_pos, hnsw_neg = hnsw_pos_state, hnsw_neg_state
    else:
        # --- 加载引用数据库 & 索引 ---
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

    # --- 执行匹配 ---
    results = id_spectrum_list(
        spectra,
        hnsw_pos, model_pos, refs_pos,
        hnsw_neg, model_neg, refs_neg,
        progress=progress
    )


    rows = []
    annotations = []
    identified_list = []
    for idx_unknow_spec, (ref_spec, dist, idx) in enumerate(results):
        if  dist >= threshold:
            meta = {k:v for k,v in ref_spec.metadata.items() if k not in ("mz","intensities")}
            smiles = meta.pop("smiles", None)
            idx = meta.pop("database_index", None)
            row = {"StructSimScore":dist, "database_index":idx, "smiles":smiles, **meta}
            annotation_df = pd.DataFrame([row])
            identified_spectrum = ref_spec


        else:
            row = {"StructSimScore": None, "database_index": None, "smiles": None}
            annotation_df = pd.DataFrame([row])
            identified_spectrum = None

        rows.append(row)
        annotations.append(annotation_df)
        identified_list.append(identified_spectrum)

    res_state["Identified Spectrum"] = identified_list
    res_state["annotation"] = annotations        # 直接成为一列

    result_df = pd.DataFrame(rows)
    gr.Info("Identified Successed")

    first_df = show_default_information(result_df)    # 保持 DataFrame 形状

    spectrum_loss_plot_fig, spectrum_plot_fig = show_default_ref_spectrum_search(res_state)
    ref_structure_fig = show_default_mol_search(res_state)
    return res_state, result_df, first_df, spectrum_loss_plot_fig, spectrum_plot_fig, ref_structure_fig

def show_information(results_df, evt: gr.SelectData):
    try:
        idx = evt.index[0]                  # 取到用户点的是第几行
        return show_default_information(results_df, idx)       # 切出对应行，保持 DataFrame 不变
    except Exception as e:
        print(f"show_information error: {e}")
        return pd.DataFrame(columns=["StructSimScore", "database_index", "smiles"])

def show_default_information(results_df, idx=0):
    try:
        return results_df.iloc[[idx]]
    except Exception as e:
        print(f"show_default_information error: {e}")
        return pd.DataFrame(columns=["StructSimScore", "database_index", "smiles"])


def save_search_csv(res_state, target_zip_file_name_state):
    gr.Info('Saving identification CSV"')
    file_list = []
    dir_path_tmp = "./tmp/result_csv_tmp/"
    dir_path = os.path.join(dir_path_tmp, uuid.uuid4().hex)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    base, ext = os.path.splitext(target_zip_file_name_state)  
    csv_filename = base + ".csv"
    csv_path = os.path.join(dir_path, csv_filename)
    all_rows = []
    # 判断是否有鉴定结果
    if res_state is None or "Identified Spectrum" not in res_state.columns:
        gr.Error("Missing Identified results, Run Identify First")
    for idx, s in enumerate(res_state["spectrum"]):
        # name = s.metadata["compound_name"]
        name = get_title_from_spectrum(spectrum=s, idx=idx)
        # 判断是否有鉴定结果，若无，则给空表
        if res_state["annotation"][idx] is None or res_state["annotation"][idx].empty:
           continue
        
        for _, row in res_state["annotation"][idx].iterrows():
            if row["StructSimScore"] is None:
                continue
            name_idx = f"{name}-{idx}"
            row_dict = {"compoundName-index": name_idx}

            for col, val in row.to_dict().items():
                row_dict[col] = val

            all_rows.append(row_dict)
    if not all_rows:
        empty_df = pd.DataFrame(columns=["distance", "database_index", "smiles"])
        empty_df.to_csv(csv_path)
        file_list.append(csv_path)
        return gr.File(file_list, visible=True), dir_path

    combined_df = pd.DataFrame(all_rows)
    combined_df.to_csv(csv_path)
    file_list.append(csv_path)
    return gr.File(file_list, visible=True), dir_path

def get_title_from_spectrum(spectrum: Spectrum, idx=None):
    if "compound_name" in spectrum.metadata.keys():
        return spectrum.metadata["compound_name"]
    else:
        if idx is not None:
            return f"Spectrum {idx}"
        else:
            return f"Spectrum {uuid.uuid4()}"

def clear_files(temp_dir=""):

    if isinstance(temp_dir, str) and temp_dir and os.path.exists(temp_dir):

        try:
            shutil.rmtree(temp_dir)  # 删除整个文件夹及其中所有文件
        except Exception as e:
            print(f"Error cleaning up directory: {e}")
    # 清空状态和文件
    return (
        None,  # 清空 res_state
        None,  # 清空 structure_state
        None,  # 清空 result_df
        None,  # 清空 navigator_obj
        None,  # 清空 information_obj
        None,  # 清空 spectrum_plot_fig
        None,  # 清空 spectrum_loss_plot_fig
        None,  # 清空 ref_structure_fig
        gr.update(visible=False, value=None),  # 清空并隐藏下载文件
    )

def clear_custom_database(out_dir):
    # 删除临时目录
    if out_dir and os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    # 清空所有 state 并禁用 Run
    return (
        None, None, None, None,    # 清 refs_pos_state..hnsw_neg_state
        gr.update(interactive=False),  # 禁用 Run
        gr.update(visible=True, value=None)  # 隐藏并清空上传框
    )
def init_custom_database(mgf_path):
    # try:
    gr.Info("Initializing custom database...")
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

    gr.Info("Custom database initialized successfully.")
    return refs_pos, refs_neg, hnsw_pos, hnsw_neg, out_dir, gr.update(interactive=True)


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
        raise gr.Error("No positive-mode spectra detected. Please check the ionmode annotation.")
    if not neg_specs:
        raise gr.Error("No negative-mode spectra detected. Please check the ionmode annotation.")
    
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
