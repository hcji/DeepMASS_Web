import logging
import pickle

import numpy as np
from gensim.models import Word2Vec
from hnswlib import Index
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
from analogSearch.spectrum_process import load_spectrum_file


def find_most_similar_spectrum(s, p, model, references, k_candidates: int = 50, topk: int = 10):
    # 计算查询向量并归一化
    query_vector = calc_vector(model, SpectrumDocument(s, n_decimals=2), allowed_missing_percentage=100)
    xq = np.array(query_vector, dtype='float32')
    norm = np.linalg.norm(xq)
    if norm == 0:
        return []
    xq /= norm

    # 一次取多个候选
    idxs, dists = p.knn_query(xq, k_candidates)
    idxs = idxs[0]
    dists = dists[0]

    results = []
    seen_smiles = set()

    for rank in range(len(idxs)):
        ref = np.array(references)[idxs[rank]]
        dist = dists[rank]

        # 算分数
        norm_distance = round(dist / 4.0, 2)
        score = round(1.0 - norm_distance, 2)

        # 提取 smiles 和 index
        smiles = None
        ref_spec_index = None
        if hasattr(ref, "metadata") and isinstance(ref.metadata, dict):
            smiles = ref.metadata.get("smiles")
            ref_spec_index = ref.metadata.get("database_index")

        # 只保留 smiles 不重复的
        if smiles and smiles not in seen_smiles:
            seen_smiles.add(smiles)
            results.append((ref, score, ref_spec_index))
            if len(results) >= topk:
                break
    # print(results)
    return results



def id_spectrum_list(spectrum_list,
                     hnsw_pos, model_pos, refs_pos,
                     hnsw_neg, model_neg, refs_neg,
                     on_progress=None
                     ):
    res = []
    for i, s in enumerate(spectrum_list, start=1):
        logging.info(f"")
        sn = None
        if "ionmode" in s.metadata.keys():
            if s.metadata["ionmode"] == "negative":
                sn = find_most_similar_spectrum(s, hnsw_neg, model_neg, refs_neg)
            else:
                sn = find_most_similar_spectrum(s, hnsw_pos, model_pos, refs_pos)
        else:
            sn = find_most_similar_spectrum(s, hnsw_pos, model_pos, refs_pos)
        res.append(sn)
        # ★ 每处理完一个查询，通知进度
        if on_progress:
            try:
                on_progress(i)
            except Exception:
                print("Error in progress callback")
    return res


if __name__ == "__main__":
    MODEL_POS_PATH = 'model/Ms2Vec_allGNPSpositive.hdf5'
    MODEL_NEG_PATH = 'model/Ms2Vec_allGNPSnegative.hdf5'
    model_pos = Word2Vec.load(MODEL_POS_PATH)
    model_neg = Word2Vec.load(MODEL_NEG_PATH)
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

    spectrums = load_spectrum_file('analogSearch_data/2.msp')
    # res = id_spectrum_list(spectrums, hnsw_pos, model_pos, refs_pos, hnsw_neg, model_neg, refs_neg)
    # for  s in res:
    #     ref_spec, dist, idx = s  # 解包元组，提取谱图对象、距离和索引
    #     print(f"Reference Spectrum: {ref_spec.metadata}")
    #     print(f"score: {dist}")
    #     print(f"Index: {idx}")
    #     print("-" * 50)  # 分隔符，用于区分不同的结果

    res = id_spectrum_list(spectrums, hnsw_pos, model_pos, refs_pos, hnsw_neg, model_neg, refs_neg)

    for qi, topk_list in enumerate(res, 0):           # 每个查询谱图的结果列表
        print(f"=== Query Spectrum index #{qi} ===")
        if not topk_list:
            print("No neighbors found.")
            print("-" * 50)
            continue

        for rank, (ref_spec, score, idx) in enumerate(topk_list, 0):
            print(f"Top{rank} | score: {score} | Index: {idx}")
            print(f"Metadata: {ref_spec.metadata}")
        print("-" * 50)
