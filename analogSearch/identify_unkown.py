import logging
import pickle

import numpy as np
from gensim.models import Word2Vec
from hnswlib import Index
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
from analogSearch.spectrum_process import load_spectrum_file


def find_most_similar_spectrum(s, p, model, references):

    query_vector = calc_vector(model, SpectrumDocument(s, n_decimals=2), allowed_missing_percentage=100)
    xq = np.array(query_vector).astype('float32')
    xq  /= np.linalg.norm(xq)

    idx, distance = p.knn_query(xq, 1)
    norm_distance = round(distance[0, 0] / 4.0, 2) # 归一化，保留两位小数

    score = 1.0 - norm_distance
    score = round(score, 2)
    ref_spec = np.array(references)[idx[0, 0]]
    ref_spec_index = ref_spec.metadata.get("database_index")
    return ref_spec, score, ref_spec_index

def id_spectrum_list(spectrum_list,
                     hnsw_pos, model_pos, refs_pos,
                     hnsw_neg, model_neg, refs_neg,
                     progress=None,):
    res = []
    for s in spectrum_list:
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
    return res


if __name__ == "__main__":
    MODEL_POS_PATH = 'model/Ms2Vec_allGNPSpositive.hdf5'
    MODEL_NEG_PATH = 'model/Ms2Vec_allGNPSnegative.hdf5'
    model_pos = Word2Vec.load(MODEL_POS_PATH)
    model_neg = Word2Vec.load(MODEL_NEG_PATH)
    with open("tmp/references_spectrums_positive.pickle", "rb") as f:
        refs_pos = pickle.load(f)
    with open("tmp/references_spectrums_negative.pickle", "rb") as f:
        refs_neg = pickle.load(f)

    hnsw_pos = Index(space="l2", dim=300)
    hnsw_pos.load_index("tmp/references_index_positive_spec2vec.bin")
    hnsw_pos.set_ef(300)
    hnsw_neg = Index(space="l2", dim=300)
    hnsw_neg.load_index("tmp/references_index_negative_spec2vec.bin")
    hnsw_neg.set_ef(300)

    spectrums = load_spectrum_file('analogSearch_data/2.msp')
    res = id_spectrum_list(spectrums, hnsw_pos, model_pos, refs_pos, hnsw_neg, model_neg, refs_neg)
    for  s in res:
        ref_spec, dist, idx = s  # 解包元组，提取谱图对象、距离和索引
        print(f"Reference Spectrum: {ref_spec.metadata}")
        print(f"score: {dist}")
        print(f"Index: {idx}")
        print("-" * 50)  # 分隔符，用于区分不同的结果