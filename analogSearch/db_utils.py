
# -*- coding: utf-8 -*-
from analogSearch.spectrum_process import load_spectrum_file
import numpy as np
import pickle
import hnswlib
import gensim
from tqdm import tqdm

from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector


def process_spectrums(file_path, pos_model_path, neg_model_path, output_spectrums_pickle, output_index_file):
    # 加载光谱文件
    spectrums = load_spectrum_file(file_path)
    
    positive_spectrums = []
    negative_spectrums = []
    
    # 根据电离模式分类光谱
    for idx, s in enumerate(spectrums):
        s.set("database_index", idx)
        ionmode = s.metadata.get("ionmode", "positive")   # 缺失时当作正谱
        if ionmode == "negative":
            negative_spectrums.append(s)
        else:
            positive_spectrums.append(s)

    # 保存分类后的光谱
    with open(output_spectrums_pickle[0], 'wb') as file:
        pickle.dump(positive_spectrums, file)

    print(f"→ Dumped {len(positive_spectrums)} positive spectrums to {output_spectrums_pickle[0]}")

    with open(output_spectrums_pickle[1], 'wb') as file:
        pickle.dump(negative_spectrums, file)
    
    print(f"→ Dumped {len(negative_spectrums)} negative spectrums to {output_spectrums_pickle[1]}")
    # 加载 Word2Vec 模型
    pos_model = gensim.models.Word2Vec.load(pos_model_path)
    calc_ms2vec_vector = lambda x: calc_vector(pos_model, SpectrumDocument(x, n_decimals=2), allowed_missing_percentage=100)

    with open(output_spectrums_pickle[0], 'rb') as file:
        reference = pickle.load(file)

    reference_vector = []
    for s in tqdm(reference):
        reference_vector.append(calc_ms2vec_vector(s))

    xb = np.array(reference_vector).astype('float32')
    xb_len =  np.linalg.norm(xb, axis=1, keepdims=True)
    xb = xb/xb_len
    dim = 300
    num_elements = len(xb)
    ids = np.arange(num_elements)

    p = hnswlib.Index(space = 'l2', dim = dim)
    p.init_index(max_elements = num_elements, ef_construction = 800, M = 64)
    p.add_items(xb, ids)
    p.set_ef(300)
    p.save_index(output_index_file[0])

    # 处理负电离模式光谱
    # file = 'model/Ms2Vec_allGNPSnegative.hdf5'
    neg_model = gensim.models.Word2Vec.load(neg_model_path)
    calc_ms2vec_vector = lambda x: calc_vector(neg_model, SpectrumDocument(x, n_decimals=2), allowed_missing_percentage=100)

    with open(output_spectrums_pickle[1], 'rb') as file:
        reference = pickle.load(file)

    reference_vector = []
    for s in tqdm(reference):
        reference_vector.append(calc_ms2vec_vector(s))

    xb = np.array(reference_vector).astype('float32')
    xb_len =  np.linalg.norm(xb, axis=1, keepdims=True)
    xb = xb/xb_len
    dim = 300
    num_elements = len(xb)
    ids = np.arange(num_elements)

    p = hnswlib.Index(space = 'l2', dim = dim)
    p.init_index(max_elements = num_elements, ef_construction = 800, M = 64)
    p.add_items(xb, ids)
    p.set_ef(300)
    p.save_index(output_index_file[1])


if __name__ == "__main__":
    # 设置文件路径和输出路径
    mgf_file_path = 'analogSearch_data/all_data.mgf'
    positive_model_path = 'model/Ms2Vec_allGNPSpositive.hdf5'
    negative_model_path = 'model/Ms2Vec_allGNPSnegative.hdf5'
    output_spectrums_pickle = ['analogSearch_data/references_spectrums_positive.pickle', 'analogSearch_data/references_spectrums_negative.pickle']
    output_index_files = ['analogSearch_data/references_index_positive_spec2vec.bin', 'analogSearch_data/references_index_negative_spec2vec.bin']
    
    # 处理光谱数据并保存
    process_spectrums(mgf_file_path, positive_model_path, negative_model_path, output_spectrums_pickle, output_index_files)
