from fastapi import Query, UploadFile, File, APIRouter, Form, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
import uuid
import time
import asyncio
import pandas as pd
import numpy as np
import pickle
import glob
import gensim.models
import hnswlib
from tqdm import tqdm
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
from backend.anal_sear.service_search import load_files, matchms_click_fn, get_title_from_spectrum
from analogSearch.spectrum_process import load_spectrum_file
from typing import Optional, Dict, Any
from backend.anal_sear.plot_utils import build_plots_for_pair
from backend.service.session_store import Store  # ★ 使用统一的 Store

import json
from backend.service.auth_deps import require_user  # ★ 新增
from pathlib import Path
# 用当前文件位置推导项目根目录（DeepMASS_Web）
BASE_DIR = Path(__file__).resolve().parents[2]

# =========================
# 统一 Store（与 comp_ident 共用 Redis / Cookie）
# =========================
store = Store(
    namespace="analsear",
    ttl_seconds=60 * 60 * 24 * 3,
    redis_host="127.0.0.1",
    redis_port=6379,
    redis_db=0,
    base_dir=str(BASE_DIR / "temp" / "session_store"),
    cookie_name="session_id",
)

# =========================
# 路由 & 常量
# =========================
router = APIRouter(
    prefix="/anal_search",
    tags=["anal_search"],
    dependencies=[Depends(require_user)]  # ★ 统一保护所有 anal_search 接口
)

MODEL_POS_PATH = str(BASE_DIR / "model" / "Ms2Vec_allGNPSpositive.hdf5")
MODEL_NEG_PATH = str(BASE_DIR / "model" / "Ms2Vec_allGNPSnegative.hdf5")

# 全部 temp 路径基于 BASE_DIR（不再用 ./temp）
TMP_DIR      = str(BASE_DIR / "temp" / "result_csv_temp") + "/"
STATE_DIR    = str(BASE_DIR / "temp" / "state_files") + "/"
DB_TMP_DIR   = str(BASE_DIR / "temp" / "database_temp") + "/"
FILE_TMP_DIR = str(BASE_DIR / "temp" / "temp_files") + "/"
TEMP_DIR     = str(BASE_DIR / "temp")
SESSION_TIMEOUT = 86400 * 3  # 3天

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(DB_TMP_DIR, exist_ok=True)
os.makedirs(FILE_TMP_DIR, exist_ok=True)

lock = asyncio.Lock()

# =========================
# 工具
# =========================
def dataframe_to_clean_records(df: pd.DataFrame):
    def convert_value(x):
        if x is None or pd.isna(x):
            return None
        if isinstance(x, (str, dict)):
            return x
        return str(x)
    return (
        df.replace([np.inf, -np.inf], np.nan)
          .astype(object)
          .where(pd.notnull(df), None)
          .applymap(convert_value)
          .to_dict(orient="records")
    )

def make_scope_key(session_id: str, tab_id: Optional[str]) -> str:
    return f"{session_id}:{tab_id or 'default'}"

def safe_name(s: str) -> str:
    return s.replace(":", "_").replace("/", "_").replace("\\", "_")

def baseline_session_state():
    return {
        "references_positive_path": None,
        "references_negative_path": None,
        "hnsw_positive_path": None,
        "hnsw_negative_path": None,
        "custom_database_path": None,
        "status": "idle",
        "status_message": "",
        "filename": None,
        "progress": {"status": "idle", "total": 0, "done": 0},
    }

def set_progress(scope_key: str, total: int, done: int, status: str, message: Optional[str] = None):
    # 直接用 Store 的进度（带 message）
    store.set_progress(scope_key, total=total, done=done, status=status, message=message)

def get_progress(scope_key: str) -> dict:
    return store.get_progress(scope_key)

# =========================
# 依赖
# =========================
async def get_session_scope(request: Request, tab_id: Optional[str] = Query(None)) -> Dict[str, Any]:
    """
    统一用 Store 的会话校验；并为 (sid:tab) 建立/更新轻量状态。
    """
    raw_sid = store.require_session(request)  # 若无/过期会 400
    scope_key = make_scope_key(raw_sid, tab_id)
    # 初始化/续期作用域状态在 Redis（不换 Cookie）
    st = store.read_state(scope_key) or {}
    if not st:
        store.update_state(scope_key, **baseline_session_state())
        print(f"[anal_search] bootstrap scope baseline for scope={scope_key}")
    else:
        store.update_state(scope_key)  # 刷新 last_accessed
    return {"sid": raw_sid, "key": scope_key, "tab_id": tab_id or "default"}

# =========================
# 后台任务
# =========================
async def process_spectrums_background(file_path, pos_model_path, neg_model_path, out_dir, scope_key):
    try:
        print(f"[BG] Start building custom DB for scope={scope_key}, file={file_path}")
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix not in ['.mgf', '.msp', '.mat']:
            msg = f"Invalid file format: {suffix}"
            store.update_state(scope_key, status="error", status_message=msg, filename=os.path.basename(file_path))
            return

        specs = load_spectrum_file(file_path)
        if not specs:
            msg = "No valid spectra found in file"
            store.update_state(scope_key, status="error", status_message=msg, filename=os.path.basename(file_path))
            return

        # 校验 + 纠错（缺 ionmode → 默认 positive）
        defaulted_ionmode = 0
        for i, s in enumerate(specs, start=1):
            if not (hasattr(s, "peaks") and s.peaks.mz.size and s.peaks.intensities.size):
                msg = f"Spectrum #{i} missing peaks or intensities"
                store.update_state(scope_key, status="error", status_message=msg, filename=os.path.basename(file_path))
                return
            if not s.metadata.get("ionmode"):
                s.set("ionmode", "positive")
                defaulted_ionmode += 1
                
        positive_specs, negative_specs = [], []
        for index, s in enumerate(specs):
            s.set("database_index", index)
            mode = (s.metadata.get("ionmode") or "positive").lower()
            (negative_specs if mode == "negative" else positive_specs).append(s)

        if not positive_specs and not negative_specs:
            msg = "No spectrum data detected in file"
            store.update_state(scope_key, status="error", status_message=msg, filename=os.path.basename(file_path))
            return

        os.makedirs(out_dir, exist_ok=True)
        positive_pkl = os.path.join(out_dir, "references_spectrums_positive.pickle")
        negative_pkl = os.path.join(out_dir, "references_spectrums_negative.pickle")
        with open(positive_pkl, "wb") as f:
            pickle.dump(positive_specs, f)
        with open(negative_pkl, "wb") as f:
            pickle.dump(negative_specs, f)

        def build_index(pkl_path, model_path, dim=300, prefix="positive"):
            if not os.path.exists(model_path):
                raise RuntimeError(f"Model file not found: {model_path}")
            with open(pkl_path, "rb") as f:
                refs = pickle.load(f)
            if not refs:
                raise RuntimeError(f"No {prefix} spectra available for indexing")
            model = gensim.models.Word2Vec.load(model_path)
            vectors = []
            for s in tqdm(refs, desc=f"Vectorizing {prefix}"):
                try:
                    v = calc_vector(model, SpectrumDocument(s, n_decimals=2), allowed_missing_percentage=100)
                    vectors.append(v)
                except Exception as e:
                    print(f"[BG] vectorize error: {e}")
                    continue
            if not vectors:
                raise RuntimeError(f"No valid vectors generated for {prefix} spectra")
            xb = np.array(vectors, dtype="float32")
            xb /= np.linalg.norm(xb, axis=1, keepdims=True)
            idxs = np.arange(len(xb))
            idx = hnswlib.Index(space="l2", dim=dim)
            idx.init_index(max_elements=len(xb), ef_construction=800, M=64)
            idx.add_items(xb, idxs)
            idx.set_ef(300)
            bin_path = os.path.join(out_dir, f"references_index_{prefix}_spec2vec.bin")
            idx.save_index(bin_path)
            return bin_path

        positive_idx = build_index(positive_pkl, pos_model_path, prefix="positive") if positive_specs else None
        negative_idx = build_index(negative_pkl, neg_model_path, prefix="negative") if negative_specs else None

        # 成功 + 告警信息
        warn_part = ""
        if defaulted_ionmode > 0:
            warn_part = f" (Note: {defaulted_ionmode} spectrum(s) with missing ionmode were set to 'positive')"

        store.update_state(
            scope_key,
            references_positive_path=positive_pkl if os.path.exists(positive_pkl) else None,
            references_negative_path=negative_pkl if os.path.exists(negative_pkl) else None,
            hnsw_positive_path=os.path.join(out_dir, "references_index_positive_spec2vec.bin") if positive_idx else None,
            hnsw_negative_path=os.path.join(out_dir, "references_index_negative_spec2vec.bin") if negative_idx else None,
            custom_database_path=out_dir,
            status="success",
            status_message=f"Custom database initialized successfully{warn_part}",
            filename=os.path.basename(file_path),
        )
        print(f"[BG] Custom DB built OK for scope={scope_key}{warn_part}")

    except Exception as e:
        msg = f"Custom database processing failed: {e}"
        print(f"[BG] Error for scope {scope_key}: {msg}")
        # 把异常也写给前端
        store.update_state(scope_key, status="error", status_message=msg, filename=os.path.basename(file_path))
        # 不要再 raise，避免“response already started”之类的噪音
        return

# =========================
# 路由
# =========================

@router.get("/start-session")
async def start_session(request: Request, response: Response):
    """
    与 comp_ident 一致：如果已有 Cookie 且状态/数据存在，则复用并续期；
    否则创建新会话。两边共享同一个 cookie_name。
    """
    sid = store.get_or_create_session(request, response)
    # 不强制建立 tab 作用域；前端首次带 tab_id 的请求会在依赖里补建
    return {"status": "success", "session_id": sid}

@router.post("/upload-custom-db")
async def upload_custom_db(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session_scope: Dict[str, Any] = Depends(get_session_scope)
):
    scope_key = session_scope["key"]
    print(f"[upload-custom-db] scope={scope_key}, filename={file.filename}")
    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix not in ['.mgf', '.msp', '.mat']:
            return JSONResponse(status_code=400, content={"status": "error", "message": f"Unsupported file format: {suffix}"})
        # 注意：UploadFile 不一定有 size 属性，这里跳过大文件判断或在前端限制

        uuid_prefix = str(uuid.uuid4())
        unique_filename = f"{safe_name(scope_key)}_{uuid_prefix}_{file.filename}"
        db_dir = os.path.join(DB_TMP_DIR, safe_name(scope_key))
        os.makedirs(db_dir, exist_ok=True)
        temp_path = os.path.join(db_dir, unique_filename)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        store.update_state(
            scope_key,
            status="processing",
            status_message=f"Processing custom database: {file.filename}",
            filename=file.filename,
        )

        out_dir = os.path.join(DB_TMP_DIR, safe_name(scope_key))
        background_tasks.add_task(process_spectrums_background, temp_path, MODEL_POS_PATH, MODEL_NEG_PATH, out_dir, scope_key)

        return {"status": "processing", "message": f"Custom database {file.filename} is being processed"}
    except Exception as e:
        print(f"[upload-custom-db] error: {e}")
        store.update_state(scope_key, status="error", status_message=f"Custom database {file.filename} upload failed: {str(e)}")
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Custom database {file.filename} upload failed: {str(e)}"})

@router.get("/custom-db-status")
async def get_custom_db_status(session_scope: Dict[str, Any] = Depends(get_session_scope)):
    scope_key = session_scope["key"]
    st = store.read_state(scope_key)
    return {
        "status": st.get("status", "idle"),
        "message": st.get("status_message", ""),
        "filename": st.get("filename")
    }

@router.get("/check-file")
async def check_file(
    uuid_filename: str = Query(...),
    session_scope: Dict[str, Any] = Depends(get_session_scope),
):
    scope_key = session_scope["key"]
    try:
        file_path = os.path.join(FILE_TMP_DIR, safe_name(scope_key), uuid_filename)
        if os.path.exists(file_path):
            return {"status": "success", "message": f"File {uuid_filename} exists"}
        else:
            return {"status": "error", "message": f"File {uuid_filename} does not exist"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to check file: {str(e)}"}

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_scope: Dict[str, Any] = Depends(get_session_scope)
):
    scope_key = session_scope["key"]
    print(f"[upload] scope={scope_key}, filename={file.filename}")
    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix not in ['.mgf', '.msp', '.mat']:
            return {"status": "error", "message": f"Unsupported file format: {suffix}"}

        uuid_prefix = str(uuid.uuid4())
        unique_filename = f"{safe_name(scope_key)}_{uuid_prefix}_{file.filename}"
        temp_dir = os.path.join(FILE_TMP_DIR, safe_name(scope_key))
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, unique_filename)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if not os.path.exists(temp_path):
            return {"status": "error", "message": f"Failed to save file: {file.filename}"}

        spectrums_df, name_list, _ = load_files([temp_path])
        if spectrums_df is None or name_list is None or len(name_list) == 0:
            return {"status": "error", "message": "File parsing failed, possibly empty or incorrect format"}

        # ★ 关键：记录原始文件名，供后续 /save-results 生成 <原文件名>.zip
        store.update_state(scope_key, last_uploaded_original=file.filename)

        return {
            "status": "success",
            "names": name_list.to_dict(),
            "uuid_filename": unique_filename,
            "message": f"File {file.filename} uploaded successfully"
        }
    except Exception as e:
        print(f"[upload] error: {e}")
        return {"status": "error", "message": f"File {file.filename} upload failed: {str(e)}"}
    finally:
        if temp_path and os.path.exists(temp_path) and not temp_path.startswith(os.path.join(FILE_TMP_DIR, safe_name(scope_key))):
            os.remove(temp_path)

@router.get("/upload-test-file")
async def upload_test_file(session_scope: Dict[str, Any] = Depends(get_session_scope)):
    scope_key = session_scope["key"]
    try:
        test_file_path = "analogSearch_data/test.mgf"
        if not os.path.exists(test_file_path):
            return {"status": "error", "message": f"Test file does not exist: {test_file_path}"}
        spectrums_df, name_list, _ = load_files([test_file_path])

        # ★ 关键：记录测试原始名
        store.update_state(scope_key, last_uploaded_original=os.path.basename(test_file_path))

        return {
            "status": "success",
            "names": name_list.to_dict(),
            "message": "Test file processed successfully"
        }
    except Exception as e:
        return {"status": "error", "message": f"Test file processing failed: {str(e)}"}

@router.post("/run")
async def run_analysis(
    threshold: float = Form(...),
    source: str = Form(...),
    db_option: str = Form(...),
    file: UploadFile = File(None),
    uuid_filename: str = Form(None),
    session_scope: Dict[str, Any] = Depends(get_session_scope)
):
    scope_key = session_scope["key"]
    sid = session_scope["sid"]
    t0 = time.time()
    print(f"[run] scope={scope_key} source={source} uuid={uuid_filename} file={(file.filename if file else None)} thr={threshold} db={db_option}")
    temp_path = None
    safe_scope = safe_name(scope_key)
    state_uuid = f"{safe_scope}_{uuid.uuid4()}"
    try:
        set_progress(scope_key, total=0, done=0, status="running", message="Loading references...")

        # 加载参考库
        if db_option == "Custom":
            st = store.read_state(scope_key)
            if st.get("status") != "success":
                return {"status": "error", "message": f"Custom database {st.get('filename') or ''} not ready: {st.get('status_message')}"}
            refs_pos_path = st.get("references_positive_path")
            refs_neg_path = st.get("references_negative_path")
            hnsw_pos_path = st.get("hnsw_positive_path")
            hnsw_neg_path = st.get("hnsw_negative_path")
            if not (refs_pos_path or refs_neg_path):
                return {"status": "error", "message": "Custom database paths missing"}

            refs_pos = pickle.load(open(refs_pos_path, "rb")) if refs_pos_path and os.path.exists(refs_pos_path) else []
            refs_neg = pickle.load(open(refs_neg_path, "rb")) if refs_neg_path and os.path.exists(refs_neg_path) else []

            hnsw_pos = None
            if hnsw_pos_path and os.path.exists(hnsw_pos_path):
                hnsw_pos = hnswlib.Index(space="l2", dim=300)
                hnsw_pos.load_index(hnsw_pos_path)
                hnsw_pos.set_ef(300)

            hnsw_neg = None
            if hnsw_neg_path and os.path.exists(hnsw_neg_path):
                hnsw_neg = hnswlib.Index(space="l2", dim=300)
                hnsw_neg.load_index(hnsw_neg_path)
                hnsw_neg.set_ef(300)
        else:
            with open("analogSearch_data/references_spectrums_positive.pickle", "rb") as f:
                refs_pos = pickle.load(f)
            with open("analogSearch_data/references_spectrums_negative.pickle", "rb") as f:
                refs_neg = pickle.load(f)
            hnsw_pos = hnswlib.Index(space="l2", dim=300)
            hnsw_pos.load_index("analogSearch_data/references_index_positive_spec2vec.bin")
            hnsw_pos.set_ef(300)
            hnsw_neg = hnswlib.Index(space="l2", dim=300)
            hnsw_neg.load_index("analogSearch_data/references_index_negative_spec2vec.bin")
            hnsw_neg.set_ef(300)

        # 处理输入
        if source == "upload":
            if not file and not uuid_filename:
                return {"status": "error", "message": "Upload mode requires a file or UUID filename"}
            if uuid_filename:
                temp_path = os.path.join(FILE_TMP_DIR, safe_scope, uuid_filename)
                if not os.path.exists(temp_path):
                    return {"status": "error", "message": f"Specified UUID file does not exist: {uuid_filename}"}
                spectrums_df, name_list, _ = load_files([temp_path])
            elif file:
                suffix = os.path.splitext(file.filename)[1].lower()
                if suffix not in ['.mgf', '.msp', '.mat']:
                    return {"status": "error", "message": f"Unsupported file format: {suffix}"}
                temp_dir = os.path.join(FILE_TMP_DIR, safe_scope)
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"{safe_scope}_{uuid.uuid4()}_{file.filename}")
                with open(temp_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                spectrums_df, name_list, _ = load_files([temp_path])
            else:
                return {"status": "error", "message": "Upload mode requires a file"}
        elif source == "test":
            test_file_path = "analogSearch_data/test.mgf"
            if not os.path.exists(test_file_path):
                return {"status": "error", "message": f"Test file does not exist: {test_file_path}"}
            spectrums_df, name_list, _ = load_files([test_file_path])
        else:
            return {"status": "error", "message": f"Invalid source parameter: {source}"}

        if spectrums_df is None or name_list is None or len(name_list) == 0:
            return {"status": "error", "message": "File parsing failed, data is empty or format is incorrect"}

        # 进度
        total_queries = len(spectrums_df)
        set_progress(scope_key, total=total_queries, done=0, status="running", message="Searching candidates...")

        # 调用检索
        (result_state, result_df, all_topk) = matchms_click_fn(
            threshold=threshold,
            res_state=spectrums_df,
            refs_pos_state=refs_pos,
            refs_neg_state=refs_neg,
            hnsw_pos_state=hnsw_pos,
            hnsw_neg_state=hnsw_neg,
            progress_cb=lambda i: set_progress(scope_key, total=total_queries, done=i, status="running", message=f"Processed {i}/{total_queries}")
        )

        set_progress(scope_key, total=total_queries, done=total_queries, status="finished", message="Completed")

        # === 统一 query_index 为 0 开始 ===
        if result_df is not None and not result_df.empty:
            if "query_index" in result_df.columns:
                try:
                    # 如果本来是 1-based（全是整数且最小值为 1），则减 1
                    if pd.to_numeric(result_df["query_index"], errors="coerce").min() == 1:
                        result_df["query_index"] = pd.to_numeric(result_df["query_index"], errors="coerce") - 1
                except Exception:
                    pass

        # 写入 bundle（供懒加载）
        state_path = os.path.join(STATE_DIR, f"{state_uuid}.pkl")
        bundle = {
            "res_state": result_state,                         # DataFrame（含 Identified Spectrum / annotation）
            "spectra": spectrums_df["spectrum"].tolist(),     # 原始查询谱图库（list）
            "all_topk": all_topk,                              # 每个查询的 TopK 列表
        }
        with open(state_path, "wb") as f:
            pickle.dump(bundle, f)

        store.update_state(scope_key)

        if result_df is None or result_df.empty:
            return {
                "status": "success",
                "names": name_list.to_dict(),
                "result": [],
                "state_uuid": state_uuid,
                "target_zip_file_name": f"{safe_scope}_results.zip",
                "message": f"No matching results found, possibly due to high threshold ({threshold}) or uninitialized database"
            }

        clean_result = dataframe_to_clean_records(result_df)
        return {
            "status": "success",
            "names": name_list.to_dict(),
            "result": clean_result,
            "state_uuid": state_uuid,
            "target_zip_file_name": f"{safe_scope}_results.zip",
            "message": "Analysis completed"
        }
    except Exception as e:
        # 进度错误态（总量/完成置 0）
        set_progress(scope_key, total=0, done=0, status="error", message=f"Run failed: {str(e)}")
        print(f"[run] error: {str(e)}")
        return {"status": "error", "message": f"Analysis failed: {str(e)}"}
    finally:
        if temp_path and os.path.exists(temp_path) and not temp_path.startswith(os.path.join(FILE_TMP_DIR, safe_scope)):
            os.remove(temp_path)
        await clean_old_states()
        t1 = time.time()
        print(f"[run] scope={scope_key} done in {t1 - t0:.2f}s")

@router.post("/save-results")
async def save_results(
    state_uuid: Optional[str] = Form(None),
    target_zip_file_name: Optional[str] = Form(None),  # 保留参数但不再用于命名
    threshold: Optional[float] = Form(None),
    session_scope: Dict[str, Any] = Depends(get_session_scope)
):
    scope_key = session_scope["key"]
    safe_scope = safe_name(scope_key)
    try:
        if not state_uuid:
            raise HTTPException(status_code=400, detail="Missing state_uuid")
        state_path = os.path.join(STATE_DIR, f"{state_uuid}.pkl")
        if not os.path.exists(state_path):
            raise HTTPException(status_code=404, detail=f"State file not found: {state_uuid}")

        with open(state_path, "rb") as f:
            obj = pickle.load(f)
        result_state = obj["res_state"] if isinstance(obj, dict) and "res_state" in obj else obj
        if result_state is None or "spectrum" not in result_state.columns or "annotation" not in result_state.columns:
            raise HTTPException(status_code=400, detail="Invalid result_state")

        # 生成明细文件
        out_dir = os.path.join(TMP_DIR, safe_scope, uuid.uuid4().hex)
        os.makedirs(out_dir, exist_ok=True)
        detail_basename = "analysis_results.xlsx"
        detail_path = os.path.join(out_dir, detail_basename)

        # 汇总所有 topK
        all_rows = []
        for idx, s in enumerate(result_state["spectrum"]):
            name = get_title_from_spectrum(spectrum=s, idx=idx)
            ann = result_state["annotation"][idx]
            if ann is None or ann.empty:
                continue
            for _, row in ann.iterrows():
                row_dict = {"compoundName-index": f"{name}-{idx}"}
                for col, val in row.to_dict().items():
                    row_dict[col] = val
                all_rows.append(row_dict)

        if not all_rows:
            df = pd.DataFrame(columns=["compoundName-index", "StructSimScore", "database_index", "smiles"])
        else:
            df = pd.DataFrame(all_rows)
            if threshold is not None:
                df = df[pd.to_numeric(df.get("StructSimScore", np.nan), errors="coerce").fillna(-np.inf) >= float(threshold)]
            if df.empty:
                keep_cols = ["compoundName-index", "StructSimScore", "database_index", "smiles"]
                for col in keep_cols:
                    if col not in df.columns:
                        df[col] = pd.Series(dtype=object)
                df = df[keep_cols]

        try:
            df.to_excel(detail_path, index=False)
        except Exception:
            # 回退 CSV
            detail_basename = "analysis_results.csv"
            detail_path = os.path.join(out_dir, detail_basename)
            df.to_csv(detail_path, index=False)

        # ★ 关键：zip 名 = 原文件名（含原后缀） + ".zip"
        st = store.read_state(scope_key) or {}
        original_name = st.get("last_uploaded_original") or "results"
        zip_name = f"{original_name}.zip"
        zip_path = os.path.join(out_dir, zip_name)

        import zipfile
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.write(detail_path, arcname=os.path.basename(detail_path))

        return {"status": "success", "message": "Results saved successfully", "file_path": zip_path}
    except Exception as e:
        print(f"[save-results] error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save results: {str(e)}")


@router.get("/download-result-file")
async def download_result_file(
    file_path: str = Query(...),
    session_scope: Dict[str, Any] = Depends(get_session_scope)
):
    scope_key = session_scope["key"]
    safe_scope = safe_name(scope_key)
    try:
        if not file_path.startswith(os.path.join(TMP_DIR, safe_scope)):
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type="application/octet-stream")
    except Exception as e:
        print(f"[download-result-file] error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

@router.post("/clear")
async def clear_state(session_scope: Dict[str, Any] = Depends(get_session_scope)):
    """清理当前 tab 作用域下的落盘 + 重置该作用域状态；不影响其他标签页"""
    scope_key = session_scope["key"]
    safe_scope = safe_name(scope_key)
    try:
        st = store.read_state(scope_key)
        custom_dir = st.get("custom_database_path")
        if custom_dir and os.path.exists(custom_dir):
            shutil.rmtree(custom_dir, ignore_errors=True)

        for base_dir in [TMP_DIR, DB_TMP_DIR, FILE_TMP_DIR]:
            d = os.path.join(base_dir, safe_scope)
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

        state_pattern = os.path.join(STATE_DIR, f"{safe_scope}_*")
        for p in glob.glob(state_pattern):
            if os.path.isfile(p):
                os.remove(p)

        store.update_state(scope_key, **baseline_session_state())

        return {"status": "success", "message": f"Scope {scope_key} cleared (tab only)."}
    except Exception as e:
        print(f"[clear] error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear state: {str(e)}")

async def clean_old_states(max_age_seconds=SESSION_TIMEOUT):
    current_time = time.time()
    # 清理过期 scope（仅 analsear 命名空间）
    try:
        pattern = f"{store.namespace}:session:*:state"
        for key in store.r.scan_iter(pattern):
            try:
                raw = store.r.get(key)
                st = json.loads(raw.decode("utf-8")) if raw else {}
                last = st.get("last_accessed", 0)
                if current_time - last > max_age_seconds:
                    store.r.delete(key)  # 只删 analsear 的状态键
            except Exception:
                continue
    except Exception as e:
        print(f"[CLEAN] redis scan failed: {e}")

    # 清 temp
    if os.path.exists(TEMP_DIR):
        for fname in os.listdir(TEMP_DIR):
            fpath = os.path.join(TEMP_DIR, fname)
            try:
                mtime = os.path.getmtime(fpath)
                if current_time - mtime > max_age_seconds:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                        print(f"[CLEAN] file: {fpath}")
                    elif os.path.isdir(fpath):
                        shutil.rmtree(fpath)
                        print(f"[CLEAN] dir: {fpath}")
            except Exception as e:
                print(f"[CLEAN] failed on {fpath}: {e}")

@router.get("/download-test-file")
async def download_test_file(session_scope: Dict[str, Any] = Depends(get_session_scope)):
    test_file_path = "analogSearch_data/test.mgf"
    if not os.path.exists(test_file_path):
        raise HTTPException(status_code=404, detail="Test file does not exist: analogSearch_data/test.mgf")
    return FileResponse(path=test_file_path, filename="test.mgf", media_type="application/octet-stream")

@router.get("/get-candidate-plots")
async def get_candidate_plots(
    state_uuid: str = Query(...),
    query_index: int = Query(..., ge=0),   # ★ 0-based
    rank: int = Query(..., ge=1),          # rank 保持 1-based（Top 1, Top 2...）
    session_scope: Dict[str, Any] = Depends(get_session_scope),
):
    """
    懒加载：返回某个查询（0-based）的第 rank 个候选
    """
    try:
        state_path = os.path.join(STATE_DIR, f"{state_uuid}.pkl")
        if not os.path.exists(state_path):
            raise HTTPException(status_code=404, detail="state not found")

        with open(state_path, "rb") as f:
            bundle = pickle.load(f)

        if not isinstance(bundle, dict) or "spectra" not in bundle or "all_topk" not in bundle:
            raise HTTPException(status_code=400, detail="state not in bundle format")

        spectra = bundle["spectra"]
        all_topk = bundle["all_topk"]

        if query_index < 0 or query_index >= len(spectra):
            raise HTTPException(status_code=400, detail="invalid query_index")
        topk_list = all_topk[query_index]
        if rank < 1 or rank > len(topk_list):
            raise HTTPException(status_code=400, detail="invalid rank")

        ref_spec, score, dbidx = topk_list[rank - 1]
        query_spec = spectra[query_index]

        spec_json, loss_json, struct_png = build_plots_for_pair(query_spec, ref_spec)

        return {
            "status": "success",
            "query_index": query_index,   # 返回 0-based
            "rank": rank,                 # 返回 1-based
            "StructSimScore": score,
            "database_index": dbidx,
            "spectrum_plot": spec_json,
            "spectrum_loss_plot": loss_json,
            "ref_structure_plot": struct_png,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[get-candidate-plots] error: {e}")
        raise HTTPException(status_code=500, detail=f"failed to build plots: {e}")

@router.get("/progress")
async def get_run_progress(session_scope: Dict[str, Any] = Depends(get_session_scope)):
    scope_key = session_scope["key"]
    prog = get_progress(scope_key)
    return {
        "status": prog.get("status", "idle"),
        "total": prog.get("total", 0),
        "done": prog.get("done", 0),
        "message": prog.get("message", ""),
    }
