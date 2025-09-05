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
from typing import Optional
from backend.anal_sear.plot_utils import build_plots_for_pair

import redis
import json

# 建立连接
r = redis.Redis(host="127.0.0.1", port=6379, db=0)

def baseline_session_state():
    return {
        "references_positive_path": None,
        "references_negative_path": None,
        "hnsw_positive_path": None,
        "hnsw_negative_path": None,
        "custom_database_path": None,
        "last_accessed": time.time(),
        "status": "idle",
        "status_message": "",
        "filename": None,
        # ★ 新增：进度
        "progress": {"status": "idle", "total": 0, "done": 0},
    }
def set_progress(session_id: str, total: int = None, done: int = None, status: str = None, message: str = None):
    st = get_state(session_id) or baseline_session_state()
    prog = st.get("progress", {"status":"idle","total":0,"done":0})
    if total is not None: prog["total"] = int(total)
    if done is not None:  prog["done"]  = int(done)
    if status:            prog["status"] = status
    st["progress"] = prog
    if message is not None:
        st["status_message"] = message
    st["last_accessed"] = time.time()
    set_state(session_id, st)

def get_progress(session_id: str) -> dict:
    st = get_state(session_id) or {}
    prog = st.get("progress", {})
    return {
        "status": prog.get("status", "idle"),
        "total":  prog.get("total", 0),
        "done":   prog.get("done", 0)
    }


def _redis_key(session_id: str) -> str:
    return f"session:{session_id}"

def set_state(session_id: str, data: dict):
    r.set(_redis_key(session_id), json.dumps(data))

def get_state(session_id: str) -> dict:
    val = r.get(_redis_key(session_id))
    if val:
        return json.loads(val)
    return {}

def update_session(session_id: str, **fields):
    st = get_state(session_id)
    if not st:
        # 如果不存在，初始化一个
        st = {
            "references_positive_path": None,
            "references_negative_path": None,
            "hnsw_positive_path": None,
            "hnsw_negative_path": None,
            "custom_database_path": None,
            "last_accessed": time.time(),
            "status": "idle",
            "status_message": "",
            "filename": None,
        }
    st.update(fields)
    set_state(session_id, st)
    return st

def delete_session(session_id: str):
    r.delete(_redis_key(session_id))

def scan_sessions(pattern: str = "session:*", count: int = 100):
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=count)
        for k in keys:
            yield k.decode() if isinstance(k, bytes) else k
        if cursor == 0:
            break

router = APIRouter(prefix="/anal_search", tags=["anal_search"])

# Preloaded model paths
MODEL_POS_PATH = 'model/Ms2Vec_allGNPSpositive.hdf5'
MODEL_NEG_PATH = 'model/Ms2Vec_allGNPSnegative.hdf5'

# Temporary directories
TMP_DIR = "./temp/result_csv_temp/"
STATE_DIR = "./temp/state_files/"
DB_TMP_DIR = "./temp/database_temp/"
FILE_TMP_DIR = "./temp/temp_files/"

TEMP_DIR = "./temp"
SESSION_TIMEOUT = 86400 * 3  # 3天
# Ensure directories exist
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(DB_TMP_DIR, exist_ok=True)
os.makedirs(FILE_TMP_DIR, exist_ok=True)

# 进程内锁
lock = asyncio.Lock()

# ---- 工具函数 ----
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

# ---- “从 Cookie 读取 session_id”的依赖（改为 Redis 校验）----
async def get_session_id(request: Request) -> str:
    session_id = request.cookies.get("session_id")
    if not session_id:
        print("Missing session_id cookie")
        raise HTTPException(status_code=400, detail="Invalid or missing session")
    st = get_state(session_id)
    if not st:
        print(f"Invalid or missing session in Redis: {session_id}")
        raise HTTPException(status_code=400, detail="Invalid or missing session")
    # 更新 last_accessed
    update_session(session_id, last_accessed=time.time())
    return session_id

# ---- 后台处理函数：将结果路径写入 Redis ----
async def process_spectrums_background(file_path, pos_model_path, neg_model_path, out_dir, session_id):
    try:
        print(f"Starting background processing for file: {file_path}")
        if not os.path.splitext(file_path)[1].lower() in ['.mgf', '.msp', '.mat']:
            raise HTTPException(status_code=400, detail=f"Invalid file format: {file_path}")
        specs = load_spectrum_file(file_path)
        if not specs:
            raise HTTPException(status_code=400, detail="No valid spectra found in file")

        for s in specs:
            if not s.metadata.get("ionmode"):
                raise HTTPException(status_code=400, detail="Missing ionmode in spectrum metadata")
            if not s.peaks.mz.size or not s.peaks.intensities.size:
                raise HTTPException(status_code=400, detail="Spectrum missing peaks or intensities")

        print(f"Loaded {len(specs)} spectra from file: {file_path}")

        positive_specs, negative_specs = [], []
        for index, s in enumerate(specs):
            s.set("database_index", index)
            mode = s.metadata.get("ionmode", "positive")
            (negative_specs if mode == "negative" else positive_specs).append(s)

        if not positive_specs and not negative_specs:
            raise HTTPException(status_code=400, detail="No spectrum data detected in file")

        os.makedirs(out_dir, exist_ok=True)
        positive_pkl = os.path.join(out_dir, "references_spectrums_positive.pickle")
        negative_pkl = os.path.join(out_dir, "references_spectrums_negative.pickle")
        with open(positive_pkl, "wb") as f:
            pickle.dump(positive_specs, f)
        with open(negative_pkl, "wb") as f:
            pickle.dump(negative_specs, f)

        def build_index(pkl_path, model_path, dim=300, prefix="positive"):
            if not os.path.exists(model_path):
                raise HTTPException(status_code=500, detail=f"Model file not found: {model_path}")
            with open(pkl_path, "rb") as f:
                refs = pickle.load(f)
            if not refs:
                raise HTTPException(status_code=400, detail=f"No {prefix} spectra available for indexing")
            model = gensim.models.Word2Vec.load(model_path)
            vectors = []
            for s in tqdm(refs, desc=f"Vectorizing {prefix}"):
                try:
                    v = calc_vector(model, SpectrumDocument(s, n_decimals=2), allowed_missing_percentage=100)
                    vectors.append(v)
                except Exception as e:
                    print(f"Error vectorizing spectrum: {str(e)}")
                    continue
            if not vectors:
                raise HTTPException(status_code=400, detail=f"No valid vectors generated for {prefix} spectra")
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

        # 将路径写入 Redis 状态
        update_session(
            session_id,
            references_positive_path=positive_pkl if os.path.exists(positive_pkl) else None,
            references_negative_path=negative_pkl if os.path.exists(negative_pkl) else None,
            hnsw_positive_path=os.path.join(out_dir, "references_index_positive_spec2vec.bin") if positive_idx else None,
            hnsw_negative_path=os.path.join(out_dir, "references_index_negative_spec2vec.bin") if negative_idx else None,
            custom_database_path=out_dir,
            status="success",
            status_message=f"Custom database initialized successfully",
            last_accessed=time.time(),
        )
        print(f"Background processing completed successfully for session_id: {session_id}")
    except Exception as e:
        print(f"Background processing error for session_id {session_id}: {str(e)}")
        update_session(
            session_id,
            status="error",
            status_message=f"Custom database processing failed: {str(e)}",
            last_accessed=time.time(),
        )
        raise

# ---- 1) Start session：设置 Cookie + 初始化 Redis 状态 ----
@router.get("/start-session")
async def start_session(request: Request, response: Response):
    # 1) 先尝试复用
    sid = request.cookies.get("session_id")
    if sid:
        st = get_state(sid)
        if st:  # Redis 里有，说明有效
            update_session(sid, last_accessed=time.time())
            # 可选：刷新 cookie 的有效期
            response.set_cookie(
                key="session_id",
                value=sid,
                httponly=True,
                max_age=3600*24*3,
                samesite="Lax",
                # secure=True  # 若走 HTTPS，需加
            )
            return {"status": "success", "session_id": sid, "reused": True}

    # 2) 没有就新建
    new_sid = str(uuid.uuid4())
    set_state(new_sid, baseline_session_state())
    response.set_cookie(
        key="session_id",
        value=new_sid,
        httponly=True,
        max_age=3600*24*3,
        samesite="Lax",
        # secure=True
    )
    return {"status": "success", "session_id": new_sid, "reused": False}

# ---- 2) 上传自定义数据库 ----
@router.post("/upload-custom-db")
async def upload_custom_db(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session_id: str = Depends(get_session_id)
):
    print(f"Received upload-custom-db: session_id={session_id}, filename={file.filename}")
    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix not in ['.mgf', '.msp', '.mat']:
            return JSONResponse(status_code=400, content={"status": "error", "message": f"Unsupported file format: {suffix}"})
        if hasattr(file, "size") and file.size and file.size > 1_000_000_000:
            return JSONResponse(status_code=400, content={"status": "error", "message": f"File too large, maximum 1GB"})

        uuid_prefix = str(uuid.uuid4())
        unique_filename = f"{session_id}_{uuid_prefix}_{file.filename}"
        db_dir = os.path.join(DB_TMP_DIR, session_id)
        os.makedirs(db_dir, exist_ok=True)
        temp_path = os.path.join(db_dir, unique_filename)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        update_session(
            session_id,
            status="processing",
            status_message=f"Processing custom database: {file.filename}",
            filename=file.filename,
            last_accessed=time.time()
        )

        out_dir = os.path.join(DB_TMP_DIR, session_id)
        background_tasks.add_task(process_spectrums_background, temp_path, MODEL_POS_PATH, MODEL_NEG_PATH, out_dir, session_id)

        return {"status": "processing", "message": f"Custom database {file.filename} is being processed"}
    except Exception as e:
        print(f"upload_custom_db error: {e}")
        update_session(
            session_id,
            status="error",
            status_message=f"Custom database {file.filename} upload failed: {str(e)}",
            last_accessed=time.time()
        )
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Custom database {file.filename} upload failed: {str(e)}"})

# ---- 3) 轮询状态 ----
@router.get("/custom-db-status")
async def get_custom_db_status(session_id: str = Depends(get_session_id)):
    st = get_state(session_id)
    return {
        "status": st.get("status", "idle"),
        "message": st.get("status_message", ""),
        "filename": st.get("filename")
    }

# ---- 4) 校验之前上传的临时文件是否仍在 ----
@router.get("/check-file")
async def check_file(
    uuid_filename: str = Query(...),
    session_id: str = Depends(get_session_id),
):
    try:
        file_path = os.path.join(FILE_TMP_DIR, session_id, uuid_filename)
        if os.path.exists(file_path):
            return {"status": "success", "message": f"File {uuid_filename} exists"}
        else:
            return {"status": "error", "message": f"File {uuid_filename} does not exist"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to check file: {str(e)}"}

# ---- 5) 上传输入文件 ----
@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id)
):
    print(f"/upload: session_id={session_id}, filename={file.filename}")
    temp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1].lower()
        if suffix not in ['.mgf', '.msp', '.mat']:
            return {"status": "error", "message": f"Unsupported file format: {suffix}"}

        uuid_prefix = str(uuid.uuid4())
        unique_filename = f"{session_id}_{uuid_prefix}_{file.filename}"
        temp_dir = os.path.join(FILE_TMP_DIR, session_id)
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, unique_filename)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        if not os.path.exists(temp_path):
            return {"status": "error", "message": f"Failed to save file: {file.filename}"}

        spectrums_df, name_list, _ = load_files([temp_path])
        if spectrums_df is None or name_list is None or len(name_list) == 0:
            return {"status": "error", "message": "File parsing failed, possibly empty or incorrect format"}

        update_session(session_id, last_accessed=time.time())

        return {
            "status": "success",
            "names": name_list.to_dict(),
            "uuid_filename": unique_filename,
            "message": f"File {file.filename} uploaded successfully"
        }
    except Exception as e:
        print(f"upload_file error: {e}")
        return {"status": "error", "message": f"File {file.filename} upload failed: {str(e)}"}
    finally:
        # 仅当路径异常（没在该 session 目录下）才删除
        if temp_path and os.path.exists(temp_path) and not temp_path.startswith(os.path.join(FILE_TMP_DIR, session_id)):
            os.remove(temp_path)

# ---- 6) 载入测试文件 ----
@router.get("/upload-test-file")
async def upload_test_file(session_id: str = Depends(get_session_id)):
    try:
        test_file_path = "analogSearch_data/test.mgf"
        if not os.path.exists(test_file_path):
            return {"status": "error", "message": f"Test file does not exist: {test_file_path}"}
        spectrums_df, name_list, _ = load_files([test_file_path])
        update_session(session_id, last_accessed=time.time())
        return {
            "status": "success",
            "names": name_list.to_dict(),
            "message": "Test file processed successfully"
        }
    except Exception as e:
        return {"status": "error", "message": f"Test file processing failed: {str(e)}"}

# ---- 7) run ----
@router.post("/run")
async def run_analysis(
    threshold: float = Form(...),
    source: str = Form(...),
    db_option: str = Form(...),
    file: UploadFile = File(None),
    uuid_filename: str = Form(None),
    session_id: str = Depends(get_session_id)
):
    start_time = time.time()
    print(f"[{start_time}] /run: source={source}, uuid_filename={uuid_filename}, file={file.filename if file else None}, threshold={threshold}, db_option={db_option}, session_id={session_id}")
    temp_path = None
    state_uuid = f"{session_id}_{str(uuid.uuid4())}"
    try:
        set_progress(session_id, total=0, done=0, status="running", message="Loading references...")

        # 加载参考库
        if db_option == "Custom":
            st = get_state(session_id)
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
                temp_path = os.path.join(FILE_TMP_DIR, session_id, uuid_filename)
                if not os.path.exists(temp_path):
                    return {"status": "error", "message": f"Specified UUID file does not exist: {uuid_filename}"}
                spectrums_df, name_list, _ = load_files([temp_path])
            elif file:
                suffix = os.path.splitext(file.filename)[1].lower()
                if suffix not in ['.mgf', '.msp', '.mat']:
                    return {"status": "error", "message": f"Unsupported file format: {suffix}"}
                temp_dir = os.path.join(FILE_TMP_DIR, session_id)
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"{session_id}_{str(uuid.uuid4())}_{file.filename}")
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

        # ★ 初始化总数
        total_queries = len(spectrums_df)
        set_progress(session_id, total=total_queries, done=0, status="running", message="Searching candidates...")

        (result_state, result_df, all_topk) = matchms_click_fn(
            threshold=threshold,
            res_state=spectrums_df,
            refs_pos_state=refs_pos,
            refs_neg_state=refs_neg,
            hnsw_pos_state=hnsw_pos,
            hnsw_neg_state=hnsw_neg,
            progress_cb=lambda i: set_progress(
                session_id,
                total=total_queries,
                done=i,
                status="running",
                message=f"Processed {i}/{total_queries}"
            )
        )
        set_progress(session_id, total=total_queries, done=total_queries, status="finished", message="Completed")

        # state_path = os.path.join(STATE_DIR, f"{state_uuid}.pkl")
        # with open(state_path, "wb") as f:
        #     pickle.dump(result_state, f)

        state_path = os.path.join(STATE_DIR, f"{state_uuid}.pkl")
        bundle = {
            "res_state": result_state,                     # DataFrame（含 Identified Spectrum / annotation）
            "spectra": spectrums_df["spectrum"].tolist(), # 原始查询谱图列表
            "all_topk": all_topk,                          # 每个查询的 TopK 列表(含 ref_spec 对象)
        }
        with open(state_path, "wb") as f:
            pickle.dump(bundle, f)

        update_session(session_id, last_accessed=time.time())

        if result_df is None or result_df.empty:
            return {
                "status": "success",
                "names": name_list.to_dict(),
                "result": [],
                "state_uuid": state_uuid,
                "target_zip_file_name": f"{session_id}_results.zip",
                "message": f"No matching results found, possibly due to high threshold ({threshold}) or uninitialized database"
            }

        clean_result = dataframe_to_clean_records(result_df)
        return {
            "status": "success",
            "names": name_list.to_dict(),
            "result": clean_result,
            "state_uuid": state_uuid,
            "target_zip_file_name": f"{session_id}_results.zip",
            "message": "Analysis completed"
        }
    except Exception as e:
        set_progress(session_id, status="error", message=f"Run failed: {str(e)}")
        print(f"run_analysis error: {str(e)}")
        return {"status": "error", "message": f"Analysis failed: {str(e)}"}
    finally:
        if temp_path and os.path.exists(temp_path) and not temp_path.startswith(os.path.join(FILE_TMP_DIR, session_id)):
            os.remove(temp_path)
        await clean_old_states()
        end_time = time.time()
        print(f"[{end_time}] Completed /run: session_id={session_id}, duration={end_time - start_time:.2f}s")

# ---- 8) 保存结果 ----
@router.post("/save-results")
async def save_results(
    state_uuid: Optional[str] = Form(None),
    target_zip_file_name: Optional[str] = Form(None),
    threshold: Optional[float] = Form(None),
    session_id: str = Depends(get_session_id)
):
    try:
        if not state_uuid:
            raise HTTPException(status_code=400, detail="Missing state_uuid")
        state_path = os.path.join(STATE_DIR, f"{state_uuid}.pkl")
        if not os.path.exists(state_path):
            raise HTTPException(status_code=404, detail=f"State file not found: {state_uuid}")

        with open(state_path, "rb") as f:
            obj = pickle.load(f)

        # 新打包格式
        if isinstance(obj, dict) and "res_state" in obj:
            result_state = obj["res_state"]
        else:
            result_state = obj  # 兼容旧格式

        if result_state is None or "spectrum" not in result_state.columns or "annotation" not in result_state.columns:
            raise HTTPException(status_code=400, detail="Invalid result_state")

        dir_path = os.path.join(TMP_DIR, session_id, uuid.uuid4().hex)
        os.makedirs(dir_path, exist_ok=True)

        # 目标扩展名改为 .xlsx
        if not target_zip_file_name:
            target_zip_file_name = f"{session_id}_results.xlsx"
        base, _ = os.path.splitext(target_zip_file_name)
        xlsx_filename = base + ".xlsx"
        xlsx_path = os.path.join(dir_path, xlsx_filename)

        # === 汇总所有 TopK 行 ===
        all_rows = []
        for idx, s in enumerate(result_state["spectrum"]):
            name = get_title_from_spectrum(spectrum=s, idx=idx)
            ann = result_state["annotation"][idx]
            if ann is None or ann.empty:
                continue
            for _, row in ann.iterrows():
                # 组装一行（保持原字段）
                row_dict = {"compoundName-index": f"{name}-{idx}"}
                for col, val in row.to_dict().items():
                    row_dict[col] = val
                all_rows.append(row_dict)

        # === 若没有任何行，写空表并返回 ===
        if not all_rows:
            empty_df = pd.DataFrame(columns=["compoundName-index", "StructSimScore", "database_index", "smiles"])
            try:
                empty_df.to_excel(xlsx_path, index=False)
            except Exception:
                # 若本机没安装 openpyxl，也允许回退到 csv
                csv_path = os.path.join(dir_path, base + ".csv")
                empty_df.to_csv(csv_path, index=False)
                return {"status": "success", "message": "No valid results found, saved empty CSV", "file_path": csv_path}
            return {"status": "success", "message": "No valid results found, saved empty Excel", "file_path": xlsx_path}

        df = pd.DataFrame(all_rows)

        # === ★ 按阈值过滤（只保留 Score ≥ threshold） ===
        if threshold is not None:
            # 字段名与你前端展示一致：StructSimScore
            df = df[pd.to_numeric(df.get("StructSimScore", np.nan), errors="coerce").fillna(-np.inf) >= float(threshold)]

        # === 可能过滤后为空 ===
        if df.empty:
            # 仍然输出空的 xlsx
            keep_cols = ["compoundName-index", "StructSimScore", "database_index", "smiles"]
            for col in keep_cols:
                if col not in df.columns:
                    df[col] = pd.Series(dtype=object)
            df = df[keep_cols]
            try:
                df.to_excel(xlsx_path, index=False)
            except Exception:
                csv_path = os.path.join(dir_path, base + ".csv")
                df.to_csv(csv_path, index=False)
                return {"status": "success", "message": f"No candidates matched threshold ≥ {threshold}, saved empty CSV", "file_path": csv_path}
            return {"status": "success", "message": f"No candidates matched threshold ≥ {threshold}, saved empty Excel", "file_path": xlsx_path}

        # === 正常写 xlsx（如无 openpyxl 则回退 csv） ===
        try:
            df.to_excel(xlsx_path, index=False)
            return {"status": "success", "message": "Results saved successfully", "file_path": xlsx_path}
        except Exception as e:
            # 回退 csv
            csv_path = os.path.join(dir_path, base + ".csv")
            df.to_csv(csv_path, index=False)
            return {"status": "success", "message": f"Excel writer not available, saved CSV instead: {e}", "file_path": csv_path}

    except Exception as e:
        print(f"Save results error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save results: {str(e)}")

# ---- 9) 下载结果文件 ----
@router.get("/download-result-file")
async def download_result_file(
    file_path: str = Query(...),
    session_id: str = Depends(get_session_id)
):
    try:
        if not file_path.startswith(os.path.join(TMP_DIR, session_id)):
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path=file_path, filename=os.path.basename(file_path), media_type="text/csv")
    except Exception as e:
        print(f"Download file error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")

# ---- 10) Clear ----
@router.post("/clear")
async def clear_state(session_id: str = Depends(get_session_id)):
    """
    仅清除落盘数据并将 Redis 会话重置为初始状态；
    不删除 Redis 键，不清除 Cookie。
    """
    try:
        # 读取当前状态，清理自定义库目录
        st = get_state(session_id)
        custom_dir = st.get("custom_database_path")
        if custom_dir and os.path.exists(custom_dir):
            shutil.rmtree(custom_dir, ignore_errors=True)

        # 清理该 session 的临时目录
        for base_dir in [TMP_DIR, DB_TMP_DIR, FILE_TMP_DIR]:
            d = os.path.join(base_dir, session_id)
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)

        # 清理该 session 的状态文件 (*.pkl)
        state_pattern = os.path.join(STATE_DIR, f"{session_id}_*")
        for p in glob.glob(state_pattern):
            if os.path.isfile(p):
                os.remove(p)

        # 重置为“空/初始”状态，保留 session_id 有效
        fresh = baseline_session_state()
        # 更新最后访问时间，避免被定时清理马上扫掉
        fresh["last_accessed"] = time.time()
        set_state(session_id, fresh)

        return {"status": "success", "message": f"Session {session_id} cleared but retained (ID kept)."}
    except Exception as e:
        print(f"Clear state error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear state: {str(e)}")


# ---- 11) 定期清理（扫描 Redis 会话并基于 last_accessed 清理落盘）----
async def clean_old_states(max_age_seconds=SESSION_TIMEOUT):
    current_time = time.time()

    # 1) 清理 Redis / 内存中的过期 session
    for key in r.scan_iter("session:*"):
        sid = key.decode().split(":", 1)[1]
        state = get_state(sid)
        if state:
            last = state.get("last_accessed", 0)
            if current_time - last > max_age_seconds:
                # 删除 temp 文件
                delete_session(sid)

    # 2) 扫描 temp 文件夹，把超过 3 天没修改的删掉
    if os.path.exists(TEMP_DIR):
        for fname in os.listdir(TEMP_DIR):
            fpath = os.path.join(TEMP_DIR, fname)
            try:
                # 文件最后修改时间
                mtime = os.path.getmtime(fpath)
                if current_time - mtime > max_age_seconds:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                        print(f"[CLEAN] 删除过期文件: {fpath}")
                    elif os.path.isdir(fpath):
                        shutil.rmtree(fpath)
                        print(f"[CLEAN] 删除过期文件夹: {fpath}")
            except Exception as e:
                print(f"[CLEAN] 删除 {fpath} 失败: {e}")

# ---- 12) 下载测试文件 ----
@router.get("/download-test-file")
async def download_test_file(session_id: str = Depends(get_session_id)):
    test_file_path = "analogSearch_data/test.mgf"
    if not os.path.exists(test_file_path):
        raise HTTPException(status_code=404, detail="Test file does not exist: analogSearch_data/test.mgf")
    # FileResponse 会自动加 Content-Disposition: attachment; filename="test.mgf"
    return FileResponse(
        path=test_file_path,
        filename="test.mgf",
        media_type="application/octet-stream"
    )

@router.get("/get-candidate-plots")
async def get_candidate_plots(
    state_uuid: str = Query(...),
    query_index: int = Query(..., ge=1),
    rank: int = Query(..., ge=1),
    session_id: str = Depends(get_session_id),
):
    """
    懒加载接口：返回某个查询(query_index) 的第 rank 个候选的三种图
    需要在 /run 时把 (spectra, all_topk) 打包存进 state 文件
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

        if query_index < 1 or query_index > len(spectra):
            raise HTTPException(status_code=400, detail="invalid query_index")
        topk_list = all_topk[query_index - 1]
        if rank < 1 or rank > len(topk_list):
            raise HTTPException(status_code=400, detail="invalid rank")

        ref_spec, score, dbidx = topk_list[rank - 1]
        query_spec = spectra[query_index - 1]

        # 生成三种图

        spec_json, loss_json, struct_png = build_plots_for_pair(query_spec, ref_spec)

        return {
            "status": "success",
            "query_index": query_index,
            "rank": rank,
            "StructSimScore": score,
            "database_index": dbidx,
            "spectrum_plot": spec_json,
            "spectrum_loss_plot": loss_json,
            "ref_structure_plot": struct_png,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"get_candidate_plots error: {e}")
        raise HTTPException(status_code=500, detail=f"failed to build plots: {e}")

@router.get("/progress")
async def get_run_progress(session_id: str = Depends(get_session_id)):
    return get_progress(session_id)
