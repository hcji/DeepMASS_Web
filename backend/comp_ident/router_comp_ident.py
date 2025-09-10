# backend/comp_ident/router_comp_ident.py
from fastapi import APIRouter, UploadFile, File, Request, Response, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List, Optional
import os, shutil, tempfile, uuid, time, zipfile
import pandas as pd
import numpy as np
from matchms import Spectrum

from backend.comp_ident.service_identity import load_files
from backend.utils.identify_unkown import identify_pos, identify_neg
from backend.comp_ident.plot_utils import plot_2_spectrum, plot_2_mol, get_formula_mass

# 与 anal_sear 共用 Cookie（会话在 Redis，小数据状态存 Redis；大对象 DataFrame 落盘为 pickle）
from backend.service.session_store import Store

router = APIRouter(prefix="/compound_identification", tags=["compound_identification"])

# 会话：Cookie 名与 anal_sear 一致；namespace 区分模块；pickle 文件根目录 base_dir
store = Store(
    namespace="compident",
    ttl_seconds=60 * 60 * 24 * 3,
    redis_host="127.0.0.1",
    redis_port=6379,
    redis_db=0,
    base_dir="temp/session_store",   # 会在 temp/session_store/compident/<sid>/spectra.pkl
    cookie_name="session_id",
)

# ----------------- 工具 -----------------
def _dataframe_to_records(df: pd.DataFrame):
    def cv(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, (str, dict)):
            return x
        return str(x)

    return (
        df.replace([np.inf, -np.inf], np.nan)
          .astype(object)
          .where(pd.notnull(df), None)
          .applymap(cv)
          .to_dict(orient="records")
    )

def _default_results_dir(sid: str):
    d = os.path.join("temp", "result_csv_temp", sid, uuid.uuid4().hex)
    os.makedirs(d, exist_ok=True)
    return d

def _as_ref_list(x):
    """
    把 metadata['reference'] 统一转为 Python list，避免 numpy array / pd.Series 触发布尔歧义。
    允许 list / tuple / np.ndarray / pd.Series，其他情况返回 []。
    """
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, (np.ndarray, pd.Series)):
        try:
            return [i for i in x.tolist() if i is not None]
        except Exception:
            return []
    # 其它类型（比如标量/字典），一律视为没有 reference
    return []

# ----------------- 会话（标签页隔离） -----------------
def _compose_sid_with_tab(sid: str, tab_id: Optional[str]) -> str:
    """把会话 id 和 tab_id 拼成复合 key；tab_id 为空时退化为原 sid。"""
    if tab_id:
        return f"{sid}:{tab_id}"
    return sid

@router.get("/start-session")
async def start_session(request: Request, response: Response):
    # 这个接口仅负责下发/续期 cookie 的 session_id
    sid = store.get_or_create_session(request, response)
    st = store.read_state(sid)
    if not st:
        store.update_state(sid, target_zip_file_name=None, last_accessed=time.time())
    return {"status": "success", "session_id": sid}

def _require_raw_sid(request: Request) -> str:
    """只校验 Cookie 会话存在，用于需要复合 sid 前的基础检查。"""
    sid = store.require_session(request)
    store.update_state(sid, last_accessed=time.time())
    return sid

def _require_composite_sid(request: Request, tab_id: Optional[str]) -> str:
    """返回复合 sid（sid:tab_id），并更新活跃时间。"""
    raw_sid = store.require_session(request)
    store.update_state(raw_sid, last_accessed=time.time())
    return _compose_sid_with_tab(raw_sid, tab_id)

# ----------------- 上传 / 测试文件 -----------------
@router.post("/upload")
async def upload_files(
    request: Request,
    response: Response,
    files: List[UploadFile] = File(...),
    tab_id: Optional[str] = Query(None),
):
    raw_sid = store.get_or_create_session(request, response)
    sid = _compose_sid_with_tab(raw_sid, tab_id)  # ★ 复合 sid（标签页隔离）

    if not files:
        return {"status": "error", "message": "No files"}

    tmp_paths = []
    try:
        for file in files:
            suffix = os.path.splitext(file.filename)[1].lower()
            if suffix not in [".mgf", ".msp", ".mat"]:
                return {"status": "error", "message": f"Unsupported file: {file.filename}"}
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            with tf as f:
                shutil.copyfileobj(file.file, f)
            tmp_paths.append(tf.name)

        spectrums_df, name_df, target_zip_file_name = load_files(tmp_paths)
        if spectrums_df is None or name_df is None or len(name_df) == 0:
            return {"status": "error", "message": "File parsing failed or empty"}

        # ★ DataFrame 落盘为 pickle（Store 内部已实现），写入复合 sid
        store.save_df(sid, spectrums_df)
        # ★ target_zip_file_name 也写到复合 sid 的 state 下（避免跨标签页串扰）
        store.update_state(sid, target_zip_file_name=target_zip_file_name, last_accessed=time.time())
        store.set_progress(sid, total=0, done=0, status="idle")

        return {"status": "success", "names": name_df.to_dict(), "message": "Upload Successful!"}
    except Exception as e:
        return {"status": "error", "message": f"Upload failed: {e}"}
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)

@router.get("/upload-test-file")
async def upload_test_file(request: Request, response: Response, tab_id: Optional[str] = Query(None)):
    raw_sid = store.get_or_create_session(request, response)
    sid = _compose_sid_with_tab(raw_sid, tab_id)
    try:
        test_file = "analogSearch_data/test.mgf"
        if not os.path.exists(test_file):
            return {"status": "error", "message": "Test file not found"}
        spectrums_df, name_df, target_zip_file_name = load_files([test_file])
        store.save_df(sid, spectrums_df)
        store.update_state(sid, target_zip_file_name=target_zip_file_name, last_accessed=time.time())
        store.set_progress(sid, total=0, done=0, status="idle")
        return {"status": "success", "names": name_df.to_dict(), "message": "Test file ready"}
    except Exception as e:
        return {"status": "error", "message": f"Test file error: {e}"}

@router.get("/download-test-file")
async def download_test_file():
    test_file = "analogSearch_data/test.mgf"
    if not os.path.exists(test_file):
        return {"status": "error", "message": "Test file not found"}
    return FileResponse(test_file, filename="test.mgf", media_type="text/plain")

@router.get("/spectrum-list")
async def spectrum_list(request: Request, tab_id: Optional[str] = Query(None)):
    sid = _require_composite_sid(request, tab_id)
    df = store.load_df(sid)
    if df is None:
        return {"status": "error", "message": "No spectra"}
    return {"status": "success", "names": df[["title"]].to_dict()}

# ----------------- 进度 -----------------
@router.get("/progress")
async def get_progress(request: Request, tab_id: Optional[str] = Query(None)):
    sid = _require_composite_sid(request, tab_id)
    prog = store.get_progress(sid)
    return {
        "status": prog.get("status", "success"),
        "total": prog.get("total", 0),
        "done": prog.get("done", 0),
    }

# ----------------- Run -----------------
@router.post("/run")
async def run_deepms(request: Request, tab_id: Optional[str] = Query(None)):
    sid = _require_composite_sid(request, tab_id)
    df = store.load_df(sid)
    if df is None:
        return {"status": "error", "message": "Please upload files first"}

    try:
        specs: List[Spectrum] = df["spectrum"].tolist()
        total = len(specs)

        # 初始化 Identified Spectrum 列（如果没有则创建，长度与 df 相同）
        if "Identified Spectrum" not in df.columns:
            df["Identified Spectrum"] = [None] * total

        store.set_progress(sid, total=total, done=0, status="running")
        store.save_df(sid, df)  # 保存初始状态，确保前端能看到 need_run -> True/已存在列

        # 逐个处理并及时保存
        for i, s in enumerate(specs):
            try:
                ionmode = (s.metadata or {}).get("ionmode", "positive") or "positive"
                sn = identify_neg(s) if str(ionmode).lower() == "negative" else identify_pos(s)
            except Exception:
                sn = None

            # 将识别结果写回 df 的对应位置（保持索引一致）
            df.at[i, "Identified Spectrum"] = sn

            # 更新进度并把当前 df 落盘
            store.set_progress(sid, total=total, done=i + 1, status="running")

            # 只间隔性落盘，减少IO
            # if (i + 1) % 50 == 0 or (i + 1) == total:
        store.save_df(sid, df)

        # 全部完成
        store.set_progress(sid, total=total, done=total, status="running")
        # ★ 注意把 tab_id 透传给 select_spectrum
        result = await select_spectrum(request, idx=0, tab_id=tab_id)
        store.set_progress(sid, total=total, done=total, status="finished")
        return result
    except Exception as e:
        store.set_progress(sid, total=0, done=0, status="error")
        return {"status": "error", "message": f"Run failed: {e}"}

# ----------------- 左侧点击：公式 + 信息 -----------------
@router.get("/select-spectrum")
async def select_spectrum(
    request: Request,
    idx: int = Query(..., ge=0),
    tab_id: Optional[str] = Query(None),
):
    sid = _require_composite_sid(request, tab_id)
    df = store.load_df(sid)
    if df is None:
        return {"status": "error", "message": "No spectra"}
    if idx >= len(df):
        return {"status": "error", "message": "Index out of range"}

    # 未识别：返回原始谱的 metadata 信息表 + 空公式表 + need_run 提示
    if "Identified Spectrum" not in df.columns:
        try:
            raw_spec: Spectrum = df["spectrum"][idx]
            md = (raw_spec.metadata or {}).copy()
        except Exception:
            md = {}
        md.pop("reference", None)
        md.pop("annotation", None)
        info_df = (
            pd.DataFrame.from_dict(md, orient="index", columns=["value"])
            .reset_index()
            .rename(columns={"index": "key"})
        )
        formula_df = pd.DataFrame(columns=["Formula", "mass", "error (mDa)"])

        return {
            "status": "success",
            "need_run": True,
            "spectrum_index": idx,
            "formula_table": _dataframe_to_records(formula_df),
            "information_table": _dataframe_to_records(info_df),
        }

    # 已识别：正常返回
    cur_spec: Spectrum = df["Identified Spectrum"][idx]
    if cur_spec is None:
        return {"status": "error", "message": "This spectrum has no identification result."}
    md = cur_spec.metadata or {}

    annotation: Optional[pd.DataFrame] = md.get("annotation")
    if annotation is None or annotation.empty or "MolecularFormula" not in annotation.columns:
        formula_df = pd.DataFrame(columns=["Formula", "mass", "error (mDa)"])
    else:
        formulas = pd.unique(annotation["MolecularFormula"])
        masses = [get_formula_mass(f) for f in formulas]
        if "parent_mass" in md:
            parent_mass = float(md["parent_mass"])
            diff_mDa = [abs(m - parent_mass) * 1000 for m in masses]
        else:
            diff_mDa = [np.nan] * len(masses)
        formula_df = pd.DataFrame(
            {"Formula": formulas, "mass": masses, "error (mDa)": diff_mDa}
        ).round(3)

    info_df = pd.DataFrame()
    try:
        d = dict(md); d.pop("reference", None); d.pop("annotation", None)
        info_df = (
            pd.DataFrame.from_dict(d, orient="index", columns=["value"])
            .reset_index()
            .rename(columns={"index": "key"})
        )
    except Exception:
        pass

    return {
        "status": "success",
        "need_run": False,
        "spectrum_index": idx,
        "formula_table": _dataframe_to_records(formula_df),
        "information_table": _dataframe_to_records(info_df),
    }

# ----------------- 选 Formula：结构表 + 默认参考 + 两张结构图 -----------------
@router.get("/structures")
async def structures(
    request: Request,
    spec_idx: int = Query(..., ge=0),
    formula_idx: int = Query(..., ge=0),
    highlight: bool = Query(True),   # ★ 新增：接收前端 Highlight 开关
    tab_id: Optional[str] = Query(None),
):
    sid = _require_composite_sid(request, tab_id)
    df = store.load_df(sid)
    if df is None:
        return {"status": "error", "message": "No spectra"}
    if "Identified Spectrum" not in df.columns:
        return {"status": "error", "message": "Not identified yet."}
    if spec_idx >= len(df):
        return {"status": "error", "message": "Index out of range"}

    cur_spec: Spectrum = df["Identified Spectrum"][spec_idx]
    md = cur_spec.metadata or {}
    annotation: Optional[pd.DataFrame] = md.get("annotation")

    structure_table = pd.DataFrame()
    ref_df = pd.DataFrame(columns=["name", "adduct", "smiles", "parent_mass", "database"])
    ann_img, ref_img = (None, None)

    if annotation is not None and not annotation.empty:
        if "MolecularFormula" in annotation.columns:
            formulas = pd.unique(annotation["MolecularFormula"])
            if 0 <= formula_idx < len(formulas):
                sel_formula = formulas[formula_idx]
                structure_table_all = annotation.loc[
                    annotation["MolecularFormula"] == sel_formula, :
                ].reset_index(drop=True)
            else:
                structure_table_all = pd.DataFrame()
        else:
            structure_table_all = annotation.reset_index(drop=True)

        structure_table = structure_table_all

        # 参考表
        current_reference = _as_ref_list(md.get("reference"))
        rows = []
        for s in current_reference:
            if s is None or not hasattr(s, "metadata"):
                continue
            smi = s.metadata.get("smiles", "")
            name = s.metadata.get("compound_name", smi)
            adduct = s.metadata.get("adduct", "")
            parent_mass = s.metadata.get("parent_mass", "")
            db = s.metadata.get("database", "")
            rows.append([name, adduct, smi, parent_mass, db])
        if rows:
            ref_df = pd.DataFrame(
                rows, columns=["name", "adduct", "smiles", "parent_mass", "database"]
            )

        # 分子图（受 highlight 影响）
        structure_smiles = None
        if "CanonicalSMILES" in structure_table.columns and len(structure_table) > 0:
            structure_smiles = structure_table.iloc[0]["CanonicalSMILES"]
        if structure_smiles and len(current_reference) > 0:
            ref_smi = getattr(current_reference[0], "metadata", {}).get("smiles") if hasattr(current_reference[0], "metadata") else None
            if ref_smi:
                ann_img, ref_img = plot_2_mol(structure_smiles, ref_smi, highlight=highlight)

    return {
        "status": "success",
        "structure_table": _dataframe_to_records(structure_table.round(3)),
        "reference_table": _dataframe_to_records(ref_df),
        "ann_img": ann_img,
        "ref_img": ref_img,
    }

# ----------------- 选 Structure：参考表 + 两张结构图 -----------------
@router.get("/reference-table")
async def reference_table(
    request: Request,
    spec_idx: int = Query(..., ge=0),
    structure_idx: int = Query(..., ge=0),
    highlight: bool = Query(True),   # ★ 新增
    tab_id: Optional[str] = Query(None),
):
    sid = _require_composite_sid(request, tab_id)
    df = store.load_df(sid)
    if df is None:
        return {"status": "error", "message": "No spectra"}
    if "Identified Spectrum" not in df.columns:
        return {"status": "error", "message": "Not identified yet."}
    if spec_idx >= len(df):
        return {"status": "error", "message": "Index out of range"}

    cur_spec: Spectrum = df["Identified Spectrum"][spec_idx]
    md = cur_spec.metadata or {}
    annotation: Optional[pd.DataFrame] = md.get("annotation")
    current_reference = _as_ref_list(md.get("reference"))

    # 找结构 SMILES
    if annotation is None or annotation.empty:
        structure_smiles = None
    else:
        try:
            structural_table = annotation.reset_index(drop=True)
            if structure_idx >= len(structural_table):
                return {"status": "error", "message": "structure_idx out of range"}
            if "CanonicalSMILES" not in structural_table.columns:
                structure_smiles = None
            else:
                structure_smiles = structural_table.loc[structure_idx, "CanonicalSMILES"]
        except Exception:
            structure_smiles = None

    # 参考整表
    rows = []
    for s in current_reference:
        if s is None or not hasattr(s, "metadata"):
            continue
        smi = s.metadata.get("smiles", "")
        name = s.metadata.get("compound_name", smi)
        adduct = s.metadata.get("adduct", "")
        parent_mass = s.metadata.get("parent_mass", "")
        db = s.metadata.get("database", "")
        rows.append([name, adduct, smi, parent_mass, db])
    ref_df = (
        pd.DataFrame(rows, columns=["name", "adduct", "smiles", "parent_mass", "database"])
        if rows
        else pd.DataFrame(columns=["name", "adduct", "smiles", "parent_mass", "database"])
    )

    # 分子图（受 highlight 影响）
    ann_img, ref_img = (None, None)
    if structure_smiles and len(current_reference) > 0:
        ref0 = current_reference[0]
        ref_smi = ref0.metadata.get("smiles") if hasattr(ref0, "metadata") else None
        if ref_smi:
            ann_img, ref_img = plot_2_mol(structure_smiles, ref_smi, highlight=highlight)

    return {
        "status": "success",
        "reference_table": _dataframe_to_records(ref_df),
        "ann_img": ann_img,
        "ref_img": ref_img,
    }

# ----------------- 选 Reference：双 Plotly + 两张分子图 -----------------
@router.get("/reference-select")
async def reference_select(
    request: Request,
    spec_idx: int = Query(..., ge=0),
    ref_idx: int = Query(..., ge=0),
    structure_smiles: Optional[str] = Query(None),
    highlight: bool = Query(True),   # ★ 新增
    tab_id: Optional[str] = Query(None),
):
    sid = _require_composite_sid(request, tab_id)
    df = store.load_df(sid)
    if df is None:
        return {"status": "error", "message": "No spectra"}
    if "Identified Spectrum" not in df.columns:
        return {"status": "error", "message": "Not identified yet."}
    if spec_idx >= len(df):
        return {"status": "error", "message": "Index out of range"}

    cur_spec: Spectrum = df["Identified Spectrum"][spec_idx]
    current_reference = _as_ref_list((cur_spec.metadata or {}).get("reference"))
    if ref_idx >= len(current_reference):
        return {"status": "error", "message": "ref_idx out of range"}

    ref_spec: Spectrum = current_reference[ref_idx]
    fig_spec = plot_2_spectrum(cur_spec, ref_spec, loss=False)
    fig_loss = plot_2_spectrum(cur_spec, ref_spec, loss=True)
    spec_json = fig_spec.to_json() if fig_spec is not None else None
    loss_json = fig_loss.to_json() if fig_loss is not None else None

    ann_img, ref_img = (None, None)
    ref_smi = ref_spec.metadata.get("smiles") if hasattr(ref_spec, "metadata") else None
    if structure_smiles and ref_smi:
        ann_img, ref_img = plot_2_mol(structure_smiles, ref_smi, highlight=highlight)

    return {
        "status": "success",
        "spectrum_plot": spec_json,
        "spectrum_loss_plot": loss_json,
        "ann_img": ann_img,
        "ref_img": ref_img,
    }

# ----------------- 保存 CSV -> ZIP -----------------
@router.post("/save")
async def save_results(request: Request, tab_id: Optional[str] = Query(None)):
    sid = _require_composite_sid(request, tab_id)
    df = store.load_df(sid)
    if df is None:
        return {"status": "error", "message": "No spectra"}

    st = store.read_state(sid)
    target_zip_file_name = (st or {}).get("target_zip_file_name") or f"{sid}_results.zip"

    out_dir = _default_results_dir(sid)
    file_list = []
    try:
        if "Identified Spectrum" not in df.columns:
            raise RuntimeError("Missing identified results")

        for idx, s in enumerate(df["Identified Spectrum"]):
            if s is None:
                # 允许个别失败，仍导出空表
                ann = pd.DataFrame(columns=["Title","MolecularFormula","CanonicalSMILES","InChIKey","DeepMass Score"])
                name = f"Spectrum_{idx}"
            else:
                name = s.metadata.get("compound_name", f"Spectrum_{idx}")
                ann = s.metadata.get("annotation")
                if ann is None:
                    ann = pd.DataFrame(columns=["Title","MolecularFormula","CanonicalSMILES","InChIKey","DeepMass Score"])

            csv_path = os.path.join(out_dir, f"{name}.csv")
            ann.to_csv(csv_path, index=True)
            file_list.append(csv_path)

        zip_path = os.path.join(out_dir, target_zip_file_name)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in file_list:
                z.write(p, arcname=os.path.basename(p))

        return {"status": "success", "file_path": zip_path, "message": "Saved successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Save failed: {e}"}

@router.get("/download")
async def download_result(path: str = Query(...)):
    # 简单防穿越：必须在 temp/result_csv_temp 下
    abspath = os.path.abspath(path)
    base = os.path.abspath(os.path.join("temp", "result_csv_temp"))
    if not abspath.startswith(base):
        raise HTTPException(403, "Forbidden")
    if not os.path.exists(abspath):
        raise HTTPException(404, "File not found")
    return FileResponse(abspath, filename=os.path.basename(abspath), media_type="application/zip")

# ----------------- 清空会话 -----------------
@router.post("/clear")
async def clear_session(request: Request, tab_id: Optional[str] = Query(None)):
    # 仅清 compound_identification 在“当前标签页”的数据；保留 cookie 的 session_id 供其他模块继续用
    sid = _require_composite_sid(request, tab_id)
    store.clear_session(sid)

    # 为避免立刻报 “Invalid/expired session”，在原始 Cookie sid 上维持活跃；不过不写任何业务数据
    raw_sid = _require_raw_sid(request)
    store.update_state(raw_sid, target_zip_file_name=None, last_accessed=time.time())

    store.set_progress(sid, total=0, done=0, status="idle")
    return {"status": "success", "message": "Cleared."}
