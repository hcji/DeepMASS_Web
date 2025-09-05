import os
import json
import time
import uuid
import pickle
from typing import Optional

import redis  # pip install redis
from fastapi import Request, Response, HTTPException


class Store:
    """
    Redis 里只放会话状态/进度（小），
    大数据 DataFrame 一律落盘为 pickle（spectra.pkl）。
    """

    def __init__(
        self,
        *,
        namespace: str,
        ttl_seconds: int = 60 * 60 * 24 * 3,
        redis_host: str = "127.0.0.1",
        redis_port: int = 6379,
        redis_db: int = 0,
        base_dir: str = "temp/session_store",
        cookie_name: str = "session_id",
    ):
        self.namespace = namespace
        self.ttl = ttl_seconds
        self.cookie_name = cookie_name

        # Redis（会话状态与进度）
        self.r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)

        # 磁盘（大对象）
        self.base_dir = os.path.join(base_dir, namespace)
        os.makedirs(self.base_dir, exist_ok=True)

    # --------- 路径 ---------
    def _session_dir(self, sid: str) -> str:
        d = os.path.join(self.base_dir, sid)
        os.makedirs(d, exist_ok=True)
        return d

    def _df_path(self, sid: str) -> str:
        return os.path.join(self._session_dir(sid), "spectra.pkl")

    # --------- Redis Key ---------
    def _key_state(self, sid: str) -> str:
        return f"{self.namespace}:session:{sid}:state"

    def _key_prog(self, sid: str) -> str:
        return f"{self.namespace}:session:{sid}:progress"

    # --------- 会话 ---------
    def get_or_create_session(self, request: Request, response: Response) -> str:
        sid = request.cookies.get(self.cookie_name)
        if sid and (self.r.exists(self._key_state(sid)) or os.path.exists(self._df_path(sid))):
            self._touch_redis(sid)
            response.set_cookie(self.cookie_name, sid, httponly=True, samesite="lax", max_age=self.ttl)
            return sid

        sid = str(uuid.uuid4())
        self._write_state(sid, {"last_accessed": time.time()})
        response.set_cookie(self.cookie_name, sid, httponly=True, samesite="lax", max_age=self.ttl)
        return sid

    def require_session(self, request: Request) -> str:
        sid = request.cookies.get(self.cookie_name)
        if not sid:
            raise HTTPException(400, "Missing session_id")
        if not (self.r.exists(self._key_state(sid)) or os.path.exists(self._df_path(sid))):
            raise HTTPException(400, "Invalid/expired session")
        self._touch_redis(sid)
        return sid

    # --------- 轻量状态 ---------
    def read_state(self, sid: str) -> dict:
        raw = self.r.get(self._key_state(sid))
        return json.loads(raw.decode("utf-8")) if raw else {}

    def update_state(self, sid: str, **fields):
        st = self.read_state(sid)
        st.update(fields)
        st["last_accessed"] = time.time()
        self._write_state(sid, st)

    # --------- 进度 ---------
    def set_progress(self, sid: str, *, total: int, done: int, status: str):
        payload = {"total": total, "done": done, "status": status, "ts": time.time()}
        self.r.setex(self._key_prog(sid), self.ttl, json.dumps(payload).encode("utf-8"))

    def get_progress(self, sid: str) -> dict:
        raw = self.r.get(self._key_prog(sid))
        return json.loads(raw.decode("utf-8")) if raw else {"status": "success", "total": 0, "done": 0}

    # --------- 大数据（始终落盘 pickle） ---------
    # def save_df(self, sid: str, df):
    #     p = self._df_path(sid)
    #     with open(p, "wb") as f:
    #         pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    #     self.update_state(sid, last_accessed=time.time())


    def save_df(self, sid: str, df):
        """
        原子写入 DataFrame 到 pickle，先写临时文件再 os.replace 到目标路径，
        避免并发读写时出现半写文件导致的 pickle 错误。
        """
        p = self._df_path(sid)
        tmp_path = p + f".tmp.{uuid.uuid4().hex}"
        # 确保目录存在
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            # 使用 NamedTemporaryFile 可能在 Windows 上有权限问题，故先打开普通文件
            with open(tmp_path, "wb") as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            # 原子替换（在同一文件系统上是原子的）
            os.replace(tmp_path, p)
            self.update_state(sid, last_accessed=time.time())
        finally:
            # 清理残留 tmp（如果有）
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def load_df(self, sid: str):
        p = self._df_path(sid)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            df = pickle.load(f)
        self.update_state(sid, last_accessed=time.time())
        return df

    # --------- 清理 ---------
    def clear_session(self, sid: str):
        # 清磁盘
        sdir = self._session_dir(sid)
        if os.path.exists(sdir):
            import shutil
            shutil.rmtree(sdir, ignore_errors=True)
        # 清 Redis 状态/进度（Cookie 保留）
        self.r.delete(self._key_state(sid))
        self.r.delete(self._key_prog(sid))

    # --------- Redis 工具 ---------
    def _write_state(self, sid: str, st: dict):
        self.r.setex(self._key_state(sid), self.ttl, json.dumps(st).encode("utf-8"))

    def _touch_redis(self, sid: str):
        st = self.read_state(sid) or {"last_accessed": time.time()}
        st["last_accessed"] = time.time()
        self._write_state(sid, st)
