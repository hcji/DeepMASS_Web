import os
import json
import time
import uuid
import pickle
from typing import Optional, Dict, Any

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
        """
        重要策略变更：
        - 如果浏览器里已有 cookie 的 sid，则**永远复用**它（不再因为本 namespace 还没有状态/文件而换新 sid）。
        - 若本 namespace 下还没有 state，则写入一个最小状态（保证 require_session 可通过）。
        """
        sid = request.cookies.get(self.cookie_name)
        if sid:
            # 本命名空间下若没有状态，就初始化一个轻量状态；否则仅续期
            if not self.r.exists(self._key_state(sid)):
                self._write_state(sid, {"last_accessed": time.time()})
            else:
                self._touch_redis(sid)
            # 统一续期 cookie
            response.set_cookie(self.cookie_name, sid, httponly=True, samesite="lax", max_age=self.ttl)
            return sid

        # 没有 cookie：新建 sid，并初始化本命名空间的轻量状态
        sid = str(uuid.uuid4())
        self._write_state(sid, {"last_accessed": time.time()})
        response.set_cookie(self.cookie_name, sid, httponly=True, samesite="lax", max_age=self.ttl)
        return sid

    def require_session(self, request: Request) -> str:
        """
        只校验浏览器是否带了 cookie 的 sid，并确保本命名空间有 state（如没有，视为过期/无效）。
        """
        sid = request.cookies.get(self.cookie_name)
        if not sid:
            raise HTTPException(400, "Missing session_id")
        if not (self.r.exists(self._key_state(sid)) or os.path.exists(self._df_path(sid))):
            # 注意：这里用的是“本命名空间”的状态/文件
            raise HTTPException(400, "Invalid/expired session")
        self._touch_redis(sid)
        return sid

    # --------- 轻量状态 ---------
    def read_state(self, sid: str) -> dict:
        raw = self.r.get(self._key_state(sid))
        return json.loads(raw.decode("utf-8")) if raw else {}

    def update_state(self, sid: str, **fields):
        """
        读出 state（可能为空），合并传入字段，并刷新 last_accessed。
        """
        st = self.read_state(sid) or {}
        if fields:
            st.update(fields)
        st["last_accessed"] = time.time()
        self._write_state(sid, st)
        return st

    # --------- 进度 ---------
    def set_progress(self, sid: str, *, total: int, done: int, status: str, message: Optional[str] = None):
        payload: Dict[str, Any] = {
            "total": int(total),
            "done": int(done),
            "status": str(status),
            "ts": time.time(),
        }
        if message is not None:
            payload["message"] = str(message)
        self.r.setex(self._key_prog(sid), self.ttl, json.dumps(payload).encode("utf-8"))

    def get_progress(self, sid: str) -> dict:
        raw = self.r.get(self._key_prog(sid))
        return json.loads(raw.decode("utf-8")) if raw else {"status": "success", "total": 0, "done": 0, "message": ""}

    # --------- 大数据（始终落盘 pickle） ---------
    def save_df(self, sid: str, df):
        """
        原子写入 DataFrame 到 pickle，先写临时文件再 os.replace 到目标路径，
        避免并发读写时出现半写文件导致的 pickle 错误。
        """
        p = self._df_path(sid)
        tmp_path = p + f".tmp.{uuid.uuid4().hex}"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, p)
            self.update_state(sid, last_accessed=time.time())
        finally:
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
