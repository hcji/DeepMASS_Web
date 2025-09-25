import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import jwt  # pip install PyJWT
from backend.load_config import GLOBAL_CONFIG

JWT_SECRET = GLOBAL_CONFIG["jwt_secret"]

JWT_ALG = "HS256"
JWT_EXPIRE_MINUTES = 24 * 60 *3  # 3天

def create_access_token(data: Dict[str, Any], expires_minutes: int = JWT_EXPIRE_MINUTES) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
