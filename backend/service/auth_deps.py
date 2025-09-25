from fastapi import Request, HTTPException, status, Depends
from typing import Optional, Dict, Any
from backend.service.jwt_tools import decode_token

ACCESS_COOKIE_NAME = "access_token"

def get_current_user(request: Request) -> Dict[str, Any]:
    token: Optional[str] = request.cookies.get(ACCESS_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return payload

def require_user(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return user
