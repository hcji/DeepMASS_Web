from fastapi import APIRouter, Response, HTTPException, status, Form, Depends
from backend.service.user_service import UserService
from backend.service.jwt_tools import create_access_token
from backend.service.auth_deps import ACCESS_COOKIE_NAME, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])

# —— 登录 ——（表单提交）
@router.post("/login")
async def login(
    response: Response,
    contact_info: str = Form(...),
    password: str = Form(...)
):
    ok = UserService().auth_login(contact_info, password)
    if not ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password.")

    token = create_access_token({"contact_info": contact_info})
    # 设置 HttpOnly Cookie
    response.set_cookie(
        key=ACCESS_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,          # 如果站点是 https，改 True
        samesite="Lax",        # 如跨站嵌入需调整
        max_age=60*60*24*3,      # 3 天
        path="/",
    )
    return {"status": "success", "message": "login ok"}

# —— 退出登录 ——（清空 Cookie）
@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(ACCESS_COOKIE_NAME, path="/")
    return {"status": "success", "message": "logout ok"}

# —— 会话自检（前端守卫用）——
@router.get("/me")
async def me(user = Depends(get_current_user)):
    return {"status": "success", "user": user}
