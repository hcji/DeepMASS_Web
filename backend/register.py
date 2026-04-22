from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.service.user_service import UserService
from backend.service.email_service import EmailSenderService
from backend.dao.database import get_db
from backend.utils.auth_router import router as auth_router
from backend.anal_sear import router_anal_sear
from backend.comp_ident import router_comp_ident
from datetime import datetime
app = FastAPI()

# 允许所有来源的CORS(解决跨域问题)
origins = [
    # 正式域名
    "http://deepmass.cn",
    "http://deepmass.cn:8000",
    "https://deepmass.cn",
    "https://deepmass.cn:8000",

    # www
    "http://www.deepmass.cn",
    "http://www.deepmass.cn:8000",
    "https://www.deepmass.cn",
    "https://www.deepmass.cn:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册
app.include_router(auth_router)
app.include_router(router_anal_sear.router)
app.include_router(router_comp_ident.router)

# 数据模型
class Register(BaseModel):
    contact_info: str
    vercode: str
    passwd: str
    confirmPassword: str
    name: str
    agreement: str


# 注册接口
@app.post("/register")
async def register(reg: Register, db: Session = Depends(get_db)):
    if reg.passwd != reg.confirmPassword:
        raise HTTPException(status_code=400, detail="两次输入的密码不一致")
    
    ok, msg = UserService(db_session=db).user_register(
        reg.contact_info, reg.passwd, reg.name, reg.vercode
    )

    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    return {"msg": msg}


# 密码重置接口使用的数据模型
class ResetPassword(BaseModel):
    contact_info: str
    vercode: str
    passwd: str
    confirmPassword: str
@app.post("/reset_password")
def reset_password(reg: ResetPassword, db: Session = Depends(get_db)):
    if reg.passwd != reg.confirmPassword:
        raise HTTPException(status_code=400, detail="两次输入的密码不一致")

    service = UserService(db_session=db)
    ok, msg = service.reset_password(reg.contact_info, reg.passwd, reg.vercode)

    if not ok:
        if msg == "用户不存在":
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)

    return {"msg": msg}

# 发送验证码接口
@app.get("/sendmail")
def sendmail(email: str, db: Session = Depends(get_db)):
    EmailSenderService(db_session=db).send_captcha(email)
    return JSONResponse({"msg": "发送成功"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.register:app", host="0.0.0.0", workers=2)
