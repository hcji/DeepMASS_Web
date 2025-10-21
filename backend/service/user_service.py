from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.dao.user_dao import UserDAO
from backend.service.code_service import CaptchaService
from backend.service.login_log_service import LoginLogService


class UserService:
    def __init__(self, db_session: Session = None):
        self.session = db_session
        self.dao = UserDAO(db_session=db_session)
    def auth_login(self, email, password):
        # 验证密码
        login_flag = self.dao.login(email, password)
        print(f"=============={login_flag}==================")
        # 登录成功，留下记录
        if login_flag:
            LoginLogService().insert_login_log(email)
        return login_flag
    def user_register(self, email, password, name, captcha):
        validate_flag = CaptchaService(db_session=self.dao.session).validate(email, captcha)
        if validate_flag:
            return {"msg": "验证码错误或已过期"}
        if self.dao.query_email_exist(email):
            return {"msg": "用户已经注册"}
        self.dao.add_user(email, password, name)
        return {"msg": "注册成功"}

    def reset_password(self, email, new_password, captcha):
        """
        重置密码接口逻辑：
          1. 验证验证码是否正确
          2. 检查用户是否存在
          3. 更新用户密码
        """
        validate_flag = CaptchaService(db_session=self.dao.session).validate(email, captcha)
        if validate_flag:
            return {"msg": "验证码错误或已过期"}
        if not self.dao.query_email_exist(email):
            return {"msg": "用户不存在"}
        update_flag = self.dao.update_password(email, new_password)
        if update_flag:
            return {"msg": "密码重置成功"}
        else:
            return {"msg": "密码重置失败"}