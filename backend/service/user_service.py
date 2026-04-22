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
            try:
                LoginLogService().insert_login_log(email)
            except Exception as e:
                print(f"[WARN] 登录日志写入失败: {e}")
        return login_flag

    def user_register(self, email, password, name, captcha):
        validate_flag = CaptchaService(db_session=self.dao.session).validate(email, captcha)
        if self.dao.query_email_exist(email):
            return False, "用户已经注册"
        
        if validate_flag:
            return False, "验证码错误或已过期"

        try:
            self.dao.add_user(email, password, name)
            return True, "注册成功"
        except Exception as e:
            print(f"[ERROR] 注册失败: {e}")
            return False, "注册失败"

    def reset_password(self, email, new_password, captcha):
        # 1. 检查用户是否存在
        if not self.dao.query_email_exist(email):
            return False, "该邮箱未注册，请先注册"

        # 2. 验证验证码
        validate_flag = CaptchaService(db_session=self.dao.session).validate(email, captcha)
        if validate_flag:
            return False, "验证码错误或已过期"

        # 3. 更新密码
        update_flag = self.dao.update_password(email, new_password)
        if update_flag:
            return True, "密码重置成功"
        else:
            return False, "密码重置失败"