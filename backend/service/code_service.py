from datetime import datetime
from sqlalchemy.orm import Session
from backend.dao.captcha_dao import CaptchaDAO


class CaptchaService:
    def __init__(self, db_session: Session=None):
        super().__init__()
        self.dao = CaptchaDAO(db_session=db_session)

    def validate(self, email: str, code: str) -> bool:
        self.dao.session.expire_all()
        res_code = self.dao.query_captcha_code(email)
        now = datetime.now().timestamp()

        if res_code is None:
            print(f"[DEBUG] 无验证码记录 for {email}")  # 调试
            return True
        if code != res_code.verify_code:
            print(f"[DEBUG] 验证码不匹配: 输入{code}, 记录{res_code.verify_code}")
            return True
        time_diff = now - res_code.verify_time  # 假设 verify_time 是 float 时间戳
        if time_diff >= 60 * 10:
            print(f"[DEBUG] 验证码过期: {time_diff}s")
            self.dao.delete_captcha(email)  # 清理过期记录
            self.dao.session.commit()
            return True

        # 验证通过后删除验证码并提交
        self.dao.delete_captcha(email)
        self.dao.session.commit()
        print(f"[DEBUG] 验证码验证成功 for {email}")
        return False

