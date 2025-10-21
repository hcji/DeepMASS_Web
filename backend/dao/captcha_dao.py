# 导入SQLAlchemy模块
# 导入SQLAlchemy模块
from datetime import datetime

from sqlalchemy import select

from backend.dao.basedao import BaseDao
from backend.entity.captcha import Code
from sqlalchemy.orm import Session

# 定义UserDao类，继承自BaseDao
class CaptchaDAO(BaseDao):
    def __init__(self, db_session: Session = None):
        super().__init__(db_session=db_session)

    def insert_log(self, email, code):
        """插入或更新验证码记录"""
        now = datetime.now().timestamp()
        # 查询是否已有记录
        existing = self.session.query(Code).filter_by(contact_info=email).first()
        if existing:
            # 更新已有记录
            existing.verify_code = code
            existing.verify_time = now
            print(f"[DEBUG] 更新验证码：{email} -> {code}")
        else:
            # 新建新记录
            new_code = Code(contact_info=email, verify_code=code, verify_time=now)
            self.session.add(new_code)
            print(f"[DEBUG] 新建验证码：{email} -> {code}")
        # self.session.commit()

    def query_captcha_code(self, email):
        res_code = (
            self.session.execute(select(Code).where(Code.contact_info.in_([email])))
            .scalars()
            .one_or_none()
        )
        return res_code



    def delete_captcha(self, email):
        self.session.query(Code).filter_by(contact_info=email).delete()
        # self.session.commit()

    def commit(self):
        self.session.commit()
