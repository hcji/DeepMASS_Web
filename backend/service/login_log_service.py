from backend.dao.login_log_dao import LoginLogDAO
from sqlalchemy.orm import Session

class LoginLogService:
    def __init__(self, db_session: Session = None):
        self.dao = LoginLogDAO(db_session=db_session)

    def insert_login_log(self, email):
        self.dao.insert_log(email)
