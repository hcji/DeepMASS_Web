from backend.dao.database import SessionLocal
from sqlalchemy.orm import Session

class BaseDao:
    def __init__(self, db_session: Session = None):
        # 如果传入了 db_session，就使用它；否则新建一个 SessionLocal
        self.session = db_session or SessionLocal()