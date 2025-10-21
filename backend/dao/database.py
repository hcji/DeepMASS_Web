from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.load_config import GLOBAL_CONFIG
Base = declarative_base()

# MySQL 连接配置
SQLALCHEMY_DATABASE_URL =  GLOBAL_CONFIG["database"]

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 依赖注入函数
def get_db():
    db = SessionLocal()
    # print("创建 SessionLocal() 对象：", id(db))
    try:
        yield db
    finally:
        db.close()
        # print("关闭 SessionLocal() 对象：", id(db))
