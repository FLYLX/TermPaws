from sqlmodel import Session, create_engine, select
from app.core.config import settings
from app.models import User, UserCreate
import uuid

# Local import to avoid circular imports
from app import crud

# Create engine with SQLite-specific configuration
_is_sqlite = "sqlite" in str(settings.SQLALCHEMY_DATABASE_URI)
engine = create_engine(
    str(settings.SQLALCHEMY_DATABASE_URI),
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    # SSE 长连接 + 并发聊天 + watchdog 下，默认 5+10 连接池会在突发流量时
    # 耗尽（"QueuePool limit reached" → 请求 500 → 消息无声丢失）。
    pool_size=30,
    max_overflow=60,
    pool_timeout=60,
)

# Register UUID type for SQLite
from sqlalchemy import types
import sqlalchemy
if sqlalchemy.__version__ >= "2.0":
    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def _connect_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=15000")
        cursor.close()


# make sure all SQLModel models are imported (app.models) before initializing DB
# otherwise, SQLModel might fail to initialize relationships properly
# for more details: https://github.com/fastapi/full-stack-fastapi-template/issues/28


def init_db(session: Session) -> None:
    if not settings.FIRST_SUPERUSER_PASSWORD:
        # 无初始密码模式：不预建超管，等待首个访问者通过 /utils/setup 设置
        return
    # Tables should be created with Alembic migrations
    # But if you don't want to use migrations, create
    # the tables un-commenting the next lines
    # from sqlmodel import SQLModel

    # This works because the models are already imported and registered from app.models
    # SQLModel.metadata.create_all(engine)

    user = session.exec(
        select(User).where(User.email == settings.FIRST_SUPERUSER)
    ).first()
    if not user:
        user_in = UserCreate(
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_superuser=True,
        )
        user = crud.create_user(session=session, user_create=user_in)
