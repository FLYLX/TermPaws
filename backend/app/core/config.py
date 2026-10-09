import os
import secrets
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AnyUrl,
    BeforeValidator,
    EmailStr,
    HttpUrl,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",") if i.strip()]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Installed mode: ~/.termpaws/.env (set by app.cli); dev mode: repo root .env
        env_file=os.environ.get("TERMPAWS_ENV_FILE") or "../.env",
        env_ignore_empty=True,
        extra="ignore",
    )
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    FRONTEND_HOST: str = "http://localhost:5173"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_cors_origins(self) -> list[str]:
        return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS] + [
            self.FRONTEND_HOST
        ]

    PROJECT_NAME: str
    SENTRY_DSN: HttpUrl | None = None
    SQLITE_DATABASE_URL: str | None = "sqlite:///./sql_app.db"
    BACKEND_LOG_DIR: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return self.SQLITE_DATABASE_URL or "sqlite:///./sql_app.db"

    SMTP_TLS: bool = True
    SMTP_SSL: bool = False
    SMTP_PORT: int = 587
    SMTP_HOST: str | None = None
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    EMAILS_FROM_EMAIL: EmailStr | None = None
    EMAILS_FROM_NAME: str | None = None

    @model_validator(mode="after")
    def _set_default_emails_from(self) -> Self:
        if not self.EMAILS_FROM_NAME:
            self.EMAILS_FROM_NAME = self.PROJECT_NAME
        return self

    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 48

    @computed_field  # type: ignore[prop-decorator]
    @property
    def emails_enabled(self) -> bool:
        return bool(self.SMTP_HOST and self.EMAILS_FROM_EMAIL)

    EMAIL_TEST_USER: EmailStr = "test@example.com"
    FIRST_SUPERUSER: EmailStr = "admin@example.com"
    FIRST_SUPERUSER_PASSWORD: str = ""

    CHROMA_PERSIST_DIR: str = str(BACKEND_DIR / "chroma_data")
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_ALLOW_REMOTE_LOAD: bool = False
    KNOWLEDGE_BASE_DIR: str = str(BACKEND_DIR / "knowledge")
    ROBOT_PLUGIN_ENABLED: bool = True
    ROBOT_BRIDGE_URL: str = "http://robot-bridge:7000"
    ROBOT_BACKEND_URL: str = "http://backend:8000"
    ROBOT_BRIDGE_SHARED_SECRET: str | None = None
    ROBOT_BRIDGE_AUTO_RELOAD: bool = True
    ROBOT_BRIDGE_HOST: str = "0.0.0.0"
    ROBOT_BRIDGE_PORT: int = 7000
    ROBOT_BACKEND_DISPATCH_TIMEOUT_SECONDS: float = 180.0
    ROBOT_BACKEND_DISPATCH_QUEUE_SIZE: int = 50
    ROBOT_BACKEND_DISPATCH_WORKERS: int = 4
    ROBOT_BACKEND_JOB_TIMEOUT_SECONDS: float = 240.0
    # 单次 LLM 调用超时（连接+流式读取共用）：中继卡顿按此快速失败，
    # 避免一次卡顿吃掉整个轮次预算。可用环境变量覆盖。
    CHAT_LLM_TIMEOUT_SECONDS: int = 60
    ROBOT_BACKEND_DISPATCH_HARD_TIMEOUT_SECONDS: float = 300.0
    AGENT_TURN_TIMEOUT_SECONDS: float = 900.0
    AGENT_STATE_STORE_PATH: str = str(BACKEND_DIR.parent / ".runtime" / "agent_state.db")
    AGENT_WATCHDOG_INTERVAL_SECONDS: float = 30.0
    AGENT_WATCHDOG_STALE_SECONDS: float = 900.0
    AGENT_WATCHDOG_BLOCKED_STALE_SECONDS: float = 3600.0
    AGENT_WATCHDOG_IDLE_RESUME_SECONDS: float = 180.0
    AGENT_WATCHDOG_WAITING_JOB_STALE_SECONDS: float = 21600.0
    ROBOT_CONVERSATION_MEMORY_DIR: str = str(BACKEND_DIR.parent / ".runtime" / "robot_conversation_memory")
    ROBOT_CONVERSATION_MEMORY_MAX_BYTES: int = 1024 * 1024
    DAEMON_HOST_PORT: int | None = 39999
    DAEMON_PUBLIC_URL: str | None = None
    DAEMON_PUBLIC_HOST: str | None = None
    DAEMON_PUBLIC_PORT: int | None = None
    DAEMON_PUBLIC_WS_SCHEME: Literal["ws", "wss"] = "ws"
    TERMINAL_WS_DEFAULT_HOST: str = "0.0.0.0"
    TERMINAL_WS_DEFAULT_PORT_START: int = 32000
    TERMINAL_WS_DEFAULT_PORT_END: int = 32111
    TERMINAL_WS_DEFAULT_HEARTBEAT_INTERVAL_SECONDS: float = 30.0

    @model_validator(mode="after")
    def _apply_robot_bridge_defaults(self) -> Self:
        # Bridge is always embedded in the backend process; self-reference URLs
        # point at the same process.
        port = os.environ.get("BACKEND_PORT", "8000")
        self.ROBOT_BRIDGE_URL = f"http://127.0.0.1:{port}/robot-bridge"
        self.ROBOT_BACKEND_URL = f"http://127.0.0.1:{port}"
        if self.ROBOT_BRIDGE_HOST == "0.0.0.0":
            self.ROBOT_BRIDGE_HOST = "127.0.0.1"
        return self

    # Security checks commented out as requested
    # def _check_default_secret(self, var_name: str, value: str | None) -> None:
    #     if value == "changethis":
    #         message = (
    #             f'The value of {var_name} is "changethis", '
    #             "for security, please change it, at least for deployments."
    #         )
    #         if self.ENVIRONMENT == "local":
    #             warnings.warn(message, stacklevel=1)
    #         else:
    #             raise ValueError(message)

    # @model_validator(mode="after")
    # def _enforce_non_default_secrets(self) -> Self:
    #     self._check_default_secret("SECRET_KEY", self.SECRET_KEY)
    #     self._check_default_secret(
    #         "FIRST_SUPERUSER_PASSWORD", self.FIRST_SUPERUSER_PASSWORD
    #     )

    #     return self


settings = Settings()  # type: ignore
