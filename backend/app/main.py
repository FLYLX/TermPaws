import logging
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.core.config import settings
from app.services import initialize_daemon_connections
from app.services.plugins import plugin_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # uvicorn 先配置 root 导致 basicConfig 变 no-op（root 停留 WARNING），
    # app.* 的 INFO 诊断日志（watchdog/sweep/resume 等）全部不可见。
    # force=True 强制统一配置，恢复运维可观测性。
    force=True,
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await plugin_manager.startup(app)

    # SPA catch-all 必须在所有插件路由（robot-bridge 等）注册之后再挂载，
    # 否则按注册顺序会优先命中 catch-all 吞掉插件路由
    _mount_frontend()

    initialize_daemon_connections()

    from app.services.agent.mcp import mcp_server_manager

    logger.info("[App] Starting MCP servers...")
    await mcp_server_manager.start_all()
    logger.info("[App] MCP servers started")

    from app.services.agent.state_restore import restore_agent_state
    from app.services.agent.task_watchdog import agent_task_watchdog

    restore_agent_state()
    agent_task_watchdog.start()

    from app.services.agent.scheduled_tasks import scheduled_task_manager

    scheduled_task_manager.start()

    yield

    scheduled_task_manager.stop()
    agent_task_watchdog.stop()

    await plugin_manager.shutdown(app)

    logger.info("[App] Stopping MCP servers...")
    await mcp_server_manager.stop_all()
    logger.info("[App] MCP servers stopped")


def custom_generate_unique_id(route: APIRoute) -> str:
    tag = route.tags[0] if route.tags else "default"
    return f"{tag}-{route.name}"


if settings.SENTRY_DSN and settings.ENVIRONMENT != "local":
    sentry_sdk.init(dsn=str(settings.SENTRY_DSN), enable_tracing=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    try:
        body = await request.body()
        logger.error(
            "[422] %s %s errors=%s body=%s",
            request.method,
            request.url.path,
            exc.errors(),
            body[:2000],
        )
    except Exception:
        logger.error("[422] %s %s errors=%s", request.method, request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# Set all CORS enabled origins
if settings.ENVIRONMENT == "local":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
elif settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)


def _mount_frontend() -> None:
    from pathlib import Path

    from fastapi.responses import FileResponse

    candidates = [
        Path(__file__).resolve().parent / "static",
        Path(__file__).resolve().parent.parent.parent / "frontend" / "dist",
    ]
    try:
        from termpaws_frontend import DIST_DIR

        candidates.insert(0, Path(DIST_DIR))
    except ImportError:
        pass
    dist = next((path for path in candidates if (path / "index.html").exists()), None)
    if dist is None:
        logger.info("[App] Frontend dist not found, API-only mode")
        return

    index_html = dist / "index.html"
    dist_root = dist.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path.startswith(("api/", "docs", "redoc", "robot-bridge/")):
            return None
        target = (dist / full_path).resolve()
        if target.is_file() and str(target).startswith(str(dist_root)):
            # hashed assets are immutable; everything else (incl. index.html) no-cache
            immutable = full_path.startswith("assets/")
            return FileResponse(
                target,
                headers={
                    "Cache-Control": "public, max-age=31536000, immutable"
                    if immutable
                    else "no-cache"
                },
            )
        return FileResponse(index_html, headers={"Cache-Control": "no-cache"})

    logger.info("[App] Frontend mounted from %s", dist)




# 注意：已移除每次请求后更新daemon连接池表的中间件
# 改为在start和stop操作中只同步特定item的连接表，减少不必要的数据传输
# 仍保留定时同步功能
