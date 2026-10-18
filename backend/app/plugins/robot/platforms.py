from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.models import Robot

from .contracts import RobotInboundMessage, RobotReplyTarget

DEFAULT_REPLY_CONTEXT_WINDOW_SECONDS = 50
MIN_REPLY_CONTEXT_WINDOW_SECONDS = 0
MAX_REPLY_CONTEXT_WINDOW_SECONDS = 3600
MAX_REPLY_REFERENCE_TEXT_LENGTH = 600


class RobotPlatformFieldPublic(BaseModel):
    key: str
    label: str
    required: bool = True
    secret: bool = False


class RobotPlatformPublic(BaseModel):
    id: str
    label: str
    provider: str = "nonebot2"
    description: str
    fields: list[RobotPlatformFieldPublic] = Field(default_factory=list)


@dataclass(frozen=True)
class RobotPlatformFieldSpec:
    key: str
    label: str
    required: bool = True
    secret: bool = False

    def to_public(self) -> RobotPlatformFieldPublic:
        return RobotPlatformFieldPublic(
            key=self.key,
            label=self.label,
            required=self.required,
            secret=self.secret,
        )


BuildInitCallback = Callable[[dict[str, Any], dict[str, Any]], None]
IdentityFromRobotCallback = Callable[[dict[str, Any]], str]
IdentityFromBotCallback = Callable[[Any], str]

@dataclass(frozen=True)
class RobotPlatformSpec:
    id: str
    label: str
    description: str
    fields: tuple[RobotPlatformFieldSpec, ...]
    adapter_module: str
    adapter_class: str
    build_init: BuildInitCallback
    robot_identity: IdentityFromRobotCallback
    bot_identity: IdentityFromBotCallback

    def to_public(self) -> RobotPlatformPublic:
        return RobotPlatformPublic(
            id=self.id,
            label=self.label,
            description=self.description,
            fields=[field.to_public() for field in self.fields],
        )


def _normalize_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_int(value: Any, *, default: int | None = None) -> int | None:
    normalized = _normalize_string(value)
    if normalized is None:
        return default
    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid integer value `{normalized}`") from exc


def _ensure_required_fields(
    platform: RobotPlatformSpec,
    credentials: dict[str, Any],
) -> None:
    missing = [
        field.label
        for field in platform.fields
        if field.required and _normalize_string(credentials.get(field.key)) is None
    ]
    if missing:
        raise ValueError(
            f"Platform `{platform.id}` requires: {', '.join(missing)}",
        )


def _append_config_item(
    init_kwargs: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    init_kwargs.setdefault(key, []).append(value)


def _merge_unique_value(
    init_kwargs: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    if value is None:
        return
    existing = init_kwargs.get(key)
    if existing is None:
        init_kwargs[key] = value
        return
    if existing != value:
        raise ValueError(f"Conflicting shared adapter config for `{key}`")


def _merge_mapping_item(
    init_kwargs: dict[str, Any],
    key: str,
    item_key: str,
    item_value: Any,
) -> None:
    if item_value is None:
        return
    mapping = init_kwargs.setdefault(key, {})
    existing = mapping.get(item_key)
    if existing is not None and existing != item_value:
        raise ValueError(f"Conflicting shared adapter config for `{key}.{item_key}`")
    mapping[item_key] = item_value


def _credential_identity(platform_id: str, value: Any) -> str:
    normalized = _normalize_string(value)
    if normalized is None:
        raise ValueError(f"Platform `{platform_id}` identity is missing")
    return f"{platform_id}:{normalized}"


def _compound_identity(platform_id: str, *parts: Any) -> str:
    normalized = [str(part).strip() for part in parts]
    return f"{platform_id}:{'|'.join(normalized)}"


def _normalize_endpoint(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _build_qq_init(init_kwargs: dict[str, Any], runtime_config: dict[str, Any]) -> None:
    from nonebot.adapters.qq.config import BotInfo

    try:
        from nonebot.adapters.qq.config import Intents
    except ImportError:
        Intents = None

    credentials = runtime_config["credentials"]
    options = runtime_config["options"]
    bot_kwargs: dict[str, Any] = {
        "id": str(credentials["app_id"]),
        "token": str(credentials["bot_token"]),
        "secret": str(credentials["app_secret"]),
        "use_websocket": _normalize_bool(options.get("use_websocket"), default=True),
    }

    bot_fields = getattr(BotInfo, "model_fields", None) or getattr(BotInfo, "__fields__", {})
    if Intents is not None and "intents" in bot_fields:
        intent_kwargs = {
            "guild_messages": _normalize_bool(options.get("guild_messages"), default=True),
            "direct_message": _normalize_bool(options.get("direct_message"), default=True),
            "c2c_group_at_messages": _normalize_bool(
                options.get("c2c_group_at_messages"),
                default=True,
            ),
        }
        intent_fields = getattr(Intents, "model_fields", None) or getattr(Intents, "__fields__", {})
        bot_kwargs["intents"] = Intents(
            **{
                key: value
                for key, value in intent_kwargs.items()
                if key in intent_fields
            }
        )

    _append_config_item(
        init_kwargs,
        "qq_bots",
        BotInfo(**bot_kwargs),
    )


def _build_telegram_init(
    init_kwargs: dict[str, Any],
    runtime_config: dict[str, Any],
) -> None:
    from nonebot.adapters.telegram.config import BotConfig

    credentials = runtime_config["credentials"]
    _append_config_item(
        init_kwargs,
        "telegram_bots",
        BotConfig(
            token=str(credentials["bot_token"]),
        ),
    )


def _build_discord_init(
    init_kwargs: dict[str, Any],
    runtime_config: dict[str, Any],
) -> None:
    from nonebot.adapters.discord.config import BotInfo

    credentials = runtime_config["credentials"]
    _append_config_item(
        init_kwargs,
        "discord_bots",
        BotInfo(
            token=str(credentials["bot_token"]),
        ),
    )


def _build_onebot_v11_init(
    init_kwargs: dict[str, Any],
    runtime_config: dict[str, Any],
) -> None:
    credentials = runtime_config["credentials"]
    _merge_unique_value(
        init_kwargs,
        "onebot_access_token",
        _normalize_string(credentials.get("access_token")),
    )
    _merge_unique_value(
        init_kwargs,
        "onebot_secret",
        _normalize_string(credentials.get("secret")),
    )


def _build_onebot_v12_init(
    init_kwargs: dict[str, Any],
    runtime_config: dict[str, Any],
) -> None:
    credentials = runtime_config["credentials"]
    _merge_unique_value(
        init_kwargs,
        "onebot_access_token",
        _normalize_string(credentials.get("access_token")),
    )
    init_kwargs.setdefault("onebot_ws_urls", set()).add(str(credentials["ws_url"]))
    _merge_mapping_item(
        init_kwargs,
        "onebot_api_roots",
        str(credentials["self_id"]),
        _normalize_string(credentials.get("api_root")),
    )


def _build_satori_init(
    init_kwargs: dict[str, Any],
    runtime_config: dict[str, Any],
) -> None:
    from nonebot.adapters.satori.config import ClientInfo

    credentials = runtime_config["credentials"]
    _append_config_item(
        init_kwargs,
        "satori_clients",
        ClientInfo(
            host=str(credentials.get("host") or "localhost"),
            port=int(credentials["port"]),
            path=str(credentials.get("path") or ""),
            token=_normalize_string(credentials.get("token")),
        ),
    )


def _build_red_init(init_kwargs: dict[str, Any], runtime_config: dict[str, Any]) -> None:
    from nonebot.adapters.red.config import BotInfo

    credentials = runtime_config["credentials"]
    _append_config_item(
        init_kwargs,
        "red_bots",
        BotInfo(
            host=str(credentials.get("host") or "localhost"),
            port=int(credentials["port"]),
            token=str(credentials["token"]),
        ),
    )


def _build_dodo_init(init_kwargs: dict[str, Any], runtime_config: dict[str, Any]) -> None:
    from nonebot.adapters.dodo.config import BotConfig

    credentials = runtime_config["credentials"]
    _append_config_item(
        init_kwargs,
        "bots",
        BotConfig(
            client_id=str(credentials["client_id"]),
            token=str(credentials["token"]),
        ),
    )


def _build_github_app_init(
    init_kwargs: dict[str, Any],
    runtime_config: dict[str, Any],
) -> None:
    from nonebot.adapters.github.config import GitHubApp

    credentials = runtime_config["credentials"]
    _append_config_item(
        init_kwargs,
        "github_apps",
        GitHubApp(
            app_id=str(credentials["app_id"]),
            private_key=str(credentials["private_key"]),
            client_id=_normalize_string(credentials.get("client_id")),
            client_secret=_normalize_string(credentials.get("client_secret")),
            webhook_secret=_normalize_string(credentials.get("webhook_secret")),
        ),
    )


def _build_github_oauth_init(
    init_kwargs: dict[str, Any],
    runtime_config: dict[str, Any],
) -> None:
    from nonebot.adapters.github.config import OAuthApp

    credentials = runtime_config["credentials"]
    _append_config_item(
        init_kwargs,
        "github_apps",
        OAuthApp(
            client_id=str(credentials["client_id"]),
            client_secret=str(credentials["client_secret"]),
            webhook_secret=_normalize_string(credentials.get("webhook_secret")),
        ),
    )


def _build_wxmp_init(init_kwargs: dict[str, Any], runtime_config: dict[str, Any]) -> None:
    from nonebot.adapters.wxmp.config import BotInfo

    credentials = runtime_config["credentials"]
    _append_config_item(
        init_kwargs,
        "wxmp_bots",
        BotInfo(
            appid=str(credentials["appid"]),
            token=str(credentials["token"]),
            secret=str(credentials["secret"]),
        ),
    )


def _build_minecraft_init(
    init_kwargs: dict[str, Any],
    runtime_config: dict[str, Any],
) -> None:
    credentials = runtime_config["credentials"]
    _merge_unique_value(
        init_kwargs,
        "minecraft_access_token",
        _normalize_string(credentials.get("access_token")),
    )
    mapping = init_kwargs.setdefault("minecraft_ws_urls", {})
    mapping.setdefault(str(credentials["self_id"]), []).append(str(credentials["ws_url"]))


def _build_vocechat_init(
    init_kwargs: dict[str, Any],
    runtime_config: dict[str, Any],
) -> None:
    from nonebot.adapters.vocechat.config import BotConfig

    credentials = runtime_config["credentials"]
    _append_config_item(
        init_kwargs,
        "vocechat_bots",
        BotConfig(
            name=_normalize_string(credentials.get("name")),
            user_id=str(credentials["user_id"]),
            server=str(credentials["server"]),
            api_key=str(credentials["api_key"]),
        ),
    )


def _build_yunhu_init(
    init_kwargs: dict[str, Any],
    runtime_config: dict[str, Any],
) -> None:
    from nonebot.adapters.yunhu.config import YunHuConfig

    credentials = runtime_config["credentials"]
    _append_config_item(
        init_kwargs,
        "yunhu_bots",
        YunHuConfig(
            app_id=str(credentials["app_id"]),
            token=str(credentials["token"]),
        ),
    )


def _build_mirai_init(
    init_kwargs: dict[str, Any],
    runtime_config: dict[str, Any],
) -> None:
    from nonebot.adapters.mirai.config import ClientInfo

    credentials = runtime_config["credentials"]
    _append_config_item(
        init_kwargs,
        "mirai_clients",
        ClientInfo(
            host=str(credentials.get("host") or "localhost"),
            port=int(credentials.get("port") or 8080),
            account=int(credentials["account"]),
            verify_key=str(credentials["verify_key"]),
        ),
    )


def _build_mail_init(init_kwargs: dict[str, Any], runtime_config: dict[str, Any]) -> None:
    from nonebot.adapters.mail.config import BotInfo, HostInfo

    credentials = runtime_config["credentials"]
    _append_config_item(
        init_kwargs,
        "mail_bots",
        BotInfo(
            id=str(credentials["id"]),
            name=str(credentials["name"]),
            password=str(credentials["password"]),
            subject=str(credentials["subject"]),
            imap=HostInfo(
                host=str(credentials["imap_host"]),
                port=int(credentials.get("imap_port") or 993),
                tls=_normalize_bool(credentials.get("imap_tls"), default=True),
            ),
            smtp=HostInfo(
                host=str(credentials["smtp_host"]),
                port=int(credentials.get("smtp_port") or 465),
                tls=_normalize_bool(credentials.get("smtp_tls"), default=True),
            ),
        ),
    )


def _bot_self_id(bot: Any) -> str:
    self_id = getattr(bot, "self_id", None)
    if self_id is not None:
        return str(self_id)
    if hasattr(bot, "get_self_id"):
        value = bot.get_self_id()
        if not inspect.isawaitable(value):
            return str(value)
    raise ValueError("Bot self_id is missing")


def _token_prefix(token: str) -> str:
    return token.split(":", 1)[0]


_PLATFORMS: dict[str, RobotPlatformSpec] = {
    "telegram": RobotPlatformSpec(
        id="telegram",
        label="Telegram",
        description="Telegram bot with a Bot Token.",
        fields=(RobotPlatformFieldSpec("bot_token", "Bot Token", secret=True),),
        adapter_module="nonebot.adapters.telegram",
        adapter_class="Adapter",
        build_init=_build_telegram_init,
        robot_identity=lambda credentials: _credential_identity(
            "telegram",
            _token_prefix(str(credentials["bot_token"])),
        ),
        bot_identity=lambda bot: _credential_identity(
            "telegram",
            _token_prefix(str(getattr(getattr(bot, "bot_config", None), "token", _bot_self_id(bot)))),
        ),
    ),
    "discord": RobotPlatformSpec(
        id="discord",
        label="Discord",
        description="Discord bot with a Bot Token.",
        fields=(RobotPlatformFieldSpec("bot_token", "Bot Token", secret=True),),
        adapter_module="nonebot.adapters.discord",
        adapter_class="Adapter",
        build_init=_build_discord_init,
        robot_identity=lambda credentials: _credential_identity(
            "discord",
            credentials.get("bot_token"),
        ),
        bot_identity=lambda bot: _credential_identity(
            "discord",
            getattr(getattr(bot, "bot_info", None), "token", None)
            or getattr(getattr(bot, "_bot_info", None), "token", None)
            or _bot_self_id(bot),
        ),
    ),
    "onebot_v11": RobotPlatformSpec(
        id="onebot_v11",
        label="QQ 接入端",
        description=(
            "填写当前登录的 QQ 号；接入端连接 /onebot/v11/ws。"
        ),
        fields=(
            RobotPlatformFieldSpec("self_id", "QQ Self ID"),
            RobotPlatformFieldSpec("access_token", "Access Token", required=False, secret=True),
            RobotPlatformFieldSpec("secret", "Secret", required=False, secret=True),
        ),
        adapter_module="nonebot.adapters.onebot.v11",
        adapter_class="Adapter",
        build_init=_build_onebot_v11_init,
        robot_identity=lambda credentials: _credential_identity(
            "onebot_v11",
            credentials.get("self_id"),
        ),
        bot_identity=lambda bot: _credential_identity("onebot_v11", _bot_self_id(bot)),
    ),
    "onebot_v12": RobotPlatformSpec(
        id="onebot_v12",
        label="OneBot V12",
        description="OneBot V12 bridge. Requires the bot self_id.",
        fields=(
            RobotPlatformFieldSpec("self_id", "Self ID"),
            RobotPlatformFieldSpec("ws_url", "WS URL"),
            RobotPlatformFieldSpec("access_token", "Access Token", required=False, secret=True),
            RobotPlatformFieldSpec("api_root", "API Root", required=False),
        ),
        adapter_module="nonebot.adapters.onebot.v12",
        adapter_class="Adapter",
        build_init=_build_onebot_v12_init,
        robot_identity=lambda credentials: _credential_identity(
            "onebot_v12",
            credentials.get("self_id"),
        ),
        bot_identity=lambda bot: _credential_identity("onebot_v12", _bot_self_id(bot)),
    ),
    "satori": RobotPlatformSpec(
        id="satori",
        label="Satori",
        description="Satori gateway for multi-platform bridging.",
        fields=(
            RobotPlatformFieldSpec("host", "Host", required=False),
            RobotPlatformFieldSpec("port", "Port"),
            RobotPlatformFieldSpec("path", "Path", required=False),
            RobotPlatformFieldSpec("token", "Token", required=False, secret=True),
        ),
        adapter_module="nonebot.adapters.satori",
        adapter_class="Adapter",
        build_init=_build_satori_init,
        robot_identity=lambda credentials: _compound_identity(
            "satori",
            credentials.get("host") or "localhost",
            credentials.get("port"),
            credentials.get("path") or "",
            credentials.get("token") or "",
        ),
        bot_identity=lambda bot: _compound_identity(
            "satori",
            getattr(getattr(bot, "info", None), "host", "localhost"),
            getattr(getattr(bot, "info", None), "port", ""),
            getattr(getattr(bot, "info", None), "path", ""),
            getattr(getattr(bot, "info", None), "token", "") or "",
        ),
    ),
    "red": RobotPlatformSpec(
        id="red",
        label="RedProtocol",
        description="RedProtocol adapter for Red-based QQ bridges.",
        fields=(
            RobotPlatformFieldSpec("host", "Host", required=False),
            RobotPlatformFieldSpec("port", "Port"),
            RobotPlatformFieldSpec("token", "Token", secret=True),
        ),
        adapter_module="nonebot.adapters.red",
        adapter_class="Adapter",
        build_init=_build_red_init,
        robot_identity=lambda credentials: _compound_identity(
            "red",
            credentials.get("host") or "localhost",
            credentials.get("port"),
            credentials.get("token"),
        ),
        bot_identity=lambda bot: _compound_identity(
            "red",
            getattr(getattr(bot, "info", None), "host", "localhost"),
            getattr(getattr(bot, "info", None), "port", ""),
            getattr(getattr(bot, "info", None), "token", ""),
        ),
    ),
    "dodo": RobotPlatformSpec(
        id="dodo",
        label="DoDo",
        description="DoDo bot with client_id and token.",
        fields=(
            RobotPlatformFieldSpec("client_id", "Client ID"),
            RobotPlatformFieldSpec("token", "Token", secret=True),
        ),
        adapter_module="nonebot.adapters.dodo",
        adapter_class="Adapter",
        build_init=_build_dodo_init,
        robot_identity=lambda credentials: _credential_identity(
            "dodo",
            credentials.get("client_id"),
        ),
        bot_identity=lambda bot: _credential_identity(
            "dodo",
            getattr(getattr(bot, "bot_config", None), "client_id", None) or _bot_self_id(bot),
        ),
    ),
    "github_app": RobotPlatformSpec(
        id="github_app",
        label="GitHub App",
        description="GitHub App for issue and webhook automation.",
        fields=(
            RobotPlatformFieldSpec("app_id", "App ID"),
            RobotPlatformFieldSpec("private_key", "Private Key", secret=True),
            RobotPlatformFieldSpec("client_id", "Client ID", required=False),
            RobotPlatformFieldSpec("client_secret", "Client Secret", required=False, secret=True),
            RobotPlatformFieldSpec("webhook_secret", "Webhook Secret", required=False, secret=True),
        ),
        adapter_module="nonebot.adapters.github",
        adapter_class="Adapter",
        build_init=_build_github_app_init,
        robot_identity=lambda credentials: _credential_identity(
            "github_app",
            credentials.get("app_id"),
        ),
        bot_identity=lambda bot: _credential_identity(
            "github_app",
            getattr(getattr(bot, "app", None), "id", None) or _bot_self_id(bot),
        ),
    ),
    "github_oauth": RobotPlatformSpec(
        id="github_oauth",
        label="GitHub OAuth",
        description="GitHub OAuth app for webhook and callback flows.",
        fields=(
            RobotPlatformFieldSpec("client_id", "Client ID"),
            RobotPlatformFieldSpec("client_secret", "Client Secret", secret=True),
            RobotPlatformFieldSpec("webhook_secret", "Webhook Secret", required=False, secret=True),
        ),
        adapter_module="nonebot.adapters.github",
        adapter_class="Adapter",
        build_init=_build_github_oauth_init,
        robot_identity=lambda credentials: _credential_identity(
            "github_oauth",
            credentials.get("client_id"),
        ),
        bot_identity=lambda bot: _credential_identity(
            "github_oauth",
            getattr(getattr(bot, "app", None), "id", None) or _bot_self_id(bot),
        ),
    ),
    "wxmp": RobotPlatformSpec(
        id="wxmp",
        label="WXMP",
        description="WeChat MP or mini-program adapter.",
        fields=(
            RobotPlatformFieldSpec("appid", "App ID"),
            RobotPlatformFieldSpec("token", "Token", secret=True),
            RobotPlatformFieldSpec("secret", "Secret", secret=True),
        ),
        adapter_module="nonebot.adapters.wxmp",
        adapter_class="Adapter",
        build_init=_build_wxmp_init,
        robot_identity=lambda credentials: _credential_identity(
            "wxmp",
            credentials.get("appid"),
        ),
        bot_identity=lambda bot: _credential_identity(
            "wxmp",
            getattr(getattr(bot, "bot_info", None), "appid", None) or _bot_self_id(bot),
        ),
    ),
    "minecraft": RobotPlatformSpec(
        id="minecraft",
        label="Minecraft",
        description="Minecraft WebSocket adapter. Requires self_id.",
        fields=(
            RobotPlatformFieldSpec("self_id", "Self ID"),
            RobotPlatformFieldSpec("ws_url", "WS URL"),
            RobotPlatformFieldSpec("access_token", "Access Token", required=False, secret=True),
        ),
        adapter_module="nonebot.adapters.minecraft",
        adapter_class="Adapter",
        build_init=_build_minecraft_init,
        robot_identity=lambda credentials: _credential_identity(
            "minecraft",
            credentials.get("self_id"),
        ),
        bot_identity=lambda bot: _credential_identity("minecraft", _bot_self_id(bot)),
    ),
    "vocechat": RobotPlatformSpec(
        id="vocechat",
        label="VoceChat",
        description="VoceChat bot with user_id, server, and api_key.",
        fields=(
            RobotPlatformFieldSpec("user_id", "User ID"),
            RobotPlatformFieldSpec("server", "Server"),
            RobotPlatformFieldSpec("api_key", "API Key", secret=True),
            RobotPlatformFieldSpec("name", "Name", required=False),
        ),
        adapter_module="nonebot.adapters.vocechat",
        adapter_class="Adapter",
        build_init=_build_vocechat_init,
        robot_identity=lambda credentials: _compound_identity(
            "vocechat",
            _normalize_endpoint(credentials.get("server")),
            credentials.get("user_id"),
        ),
        bot_identity=lambda bot: _compound_identity(
            "vocechat",
            _normalize_endpoint(getattr(bot, "server_base", "")),
            getattr(bot, "user_id", None) or _bot_self_id(bot),
        ),
    ),
    "yunhu": RobotPlatformSpec(
        id="yunhu",
        label="YunHu",
        description="YunHu bot with app_id and token.",
        fields=(
            RobotPlatformFieldSpec("app_id", "App ID"),
            RobotPlatformFieldSpec("token", "Token", secret=True),
        ),
        adapter_module="nonebot.adapters.yunhu",
        adapter_class="Adapter",
        build_init=_build_yunhu_init,
        robot_identity=lambda credentials: _credential_identity(
            "yunhu",
            credentials.get("app_id"),
        ),
        bot_identity=lambda bot: _credential_identity(
            "yunhu",
            getattr(getattr(bot, "bot_config", None), "app_id", None) or _bot_self_id(bot),
        ),
    ),
    "mirai": RobotPlatformSpec(
        id="mirai",
        label="Mirai",
        description="Mirai bridge for QQ protocol bots.",
        fields=(
            RobotPlatformFieldSpec("account", "Account"),
            RobotPlatformFieldSpec("verify_key", "Verify Key", secret=True),
            RobotPlatformFieldSpec("host", "Host", required=False),
            RobotPlatformFieldSpec("port", "Port", required=False),
        ),
        adapter_module="nonebot.adapters.mirai",
        adapter_class="Adapter",
        build_init=_build_mirai_init,
        robot_identity=lambda credentials: _credential_identity(
            "mirai",
            credentials.get("account"),
        ),
        bot_identity=lambda bot: _credential_identity(
            "mirai",
            getattr(getattr(bot, "info", None), "account", None) or _bot_self_id(bot),
        ),
    ),
    "mail": RobotPlatformSpec(
        id="mail",
        label="Mail",
        description="Mail bot using inbox events as triggers.",
        fields=(
            RobotPlatformFieldSpec("id", "ID"),
            RobotPlatformFieldSpec("name", "Name"),
            RobotPlatformFieldSpec("password", "Password", secret=True),
            RobotPlatformFieldSpec("subject", "Subject"),
            RobotPlatformFieldSpec("imap_host", "IMAP Host"),
            RobotPlatformFieldSpec("imap_port", "IMAP Port", required=False),
            RobotPlatformFieldSpec("imap_tls", "IMAP TLS", required=False),
            RobotPlatformFieldSpec("smtp_host", "SMTP Host"),
            RobotPlatformFieldSpec("smtp_port", "SMTP Port", required=False),
            RobotPlatformFieldSpec("smtp_tls", "SMTP TLS", required=False),
        ),
        adapter_module="nonebot.adapters.mail",
        adapter_class="Adapter",
        build_init=_build_mail_init,
        robot_identity=lambda credentials: _credential_identity("mail", credentials.get("id")),
        bot_identity=lambda bot: _credential_identity(
            "mail",
            getattr(getattr(bot, "bot_info", None), "id", None) or _bot_self_id(bot),
        ),
    ),
}

_SUPPORTED_ROBOT_PLATFORM_IDS = ("onebot_v11",)

_PLATFORM_ALIASES = {
    "qq": "onebot_v11",
    "qqofficial": "onebot_v11",
    "napcat": "onebot_v11",
    "lagrange": "onebot_v11",
    "lagrangeonebot": "onebot_v11",
    "sonwluma": "onebot_v11",
    "telegram": "telegram",
    "tg": "telegram",
    "discord": "discord",
    "onebot11": "onebot_v11",
    "onebot_v11": "onebot_v11",
    "onebot-v11": "onebot_v11",
    "ob11": "onebot_v11",
    "onebot12": "onebot_v12",
    "onebot_v12": "onebot_v12",
    "onebot-v12": "onebot_v12",
    "ob12": "onebot_v12",
    "satori": "satori",
    "red": "red",
    "redprotocol": "red",
    "dodo": "dodo",
    "githubapp": "github_app",
    "github_app": "github_app",
    "githuboauth": "github_oauth",
    "github_oauth": "github_oauth",
    "wxmp": "wxmp",
    "minecraft": "minecraft",
    "vocechat": "vocechat",
    "yunhu": "yunhu",
    "mirai": "mirai",
    "mail": "mail",
}

_ADAPTER_NAME_TO_PLATFORM = {
    "telegram": "telegram",
    "discord": "discord",
    "onebot v11": "onebot_v11",
    "onebot v12": "onebot_v12",
    "satori": "satori",
    "redprotocol": "red",
    "dodo": "dodo",
    "wxmp": "wxmp",
    "minecraft": "minecraft",
    "vocechat": "vocechat",
    "yunhu": "yunhu",
    "mirai": "mirai",
    "mail": "mail",
}


def get_robot_platform(platform_id: str) -> RobotPlatformSpec:
    normalized = normalize_robot_platform_id(platform_id)
    if normalized not in _SUPPORTED_ROBOT_PLATFORM_IDS:
        raise ValueError(
            f"Unsupported robot platform `{platform_id}`. "
            "Only OneBot V11 QQ connectors are currently supported."
        )
    try:
        return _PLATFORMS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported robot platform `{platform_id}`") from exc


def list_supported_robot_platforms() -> list[RobotPlatformPublic]:
    return [
        _PLATFORMS[platform_id].to_public()
        for platform_id in _SUPPORTED_ROBOT_PLATFORM_IDS
    ]


def normalize_robot_platform_id(platform_id: str | None) -> str:
    normalized = _normalize_string(platform_id)
    if normalized is None:
        raise ValueError("Robot platform is required")
    key = normalized.lower().replace(" ", "").replace("-", "_")
    try:
        return _PLATFORM_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported robot platform `{platform_id}`") from exc


def extract_robot_credentials(robot: Robot) -> dict[str, Any]:
    config = robot.config if isinstance(robot.config, dict) else {}
    raw_credentials = (
        config.get("credentials") if isinstance(config.get("credentials"), dict) else {}
    )
    credentials = {
        str(key): value
        for key, value in raw_credentials.items()
        if value is not None and value != ""
    }

    return credentials


def normalize_robot_config(
    platform_id: str,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    platform = get_robot_platform(platform_id)
    normalized_config = config if isinstance(config, dict) else {}
    raw_credentials = (
        normalized_config.get("credentials")
        if isinstance(normalized_config.get("credentials"), dict)
        else {}
    )
    raw_options = (
        normalized_config.get("options")
        if isinstance(normalized_config.get("options"), dict)
        else {}
    )

    allowed_fields = {field.key for field in platform.fields}
    credentials = {
        str(key): value
        for key, value in raw_credentials.items()
        if str(key) in allowed_fields and value is not None and value != ""
    }
    _ensure_required_fields(platform, credentials)

    options = dict(raw_options)
    options.pop("route_key", None)
    raw_reply_context_window_seconds = options.get("reply_context_window_seconds")
    try:
        reply_context_window_seconds = int(
            raw_reply_context_window_seconds
            if raw_reply_context_window_seconds is not None
            else DEFAULT_REPLY_CONTEXT_WINDOW_SECONDS
        )
    except (TypeError, ValueError):
        reply_context_window_seconds = DEFAULT_REPLY_CONTEXT_WINDOW_SECONDS
    options["reply_context_window_seconds"] = min(
        MAX_REPLY_CONTEXT_WINDOW_SECONDS,
        max(MIN_REPLY_CONTEXT_WINDOW_SECONDS, reply_context_window_seconds),
    )

    return {
        "credentials": credentials,
        "options": options,
    }


def validate_robot_platform_config(
    platform_id: str,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    return normalize_robot_config(platform_id, config)


def get_robot_runtime_config(robot: Robot) -> dict[str, Any]:
    platform_id = normalize_robot_platform_id(robot.platform or robot.protocol)
    config = robot.config if isinstance(robot.config, dict) else {}
    options = config.get("options") if isinstance(config.get("options"), dict) else {}
    runtime_config = {
        "credentials": extract_robot_credentials(robot),
        "options": dict(options),
    }
    return validate_robot_platform_config(platform_id, runtime_config)


def build_nonebot_init_kwargs(robots: list[Robot]) -> dict[str, Any]:
    init_kwargs: dict[str, Any] = {}
    for robot in robots:
        platform_id = normalize_robot_platform_id(robot.platform or robot.protocol)
        platform = get_robot_platform(platform_id)
        platform.build_init(init_kwargs, get_robot_runtime_config(robot))
    return init_kwargs


def register_nonebot_adapters(driver: Any, robots: list[Robot]) -> None:
    adapter_imports: set[tuple[str, str]] = set()
    for robot in robots:
        platform = get_robot_platform(robot.platform or robot.protocol)
        adapter_imports.add((platform.adapter_module, platform.adapter_class))

    for module_name, class_name in adapter_imports:
        module = __import__(module_name, fromlist=[class_name])
        adapter_class = getattr(module, class_name)
        driver.register_adapter(adapter_class)


def resolve_robot_identity(platform_id: str, robot: Robot) -> str:
    platform = get_robot_platform(platform_id)
    return platform.robot_identity(get_robot_runtime_config(robot)["credentials"])


def resolve_platform_from_bot(bot: Any) -> str | None:
    adapter_name = str(bot.adapter.get_name()).strip().lower()
    return _ADAPTER_NAME_TO_PLATFORM.get(adapter_name)


def resolve_bot_identity(bot: Any) -> str:
    platform_id = resolve_platform_from_bot(bot)
    if platform_id is None:
        raise ValueError("Unsupported bot adapter")
    return get_robot_platform(platform_id).bot_identity(bot)


def _extract_event_plain_text(event: Any) -> str:
    get_plaintext = getattr(event, "get_plaintext", None)
    if callable(get_plaintext):
        return str(get_plaintext() or "").strip()

    get_message = getattr(event, "get_message", None)
    if callable(get_message):
        message = get_message()
        extract_plain_text = getattr(message, "extract_plain_text", None)
        if callable(extract_plain_text):
            return str(extract_plain_text() or "").strip()
        return str(message or "").strip()

    return str(getattr(event, "message", "") or "").strip()


def _cq_escape(value: Any) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace(",", "&#44;")
    )


def _segment_to_onebot_text(segment: Any) -> str:
    segment_type = _segment_type(segment)
    data = _segment_data(segment)
    if not segment_type:
        return str(segment or "")
    if segment_type == "text":
        return str(data.get("text") or "")

    params = ",".join(
        f"{key}={_cq_escape(value)}"
        for key, value in data.items()
        if value not in (None, "")
    )
    if params:
        return f"[CQ:{segment_type},{params}]"
    return f"[CQ:{segment_type}]"


def _event_message_as_onebot_text(event: Any) -> str:
    return "".join(
        _segment_to_onebot_text(segment)
        for segment in _event_message_segments(event)
    ).strip()


def _extract_event_text(event: Any) -> str:
    raw_message = str(getattr(event, "raw_message", "") or "").strip()
    if raw_message:
        return raw_message

    segment_text = _event_message_as_onebot_text(event)
    if segment_text:
        return segment_text

    return _extract_event_plain_text(event)

def _event_attr_text(event: Any, name: str) -> str:
    return str(getattr(event, name, "") or "").strip()


def _clean_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "")}


def _extract_conversation_metadata(
    platform_id: str,
    event: Any,
    target_data: dict[str, Any],
) -> dict[str, Any]:
    user_id = _event_user_id(event)
    message_type = _event_attr_text(event, "message_type").lower()
    group_id = _event_attr_text(event, "group_id")
    guild_id = _event_attr_text(event, "guild_id")
    channel_id = _event_attr_text(event, "channel_id")
    target_id = str(target_data.get("id") or "").strip()
    parent_id = str(target_data.get("parent_id") or "").strip()

    if platform_id == "onebot_v11":
        if message_type == "private" or bool(target_data.get("private")):
            conversation_id = user_id or target_id
            return _clean_mapping(
                {
                    "platform": platform_id,
                    "type": "private",
                    "id": conversation_id,
                    "target_type": "private",
                    "target_id": conversation_id,
                    "user_id": user_id,
                    "message_type": message_type or "private",
                }
            )
        if group_id or message_type == "group":
            conversation_id = group_id or parent_id or target_id
            return _clean_mapping(
                {
                    "platform": platform_id,
                    "type": "group",
                    "id": conversation_id,
                    "target_type": "group",
                    "target_id": conversation_id,
                    "group_id": group_id,
                    "user_id": user_id,
                    "message_type": message_type or "group",
                }
            )

    if bool(target_data.get("private")):
        conversation_id = user_id or target_id
        return _clean_mapping(
            {
                "platform": platform_id,
                "type": "private",
                "id": conversation_id,
                "target_type": "private",
                "target_id": conversation_id,
                "user_id": user_id,
                "message_type": message_type or "private",
            }
        )
    if bool(target_data.get("channel")):
        conversation_id = parent_id or channel_id or target_id
        return _clean_mapping(
            {
                "platform": platform_id,
                "type": "channel",
                "id": conversation_id,
                "target_type": "channel",
                "target_id": channel_id or target_id,
                "guild_id": guild_id or parent_id,
                "channel_id": channel_id or target_id,
                "user_id": user_id,
                "message_type": message_type or "channel",
            }
        )

    conversation_id = group_id or parent_id or target_id
    return _clean_mapping(
        {
            "platform": platform_id,
            "type": "group",
            "id": conversation_id,
            "target_type": "group",
            "target_id": conversation_id,
            "group_id": group_id,
            "user_id": user_id,
            "message_type": message_type or "group",
        }
    )


def _extract_sender_key(
    platform_id: str,
    event: Any,
    target_data: dict[str, Any],
    conversation_data: dict[str, Any] | None = None,
) -> str:
    conversation_data = conversation_data or {}
    user_id = str(conversation_data.get("user_id") or _event_user_id(event)).strip()
    conversation_type = str(conversation_data.get("type") or "").strip().lower()
    conversation_id = str(conversation_data.get("id") or "").strip()
    if conversation_type and conversation_id:
        if conversation_type == "private":
            return f"{platform_id}:private:{user_id or conversation_id}"
        return f"{platform_id}:{conversation_type}:{conversation_id}:{user_id or conversation_id}"

    target_id = str(target_data.get("id") or "")
    parent_id = str(target_data.get("parent_id") or "")
    if bool(target_data.get("private")):
        return f"{platform_id}:private:{user_id or target_id}"
    scope = "channel" if bool(target_data.get("channel")) else "group"
    return f"{platform_id}:{scope}:{parent_id or target_id}:{user_id or target_id}"


def _event_user_id(event: Any) -> str:
    get_user_id = getattr(event, "get_user_id", None)
    if callable(get_user_id):
        try:
            return str(get_user_id() or "").strip()
        except Exception:
            return ""
    return str(getattr(event, "user_id", "") or "").strip()


def _dump_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)

    for method_name in ("model_dump", "dict"):
        dump_method = getattr(value, method_name, None)
        if callable(dump_method):
            try:
                dumped = dump_method()
            except Exception:
                continue
            if isinstance(dumped, dict):
                return dict(dumped)

    return {}


def _extract_sender_metadata(platform_id: str, event: Any) -> dict[str, Any]:
    sender_data = _dump_mapping(getattr(event, "sender", None))
    user_id = _event_user_id(event) or str(sender_data.get("user_id") or "").strip()
    nickname = str(sender_data.get("nickname") or "").strip()
    card = str(sender_data.get("card") or "").strip()
    display_name = card or nickname or user_id

    metadata: dict[str, Any] = {
        "platform": platform_id,
        "user_id": user_id,
        "display_name": display_name,
    }
    if nickname:
        metadata["nickname"] = nickname
    if card:
        metadata["card"] = card

    for key in ("role", "title", "sex", "age", "area", "level"):
        value = sender_data.get(key)
        if value not in (None, ""):
            metadata[key] = value

    message_type = str(getattr(event, "message_type", "") or "").strip()
    if message_type:
        metadata["message_type"] = message_type
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def _bot_self_ids(bot: Any, event: Any) -> set[str]:
    values = [
        getattr(bot, "self_id", None),
        getattr(bot, "user_id", None),
        getattr(bot, "id", None),
        getattr(event, "self_id", None),
    ]
    return {str(value).strip() for value in values if str(value or "").strip()}


def _event_message_segments(event: Any) -> list[Any]:
    get_message = getattr(event, "get_message", None)
    message = get_message() if callable(get_message) else getattr(event, "message", None)
    if message is None or isinstance(message, str):
        return []

    try:
        return list(message)
    except TypeError:
        return []


def _segment_type(segment: Any) -> str:
    if isinstance(segment, dict):
        return str(segment.get("type") or "").strip().lower()
    return str(getattr(segment, "type", "") or "").strip().lower()


def _segment_data(segment: Any) -> dict[str, Any]:
    data = segment.get("data") if isinstance(segment, dict) else getattr(segment, "data", None)
    return data if isinstance(data, dict) else {}


def _serialize_event_message_segments(event: Any) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for segment in _event_message_segments(event):
        segment_type = _segment_type(segment)
        data = _segment_data(segment)
        if not segment_type and not data:
            continue
        segments.append({"type": segment_type, "data": dict(data)})
    return segments

def _clean_mention_value(value: Any) -> str:
    return str(value or "").strip()


def _mention_from_data(data: dict[str, Any]) -> dict[str, str] | None:
    mention_id = _clean_mention_value(
        data.get("qq") or data.get("id") or data.get("user_id")
    )
    if not mention_id:
        return None

    mention: dict[str, str] = {"id": mention_id, "qq": mention_id}
    name = _clean_mention_value(
        data.get("name")
        or data.get("display_name")
        or data.get("nickname")
        or data.get("card")
    )
    if name:
        mention["name"] = name
    return mention


def _append_mention(mentions: list[dict[str, str]], mention: dict[str, str]) -> None:
    mention_id = mention.get("id", "")
    for existing in mentions:
        if existing.get("id") != mention_id:
            continue
        if mention.get("name") and not existing.get("name"):
            existing["name"] = mention["name"]
        return
    mentions.append(mention)


def _parse_cq_params(raw_params: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for part in raw_params.split(","):
        key, separator, value = part.partition("=")
        if separator:
            params[key.strip()] = value.strip()
    return params


def _extract_event_mentions(event: Any) -> list[dict[str, str]]:
    mentions: list[dict[str, str]] = []
    for segment in _event_message_segments(event):
        if _segment_type(segment) != "at":
            continue
        mention = _mention_from_data(_segment_data(segment))
        if mention is not None:
            _append_mention(mentions, mention)

    raw_message = str(getattr(event, "raw_message", "") or "")
    for match in re.finditer(r"\[CQ:at,([^\]]+)\]", raw_message):
        mention = _mention_from_data(_parse_cq_params(match.group(1)))
        if mention is not None:
            _append_mention(mentions, mention)
    return mentions


def _raw_message_mentions_bot(event: Any, self_ids: set[str]) -> bool:
    raw_message = str(getattr(event, "raw_message", "") or "")
    return any(
        f"[CQ:at,qq={self_id}]" in raw_message
        or f"[CQ:at,qq={self_id}," in raw_message
        for self_id in self_ids
    )


def _candidate_reply_sender_ids(reply_data: dict[str, Any]) -> set[str]:
    sender = reply_data.get("sender")
    sender_data = sender if isinstance(sender, dict) else _dump_mapping(sender)
    candidates = {
        reply_data.get("sender_id"),
        reply_data.get("user_id"),
        reply_data.get("qq"),
        reply_data.get("message_sender_id"),
        sender_data.get("sender_id"),
        sender_data.get("user_id"),
        sender_data.get("qq"),
        sender_data.get("id"),
    }
    return {str(value).strip() for value in candidates if str(value or "").strip()}


def _bounded_reply_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    if len(text) <= MAX_REPLY_REFERENCE_TEXT_LENGTH:
        return text
    return f"{text[:MAX_REPLY_REFERENCE_TEXT_LENGTH].rstrip()}..."


def _message_value_to_onebot_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "".join(_segment_to_onebot_text(segment) for segment in value)

    extract_plain_text = getattr(value, "extract_plain_text", None)
    if callable(extract_plain_text):
        try:
            plain_text = str(extract_plain_text() or "").strip()
        except Exception:
            plain_text = ""
        if plain_text:
            return plain_text

    try:
        segments = list(value)
    except TypeError:
        return str(value or "")
    if not segments:
        return ""
    return "".join(_segment_to_onebot_text(segment) for segment in segments)


def _reply_text_from_data(reply_data: dict[str, Any]) -> str:
    for key in ("raw_message", "plain_text", "text", "content"):
        text = _bounded_reply_text(reply_data.get(key))
        if text:
            return text
    return _bounded_reply_text(_message_value_to_onebot_text(reply_data.get("message")))


def _reply_sender_metadata(reply_data: dict[str, Any]) -> dict[str, Any]:
    sender = reply_data.get("sender")
    sender_data = sender if isinstance(sender, dict) else _dump_mapping(sender)
    user_id = str(
        sender_data.get("user_id")
        or sender_data.get("sender_id")
        or sender_data.get("qq")
        or sender_data.get("id")
        or reply_data.get("sender_id")
        or reply_data.get("user_id")
        or reply_data.get("qq")
        or ""
    ).strip()
    nickname = str(sender_data.get("nickname") or "").strip()
    card = str(sender_data.get("card") or "").strip()
    display_name = str(
        sender_data.get("display_name") or card or nickname or user_id
    ).strip()
    return _clean_mapping(
        {
            "user_id": user_id,
            "display_name": display_name,
            "nickname": nickname,
            "card": card,
        }
    )


def _first_reply_segment_data(event: Any) -> dict[str, Any]:
    for segment in _event_message_segments(event):
        if _segment_type(segment) == "reply":
            return dict(_segment_data(segment))
    return {}


def _raw_reply_segment_data(event: Any) -> dict[str, str]:
    raw_message = str(getattr(event, "raw_message", "") or "")
    match = re.search(r"\[CQ:reply,([^\]]+)\]", raw_message)
    return _parse_cq_params(match.group(1)) if match else {}


def _extract_reply_metadata(event: Any) -> dict[str, Any]:
    reply_data = _dump_mapping(getattr(event, "reply", None))
    segment_data = _first_reply_segment_data(event)
    raw_segment_data = _raw_reply_segment_data(event)

    message_id = str(
        reply_data.get("message_id")
        or reply_data.get("id")
        or reply_data.get("message_seq")
        or reply_data.get("real_id")
        or segment_data.get("message_id")
        or segment_data.get("id")
        or raw_segment_data.get("message_id")
        or raw_segment_data.get("id")
        or ""
    ).strip()
    source = str(
        reply_data.get("source")
        or reply_data.get("message_id")
        or reply_data.get("id")
        or segment_data.get("source")
        or segment_data.get("id")
        or raw_segment_data.get("id")
        or ""
    ).strip()

    metadata = _clean_mapping(
        {
            "message_id": message_id,
            "source": source,
            "text": _reply_text_from_data(reply_data),
        }
    )
    sender_data = _reply_sender_metadata(reply_data)
    if sender_data:
        metadata["sender"] = sender_data
    if segment_data:
        metadata["segment"] = _clean_mapping(dict(segment_data))
    if raw_segment_data:
        metadata["raw_segment"] = raw_segment_data
    return metadata


def _event_replies_to_bot(bot: Any, event: Any) -> bool:
    self_ids = _bot_self_ids(bot, event)
    if not self_ids:
        return False

    reply_data = _dump_mapping(getattr(event, "reply", None))
    if _candidate_reply_sender_ids(reply_data).intersection(self_ids):
        return True

    for segment in _event_message_segments(event):
        if _segment_type(segment) != "reply":
            continue
        if _candidate_reply_sender_ids(_segment_data(segment)).intersection(self_ids):
            return True

    return False


def _event_mentions_bot(bot: Any, event: Any) -> bool:
    self_ids = _bot_self_ids(bot, event)
    explicit_mentions = _extract_event_mentions(event)
    if explicit_mentions:
        if not self_ids:
            return False
        return any(
            str(mention.get("id") or mention.get("qq") or "").strip() in self_ids
            for mention in explicit_mentions
        )

    if self_ids and _raw_message_mentions_bot(event, self_ids):
        return True

    to_me = getattr(event, "to_me", False)
    if callable(to_me):
        try:
            to_me = to_me()
        except Exception:
            to_me = False
    message_type = str(getattr(event, "message_type", "") or "").strip().lower()
    if bool(to_me) and message_type not in {"group", "guild", "channel"}:
        return True

    return False


def build_inbound_message(
    platform_id: str,
    bot: Any,
    event: Any,
) -> RobotInboundMessage | None:
    text = _extract_event_text(event)
    mentioned_bot = _event_mentions_bot(bot, event)
    replied_to_bot = _event_replies_to_bot(bot, event)
    if not text:
        if mentioned_bot:
            text = "[mention_bot]"
        elif replied_to_bot:
            text = "[reply_to_bot]"
        else:
            return None

    from nonebot_plugin_alconna import get_message_id, get_target

    target = get_target(event, bot)
    target_data = target.dump()
    target_data["source"] = target.source or get_message_id(event, bot)
    conversation_data = _extract_conversation_metadata(platform_id, event, target_data)
    for key in ("message_type", "group_id", "user_id", "guild_id", "channel_id"):
        value = conversation_data.get(key)
        if value not in (None, ""):
            target_data.setdefault(key, value)
    metadata: dict[str, Any] = {
        "target": target_data,
        "sender": _extract_sender_metadata(platform_id, event),
        "conversation": conversation_data,
    }
    native_message = _clean_mapping(
        {
            "format": platform_id,
            "raw_message": str(getattr(event, "raw_message", "") or "").strip(),
            "plain_text": _extract_event_plain_text(event),
            "segments": _serialize_event_message_segments(event),
        }
    )
    if native_message:
        metadata["message"] = native_message
    reply_metadata = _extract_reply_metadata(event)
    if reply_metadata:
        metadata["reply"] = reply_metadata
    bot_self_ids = sorted(_bot_self_ids(bot, event))
    if bot_self_ids:
        metadata["bot_self_ids"] = bot_self_ids
    mentions = _extract_event_mentions(event)
    if mentions:
        metadata["mentions"] = mentions
    if mentioned_bot:
        metadata["mentioned_bot"] = True
    if replied_to_bot:
        metadata["replied_to_bot"] = True

    return RobotInboundMessage(
        sender_key=_extract_sender_key(
            platform_id,
            event,
            target_data,
            conversation_data=conversation_data,
        ),
        text=text,
        reply_target=RobotReplyTarget(
            target_type="universal",
            target_id=str(target.id),
            metadata=metadata,
        ),
    )


def _normalize_onebot_target_id(value: str) -> int | str:
    normalized = str(value or "").strip()
    return int(normalized) if normalized.isdigit() else normalized


_CQ_AT_PATTERN = re.compile(r"\[CQ:at,([^\]]+)\]")


def _parse_cq_params(raw_params: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for part in str(raw_params or "").split(","):
        key, separator, value = part.partition("=")
        if not separator:
            continue
        normalized_key = key.strip()
        if not normalized_key:
            continue
        params[normalized_key] = value.strip()
    return params


def _append_onebot_text_segment(
    segments: list[dict[str, dict[str, str]]],
    text: str,
) -> None:
    if not text:
        return
    if segments and segments[-1].get("type") == "text":
        segments[-1]["data"]["text"] += text
        return
    segments.append({"type": "text", "data": {"text": text}})


def _onebot_group_message_payload(text: str) -> str | list[dict[str, dict[str, str]]]:
    normalized_text = str(text or "")
    segments: list[dict[str, dict[str, str]]] = []
    has_at_segment = False
    cursor = 0

    for match in _CQ_AT_PATTERN.finditer(normalized_text):
        _append_onebot_text_segment(segments, normalized_text[cursor : match.start()])
        params = _parse_cq_params(match.group(1))
        qq = str(params.get("qq") or "").strip()
        if qq:
            segments.append({"type": "at", "data": {"qq": qq}})
            has_at_segment = True
        else:
            _append_onebot_text_segment(segments, match.group(0))
        cursor = match.end()

    if not has_at_segment:
        return normalized_text

    _append_onebot_text_segment(segments, normalized_text[cursor:])
    return segments


def _onebot_group_id_from_target_data(target_data: dict[str, Any]) -> int | str | None:
    message_type = str(target_data.get("message_type") or "").strip().lower()
    group_id = str(target_data.get("group_id") or "").strip()
    if message_type == "group":
        group_id = group_id or str(
            target_data.get("parent_id") or target_data.get("id") or ""
        ).strip()
    if not group_id:
        return None
    return _normalize_onebot_target_id(group_id)


def _prepare_unimessage_target_data(target_data: dict[str, Any]) -> tuple[dict[str, Any], str]:
    loaded_target_data = dict(target_data)
    source = str(loaded_target_data.pop("source", "") or "")
    for key in ("message_type", "group_id", "user_id", "guild_id", "channel_id"):
        loaded_target_data.pop(key, None)
    return loaded_target_data, source


async def _send_onebot_text_with_explicit_target(
    bot: Any,
    target: RobotReplyTarget,
    text: str,
    image_b64: str = "",
) -> bool:
    adapter_name = str(bot.adapter.get_name()).strip().lower()
    if adapter_name != "onebot v11":
        return False

    connections = getattr(bot.adapter, "connections", {})
    self_id = str(getattr(bot, "self_id", "") or "")
    if self_id not in connections:
        raise ConnectionError(
            "OneBot reverse WebSocket is not connected; "
            "The QQ connector may have disconnected before the message was sent"
        )

    target_type = str(target.target_type or "").strip().lower()
    target_id = _normalize_onebot_target_id(target.target_id)
    if target_type == "group":
        await bot.call_api(
            "send_group_msg",
            group_id=target_id,
            message=_onebot_message_segments(text, image_b64),
        )
        return True
    if target_type in {"private", "c2c", "friend", "direct", "direct_message"}:
        await bot.call_api(
            "send_private_msg",
            user_id=target_id,
            message=_onebot_message_segments(text, image_b64) if image_b64 else text,
        )
        return True
    return False


def _onebot_message_segments(text: str, image_b64: str = "") -> list[dict[str, Any]] | str:
    """文本 + 可选图片（base64）组装 OneBot 消息段。无图时保持原逻辑。"""
    if not image_b64:
        return _onebot_group_message_payload(text)
    segments: list[dict[str, Any]] = []
    text_payload = _onebot_group_message_payload(text)
    if isinstance(text_payload, list):
        segments.extend(text_payload)
    elif text_payload.strip():
        segments.append({"type": "text", "data": {"text": text_payload}})
    segments.append({"type": "image", "data": {"file": f"base64://{image_b64}"}})
    return segments


async def send_text_with_bot(
    bot: Any,
    target: RobotReplyTarget,
    text: str,
    *,
    image_b64: str = "",
) -> None:
    normalized_text = (text or "").strip()
    if not normalized_text and not image_b64:
        return

    target_data = target.metadata.get("target")
    if isinstance(target_data, dict):
        adapter_name = str(bot.adapter.get_name()).strip().lower()
        if adapter_name == "onebot v11":
            connections = getattr(bot.adapter, "connections", {})
            self_id = str(getattr(bot, "self_id", "") or "")
            if self_id not in connections:
                raise ConnectionError(
                    "OneBot reverse WebSocket is not connected; "
                    "The QQ connector may have disconnected before the reply was sent"
                )

        onebot_group_id = (
            _onebot_group_id_from_target_data(target_data)
            if adapter_name == "onebot v11"
            else None
        )
        if onebot_group_id is not None:
            group_payload = _onebot_message_segments(normalized_text, image_b64)
            if not isinstance(group_payload, str):
                await bot.call_api(
                    "send_group_msg",
                    group_id=onebot_group_id,
                    message=group_payload,
                )
                return

        from nonebot_plugin_alconna import UniMessage
        from nonebot_plugin_alconna.uniseg import Target

        loaded_target_data, source = _prepare_unimessage_target_data(target_data)
        uni_target = Target.load(loaded_target_data)
        if source:
            uni_target.source = source
        await UniMessage(normalized_text).send(target=uni_target, bot=bot)
        return

    if await _send_onebot_text_with_explicit_target(bot, target, normalized_text, image_b64):
        return

    raise ValueError("Unsupported reply target for current bot")
