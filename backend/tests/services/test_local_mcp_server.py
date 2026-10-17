from app.services.agent.mcp.local_server import LocalMCPServer
from app.plugins.robot.contracts import RobotReplyTarget
from app.plugins.robot.mcp.context import (
    RobotMCPContext,
    register_robot_mcp_context,
    unregister_robot_mcp_context,
)
from app.services.agent.session import (
    is_background_job_started_result,
)
from app.services.agent.prompts.system import get_system_prompt
from app.services.agent.skills import skill_loader
from app.services.terminal_runtime_state import TerminalRuntimeState


def _allow_main_terminal(monkeypatch, server: LocalMCPServer) -> None:
    monkeypatch.setattr(
        server,
        "_ensure_terminal_input_handler",
        lambda _item_id: True,
    )


def _allow_live_terminal_room(monkeypatch) -> None:
    import app.services.terminal_runtime_state as runtime_state_module

    monkeypatch.setattr(
        runtime_state_module,
        "get_terminal_runtime_state",
        lambda item_id: TerminalRuntimeState(
            item_id=item_id,
            active=True,
            daemon_connected=True,
            terminal_process_active=True,
            process_status="running",
            backend_room_connected=True,
            permanent_subscriber_count=1,
            reason="active",
        ),
    )


def test_get_terminal_status_uses_live_socket_room_state(monkeypatch) -> None:
    import app.services.terminal_runtime_state as runtime_state_module

    monkeypatch.setattr(
        runtime_state_module,
        "get_terminal_runtime_state",
        lambda item_id: TerminalRuntimeState(
            item_id=item_id,
            active=False,
            daemon_connected=True,
            terminal_process_active=True,
            process_status="running",
            backend_room_connected=False,
            permanent_subscriber_count=0,
            reason="backend_room_not_connected",
        ),
    )

    server = LocalMCPServer()
    result = server.call_tool(
        "get_terminal_status",
        {"item_id": "item-1"},
    )

    assert "终端未启动或未连接" in result[0]["text"]
    assert "Socket Room" in result[0]["text"]
    assert result[1]["terminal_active"] is False
    assert result[1]["backend_room_connected"] is False


def test_terminal_input_handler_rejects_stale_handler_without_live_room(
    monkeypatch,
) -> None:
    import app.services.terminal_runtime_state as runtime_state_module
    from app.services.socket_pool.input_center import input_center

    server = LocalMCPServer()
    monkeypatch.setattr(
        runtime_state_module,
        "get_terminal_runtime_state",
        lambda item_id: TerminalRuntimeState(
            item_id=item_id,
            active=False,
            daemon_connected=False,
            reason="daemon_not_connected",
        ),
    )
    monkeypatch.setattr(input_center, "has_handler", lambda _item_id: True)
    removed = []
    monkeypatch.setattr(
        input_center,
        "unregister_all_by_item",
        lambda item_id: removed.append(item_id) or 1,
    )

    assert server._ensure_terminal_input_handler("item-1") is False
    assert removed == ["item-1"]


def test_execute_command_reports_specific_live_terminal_failure(monkeypatch) -> None:
    import app.services.terminal_runtime_state as runtime_state_module

    server = LocalMCPServer()
    monkeypatch.setattr(server, "_ensure_terminal_input_handler", lambda _item_id: False)
    monkeypatch.setattr(
        runtime_state_module,
        "get_terminal_runtime_state",
        lambda item_id: TerminalRuntimeState(
            item_id=item_id,
            active=False,
            daemon_connected=True,
            terminal_process_active=True,
            process_status="running",
            backend_room_connected=False,
            reason="backend_room_not_connected",
        ),
    )

    result = server._execute_command({"item_id": "item-1", "command": "ls"})

    assert "终端进程存在" in result[0]["text"]
    assert "Backend 没有进入对应的 Socket Room" in result[0]["text"]
    assert "命令没有发送" in result[0]["text"]


def test_run_job_reports_structured_terminal_unavailable_failure(monkeypatch) -> None:
    server = LocalMCPServer()
    monkeypatch.setattr(server, "_ensure_terminal_input_handler", lambda _item_id: False)
    monkeypatch.setattr(
        server,
        "_terminal_unavailable_message",
        lambda _item_id: "main terminal is inactive",
    )

    result = server._run_job({"item_id": "item-1", "command": "cat /etc/os-release"})

    assert result[0] == {"type": "text", "text": "main terminal is inactive"}
    assert result[1]["type"] == "metadata"
    assert result[1]["command_dispatch_failed"] is True
    assert result[1]["terminal_unavailable"] is True
    assert result[1]["reason"] == "terminal_unavailable"
    assert result[1]["command_sent"] is False


def test_read_chat_history_tool_returns_recent_trimmed_context(monkeypatch) -> None:
    import app.services.agent.history.chat as chat_history

    long_terminal_output = "download progress\n" + ("x" * 2200)
    monkeypatch.setattr(
        chat_history,
        "get_chat_messages",
        lambda item_id: [
            {
                "type": "chat_user",
                "role": "user",
                "timestamp": "2026-07-10T10:00:00",
                "content": "install java",
            },
            {
                "type": "terminal_output",
                "role": "terminal",
                "timestamp": "2026-07-10T10:01:00",
                "content": long_terminal_output,
            },
            {
                "type": "agent_response",
                "role": "assistant",
                "timestamp": "2026-07-10T10:02:00",
                "content": "java install failed",
            },
        ],
    )
    monkeypatch.setattr(
        chat_history,
        "get_latest_session_summary",
        lambda item_id: {"content": "Current task: install Java for the server."},
    )

    server = LocalMCPServer()
    result = server.call_tool(
        "read_chat_history",
        {
            "item_id": "item-1",
            "limit": 2,
        },
    )

    text = result[0]["text"]
    assert "TermPaws item chat history" in text
    assert "Latest session summary" in text
    assert "Current task: install Java" in text
    assert "install java" not in text
    assert "terminal_output" in text
    assert "java install failed" in text
    assert "...<truncated>..." in text

    tool = next(tool for tool in server.list_tools() if tool["name"] == "read_chat_history")
    assert tool["skip_memory"] is True
    assert "context recovery" in tool["description"]
    assert "query" in tool["inputSchema"]["properties"]


def test_save_memory_uses_type_ttl_and_verified_metadata(monkeypatch) -> None:
    import importlib

    vector_store_module = importlib.import_module("app.services.agent.memory.vector_store")

    captured: dict[str, object] = {}

    def fake_add_memory(**kwargs):
        captured.update(kwargs)
        return "memory-123456"

    monkeypatch.setattr(vector_store_module.vector_store, "add_memory", fake_add_memory)

    server = LocalMCPServer()
    result = server.call_tool(
        "save_memory",
        {
            "item_id": "item-1",
            "content": "你叫大狗",
            "memory_type": "fact",
        },
    )

    assert "memory-1" in result[0]["text"]
    assert captured["handler_id"] == "item-1"
    assert captured["content"] == "你叫大狗"
    assert captured["memory_type"] == "fact"
    assert captured["ttl_days"] == 90
    assert captured["allow_duplicate"] is True
    metadata = captured["metadata"]
    assert metadata["type"] == "agent_saved"
    assert metadata["source"] == "local_agent_saved"
    assert metadata["verified"] is True
    assert metadata["content_hash"]


def test_save_memory_rejects_removed_task_type() -> None:
    result = LocalMCPServer().call_tool(
        "save_memory",
        {
            "item_id": "item-1",
            "content": "安装 Java",
            "memory_type": "task",
        },
    )

    assert result == [{"type": "text", "text": "Error: invalid memory_type: task"}]


def test_execute_command_only_reports_dispatch(monkeypatch) -> None:
    import importlib

    import app.services.socket_pool as socket_pool

    input_center_module = importlib.import_module("app.services.socket_pool.input_center")

    sent: dict[str, str] = {}

    class FakeInputSDK:
        def send(self, item_id: str, command: str) -> bool:
            sent["item_id"] = item_id
            sent["command"] = command
            return True

    monkeypatch.setattr(socket_pool, "InputSDK", FakeInputSDK)
    monkeypatch.setattr(input_center_module.input_center, "has_handler", lambda item_id: True)
    _allow_live_terminal_room(monkeypatch)

    server = LocalMCPServer()
    result = server.call_tool(
        "execute_command",
        {
            "item_id": "item-1",
            "command": "echo 23231",
        },
    )

    assert result == [
        {
            "type": "text",
            "text": "命令已发送: echo 23231。终端反馈只送一轮（约 20-30 秒窗口），之后的持续输出不会再唤起你。要等长过程的特定输出（启动完成、编译结束等），用 run_job 跑等待式命令（如 timeout 120 tail -f 日志 | grep -m1 标记），结果会自动回调；不要空等主终端。",
        }
    ]
    assert sent == {"item_id": "item-1", "command": "echo 23231\n"}
    tool = next(tool for tool in server.list_tools() if tool["name"] == "execute_command")
    props = tool["inputSchema"]["properties"]
    assert "timeout_seconds" in props
    assert "auto_interrupt_on_timeout" in props
    assert props["auto_interrupt_on_timeout"]["default"] is False
    assert "run_job" in tool["description"]
    assert "一次只发一条" in tool["description"]

def test_execute_command_sends_shell_like_input_to_active_console(
    monkeypatch,
) -> None:
    import importlib

    import app.services.socket_pool as socket_pool
    from app.services.agent.session import EXECUTE_COMMAND_TOOL_NAME, agent_session_manager

    input_center_module = importlib.import_module("app.services.socket_pool.input_center")
    item_id = "item-console-shell-query"
    sent: dict[str, str] = {}

    class FakeInputSDK:
        def send(self, item_id: str, command: str) -> bool:
            sent["item_id"] = item_id
            sent["command"] = command
            return True

    monkeypatch.setattr(socket_pool, "InputSDK", FakeInputSDK)
    monkeypatch.setattr(input_center_module.input_center, "has_handler", lambda item_id: True)
    _allow_live_terminal_room(monkeypatch)

    agent_session_manager.remove_session(item_id)
    session = agent_session_manager.get_or_create_session(item_id, "handler-1")
    monkeypatch.setattr(session, "_schedule_pending_command_recheck", lambda *args, **kwargs: None)
    monkeypatch.setattr(session, "_get_log_line_count", lambda: 0)
    session.mark_terminal_command_dispatched(
        EXECUTE_COMMAND_TOOL_NAME,
        {"command": "java -jar paper-server.jar nogui"},
    )

    server = LocalMCPServer()
    monkeypatch.setattr(
        server,
        "_run_job",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("active console input must not be auto-routed")
        ),
    )

    try:
        result = server.call_tool(
            "execute_command",
            {
                "item_id": item_id,
                "command": "ls -la",
            },
        )
    finally:
        agent_session_manager.remove_session(item_id)

    assert sent == {"item_id": item_id, "command": "ls -la\n"}
    assert result == [
        {
            "type": "text",
            "text": "命令已发送: ls -la。终端反馈只送一轮（约 20-30 秒窗口），之后的持续输出不会再唤起你。要等长过程的特定输出（启动完成、编译结束等），用 run_job 跑等待式命令（如 timeout 120 tail -f 日志 | grep -m1 标记），结果会自动回调；不要空等主终端。",
        }
    ]


def test_execute_command_sends_cd_then_server_launcher_when_console_context_is_stale(
    monkeypatch,
) -> None:
    import importlib

    import app.services.socket_pool as socket_pool
    from app.services.agent.session import EXECUTE_COMMAND_TOOL_NAME, agent_session_manager

    input_center_module = importlib.import_module("app.services.socket_pool.input_center")
    item_id = "item-stale-console-launch"
    sent: dict[str, str] = {}

    class FakeInputSDK:
        def send(self, item_id: str, command: str) -> bool:
            sent["item_id"] = item_id
            sent["command"] = command
            return True

    monkeypatch.setattr(socket_pool, "InputSDK", FakeInputSDK)
    monkeypatch.setattr(input_center_module.input_center, "has_handler", lambda item_id: True)
    _allow_live_terminal_room(monkeypatch)

    agent_session_manager.remove_session(item_id)
    session = agent_session_manager.get_or_create_session(item_id, "handler-1")
    monkeypatch.setattr(session, "_schedule_pending_command_recheck", lambda *args, **kwargs: None)
    monkeypatch.setattr(session, "_get_log_line_count", lambda: 0)
    session.mark_terminal_command_dispatched(
        EXECUTE_COMMAND_TOOL_NAME,
        {"command": "bash run.sh"},
    )

    try:
        result = LocalMCPServer().call_tool(
            "execute_command",
            {
                "item_id": item_id,
                "command": "cd temp_extract && bash run.sh",
            },
        )
    finally:
        agent_session_manager.remove_session(item_id)

    assert sent == {
        "item_id": item_id,
        "command": "cd temp_extract && bash run.sh\n",
    }
    assert result[0]["type"] == "text"
    assert "cd temp_extract && bash run.sh" in result[0]["text"]


def test_system_prompt_forbids_claiming_command_success_without_confirmation() -> None:
    skill_loader.reload()
    prompt = get_system_prompt()

    assert "不代表成功" in prompt
    assert "没有新证据不要断言结果" in prompt
    assert "run_job" in prompt
    assert "execute_command" in prompt



def test_execute_command_reports_disconnected_without_handler(monkeypatch) -> None:
    import importlib

    import app.services.socket_pool as socket_pool

    input_center_module = importlib.import_module("app.services.socket_pool.input_center")

    class FakeInputSDK:
        def send(self, item_id: str, command: str) -> bool:
            raise AssertionError("send should not be called without a terminal handler")

    monkeypatch.setattr(socket_pool, "InputSDK", FakeInputSDK)
    monkeypatch.setattr(input_center_module.input_center, "has_handler", lambda item_id: False)
    _allow_live_terminal_room(monkeypatch)

    server = LocalMCPServer()
    restore_calls: list[str] = []
    monkeypatch.setattr(
        server,
        "_restore_existing_terminal_input",
        lambda item_id: restore_calls.append(item_id) or False,
    )
    result = server.call_tool(
        "execute_command",
        {
            "item_id": "item-1",
            "command": "echo 23231",
        },
    )

    assert result[0] == {
        "type": "text",
        "text": (
            "终端已启动并已连接，但 Backend 输入处理器不可用，"
            "自动恢复失败。命令没有发送。"
        ),
    }
    assert result[1]["command_dispatch_failed"] is True
    assert result[1]["reason"] == "terminal_unavailable"
    assert result[1]["command_sent"] is False
    assert restore_calls == ["item-1"]


def test_execute_command_restores_existing_terminal_input_handler(monkeypatch) -> None:
    import importlib

    import app.services.socket_pool as socket_pool

    input_center_module = importlib.import_module("app.services.socket_pool.input_center")

    sent: dict[str, str] = {}
    handler_checks = {"value": 0}
    restore_calls: list[str] = []

    class FakeInputSDK:
        def send(self, item_id: str, command: str) -> bool:
            sent["item_id"] = item_id
            sent["command"] = command
            return True

    def fake_has_handler(item_id: str) -> bool:
        handler_checks["value"] += 1
        return handler_checks["value"] >= 2

    monkeypatch.setattr(socket_pool, "InputSDK", FakeInputSDK)
    monkeypatch.setattr(input_center_module.input_center, "has_handler", fake_has_handler)
    _allow_live_terminal_room(monkeypatch)

    server = LocalMCPServer()
    monkeypatch.setattr(
        server,
        "_restore_existing_terminal_input",
        lambda item_id: restore_calls.append(item_id) or True,
    )

    result = server.call_tool(
        "execute_command",
        {
            "item_id": "item-1",
            "command": "echo restored",
        },
    )

    assert restore_calls == ["item-1"]
    assert sent == {"item_id": "item-1", "command": "echo restored\n"}
    assert result == [
        {
            "type": "text",
            "text": "命令已发送: echo restored。终端反馈只送一轮（约 20-30 秒窗口），之后的持续输出不会再唤起你。要等长过程的特定输出（启动完成、编译结束等），用 run_job 跑等待式命令（如 timeout 120 tail -f 日志 | grep -m1 标记），结果会自动回调；不要空等主终端。",
        }
    ]


def test_terminal_input_filter_rule_tool_adds_noise_block_rule(db) -> None:
    from app.models import Item
    from app.services.filters.input_filter import InputFilter, InputFilterConfig
    from tests.utils.item import create_random_item

    item = create_random_item(db)
    server = LocalMCPServer()

    result = server.call_tool(
        "add_terminal_input_filter_rule",
        {
            "item_id": str(item.id),
            "name": "noise_ftb_backups",
            "regex_patterns": [
                r"FTBBackups/]: Attempting to create an automatic backup"
            ],
            "reason": "Repeated automatic backup status line.",
        },
    )

    assert "Added terminal output -> Agent input filter rule" in result[0]["text"]
    updated = db.get(Item, item.id)
    assert updated is not None
    db.refresh(updated)
    assert updated.input_filter_enabled is True
    assert "noise_ftb_backups" in updated.input_filter_rules

    input_filter = InputFilter(InputFilterConfig.from_item(updated))
    filtered = input_filter.filter(
        {
            "stdout": (
                "[22:00:00] [ftbbackups2_Worker-1/INFO] "
                "[ne.cr.ft.FTBBackups/]: "
                "Attempting to create an automatic backup\n"
            ),
            "stderr": "",
        }
    )
    assert filtered is None

    list_result = server.call_tool(
        "list_terminal_input_filter_rules",
        {"item_id": str(item.id)},
    )
    assert "noise_ftb_backups" in list_result[0]["text"]
    assert "FTBBackups" in list_result[0]["text"]


def test_terminal_filter_rule_tool_lists_input_and_output_rules(db) -> None:
    from tests.utils.item import create_random_item

    item = create_random_item(db)
    item.input_filter_enabled = True
    item.input_filter_rules = {
        "noise_backup": {
            "regex_patterns": [r"Attempting to create an automatic backup"],
            "action_type": "block",
            "reason": "Repeated backup line.",
        }
    }
    item.output_filter_enabled = True
    item.output_filter_rules = {
        "dangerous_rm": {
            "regex_patterns": [r"rm\s+-rf\s+/"],
            "action_type": "block",
            "reason": "Protect root filesystem.",
        }
    }
    db.add(item)
    db.commit()

    server = LocalMCPServer()
    result = server.call_tool(
        "list_terminal_filter_rules",
        {"item_id": str(item.id)},
    )

    text = result[0]["text"]
    assert "Input filter (terminal output -> Agent) enabled=True; rules=1" in text
    assert "noise_backup" in text
    assert "Attempting to create an automatic backup" in text
    assert "Output filter (Agent command -> terminal) enabled=True; rules=1" in text
    assert "dangerous_rm" in text
    assert r"rm\s+-rf\s+/" in text


def test_terminal_input_filter_rule_tools_delete_and_clear_rules(db) -> None:
    from app.models import Item
    from tests.utils.item import create_random_item

    item = create_random_item(db)
    server = LocalMCPServer()

    server.call_tool(
        "add_terminal_input_filter_rule",
        {
            "item_id": str(item.id),
            "name": "noise_one",
            "regex_patterns": [r"noise one"],
        },
    )
    server.call_tool(
        "add_terminal_input_filter_rule",
        {
            "item_id": str(item.id),
            "name": "noise_two",
            "regex_patterns": [r"noise two"],
        },
    )

    delete_result = server.call_tool(
        "delete_terminal_input_filter_rule",
        {"item_id": str(item.id), "name": "noise_one"},
    )
    assert "Deleted terminal output -> Agent input filter rule `noise_one`" in delete_result[0]["text"]

    updated = db.get(Item, item.id)
    assert updated is not None
    db.refresh(updated)
    assert "noise_one" not in updated.input_filter_rules
    assert "noise_two" in updated.input_filter_rules
    assert updated.input_filter_enabled is True

    clear_result = server.call_tool(
        "clear_terminal_input_filter_rules",
        {"item_id": str(item.id)},
    )
    assert "Cleared terminal output -> Agent input filter rules" in clear_result[0]["text"]

    db.refresh(updated)
    assert updated.input_filter_rules == {}
    assert updated.input_filter_enabled is False


def test_run_job_blocks_when_main_terminal_stopped(monkeypatch) -> None:
    server = LocalMCPServer()
    monkeypatch.setattr(
        server,
        "_ensure_terminal_input_handler",
        lambda _item_id: False,
    )
    monkeypatch.setattr(
        server,
        "_get_item_daemon_context",
        lambda _item_id: (_ for _ in ()).throw(
            AssertionError("inactive terminal must not contact the daemon job API")
        ),
    )
    monkeypatch.setattr(
        server,
        "_start_background_job_thread",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("inactive terminal must not create a background thread")
        ),
    )

    result = server.call_tool(
        "run_job",
        {
            "item_id": "item-terminal-stopped",
            "command": "java -version",
        },
    )

    assert result[0] == {
        "type": "text",
        "text": "终端未启动或未连接。终端 Item ID 无效。命令没有发送。",
    }
    assert result[1]["command_dispatch_failed"] is True
    assert result[1]["reason"] == "terminal_unavailable"
    assert result[1]["command_sent"] is False
    assert not any(item.get("background_job_started") for item in result)


def test_run_job_blocks_superseded_reply_ticket(monkeypatch) -> None:
    from app.services.agent.reply_ticket import reply_ticket_manager

    server = LocalMCPServer()
    _allow_main_terminal(monkeypatch, server)
    monkeypatch.setattr(
        server,
        "_get_item_daemon_context",
        lambda _item_id: (_ for _ in ()).throw(
            AssertionError("superseded task must not contact the daemon job API")
        ),
    )
    reply_ticket_manager.reset()

    result = server.call_tool(
        "run_job",
        {
            "item_id": "item-1",
            "command": "wget https://old.example/jdk.tar.gz",
            "_reply_ticket_id": "superseded-ticket",
        },
    )

    assert len(result) == 1
    assert "superseded by a newer message" in result[0]["text"]
    assert not any(item.get("background_job_started") for item in result)


def test_run_job_is_always_asynchronous_even_when_wait_requested(monkeypatch) -> None:
    from types import SimpleNamespace

    server = LocalMCPServer()
    _allow_main_terminal(monkeypatch, server)
    captured: dict[str, object] = {}

    class FakeConnection:
        def run_job_http(self, **kwargs):
            raise AssertionError("run_job must not wait synchronously")

    connection = FakeConnection()

    monkeypatch.setattr(
        server,
        "_get_item_daemon_context",
        lambda item_id: (
            SimpleNamespace(owner_id="user-1", working_directory="/workspace/item", title="Test Item"),
            connection,
        ),
    )
    monkeypatch.setattr(
        server,
        "_start_background_job_thread",
        lambda **kwargs: captured.update(kwargs),
    )

    result = server.call_tool(
        "run_job",
        {
            "item_id": "item-1",
            "command": "bun install",
            "timeout_seconds": 12,
            "tail_lines": 5,
            "wait_for_completion": True,
        },
    )

    assert result[1]["background_job_started"] is True
    assert captured["item_id"] == "item-1"
    assert captured["command"] == "bun install"
    assert captured["connection"] is connection
    assert captured["request_kwargs"] == {
        "item_uuid": "item-1",
        "user_uuid": "user-1",
        "command": "bun install",
            "working_directory": "/workspace/item",
            "timeout_seconds": 12,
            "tail_lines": 5,
            "item_title": "Test Item",
        }
    run_job_tool = next(tool for tool in server.list_tools() if tool["name"] == "run_job")
    assert "wait_for_completion" not in run_job_tool["inputSchema"]["properties"]


def test_tool_descriptions_guide_foreground_background_command_choice() -> None:
    server = LocalMCPServer()

    execute_tool = next(tool for tool in server.list_tools() if tool["name"] == "execute_command")
    run_job_tool = next(tool for tool in server.list_tools() if tool["name"] == "run_job")

    assert "run_job" in execute_tool["description"]
    assert "后台" in run_job_tool["description"]
    assert "cancel_job" in run_job_tool["description"]
    assert "timeout_seconds" in run_job_tool["description"]


def test_run_job_defaults_to_background_and_notifies_session(monkeypatch) -> None:
    import threading
    from types import SimpleNamespace

    from app.services.agent.session import RUN_JOB_TOOL_NAME, agent_session_manager

    item_id = "item-background-job"
    server = LocalMCPServer()
    _allow_main_terminal(monkeypatch, server)
    delivered: list[object] = []
    started = threading.Event()
    allow_finish = threading.Event()
    done = threading.Event()

    class FakeConnection:
        def run_job_http(self, **kwargs):
            started.set()
            self.command = kwargs["command"]
            return {"success": True, "job_id": "job-bg"}

        def get_job_result_http(self, **kwargs):
            assert allow_finish.wait(2)
            return {
                "success": True,
                "status": "finished",
                "result": {
                    "success": True,
                    "job_id": "job-bg",
                    "command": self.command,
                    "cwd": "/workspace/item",
                    "exit_code": 0,
                    "timed_out": False,
                    "duration_seconds": 3.0,
                    "output_tail": "install complete",
                },
            }


    monkeypatch.setattr(
        server,
        "_get_item_daemon_context",
        lambda item_id: (
            SimpleNamespace(owner_id="user-1", working_directory="/workspace/item", title="Test Item"),
            FakeConnection(),
        ),
    )

    agent_session_manager.remove_session(item_id)
    session = agent_session_manager.get_or_create_session(item_id, "handler-1")

    def fake_process_input(input_msg):
        delivered.append(input_msg)
        done.set()

    monkeypatch.setattr(session, "process_input", fake_process_input)
    try:
        result = server.call_tool(
            "run_job",
            {"item_id": item_id, "command": "apt-get install -y temurin-17-jdk"},
        )
        assert started.wait(2)
        assert session.has_running_terminal_job() is True
        allow_finish.set()
        assert done.wait(2)
    finally:
        allow_finish.set()
        agent_session_manager.remove_session(item_id)

    assert "后台任务已启动" in result[0]["text"]
    assert result[1]["type"] == "metadata"
    assert result[1]["background_job_started"] is True
    assert session.has_running_terminal_job() is False
    assert len(delivered) == 1
    assert delivered[0].input_type.value == "terminal"
    assert "Background terminal job completed" in delivered[0].content
    assert "install complete" in delivered[0].content


def test_background_run_job_queues_robot_completion(monkeypatch) -> None:
    import threading
    from types import SimpleNamespace

    from app.plugins.robot.service import robot_service

    item_id = "item-background-job-robot"
    server = LocalMCPServer()
    _allow_main_terminal(monkeypatch, server)
    queued: list[dict] = []
    started = threading.Event()
    done = threading.Event()
    target = RobotReplyTarget(
        target_type="private",
        target_id="2537134688",
        metadata={"mentioned_bot": True},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="00000000-0000-0000-0000-000000000001",
            sender_key="2537134688",
            reply_target=target,
            conversation_key="private:2537134688",
            conversation_generation=7,
            reply_requires_awake=True,
        )
    )

    class FakeConnection:
        def run_job_http(self, **kwargs):
            started.set()
            self.command = kwargs["command"]
            return {"success": True, "job_id": "job-bg-robot"}

        def get_job_result_http(self, **kwargs):
            return {
                "success": True,
                "status": "finished",
                "result": {
                    "success": True,
                    "job_id": "job-bg-robot",
                    "command": self.command,
                    "cwd": "/workspace/item",
                    "exit_code": 0,
                    "timed_out": False,
                    "duration_seconds": 3.0,
                    "output_tail": "temurin installed",
                },
            }

    def fake_enqueue_background_job_result(**kwargs):
        queued.append(kwargs)
        done.set()
        return True

    monkeypatch.setattr(
        server,
        "_get_item_daemon_context",
        lambda item_id: (
            SimpleNamespace(owner_id="user-1", working_directory="/workspace/item", title="Test Item"),
            FakeConnection(),
        ),
    )
    monkeypatch.setattr(
        robot_service,
        "enqueue_background_job_result",
        fake_enqueue_background_job_result,
    )

    try:
        result = server.call_tool(
            "run_job",
            {
                "item_id": item_id,
                "command": "apt-get install -y temurin-17-jdk",
                "_robot_context_token": token,
            },
        )
        assert started.wait(2)
        assert done.wait(2)
    finally:
        unregister_robot_mcp_context(token)

    assert "后台" in result[0]["text"] or "鍚庡彴" in result[0]["text"]
    assert len(queued) == 1
    assert queued[0]["robot_id"] == "00000000-0000-0000-0000-000000000001"
    assert queued[0]["item_id"] == item_id
    assert queued[0]["sender_key"] == "2537134688"
    assert queued[0]["conversation_key"] == "private:2537134688"
    assert queued[0]["conversation_generation"] == 7
    assert queued[0]["reply_target"]["target_id"] == "2537134688"
    assert "temurin installed" in queued[0]["message"]


def test_background_run_job_queues_robot_result_instead_of_direct_raw_reply(
    monkeypatch,
) -> None:
    import threading
    from types import SimpleNamespace

    from app.plugins.robot.bridge_client import robot_bridge_client
    from app.plugins.robot.conversation_memory import robot_conversation_memory
    from app.plugins.robot.service import robot_service
    from app.services.agent.reply_ticket import reply_ticket_manager

    item_id = "item-background-job-ticket"
    server = LocalMCPServer()
    _allow_main_terminal(monkeypatch, server)
    started = threading.Event()
    queued_event = threading.Event()
    cleared: list[str] = []
    sent: list[tuple[str, RobotReplyTarget, str]] = []
    queued_results: list[dict] = []
    memory_writes: list[tuple[str, str, str]] = []
    target = RobotReplyTarget(
        target_type="private",
        target_id="2537134688",
        metadata={"conversation": {"type": "private", "id": "2537134688"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="00000000-0000-0000-0000-000000000001",
            sender_key="2537134688",
            reply_target=target,
            conversation_key="private:2537134688",
            conversation_generation=8,
            reply_requires_awake=True,
        )
    )
    fake_agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="00000000-0000-0000-0000-000000000001",
            robot_context_token=token,
            reply_ticket_id="",
        )
    )

    class FakeConnection:
        def run_job_http(self, **kwargs):
            started.set()
            self.command = kwargs["command"]
            return {"success": True, "job_id": "job-bg-ticket"}

        def get_job_result_http(self, **kwargs):
            return {
                "success": True,
                "status": "finished",
                "result": {
                    "success": True,
                    "job_id": "job-bg-ticket",
                    "command": self.command,
                    "cwd": "/workspace/item",
                    "exit_code": 0,
                    "timed_out": False,
                    "duration_seconds": 2.5,
                    "output_tail": "temurin installed",
                },
            }

    monkeypatch.setattr(
        server,
        "_get_item_daemon_context",
        lambda item_id: (
            SimpleNamespace(owner_id="user-1", working_directory="/workspace/item", title="Test Item"),
            FakeConnection(),
        ),
    )
    monkeypatch.setattr(
        robot_service,
        "register_background_job_reply",
        lambda **_kwargs: "pending-ticket-1",
    )
    monkeypatch.setattr(
        robot_service,
        "clear_background_job_reply",
        lambda **kwargs: cleared.append(kwargs["pending_reply_id"]),
    )
    monkeypatch.setattr(
        robot_service,
        "enqueue_background_job_result",
        lambda **kwargs: (
            queued_results.append(kwargs),
            queued_event.set(),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        robot_bridge_client,
        "send_message",
        lambda robot_id, target, text: sent.append((str(robot_id), target, text)),
    )
    monkeypatch.setattr(
        robot_conversation_memory,
        "append_assistant_message",
        lambda robot_id, conversation_key, text: memory_writes.append(
            (str(robot_id), conversation_key, text)
        ),
    )

    reply_ticket_manager.reset()
    try:
        ticket = reply_ticket_manager.create_for_agent(
            fake_agent,
            item_id=item_id,
            handler_id="handler-1",
            message="安装 temurin java17",
        )
        result = server.call_tool(
            "run_job",
            {
                "item_id": item_id,
                "command": "apt-get install -y temurin-17-jdk",
                "_robot_context_token": token,
                "_reply_ticket_id": ticket.ticket_id,
            },
        )
        assert started.wait(2)
        assert queued_event.wait(2)
    finally:
        unregister_robot_mcp_context(token)
        reply_ticket_manager.reset()

    assert "后台任务已启动" in result[0]["text"]
    assert cleared == []
    assert sent == []
    assert memory_writes == []
    assert len(queued_results) == 1
    assert queued_results[0]["robot_id"] == "00000000-0000-0000-0000-000000000001"
    assert queued_results[0]["conversation_key"] == "private:2537134688"
    assert queued_results[0]["reply_ticket_id"] == ticket.ticket_id
    assert "后台终端任务结果" in queued_results[0]["message"]
    assert "apt-get install -y temurin-17-jdk" in queued_results[0]["message"]
    assert "temurin installed" in queued_results[0]["message"]


def test_background_job_reply_ticket_message_hides_command_and_keeps_result() -> None:
    server = LocalMCPServer()

    message = server._format_background_job_reply_ticket_message(
        "find / -name server.jar",
        {
            "success": True,
            "exit_code": 0,
            "duration_seconds": 0.2,
            "output_tail": "./server.jar",
        },
    )

    assert "后台任务已完成" in message
    assert "./server.jar" in message
    assert "find / -name server.jar" not in message


def test_background_run_job_passes_web_reply_ticket_to_terminal_feedback(
    monkeypatch,
) -> None:
    import threading
    from types import SimpleNamespace

    from app.services.agent.reply_ticket import reply_ticket_manager
    from app.services.agent.session import agent_session_manager

    item_id = "item-background-job-web-ticket"
    server = LocalMCPServer()
    _allow_main_terminal(monkeypatch, server)
    started = threading.Event()
    delivered = threading.Event()
    processed_inputs: list[object] = []

    class FakeConnection:
        def run_job_http(self, **kwargs):
            started.set()
            self.command = kwargs["command"]
            return {"success": True, "job_id": "job-web-ticket"}

        def get_job_result_http(self, **kwargs):
            return {
                "success": True,
                "status": "finished",
                "result": {
                    "success": False,
                    "error": "daemon job failed",
                    "job_id": "job-web-ticket",
                    "command": self.command,
                    "cwd": "/workspace/item",
                    "exit_code": 127,
                    "timed_out": False,
                    "duration_seconds": 0.02,
                    "output_tail": "/bin/sh: 1: java: not found",
                },
            }

    monkeypatch.setattr(
        server,
        "_get_item_daemon_context",
        lambda item_id: (
            SimpleNamespace(owner_id="user-1", working_directory="/workspace/item", title="Test Item"),
            FakeConnection(),
        ),
    )

    agent_session_manager.remove_session(item_id)
    session = agent_session_manager.get_or_create_session(item_id, "handler-1")

    def fake_process_input(input_msg):
        processed_inputs.append(input_msg)
        delivered.set()

    monkeypatch.setattr(session, "process_input", fake_process_input)
    fake_agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="",
            robot_context_token="",
            reply_ticket_id="",
        )
    )

    reply_ticket_manager.reset()
    try:
        ticket = reply_ticket_manager.create_for_agent(
            fake_agent,
            item_id=item_id,
            handler_id="handler-1",
            message="java 还在不",
        )
        result = server.call_tool(
            "run_job",
            {
                "item_id": item_id,
                "command": "java -version 2>&1",
                "_reply_ticket_id": ticket.ticket_id,
            },
        )
        assert started.wait(2)
        assert delivered.wait(2)
    finally:
        agent_session_manager.remove_session(item_id)
        reply_ticket_manager.reset()

    assert result[0]["type"] == "text"
    assert len(processed_inputs) == 1
    assert processed_inputs[0].reply_ticket_id == ticket.ticket_id
    assert processed_inputs[0].content.startswith("[Background terminal job failed]")


def test_web_reply_ticket_blocks_robot_send_for_terminal_feedback() -> None:
    from types import SimpleNamespace

    from app.services.agent.reply_ticket import reply_ticket_manager
    from app.services.agent.session import agent_session_manager

    item_id = "item-web-ticket-blocks-qq"
    session = agent_session_manager.get_or_create_session(item_id, "handler-1")
    fake_agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="",
            robot_context_token="",
            reply_ticket_id="",
        )
    )

    reply_ticket_manager.reset()
    try:
        ticket = reply_ticket_manager.create_for_agent(
            fake_agent,
            item_id=item_id,
            handler_id="handler-1",
            message="java 还在不",
        )

        warning = session._robot_send_blocked_by_reply_ticket(
            ticket.ticket_id,
            "mcp_robot_send_message",
        )
        prompt = session._build_reply_ticket_prompt(ticket.ticket_id)
    finally:
        agent_session_manager.remove_session(item_id)
        reply_ticket_manager.reset()

    assert "QQ" in warning
    assert "TermPaws web chat" in warning
    assert "REPLY ROUTING" in prompt


def test_list_jobs_reports_active_daemon_jobs(monkeypatch) -> None:
    from types import SimpleNamespace

    server = LocalMCPServer()
    captured: dict[str, object] = {}

    class FakeConnection:
        def list_jobs_http(self, **kwargs):
            captured.update(kwargs)
            return {
                "success": True,
                "jobs": [
                    {
                        "job_id": "job-7",
                        "item_uuid": "item-1",
                        "command": "apt-get install -y temurin-17-jdk",
                        "elapsed_seconds": 12.4,
                        "cancel_requested": False,
                        "output_tail": "Reading package lists...\nInstalling temurin-17-jdk",
                    }
                ],
                "count": 1,
            }

    monkeypatch.setattr(
        server,
        "_get_item_daemon_context",
        lambda item_id: (
            SimpleNamespace(owner_id="user-1", working_directory="/workspace/item", title="Test Item"),
            FakeConnection(),
        ),
    )

    result = server.call_tool("list_jobs", {"item_id": "item-1"})

    assert captured == {"item_uuid": "item-1"}
    assert len(result) == 1
    text = result[0]["text"]
    assert "job_id=job-7" in text
    assert "apt-get install -y temurin-17-jdk" in text
    assert "Installing temurin-17-jdk" in text


def test_cancel_job_tool_cancels_selected_daemon_job(monkeypatch) -> None:
    from types import SimpleNamespace

    server = LocalMCPServer()
    captured: list[dict[str, object]] = []

    class FakeConnection:
        def cancel_job_http(self, **kwargs):
            captured.append(kwargs)
            return {"success": True, "cancelled": True, "job_id": kwargs["job_id"]}

    monkeypatch.setattr(
        server,
        "_get_item_daemon_context",
        lambda item_id: (
            SimpleNamespace(owner_id="user-1", working_directory="/workspace/item", title="Test Item"),
            FakeConnection(),
        ),
    )

    result = server.call_tool(
        "cancel_job",
        {"item_id": "item-1", "job_id": "job-7"},
    )

    assert captured == [{"item_uuid": "item-1", "job_id": "job-7"}]
    assert "job-7" in result[0]["text"]


def test_run_job_allows_distinct_background_jobs_but_blocks_duplicates(monkeypatch) -> None:
    import threading
    from types import SimpleNamespace

    from app.services.agent.session import agent_session_manager

    item_id = "item-running-job"
    server = LocalMCPServer()
    _allow_main_terminal(monkeypatch, server)
    commands: list[str] = []
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()

    class FakeConnection:
        def __init__(self):
            self.commands_by_job = {}

        def run_job_http(self, **kwargs):
            command = kwargs["command"]
            commands.append(command)
            session = agent_session_manager.get_session(item_id)
            assert session is not None
            assert session.has_running_terminal_job() is True
            job_id = f"job-{len(commands)}"
            self.commands_by_job[job_id] = command
            return {"success": True, "job_id": job_id}

        def get_job_result_http(self, **kwargs):
            command = self.commands_by_job.get(kwargs["job_id"], "")
            if command == "curl https://example.test/file -o file":
                first_started.set()
                release_first.wait(2)
            if command == "python -m pip install demo-package":
                second_started.set()
            return {
                "success": True,
                "status": "finished",
                "result": {
                    "success": True,
                    "job_id": kwargs["job_id"],
                    "command": command,
                    "cwd": "/workspace/item",
                    "exit_code": 0,
                    "timed_out": False,
                    "duration_seconds": 2.0,
                    "output_tail": "done",
                },
            }

    monkeypatch.setattr(
        server,
        "_get_item_daemon_context",
        lambda item_id: (
            SimpleNamespace(owner_id="user-1", working_directory="/workspace/item", title="Test Item"),
            FakeConnection(),
        ),
    )

    agent_session_manager.remove_session(item_id)
    session = agent_session_manager.get_or_create_session(item_id, "handler-1")
    try:
        first_result = server.call_tool(
            "run_job",
            {
                "item_id": item_id,
                "command": "curl https://example.test/file -o file",
            },
        )
        assert first_started.wait(2)
        duplicate_result = server.call_tool(
            "run_job",
            {
                "item_id": item_id,
                "command": "curl https://example.test/file -o file",
            },
        )
        second_result = server.call_tool(
            "run_job",
            {
                "item_id": item_id,
                "command": "python -m pip install demo-package",
            },
        )
        assert second_started.wait(2)
        release_first.set()
    finally:
        release_first.set()
        agent_session_manager.remove_session(item_id)

    assert first_result[0]["type"] == "text"
    assert duplicate_result[0]["type"] == "text"
    assert second_result[0]["type"] == "text"
    # duplicates are no longer blocked; the agent sees a running-jobs hint instead
    assert commands.count("curl https://example.test/file -o file") == 2
    assert "后台任务" in duplicate_result[0]["text"]
    assert "python -m pip install demo-package" in commands


def test_interrupt_command_cancels_running_daemon_job(monkeypatch) -> None:
    from types import SimpleNamespace

    from app.services.agent.session import RUN_JOB_TOOL_NAME, agent_session_manager

    item_id = "item-cancel-job"
    server = LocalMCPServer()
    cancelled: list[dict] = []

    class FakeConnection:
        def cancel_job_http(self, **kwargs):
            cancelled.append(kwargs)
            return {"success": True, "cancelled": True, "job_id": "job-1"}

    monkeypatch.setattr(
        server,
        "_get_item_daemon_context",
        lambda item_id: (
            SimpleNamespace(owner_id="user-1", working_directory="/workspace/item", title="Test Item"),
            FakeConnection(),
        ),
    )

    agent_session_manager.remove_session(item_id)
    session = agent_session_manager.get_or_create_session(item_id, "handler-1")
    session.mark_terminal_job_started(
        RUN_JOB_TOOL_NAME,
        {"item_id": item_id, "command": "apt-get install -y temurin-17-jdk"},
    )
    try:
        result = server.call_tool("interrupt_command", {"item_id": item_id})
    finally:
        agent_session_manager.remove_session(item_id)

    assert cancelled == [{"item_uuid": item_id}]
    assert "\u540e\u53f0\u4efb\u52a1\u5df2\u4e2d\u65ad" in result[0]["text"]
    assert session.has_running_terminal_job() is False


def test_run_job_tool_is_registered_for_long_jobs() -> None:
    server = LocalMCPServer()
    tool = next(tool for tool in server.list_tools() if tool["name"] == "run_job")

    assert "后台一次性命令" in tool["description"]
    assert tool["inputSchema"]["properties"]["timeout_seconds"]["default"] == 1200
    assert tool["skip_memory"] is True


def test_local_compress_memories_merges_and_deletes(monkeypatch) -> None:
    from app.services.agent.memory.vector_store import vector_store

    server = LocalMCPServer()
    tools = {tool["name"]: tool for tool in server.list_tools()}
    assert "compress_memories" in tools

    existing = [
        {
            "id": "aaaa1111-0000-4000-8000-000000000001",
            "content": "端口是 43906",
            "metadata": {"memory_type": "fact"},
        },
        {
            "id": "bbbb2222-0000-4000-8000-000000000002",
            "content": "服务器端口其实是 43906",
            "metadata": {"memory_type": "fact"},
        },
    ]
    added: dict[str, object] = {}
    deleted: list[str] = []

    monkeypatch.setattr(
        vector_store,
        "get_all_memories",
        lambda *_args, **_kwargs: existing,
    )

    def fake_add_memory(**kwargs):
        added.update(kwargs)
        return "cccc3333-0000-4000-8000-000000000003"

    def fake_delete_memory(memory_id):
        deleted.append(str(memory_id))
        return True

    monkeypatch.setattr(vector_store, "add_memory", fake_add_memory)
    monkeypatch.setattr(vector_store, "delete_memory", fake_delete_memory)

    result = server.call_tool(
        "compress_memories",
        {
            "item_id": "item-1",
            "memory_ids": ["aaaa1111", "bbbb2222"],
            "content": "服务器端口是 43906",
        },
    )

    text = result[0]["text"]
    assert "压缩" in text
    assert added["handler_id"] == "item-1"
    assert added["content"] == "服务器端口是 43906"
    assert added["memory_type"] == "fact"
    assert added["metadata"]["source"] == "local_agent_compress"
    assert deleted == [memory["id"] for memory in existing]


def test_local_compress_memories_requires_two_ids() -> None:
    server = LocalMCPServer()

    result = server.call_tool(
        "compress_memories",
        {
            "item_id": "item-1",
            "memory_ids": ["aaaa1111"],
            "content": "合并后的记忆",
        },
    )

    assert "至少" in result[0]["text"]


def test_job_results_merge_into_one_batch_on_turn_end(monkeypatch) -> None:
    from app.services.agent.mcp import local_server as mcp_module

    sent: list[str] = []
    monkeypatch.setattr(
        mcp_module.local_mcp_server,
        "_deliver_background_job_to_robot",
        lambda **kwargs: sent.append(kwargs.get("message_override") or "") or True,
    )
    context = {
        "robot_id": "r1",
        "conversation_key": "group:g1",
        "sender_key": "s1",
        "reply_target": {},
    }
    mcp_module.buffer_background_job_result(
        "item-1",
        {
            "command": "apt-get update",
            "result": {"success": True, "exit_code": 0, "duration_seconds": 2.0},
            "robot_job_context": context,
            "conversation_key": "group:g1",
            "feedback": "fb-1",
            "reply_ticket_id": "ticket-1",
        },
    )
    mcp_module.buffer_background_job_result(
        "item-1",
        {
            "command": "apt-get install -y openjdk-21-jdk-headless",
            "result": {
                "success": True,
                "exit_code": 0,
                "duration_seconds": 10.0,
                "output_tail": "Setting up openjdk",
            },
            "robot_job_context": context,
            "conversation_key": "group:g1",
            "feedback": "fb-2",
            "reply_ticket_id": "ticket-2",
        },
    )

    assert mcp_module.flush_job_results_for_turn_end("item-1", "group:g1") is True
    assert len(sent) == 1, "two finished jobs should merge into one batch message"
    assert "batch: 2 jobs finished" in sent[0]
    assert "apt-get update" in sent[0]
    assert "apt-get install -y openjdk-21-jdk-headless" in sent[0]
    # buffer drained
    assert mcp_module.flush_job_results_for_turn_end("item-1", "group:g1") is False


def test_single_job_result_flushes_as_is(monkeypatch) -> None:
    from app.services.agent.mcp import local_server as mcp_module

    sent: list[str] = []
    monkeypatch.setattr(
        mcp_module.local_mcp_server,
        "_deliver_background_job_to_robot",
        lambda **kwargs: sent.append(kwargs.get("message_override") or "") or True,
    )
    context = {
        "robot_id": "r1",
        "conversation_key": "group:g1",
        "sender_key": "s1",
        "reply_target": {},
    }
    entry = {
        "command": "apt-get update",
        "result": {"success": True, "exit_code": 0, "duration_seconds": 2.0},
        "robot_job_context": context,
        "conversation_key": "group:g1",
        "feedback": "job done text",
        "reply_ticket_id": "ticket-1",
    }

    assert mcp_module.flush_background_job_results_for_entries("item-1", [entry]) is True
    assert len(sent) == 1
    assert "apt-get update" in sent[0]


def test_session_turn_busy_detection() -> None:
    import queue as queue_module
    import threading
    from types import SimpleNamespace

    from app.services.agent.mcp.local_server import _session_turn_busy
    from app.services.agent.session import SessionState

    session = SimpleNamespace(
        lock=threading.RLock(),
        state=SessionState.IDLE,
        input_queue=queue_module.Queue(),
    )
    assert _session_turn_busy(session) is False
    session.state = SessionState.RUNNING
    assert _session_turn_busy(session) is True
    session.state = SessionState.IDLE
    session.input_queue.put("x")
    assert _session_turn_busy(session) is True
    assert _session_turn_busy(None) is False


def test_cancel_background_jobs_for_item_filters_commands(monkeypatch) -> None:
    from app.services.agent.mcp.local_server import (
        cancel_background_jobs_for_item,
        local_mcp_server,
    )

    cancel_calls: list[str] = []

    class FakeConnection:
        def list_jobs_http(self, **_kwargs):
            return {
                "success": True,
                "jobs": [
                    {"job_id": "j1", "command": "apt-get install java"},
                    {"job_id": "j2", "command": "npm install"},
                ],
            }

        def cancel_job_http(self, *, job_id, **_kwargs):
            cancel_calls.append(job_id)
            return {"success": True}

    monkeypatch.setattr(
        local_mcp_server,
        "_get_item_daemon_context",
        lambda _item_id: (None, FakeConnection()),
    )

    cancelled = cancel_background_jobs_for_item(
        "item-1",
        commands={"apt-get install java"},
    )
    assert cancelled == 1
    assert cancel_calls == ["j1"]

    cancel_calls.clear()
    cancelled = cancel_background_jobs_for_item("item-1", commands=None)
    assert cancelled == 2
    assert cancel_calls == ["j1", "j2"]


def test_cancel_background_jobs_for_item_daemon_down(monkeypatch) -> None:
    from app.services.agent.mcp.local_server import (
        cancel_background_jobs_for_item,
        local_mcp_server,
    )

    monkeypatch.setattr(
        local_mcp_server,
        "_get_item_daemon_context",
        lambda _item_id: (_ for _ in ()).throw(RuntimeError("daemon down")),
    )
    assert cancel_background_jobs_for_item("item-1") == 0


def _patch_reply_ticket_manager(monkeypatch):
    from types import SimpleNamespace

    import app.services.agent.reply_ticket as reply_ticket_module

    manager = reply_ticket_module.ReplyTicketManager()
    monkeypatch.setattr(reply_ticket_module, "reply_ticket_manager", manager)

    def make_ticket(item_id: str = "item-1", message: str = "request"):
        agent = SimpleNamespace(
            _context=SimpleNamespace(
                robot_id="",
                robot_context_token="",
                reply_ticket_id="",
            )
        )
        return manager.create_for_agent(
            agent,
            item_id=item_id,
            handler_id="handler-1",
            message=message,
            source_type="web",
        )

    return manager, make_ticket


def test_update_plan_writes_to_owning_reply_ticket(monkeypatch) -> None:
    manager, make_ticket = _patch_reply_ticket_manager(monkeypatch)
    ticket_a = make_ticket(message="task A")
    ticket_b = make_ticket(message="task B")
    server = LocalMCPServer()

    result = server.call_tool(
        "update_plan",
        {
            "item_id": "item-1",
            "_reply_ticket_id": ticket_a.ticket_id,
            "plan": [
                {"step": "download", "status": "completed"},
                {"step": "install", "status": "in_progress"},
                {"step": "", "status": "pending"},
                {"step": "verify", "status": "bogus-status"},
            ],
        },
    )

    assert "Plan updated:" in result[0]["text"]
    # Normalization keeps empty steps out and coerces unknown statuses.
    assert ticket_a.plan == [
        {"step": "download", "status": "completed"},
        {"step": "install", "status": "in_progress"},
        {"step": "verify", "status": "pending"},
    ]
    # The other ticket of the same item is untouched.
    assert ticket_b.plan == []


def test_update_plan_all_completed_clears_ticket_plan(monkeypatch) -> None:
    _manager, make_ticket = _patch_reply_ticket_manager(monkeypatch)
    ticket = make_ticket()
    ticket.plan = [{"step": "install", "status": "in_progress"}]
    server = LocalMCPServer()

    result = server.call_tool(
        "update_plan",
        {
            "item_id": "item-1",
            "_reply_ticket_id": ticket.ticket_id,
            "plan": [{"step": "install", "status": "completed"}],
        },
    )

    assert ticket.plan == []
    assert "All steps complete" in result[0]["text"]


def test_update_plan_without_ticket_id_is_rejected(monkeypatch) -> None:
    manager, make_ticket = _patch_reply_ticket_manager(monkeypatch)
    ticket = make_ticket()
    server = LocalMCPServer()

    result = server.call_tool(
        "update_plan",
        {"item_id": "item-1", "plan": [{"step": "s", "status": "pending"}]},
    )
    assert "无法关联 ticket" in result[0]["text"]
    assert ticket.plan == []


def test_update_plan_on_expired_ticket_renews_it(monkeypatch) -> None:
    manager, make_ticket = _patch_reply_ticket_manager(monkeypatch)
    make_ticket()
    server = LocalMCPServer()

    result = server.call_tool(
        "update_plan",
        {
            "item_id": "item-1",
            "_reply_ticket_id": "missing-ticket",
            "plan": [{"step": "install", "status": "in_progress"}],
        },
    )
    assert "Plan updated" in result[0]["text"]
    renewed = manager.get("missing-ticket")
    assert renewed is not None
    assert renewed.ticket_id != "missing-ticket"
    assert renewed.plan == [{"step": "install", "status": "in_progress"}]


def test_update_plan_identical_write_is_deduped(monkeypatch) -> None:
    _manager, make_ticket = _patch_reply_ticket_manager(monkeypatch)
    ticket = make_ticket()
    server = LocalMCPServer()
    plan = [
        {"step": "download", "status": "completed"},
        {"step": "install", "status": "in_progress"},
    ]
    args = {"item_id": "item-1", "_reply_ticket_id": ticket.ticket_id}

    first = server.call_tool("update_plan", {**args, "plan": plan})
    assert "Plan updated:" in first[0]["text"]

    second = server.call_tool("update_plan", {**args, "plan": plan})
    assert "无变化" in second[0]["text"]
    assert ticket.plan == [
        {"step": "download", "status": "completed"},
        {"step": "install", "status": "in_progress"},
    ]

    changed = server.call_tool(
        "update_plan",
        {
            **args,
            "plan": [
                {"step": "download", "status": "completed"},
                {"step": "install", "status": "completed"},
            ],
        },
    )
    assert "无变化" not in changed[0]["text"]


def test_update_plan_warns_when_open_step_silently_dropped(monkeypatch) -> None:
    _manager, make_ticket = _patch_reply_ticket_manager(monkeypatch)
    ticket = make_ticket()
    server = LocalMCPServer()
    args = {"item_id": "item-1", "_reply_ticket_id": ticket.ticket_id}

    server.call_tool(
        "update_plan",
        {
            **args,
            "plan": [
                {"step": "启动服务器并验证", "status": "in_progress"},
                {"step": "关服并汇报结果", "status": "pending"},
            ],
        },
    )
    result = server.call_tool(
        "update_plan",
        {
            **args,
            "plan": [
                {"step": "启动服务器并验证", "status": "completed"},
                {"step": "汇报最终结果", "status": "in_progress"},
            ],
        },
    )
    text = result[0]["text"]
    assert "凭空消失" in text
    assert "关服并汇报结果" in text

    # 显式 cancelled 不算蒸发，不告警
    result2 = server.call_tool(
        "update_plan",
        {
            **args,
            "plan": [
                {"step": "启动服务器并验证", "status": "completed"},
                {"step": "汇报最终结果", "status": "completed"},
            ],
        },
    )
    assert "凭空消失" not in result2[0]["text"]


def test_update_plan_closure_audit_echoes_original_request(monkeypatch) -> None:
    _manager, make_ticket = _patch_reply_ticket_manager(monkeypatch)
    ticket = make_ticket(message="跑个构建然后关服再汇报")
    ticket.plan = [{"step": "构建", "status": "in_progress"}]
    server = LocalMCPServer()

    result = server.call_tool(
        "update_plan",
        {
            "item_id": "item-1",
            "_reply_ticket_id": ticket.ticket_id,
            "plan": [{"step": "构建", "status": "completed"}],
        },
    )
    assert "收尾审计" in result[0]["text"]
    assert "跑个构建然后关服再汇报" in result[0]["text"]

    ticket.plan = [{"step": "汇报", "status": "in_progress"}]
    result2 = server.call_tool(
        "update_plan",
        {
            "item_id": "item-1",
            "_reply_ticket_id": ticket.ticket_id,
            "plan": [],
        },
    )
    assert "Plan completed." in result2[0]["text"]
    assert "收尾审计" in result2[0]["text"]


def test_closure_audit_includes_action_ledger_and_undelivered(monkeypatch) -> None:
    manager, make_ticket = _patch_reply_ticket_manager(monkeypatch)
    ticket = make_ticket(message="构建然后关服")
    ticket.plan = [{"step": "构建", "status": "in_progress"}]
    manager.record_action(ticket.ticket_id, "job OK: gradle build")
    manager.record_action(ticket.ticket_id, "job FAIL(exit=127): curl → not found")
    manager.mark_report_undelivered(ticket.ticket_id, "500")
    server = LocalMCPServer()

    result = server.call_tool(
        "update_plan",
        {
            "item_id": "item-1",
            "_reply_ticket_id": ticket.ticket_id,
            "plan": [{"step": "构建", "status": "completed"}],
        },
    )
    text = result[0]["text"]
    assert "收尾审计" in text
    assert "构建然后关服" in text
    assert "job OK: gradle build" in text
    assert "job FAIL(exit=127): curl → not found" in text
    assert "未送达" in text


def test_record_action_caps_ledger_and_report_flags_roundtrip(monkeypatch) -> None:
    manager, make_ticket = _patch_reply_ticket_manager(monkeypatch)
    ticket = make_ticket(message="x")

    for i in range(15):
        manager.record_action(ticket.ticket_id, f"action-{i}")
    assert len(ticket.action_ledger) == 12
    assert ticket.action_ledger[0] == "action-3"

    manager.mark_report_undelivered(ticket.ticket_id, "500")
    undelivered = manager.undelivered_reports_for_item("item-1")
    assert len(undelivered) == 1
    assert undelivered[0]["ticket_id"] == ticket.ticket_id

    manager.mark_report_delivered(ticket.ticket_id)
    assert manager.undelivered_reports_for_item("item-1") == []
    assert any("已送达" in line for line in ticket.action_ledger)


def test_plan_reminder_reads_plan_of_owning_ticket(monkeypatch) -> None:
    _manager, make_ticket = _patch_reply_ticket_manager(monkeypatch)
    ticket_a = make_ticket(message="task A")
    ticket_b = make_ticket(message="task B")
    ticket_a.plan = [{"step": "plan A step", "status": "in_progress"}]
    ticket_b.plan = [{"step": "plan B step", "status": "pending"}]
    server = LocalMCPServer()

    reminder_b = server._plan_reminder("item-1", ticket_b.ticket_id)
    assert "plan B step" in reminder_b
    assert "plan A step" not in reminder_b

    reminder_a = server._plan_reminder("item-1", ticket_a.ticket_id)
    assert "plan A step" in reminder_a

    assert server._plan_reminder("item-1", "") == ""
    assert server._plan_reminder("item-1", "missing-ticket") == ""
