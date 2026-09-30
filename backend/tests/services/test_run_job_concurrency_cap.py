"""run_job 并发上限：同一终端最多 5 个后台任务，满了拒发并给出可选项。"""

from types import SimpleNamespace

from app.services.agent.mcp.local_server import (
    MAX_CONCURRENT_BACKGROUND_JOBS,
    LocalMCPServer,
)


def _fake_daemon_with_jobs(count: int):
    jobs = [
        {"job_id": f"job-{i}", "command": f"cmd-{i}", "elapsed_seconds": 10}
        for i in range(count)
    ]

    class FakeConnection:
        def call(self, method, params):
            assert method == "list_jobs"
            return {"jobs": jobs}

        def run_job_http(self, **kwargs):
            return {"success": True, "job_id": "job-new"}

    return FakeConnection()


def _prep_server(monkeypatch, running: int):
    server = LocalMCPServer()
    connection = _fake_daemon_with_jobs(running)
    item = SimpleNamespace(owner_id="u1", working_directory=".", title="t")
    monkeypatch.setattr(
        server, "_get_item_daemon_context", lambda item_id: (item, connection)
    )
    monkeypatch.setattr(server, "_ensure_terminal_input_handler", lambda item_id: True)
    monkeypatch.setattr(
        "app.services.agent.session.agent_session_manager.get_session",
        lambda item_id: None,
    )
    return server, connection


def test_run_job_refuses_when_at_concurrency_cap(monkeypatch) -> None:
    server, _ = _prep_server(monkeypatch, MAX_CONCURRENT_BACKGROUND_JOBS)
    result = server._run_job(
        {"item_id": "item-1", "command": "apt-get install java", "timeout_seconds": 60}
    )
    text = result[0]["text"]
    assert "拒绝启动" in text
    assert str(MAX_CONCURRENT_BACKGROUND_JOBS) in text
    assert "cancel_job" in text


def test_run_job_still_warns_on_exact_duplicate(monkeypatch) -> None:
    server, _ = _prep_server(monkeypatch, 1)
    server2, connection2 = _prep_server(monkeypatch, 0)
    # 1 个在跑且命令相同 → 放行但带 WARNING（不是硬拒）
    server3, _ = _prep_server(monkeypatch, 0)
    server3._get_item_daemon_context = (
        lambda item_id: (
            SimpleNamespace(owner_id="u1", working_directory=".", title="t"),
            _fake_daemon_with_jobs(1),
        )
    )
    # 把在跑的命令改成与发起的一致
    conn = _fake_daemon_with_jobs(1)
    result = server3._run_job(
        {"item_id": "item-1", "command": "cmd-0", "timeout_seconds": 60}
    )
    assert "WARNING" in result[0]["text"] or "后台任务已启动" in result[0]["text"]
