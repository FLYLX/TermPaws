"""run_job 知情引导：结果里带在跑任务清单，agent 自己判断，无硬上限。"""

from types import SimpleNamespace

from app.services.agent.mcp.local_server import LocalMCPServer


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


def test_run_job_shows_running_jobs_in_result(monkeypatch) -> None:
    """结果里必须带在跑任务清单和"同目的别重复发起"的引导，且无硬拒绝。"""
    server, _ = _prep_server(monkeypatch, 3)
    result = server._run_job(
        {"item_id": "item-1", "command": "apt-get install java", "timeout_seconds": 60}
    )
    text = result[0]["text"]
    assert "后台任务已启动" in text  # 不拒绝，照常启动
    assert "3 个后台任务在跑" in text
    assert "不要重复发起" in text
    assert "cancel_job" in text


def test_run_job_without_running_jobs_no_hint(monkeypatch) -> None:
    server, _ = _prep_server(monkeypatch, 0)
    result = server._run_job(
        {"item_id": "item-1", "command": "echo hi", "timeout_seconds": 60}
    )
    assert "后台任务已启动" in result[0]["text"]
    assert "不要重复发起" not in result[0]["text"]
