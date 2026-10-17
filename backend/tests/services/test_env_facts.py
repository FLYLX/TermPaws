"""环境事实沉淀：缺失工具黑名单的自动写入、计数、放行口与恢复。"""

from sqlmodel import Session

from app.core.db import engine
from app.services.filters.env_facts import (
    ENV_RULE_PREFIX,
    extract_missing_tool,
    extract_verified_tool,
    list_missing_tools,
    sediment_job_result,
)
from app.services.filters.output_filter import OutputFilter, OutputFilterConfig
from app.models import Item


def _item_rules(item_id: str) -> dict:
    with Session(engine) as db:
        item = db.get(Item, __import__("uuid").UUID(item_id))
        return dict(item.output_filter_rules or {}), item.output_filter_enabled


def _filter_for(rules: dict) -> OutputFilter:
    from app.services.filters.output_filter import FilterRule

    filters = [
        FilterRule(
            name=name,
            regex_patterns=cfg.get("regex_patterns", []),
            action_type=cfg.get("action_type", "ignore"),
            reason=cfg.get("reason", ""),
        )
        for name, cfg in rules.items()
        if isinstance(cfg, dict)
    ]
    return OutputFilter(OutputFilterConfig(enabled=True, filters=filters))


def test_extract_missing_tool_shell_variants() -> None:
    assert extract_missing_tool("bash: curl: command not found") == "curl"
    assert extract_missing_tool("/bin/sh: 11: stop: not found") == "stop"
    assert extract_missing_tool("sh: 1: wget: not found") == "wget"
    assert extract_missing_tool("'unzip' is not recognized as an internal or external command") == "unzip"
    assert extract_missing_tool("") == ""
    assert extract_missing_tool("everything is fine") == ""


def test_extract_verified_tool_probes() -> None:
    assert extract_verified_tool("which python3") == "python3"
    assert extract_verified_tool("command -v curl") == "curl"
    assert extract_verified_tool("cd /tmp && which git") == "git"
    assert extract_verified_tool("java --version") == "java"
    assert extract_verified_tool("curl http://example.com") == ""


def test_sediment_writes_blacklist_rule_and_blocks_usage(db) -> None:
    from tests.utils.item import create_random_item

    item = create_random_item(db)
    summary = sediment_job_result(
        str(item.id),
        "curl -sL https://example.com -o /tmp/f.zip",
        success=False,
        exit_code=127,
        output_tail="/bin/sh: 4: curl: not found",
        job_id="abcdef123456",
    )
    assert summary["blacklisted"] == "curl"

    rules, enabled = _item_rules(str(item.id))
    assert enabled is True
    rule_name = f"{ENV_RULE_PREFIX}curl"
    assert rule_name in rules
    assert rules[rule_name]["action_type"] == "block"
    assert rules[rule_name]["count"] == 1

    filt = _filter_for(rules)
    blocked = filt.filter("curl https://example.com")
    assert blocked.is_blocked
    assert "curl" in blocked.reason
    assert "command -v curl" in blocked.reason

    assert filt.filter("command -v curl").is_allowed
    assert filt.filter("which curl").is_allowed
    assert filt.filter("apt install curl").is_allowed
    assert filt.filter("curl --version").is_allowed
    assert filt.filter("echo hello").is_allowed


def test_sediment_increments_count_on_repeat(db) -> None:
    from tests.utils.item import create_random_item

    item = create_random_item(db)
    for _ in range(2):
        sediment_job_result(
            str(item.id),
            "wget https://example.com",
            success=False,
            exit_code=127,
            output_tail="sh: 1: wget: not found",
            job_id="j1",
        )
    rules, _ = _item_rules(str(item.id))
    assert rules[f"{ENV_RULE_PREFIX}wget"]["count"] == 2
    entries = list_missing_tools(rules)
    assert entries[0]["tool"] == "wget"
    assert entries[0]["count"] == 2


def test_sediment_recovers_on_successful_verification(db) -> None:
    from tests.utils.item import create_random_item

    item = create_random_item(db)
    sediment_job_result(
        str(item.id),
        "curl https://example.com",
        success=False,
        exit_code=127,
        output_tail="bash: curl: command not found",
        job_id="j1",
    )
    rules, _ = _item_rules(str(item.id))
    assert f"{ENV_RULE_PREFIX}curl" in rules

    summary = sediment_job_result(
        str(item.id),
        "command -v curl",
        success=True,
        exit_code=0,
        output_tail="/usr/bin/curl",
        job_id="j2",
    )
    assert summary["recovered"] == "curl"
    assert summary["verified"] == "curl"
    rules, _ = _item_rules(str(item.id))
    assert f"{ENV_RULE_PREFIX}curl" not in rules


def test_sediment_ignores_successful_unrelated_jobs(db) -> None:
    from tests.utils.item import create_random_item

    item = create_random_item(db)
    summary = sediment_job_result(
        str(item.id),
        "echo curl is a nice word",
        success=True,
        exit_code=0,
        output_tail="curl is a nice word",
        job_id="j1",
    )
    assert summary["blacklisted"] == ""
    assert summary["recovered"] == ""
    rules, _ = _item_rules(str(item.id))
    assert not any(name.startswith(ENV_RULE_PREFIX) for name in rules)
