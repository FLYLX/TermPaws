"""压测运行器：跑场景、断言路由与远端文件状态、出报告。

用法（需要本地已起 termpaws + daemon + FTB 服务器终端）：
    python -m scripts.stress.run dual_group_same_property
    python -m scripts.stress.run --all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from .onebot_sim import OnebotSim
from .scenarios import SCENARIOS
from .termman_api import TermManApi

WS_URL = "ws://localhost:28888/robot-bridge/onebot/v11/ws"
ITEM_ID = "014c5e0e-4965-4dda-8f43-e10f76485355"
REPORT_DIR = Path(__file__).resolve().parent / "reports"


def _check_routes(scenario: dict, sends) -> list[str]:
    failures: list[str] = []
    for rule in scenario.get("expect_routes", []):
        ttype, tid = rule["target"]
        hits = [
            s for s in sends
            if (s.group_id if ttype == "group" else s.user_id) == tid
        ]
        blob = "\n".join(s.message for s in hits)
        for kw in rule.get("must", []):
            if kw not in blob:
                failures.append(f"{ttype}:{tid} 的回复里没找到「{kw}」（共 {len(hits)} 条）")
        for s in hits:
            for kw in rule.get("forbid", []):
                if kw in s.message:
                    failures.append(f"{ttype}:{tid} 的回复里出现禁词「{kw}」: {s.message[:80]}")
    forbidden = scenario.get("forbid_global", [])
    for s in sends:
        for kw in forbidden:
            if kw in s.message:
                failures.append(f"串味：{s.group_id or s.user_id} 收到「{kw}」: {s.message[:80]}")
    return failures


def _check_files(api: TermManApi, scenario: dict) -> list[str]:
    failures: list[str] = []
    for path, musts, forbids in scenario.get("expect_files", []):
        try:
            content = api.read_remote_file(path)
        except Exception as exc:
            failures.append(f"读不到远端文件 {path}: {exc}")
            continue
        for kw in musts:
            if kw not in content:
                failures.append(f"{path} 缺少「{kw}」")
        for kw in forbids:
            if kw in content:
                failures.append(f"{path} 不应包含「{kw}」")
    return failures


def run_scenario(name: str, api: TermManApi) -> dict:
    scenario = SCENARIOS[name]
    sim = OnebotSim(WS_URL, access_token="test", self_id=123456)
    for delay, message in scenario.get("web", []):
        if delay == 0:
            api.chat_async(message)
        else:
            import threading

            threading.Timer(delay, api.chat_async, args=(message,)).start()
    asyncio.run(sim.run(scenario["steps"], duration_s=scenario["wait_s"]))
    time.sleep(5)
    failures = _check_routes(scenario, sim.sends)
    failures.extend(_check_files(api, scenario))
    return {
        "name": name,
        "ok": not failures,
        "failures": failures,
        "sends": [vars(s) for s in sim.sends],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", nargs="?", help="场景名；--all 跑全部")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    api = TermManApi("http://localhost:28888", "admin@example.com",
                     "BM-vmvbo66xR2G4Y", ITEM_ID)
    names = list(SCENARIOS) if args.all else [args.scenario]
    if not names or not names[0]:
        parser.error("指定场景名或 --all")

    REPORT_DIR.mkdir(exist_ok=True)
    results = []
    for name in names:
        print(f"=== scenario {name} ===", flush=True)
        result = run_scenario(name, api)
        results.append(result)
        status = "PASS" if result["ok"] else "FAIL"
        print(f"[{status}] {name}: {len(result['sends'])} sends, "
              f"{len(result['failures'])} failures", flush=True)
        for failure in result["failures"]:
            print(f"  - {failure}", flush=True)
        out = REPORT_DIR / f"{name}-{int(time.time())}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    failed = [r["name"] for r in results if not r["ok"]]
    print(f"\n== summary: {len(results) - len(failed)}/{len(results)} passed"
          + (f", failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
