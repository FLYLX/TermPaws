"""暴力压测场景 + 断言。所有断言只看路由和最终文件状态，不看话术。

新增场景时往 SCENARIOS 里加一项即可。每个场景：
  name: 名字
  steps: [(delay_s, OnebotSim.group/private(...))]  QQ 侧消息时间线
  web: [(delay_s, "消息")]                            web 侧任务（异步发送）
  wait_s: 总观察时长
  expect_files: [(远端路径, 必含子串列表, 禁含子串列表)]
  expect_routes: [{target: ("group", 1001), must: ["关键词"], forbid: ["关键词"]}]
      路由断言：该 target 至少收到一条含全部 must 关键词的发送，
      且所有发送里不出现 forbid 关键词。
  forbid_global: 任何会话的回复都不应出现的关键词（如别的任务的标志词）
"""
from __future__ import annotations

from .onebot_sim import OnebotSim

BASE = "opt/termpaws-daemon/src/workdir/1/ftb_forge"


def _creeper_file() -> str:
    return f"{BASE}/world/datapacks/creeper_resist/data/creeper_resist/functions/tick.mcfunction"


def _tick_file() -> str:
    return f"{BASE}/world/datapacks/jump_speed/data/jump_speed/functions/tick.mcfunction"


SCENARIOS = {
    # 双群同改一项配置的不同值 + 改口 + 催促：路由隔离与交接
    "dual_group_same_property": {
        "steps": [
            (0, OnebotSim.group(1001, 2001, "甲", "paws 把 view-distance 改成 11，改完确认")),
            (2, OnebotSim.group(1002, 2003, "丙", "paws 把 view-distance 改成 13，改完确认")),
            (30, OnebotSim.group(1001, 2001, "甲", "paws 好了吗")),
            (35, OnebotSim.group(1002, 2003, "丙", "paws 我这边呢")),
            (120, OnebotSim.group(1001, 2001, "甲", "paws 再改成 12 吧")),
        ],
        "web": [],
        "wait_s": 900,
        "expect_files": [(f"{BASE}/server.properties", ["view-distance=12"], [])],
        "expect_routes": [
            {"target": ("group", 1001), "must": ["12"], "forbid": []},
            {"target": ("group", 1002), "must": ["13"], "forbid": []},
        ],
        "forbid_global": [],
    },
    # 私聊长任务 + 原样重发 + 催促：重复压制
    "duplicate_pressure": {
        "steps": [
            (0, OnebotSim.private(3001, "戊", "paws 把 creeper_resist 的抗性持续时间改成 7 秒，改完 reload 验证再告诉我")),
            (35, OnebotSim.private(3001, "戊", "paws 把 creeper_resist 的抗性持续时间改成 7 秒，改完 reload 验证再告诉我")),
            (70, OnebotSim.private(3001, "戊", "paws 搞定了吗")),
        ],
        "web": [],
        "wait_s": 600,
        "expect_files": [(_creeper_file(), ["resistance 7 "], [])],
        "expect_routes": [
            {"target": ("private", 3001), "must": ["7"], "forbid": []},
        ],
        "forbid_global": [],
    },
    # 全家桶：web 长任务 + 双群并发 + 私聊重复 + 新用户 + 多轮追问
    "kitchen_sink": {
        "steps": [
            (0, OnebotSim.group(1001, 2001, "甲", "paws 把 max-players 改成 90，改完确认")),
            (2, OnebotSim.group(1002, 2003, "丙", "paws 把 max-players 改成 100，改完确认")),
            (3, OnebotSim.private(3001, "戊", "paws 把 skeleton_regen 的再生改成 2 秒，改完 reload 验证再告诉我")),
            (5, OnebotSim.private(3002, "己", "paws 服务器难度现在是什么")),
            (30, OnebotSim.private(3001, "戊", "paws 把 skeleton_regen 的再生改成 2 秒，改完 reload 验证再告诉我")),
            (45, OnebotSim.group(1001, 2001, "甲", "paws 好了没")),
            (50, OnebotSim.group(1002, 2003, "丙", "paws 我这边改了吗")),
            (120, OnebotSim.group(1001, 2001, "甲", "paws 改成 85 吧")),
        ],
        "web": [(0, "WS：把「每日签到」章节里收集泥土的任务需求改成 24 个，ftbquests reload 确认后回复我")],
        "wait_s": 1200,
        "expect_files": [
            (f"{BASE}/server.properties", ["max-players=85"], []),
            (f"{BASE}/world/datapacks/skeleton_regen/data/skeleton_regen/functions/give_regen.mcfunction",
             ["regeneration 2 "], []),
            (f"{BASE}/config/ftbquests/quests/chapters/meiriqiandao.snbt", ["count: 24L"], []),
        ],
        "expect_routes": [
            {"target": ("group", 1001), "must": ["85"], "forbid": []},
            {"target": ("group", 1002), "must": ["100"], "forbid": ["85"]},
            {"target": ("private", 3001), "must": ["2"], "forbid": []},
            {"target": ("private", 3002), "must": [], "forbid": ["85", "90", "skeleton"]},
        ],
        "forbid_global": [],
    },
}
