"""OneBot V11 反向 WS 模拟器：不依赖真实 QQ/NapCat 的压测客户端。

模拟 NapCat 连接 TermPaws 的 robot-bridge，发送伪造的群/私聊消息事件，
并把 bot 侧 API 调用（send_msg 等）记录下来作为路由断言依据。

用法：
    sim = OnebotSim("ws://localhost:28888/robot-bridge/onebot/v11/ws",
                    access_token="test", self_id=123456)
    await sim.run(scenario_steps, duration_s=600)

scenario_steps: [(delay_s, spec)] — spec 用 OnebotSim.group(...) / .private(...)
返回 sends: [{ts, action, group_id, user_id, message}]（含收到时间）。
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class CapturedSend:
    ts: str
    action: str
    group_id: int | None
    user_id: int | None
    message: str
    has_image: bool = False


class OnebotSim:
    def __init__(self, ws_url: str, *, access_token: str, self_id: int):
        self.ws_url = ws_url
        self.access_token = access_token
        self.self_id = self_id
        self.sends: list[CapturedSend] = []
        self._mid = 1000
        self._echo_seq = 0
        # on_send 回调用于触发多轮追问（收到某会话的回复后自动发下一条）
        self.on_send = None  # type: ignore[assignment]

    @staticmethod
    def group(group_id: int, user_id: int, nickname: str, text: str) -> dict[str, Any]:
        return {"kind": "group", "group_id": group_id, "user_id": user_id,
                "nickname": nickname, "text": text}

    @staticmethod
    def private(user_id: int, nickname: str, text: str) -> dict[str, Any]:
        return {"kind": "private", "user_id": user_id, "nickname": nickname,
                "text": text}

    def _event(self, spec: dict[str, Any]) -> dict[str, Any]:
        self._mid += 1
        ev: dict[str, Any] = {
            "time": int(time.time()),
            "self_id": self.self_id,
            "post_type": "message",
            "message_type": spec["kind"],
            "sub_type": "normal",
            "message_id": self._mid,
            "user_id": spec["user_id"],
            "raw_message": spec["text"],
            "message": [{"type": "text", "data": {"text": spec["text"]}}],
            "sender": {"user_id": spec["user_id"], "nickname": spec["nickname"]},
            "font": 0,
        }
        if spec["kind"] == "group":
            ev["group_id"] = spec["group_id"]
        return ev

    async def run(self, steps: list[tuple[float, dict[str, Any]]], *,
                  duration_s: float,
                  web_steps: list[tuple[float, Any]] | None = None):
        """web_steps: [(delay_s, 无参回调)] 与 QQ 时间线共用同一个调度循环，
        避免 Timer 线程在 WMI 托管进程里被回收。"""
        import websockets

        headers = {"Authorization": f"Bearer {self.access_token}",
                   "X-Self-ID": str(self.self_id)}
        async with websockets.connect(self.ws_url, additional_headers=headers,
                                      ping_interval=None) as ws:
            t0 = time.time()
            sent = 0
            web_sent = 0
            web_steps = web_steps or []
            deadline = t0 + duration_s
            while time.time() < deadline:
                now = time.time() - t0
                while sent < len(steps) and now >= steps[sent][0]:
                    await ws.send(json.dumps(self._event(steps[sent][1]),
                                             ensure_ascii=False))
                    sent += 1
                while web_sent < len(web_steps) and now >= web_steps[web_sent][0]:
                    web_steps[web_sent][1]()
                    web_sent += 1
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10)
                except asyncio.TimeoutError:
                    continue
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                action = data.get("action")
                if not action:
                    continue
                params = data.get("params") or {}
                if str(action).startswith("send"):
                    msg = params.get("message")
                    has_image = False
                    if isinstance(msg, list):
                        has_image = any(
                            isinstance(s, dict) and s.get("type") == "image"
                            for s in msg
                        )
                        msg = "".join(s.get("data", {}).get("text", "")
                                      for s in msg
                                      if isinstance(s, dict) and s.get("type") == "text")
                    rec = CapturedSend(
                        ts=time.strftime("%H:%M:%S"),
                        action=str(action),
                        group_id=params.get("group_id"),
                        user_id=params.get("user_id"),
                        message=str(msg)[:400],
                        has_image=has_image,
                    )
                    self.sends.append(rec)
                    if self.on_send:
                        followups = self.on_send(rec)
                        if followups:
                            for spec in followups:
                                await ws.send(json.dumps(self._event(spec),
                                                         ensure_ascii=False))
                self._echo_seq += 1
                await ws.send(json.dumps({
                    "status": "ok", "retcode": 0,
                    "data": {"message_id": 9000 + self._echo_seq},
                    "echo": data.get("echo"),
                }))
