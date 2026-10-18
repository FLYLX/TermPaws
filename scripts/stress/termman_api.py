"""TermPaws HTTP API 封装（压测用）：登录、web 聊天、经 daemon 读远端文件。"""
from __future__ import annotations

import requests


class TermManApi:
    def __init__(self, base: str, email: str, password: str, item_id: str):
        self.base = base.rstrip("/")
        self.item_id = item_id
        r = requests.post(f"{self.base}/api/v1/login/access-token",
                          data={"username": email, "password": password}, timeout=30)
        r.raise_for_status()
        self._headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    def chat(self, message: str, timeout: int = 3600) -> str:
        r = requests.post(f"{self.base}/api/v1/chat/{self.item_id}",
                          headers=self._headers, json={"message": message},
                          timeout=timeout)
        r.raise_for_status()
        return str(r.json().get("content", ""))

    def chat_async(self, message: str) -> None:
        """发出去就完事（后台任务场景），不等最终汇报。"""
        import threading

        threading.Thread(target=self.chat, args=(message,), daemon=True).start()

    def read_remote_file(self, path: str) -> str:
        """经 TermMan 文件 API 读 daemon 侧文件（路径相对 daemon 根，不带前导/）。"""
        r = requests.get(
            f"{self.base}/api/v1/items/{self.item_id}/files/content",
            headers=self._headers, params={"path": path}, timeout=60,
        )
        r.raise_for_status()
        return str(r.json().get("content", ""))
