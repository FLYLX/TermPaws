import os
import re
import select
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core import config, get_socket_service
from service.item_path_service import ItemPathError, item_path_service
from service.terminal_manager import daemon_log_manager
from utils.logger import logger


class JobRunnerError(Exception):
    pass


@dataclass
class ActiveJob:
    job_id: str
    item_uuid: str
    command: str
    pid: int
    started_at: datetime
    cancel_requested: bool = False
    output_tail: deque[str] | None = None


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ERROR_HINT_RE = re.compile(
    r"\b(?:error|failed|failure|exception|traceback|denied|not found|timed out|timeout|aborted|cancelled)\b",
    re.IGNORECASE,
)
PROGRESS_LINE_PATTERNS = (
    re.compile(r"^\s*%\s+Total\s+%\s+Received\b", re.IGNORECASE),
    re.compile(
        r"^\s*\d{1,3}\s+\d+(?:\.\d+)?[kmg]?\s+\d{1,3}\s+\d+(?:\.\d+)?[kmg]?\s+\d{1,3}\s+",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*\d{1,3}%\|.*\|"),
    re.compile(r".*\d{1,3}%.*(?:\[[\s=>#.-]+\]|[=#>]{2,}|\|.*\|).*"),
    re.compile(r"^\s*\d+(?:\.\d+)?\s*[KMG]B?\s+.*\d{1,3}%", re.IGNORECASE),
    re.compile(
        r"^\s*(?:downloading|downloaded|fetching|receiving|extracting|installing|building|preparing)\b.*(?:\d{1,3}%|\d+(?:\.\d+)?\s*(?:kb|mb|gb)|/s|it/s)",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(?:get|hit|ign):\d+\s+", re.IGNORECASE),
    re.compile(r"^\s*(?:\||/|-|\\)\s*$"),
    re.compile(r"^\s*(?:\||/|-|\\)\s+(?:download|fetch|install|build|extract)\b", re.IGNORECASE),
)


def _strip_terminal_controls(value: str) -> str:
    value = ANSI_ESCAPE_RE.sub("", value or "")
    value = value.replace("\b", "")
    value = CONTROL_CHARS_RE.sub("", value)
    return value


def is_progress_noise_line(line: str) -> bool:
    stripped = _strip_terminal_controls(line).strip()
    if not stripped:
        return False
    if ERROR_HINT_RE.search(stripped):
        return False
    if any(pattern.match(stripped) for pattern in PROGRESS_LINE_PATTERNS):
        return True
    return bool(
        re.search(r"\d{1,3}%", stripped)
        and re.search(r"(?:/s|eta|remaining|\[[\s=>#.-]+\])", stripped, re.IGNORECASE)
    )


class JobRunner:
    DEFAULT_TIMEOUT_SECONDS = 600
    MAX_TIMEOUT_SECONDS = 3600
    DEFAULT_TAIL_LINES = 80
    MAX_TAIL_LINES = 300
    HEARTBEAT_INTERVAL_SECONDS = 30
    FINISHED_RESULT_TTL_SECONDS = 3600
    FINISHED_RESULT_MAX_COUNT = 200

    def __init__(self):
        self.encoding = config.get("TERMINAL_ENCODING", "utf-8")
        self._active_jobs: dict[str, ActiveJob] = {}
        self._finished_results: dict[str, tuple[float, dict[str, Any]]] = {}
        self._jobs_lock = threading.RLock()

    def _run_job_impl(
        self,
        *,
        user_uuid: str,
        item_uuid: str,
        command: str,
        working_directory: Optional[str] = None,
        timeout_seconds: int | float | None = None,
        tail_lines: int | None = None,
        env: dict[str, str] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        job_id = job_id or uuid.uuid4().hex[:12]
        command = (command or "").strip()
        if not command:
            return {"success": False, "error": "command is required", "job_id": job_id}

        timeout = self._coerce_timeout(timeout_seconds)
        tail_limit = self._coerce_tail_lines(tail_lines)
        started_at = datetime.now()
        start_monotonic = time.monotonic()
        pid: int | None = None
        process: subprocess.Popen[bytes] | None = None
        stdout_fd: int | None = None
        timed_out = False
        cancelled = False
        cancel_terminate_sent = False
        exit_code: int | None = None
        tail = deque(maxlen=tail_limit)
        output_bytes = 0
        read_iterations = 0
        cwd = ""

        logger.info(
            f"[JobRunner] Starting job: job_id={job_id} item={item_uuid} user={user_uuid} "
            f"timeout={timeout} tail_lines={tail_limit} command={command!r} "
            f"working_directory={working_directory!r}"
        )

        try:
            cwd = self._resolve_workdir(user_uuid, item_uuid, working_directory)
            self._write_job_log(
                item_uuid,
                f"\n{'=' * 60}\n"
                f"[{started_at.strftime('%Y-%m-%d %H:%M:%S')}] Job started\n"
                f"  Job ID: {job_id}\n"
                f"  Working Directory: {cwd}\n"
                f"  Command: {command}\n"
                f"{'=' * 60}\n",
            )
            self._broadcast(
                item_uuid,
                {
                    "stdout": f"[{started_at.strftime('%Y-%m-%d %H:%M:%S')}] Job started: {command}\n",
                    "stderr": "",
                    "source": "job",
                    "job_id": job_id,
                    "status": "running",
                },
            )

            process = self._start_process(cwd, command, env)
            pid = process.pid
            if process.stdout is None:
                raise JobRunnerError("job process stdout pipe was not created")
            stdout_fd = process.stdout.fileno()
            os.set_blocking(stdout_fd, False)
            logger.info(f"[JobRunner] subprocess started: job_id={job_id} pid={pid} cwd={cwd}")
            self._register_active_job(
                item_uuid=item_uuid,
                job_id=job_id,
                command=command,
                pid=pid,
                started_at=started_at,
                output_tail=tail,
            )

            pending_line = ""
            last_output_monotonic = start_monotonic
            last_heartbeat_monotonic = start_monotonic
            while True:
                now = time.monotonic()
                if self._is_cancel_requested(job_id):
                    cancelled = True
                    if not cancel_terminate_sent:
                        cancel_terminate_sent = True
                        logger.warning(
                            f"[JobRunner] Job cancel requested: job_id={job_id} pid={pid} command={command!r}"
                        )
                        self._terminate_process(pid)

                if not timed_out and now - start_monotonic > timeout:
                    timed_out = True
                    logger.warning(
                        f"[JobRunner] Job timeout: job_id={job_id} pid={pid} timeout={timeout} command={command!r}"
                    )
                    self._terminate_process(pid)

                ready, _, _ = select.select([stdout_fd], [], [], 0.1)
                if ready:
                    try:
                        chunk = os.read(stdout_fd, 4096)
                    except BlockingIOError:
                        chunk = b""
                    except OSError:
                        chunk = b""

                    if chunk:
                        last_output_monotonic = now
                        read_iterations += 1
                        output_bytes += len(chunk)
                        decoded = chunk.decode(self.encoding, errors="replace")
                        pending_line = self._handle_output_chunk(
                            item_uuid=item_uuid,
                            job_id=job_id,
                            decoded=decoded,
                            pending_line=pending_line,
                            tail=tail,
                        )

                exit_code = process.poll()
                if exit_code is not None:
                    pending_line = self._drain_process_output(
                        stdout_fd=stdout_fd,
                        item_uuid=item_uuid,
                        job_id=job_id,
                        pending_line=pending_line,
                        tail=tail,
                    )
                    logger.info(
                        f"[JobRunner] Job process exited: job_id={job_id} pid={pid} "
                        f"exit_code={exit_code} timed_out={timed_out}"
                    )
                    break

                if timed_out and process.poll() is None:
                    try:
                        exit_code = process.wait(timeout=1)
                        pending_line = self._drain_process_output(
                            stdout_fd=stdout_fd,
                            item_uuid=item_uuid,
                            job_id=job_id,
                            pending_line=pending_line,
                            tail=tail,
                        )
                        break
                    except subprocess.TimeoutExpired:
                        self._terminate_process(pid)

                if (
                    now - last_output_monotonic >= self.HEARTBEAT_INTERVAL_SECONDS
                    and now - last_heartbeat_monotonic >= self.HEARTBEAT_INTERVAL_SECONDS
                ):
                    last_heartbeat_monotonic = now
                    self._broadcast_job_heartbeat(
                        item_uuid=item_uuid,
                        job_id=job_id,
                        idle_seconds=round(now - last_output_monotonic),
                        elapsed_seconds=round(now - start_monotonic),
                        timeout_seconds=timeout,
                    )

            if pending_line.strip():
                self._append_output_line(item_uuid, job_id, pending_line, tail)

            # cancel_job terminates the process immediately; the loop above can
            # break on the same iteration before re-checking the cancel flag.
            # Re-check here (the job is still registered) so the finished
            # result is authoritatively marked cancelled.
            if not cancelled:
                cancelled = self._is_cancel_requested(job_id)

        except Exception as exc:
            logger.error(f"[JobRunner] Job failed before completion: job_id={job_id} item={item_uuid} error={exc}")
            return {
                "success": False,
                "error": str(exc),
                "job_id": job_id,
                "item_uuid": item_uuid,
                "command": command,
                "cwd": cwd,
            }
        finally:
            self._unregister_active_job(job_id)
            if process and process.stdout:
                try:
                    process.stdout.close()
                except Exception:
                    pass
            self._reap_zombies()

        finished_at = datetime.now()
        duration_seconds = round(time.monotonic() - start_monotonic, 3)
        exit_code = exit_code if exit_code is not None else -1
        output_tail = "\n".join(tail)
        footer = (
            f"\n{'=' * 60}\n"
            f"[{finished_at.strftime('%Y-%m-%d %H:%M:%S')}] Job finished\n"
            f"  Job ID: {job_id}\n"
            f"  Exit Code: {exit_code}\n"
            f"  Timed Out: {timed_out}\n"
            f"  Cancelled: {cancelled}\n"
            f"  Duration: {duration_seconds}s\n"
            f"{'=' * 60}\n"
        )
        self._write_job_log(item_uuid, footer)
        self._broadcast(
            item_uuid,
            {
                "stdout": footer,
                "stderr": "",
                "source": "job",
                "job_id": job_id,
                "status": "finished",
                "exit_code": exit_code,
                "timed_out": timed_out,
                "cancelled": cancelled,
            },
        )
        logger.info(
            f"[JobRunner] Job completed: job_id={job_id} item={item_uuid} exit_code={exit_code} "
            f"timed_out={timed_out} duration={duration_seconds} output_bytes={output_bytes} "
            f"read_iterations={read_iterations} tail_lines={len(tail)}"
        )
        return {
            "success": True,
            "job_id": job_id,
            "item_uuid": item_uuid,
            "command": command,
            "cwd": cwd,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "cancelled": cancelled,
            "duration_seconds": duration_seconds,
            "output_tail": output_tail,
            "tail_lines": len(tail),
            "output_bytes": output_bytes,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
        }

    def run_job(self, **kwargs: Any) -> dict[str, Any]:
        """Run a job synchronously and buffer its result for polling.

        Wraps _run_job_impl so every terminal outcome (success or failure)
        lands in the finished-result buffer, letting backend callers recover
        the result via get_job_result() even after a backend restart.
        """
        result = self._run_job_impl(**kwargs)
        self._store_finished_result(str(result.get("job_id") or ""), result)
        return result

    def start_job(self, **kwargs: Any) -> dict[str, Any]:
        """Start a job in a daemon-side thread and return its job_id at once."""
        command = str(kwargs.get("command") or "").strip()
        job_id = uuid.uuid4().hex[:12]
        if not command:
            return {"success": False, "error": "command is required", "job_id": job_id}
        thread = threading.Thread(
            target=self.run_job,
            kwargs={**kwargs, "command": command, "job_id": job_id},
            name=f"termpaws-daemon-job-{job_id}",
            daemon=True,
        )
        thread.start()
        logger.info(
            f"[JobRunner] Job scheduled asynchronously: job_id={job_id} "
            f"item={kwargs.get('item_uuid')} command={command!r}"
        )
        return {"success": True, "job_id": job_id, "command": command}

    def _store_finished_result(self, job_id: str, result: dict[str, Any]) -> None:
        if not job_id:
            return
        with self._jobs_lock:
            self._prune_finished_results_locked()
            self._finished_results[job_id] = (time.monotonic(), result)

    def _prune_finished_results_locked(self) -> None:
        cutoff = time.monotonic() - self.FINISHED_RESULT_TTL_SECONDS
        stale = [
            key
            for key, (stored_at, _result) in self._finished_results.items()
            if stored_at < cutoff
        ]
        for key in stale:
            self._finished_results.pop(key, None)
        overflow = len(self._finished_results) - self.FINISHED_RESULT_MAX_COUNT
        if overflow > 0:
            oldest = sorted(
                self._finished_results.items(), key=lambda item: item[1][0]
            )[:overflow]
            for key, _entry in oldest:
                self._finished_results.pop(key, None)

    def get_job_result(self, job_id: str) -> dict[str, Any]:
        """Poll the buffered result of a job started via start_job()."""
        job_id = str(job_id or "").strip()
        if not job_id:
            return {
                "success": False,
                "status": "unknown",
                "error": "job_id is required",
            }
        with self._jobs_lock:
            self._prune_finished_results_locked()
            entry = self._finished_results.get(job_id)
            active = self._active_jobs.get(job_id)
        if entry is not None:
            return {
                "success": True,
                "status": "finished",
                "job_id": job_id,
                "result": entry[1],
            }
        if active is not None:
            return {"success": True, "status": "running", "job_id": job_id}
        return {
            "success": False,
            "status": "unknown",
            "job_id": job_id,
            "error": "job result unavailable",
        }

    def list_jobs(self, *, item_uuid: str | None = None) -> dict[str, Any]:
        now = datetime.now()
        with self._jobs_lock:
            jobs = [
                {
                    "job_id": job.job_id,
                    "item_uuid": job.item_uuid,
                    "command": job.command,
                    "pid": job.pid,
                    "started_at": job.started_at.isoformat(),
                    "elapsed_seconds": round((now - job.started_at).total_seconds(), 3),
                    "cancel_requested": job.cancel_requested,
                    "output_tail": "\n".join(job.output_tail or []),
                    "tail_lines": len(job.output_tail or []),
                }
                for job in self._active_jobs.values()
                if item_uuid is None or job.item_uuid == item_uuid
            ]
        jobs.sort(key=lambda job: str(job.get("started_at") or ""), reverse=True)
        return {
            "success": True,
            "jobs": jobs,
            "count": len(jobs),
            "item_uuid": item_uuid,
        }

    def cancel_job(self, *, item_uuid: str, job_id: str | None = None) -> dict[str, Any]:
        with self._jobs_lock:
            active_job = None
            if job_id:
                candidate = self._active_jobs.get(job_id)
                if candidate and candidate.item_uuid == item_uuid:
                    active_job = candidate
            else:
                active_job = next(
                    (job for job in self._active_jobs.values() if job.item_uuid == item_uuid),
                    None,
                )

            if not active_job:
                return {
                    "success": False,
                    "cancelled": False,
                    "error": "No running job for item",
                    "item_uuid": item_uuid,
                    "job_id": job_id or "",
                }

            active_job.cancel_requested = True
            pid = active_job.pid
            resolved_job_id = active_job.job_id
            command = active_job.command

        logger.warning(
            f"[JobRunner] Cancelling active job: job_id={resolved_job_id} item={item_uuid} pid={pid}"
        )
        self._terminate_process(pid)
        return {
            "success": True,
            "cancelled": True,
            "item_uuid": item_uuid,
            "job_id": resolved_job_id,
            "command": command,
        }

    def _register_active_job(
        self,
        *,
        item_uuid: str,
        job_id: str,
        command: str,
        pid: int,
        started_at: datetime,
        output_tail: deque[str] | None = None,
    ) -> None:
        with self._jobs_lock:
            self._active_jobs[job_id] = ActiveJob(
                job_id=job_id,
                item_uuid=item_uuid,
                command=command,
                pid=pid,
                started_at=started_at,
                output_tail=output_tail,
            )

    def _unregister_active_job(self, job_id: str) -> None:
        with self._jobs_lock:
            self._active_jobs.pop(job_id, None)

    def _is_cancel_requested(self, job_id: str) -> bool:
        with self._jobs_lock:
            active_job = self._active_jobs.get(job_id)
            return bool(active_job and active_job.cancel_requested)

    def _resolve_workdir(self, user_uuid: str, item_uuid: str, working_directory: str | None) -> str:
        # job 跟随终端实时 cwd：终端 cd 到哪，job 就跑在哪。
        # 绝对路径原样使用——不做 item 沙箱重映射（交互终端本来就不沙箱，
        # job 却静默 remap 成另一个目录，两边对不上就是 bug）。
        if working_directory:
            candidate = Path(str(working_directory).replace("\\", "/").strip()).resolve(
                strict=False
            )
            if candidate.is_absolute():
                candidate.mkdir(parents=True, exist_ok=True)
                return str(candidate)
        try:
            return str(
                item_path_service.resolve_workdir(
                    user_uuid=user_uuid,
                    item_uuid=item_uuid,
                    working_directory=working_directory,
                    create=True,
                )
            )
        except ItemPathError as exc:
            raise JobRunnerError(f"invalid working_directory: {exc}") from exc

    def _build_child_env(self, env: dict[str, str] | None) -> dict[str, str]:
        child_env = os.environ.copy()
        child_env.update(
            {
                "TERM": "dumb",
                "COLUMNS": "120",
                "LINES": "30",
                "LANG": child_env.get("LANG", "C.UTF-8"),
                "LC_ALL": child_env.get("LC_ALL", "C.UTF-8"),
                "LC_CTYPE": child_env.get("LC_CTYPE", child_env.get("LC_ALL", "C.UTF-8")),
                "CI": child_env.get("CI", "1"),
                "DEBIAN_FRONTEND": child_env.get("DEBIAN_FRONTEND", "noninteractive"),
                "APT_LISTCHANGES_FRONTEND": child_env.get("APT_LISTCHANGES_FRONTEND", "none"),
                "NEEDRESTART_MODE": child_env.get("NEEDRESTART_MODE", "a"),
                "GPG_TTY": "",
                "PYTHONUTF8": child_env.get("PYTHONUTF8", "1"),
                "PYTHONIOENCODING": child_env.get("PYTHONIOENCODING", "utf-8"),
                "PYTHONUNBUFFERED": child_env.get("PYTHONUNBUFFERED", "1"),
            }
        )
        extra_paths = ["/usr/games", "/usr/local/games"]
        current_path = child_env.get("PATH", "")
        for extra in extra_paths:
            if extra not in current_path.split(":"):
                current_path = f"{current_path}:{extra}" if current_path else extra
        child_env["PATH"] = current_path
        if env:
            child_env.update({str(key): str(value) for key, value in env.items()})
        return child_env

    def _start_process(
        self,
        cwd: str,
        command: str,
        env: dict[str, str] | None,
    ) -> subprocess.Popen[bytes]:
        shell = config.get("TERMINAL_SHELL", "/bin/bash")
        return subprocess.Popen(
            [shell, "-lc", command],
            cwd=cwd,
            env=self._build_child_env(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _drain_process_output(
        self,
        *,
        stdout_fd: int,
        item_uuid: str,
        job_id: str,
        pending_line: str,
        tail: deque[str],
    ) -> str:
        while True:
            try:
                chunk = os.read(stdout_fd, 4096)
            except BlockingIOError:
                return pending_line
            except OSError:
                return pending_line
            if not chunk:
                return pending_line
            decoded = chunk.decode(self.encoding, errors="replace")
            pending_line = self._handle_output_chunk(
                item_uuid=item_uuid,
                job_id=job_id,
                decoded=decoded,
                pending_line=pending_line,
                tail=tail,
            )

    def _handle_output_chunk(
        self,
        *,
        item_uuid: str,
        job_id: str,
        decoded: str,
        pending_line: str,
        tail: deque[str],
    ) -> str:
        text = decoded.replace("\r\n", "\n")
        self._broadcast(
            item_uuid,
            {
                "stdout": text,
                "stderr": "",
                "source": "job",
                "job_id": job_id,
                "status": "running",
            },
        )
        pending_line += text.replace("\r", "\n")
        lines = pending_line.split("\n")
        pending_line = lines[-1]
        for line in lines[:-1]:
            if line.strip():
                self._append_output_line(item_uuid, job_id, line, tail)
        return pending_line

    def _append_output_line(self, item_uuid: str, job_id: str, line: str, tail: deque[str]):
        clean_line = line.rstrip("\n")
        if is_progress_noise_line(clean_line):
            return
        tail.append(clean_line)
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        self._write_job_log(item_uuid, f"{timestamp} [job:{job_id}] {clean_line}\n")

    def _broadcast_job_heartbeat(
        self,
        *,
        item_uuid: str,
        job_id: str,
        idle_seconds: int,
        elapsed_seconds: int,
        timeout_seconds: int,
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._broadcast(
            item_uuid,
            {
                "stdout": (
                    f"[{timestamp}] 后台 Job 仍在运行：{idle_seconds}s 没有新输出"
                    f"（已运行 {elapsed_seconds}s / 超时 {timeout_seconds}s）\n"
                ),
                "stderr": "",
                "source": "job",
                "job_id": job_id,
                "status": "running",
                "idle_seconds": idle_seconds,
                "elapsed_seconds": elapsed_seconds,
            },
        )

    def _write_job_log(self, item_uuid: str, content: str):
        if not daemon_log_manager.write_to_log(item_uuid, content):
            logger.warning(f"[JobRunner] Failed to write daemon log: item={item_uuid}")

    def _broadcast(self, item_uuid: str, data: dict[str, Any]):
        try:
            socket_service = get_socket_service()
            if socket_service:
                socket_service.sync_broadcast(item_uuid, "stream", data)
        except Exception as exc:
            logger.error(f"[JobRunner] Broadcast failed: item={item_uuid} error={exc}")

    @staticmethod
    def _reap_zombies() -> None:
        """Reap any zombie child processes to prevent accumulation."""
        try:
            while True:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
        except ChildProcessError:
            pass

    def _terminate_process(self, pid: int | None):
        if pid is None:
            return
        for sig, delay in ((signal.SIGTERM, 0.5), (signal.SIGKILL, 0.0)):
            delivered = False
            try:
                child_pgid = os.getpgid(pid)
                if child_pgid != os.getpgrp():
                    os.killpg(child_pgid, sig)
                    delivered = True
                else:
                    logger.debug(
                        f"[JobRunner] Skip process-group signal {sig} for pid={pid}; child shares daemon pgid"
                    )
            except ProcessLookupError:
                return
            except Exception as exc:
                logger.debug(f"[JobRunner] Failed to send signal {sig} to process group for pid={pid}: {exc}")

            if not delivered:
                try:
                    os.kill(pid, sig)
                    delivered = True
                except ProcessLookupError:
                    return
                except Exception as exc:
                    logger.warning(f"[JobRunner] Failed to send signal {sig} to pid={pid}: {exc}")

            if delivered and delay:
                time.sleep(delay)

    def _status_to_exit_code(self, status: int) -> int:
        if os.WIFEXITED(status):
            return os.WEXITSTATUS(status)
        if os.WIFSIGNALED(status):
            return -os.WTERMSIG(status)
        return -1

    def _coerce_timeout(self, value: int | float | None) -> int:
        try:
            timeout = int(float(value)) if value is not None else self.DEFAULT_TIMEOUT_SECONDS
        except (TypeError, ValueError):
            timeout = self.DEFAULT_TIMEOUT_SECONDS
        return max(1, min(timeout, self.MAX_TIMEOUT_SECONDS))

    def _coerce_tail_lines(self, value: int | None) -> int:
        try:
            lines = int(value) if value is not None else self.DEFAULT_TAIL_LINES
        except (TypeError, ValueError):
            lines = self.DEFAULT_TAIL_LINES
        return max(1, min(lines, self.MAX_TAIL_LINES))


job_runner = JobRunner()
