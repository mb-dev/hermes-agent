"""Borrow a connection to Codex's shared local daemon without owning its lifecycle."""

from __future__ import annotations

import json
import subprocess
import threading

from websockets.exceptions import ConnectionClosed, InvalidHandshake
from websockets.sync.client import unix_connect

from agent.memory_provider import spawn_context_thread
from agent.transports.codex_app_server import CodexAppServerClient, CodexAppServerTransportError
from tools.environments.local import hermes_subprocess_env


def _daemon_socket(codex_bin: str, codex_home: str | None) -> str:
    """Ask the installed CLI for its daemon endpoint; never start or restart it."""
    env = hermes_subprocess_env()
    if codex_home:
        env["CODEX_HOME"] = codex_home
    result = subprocess.run(
        [codex_bin, "app-server", "daemon", "version"], capture_output=True, text=True,
        timeout=10, stdin=subprocess.DEVNULL, env=env, check=False,
    )
    status = json.loads(result.stdout) if result.returncode == 0 else {}
    if status.get("status") != "running" or not status.get("socketPath"):
        raise ValueError("Codex daemon is unavailable; run `codex app-server daemon start` first")
    return status["socketPath"]


class CodexDaemonClient(CodexAppServerClient):
    """The existing RPC protocol over a Unix WebSocket, with connection-only cleanup."""

    def __init__(
        self, codex_bin: str = "codex", codex_home: str | None = None, *, socket_path: str = "",
    ) -> None:
        self._init_protocol_state()
        self._disconnected = threading.Event()
        try:
            path = socket_path or _daemon_socket(codex_bin, codex_home)
            self._socket = unix_connect(path, open_timeout=10, close_timeout=3, max_size=None)
        except (OSError, ValueError, TimeoutError, InvalidHandshake, subprocess.SubprocessError) as exc:
            raise CodexAppServerTransportError(
                code=-32000, message=f"Cannot connect to Codex daemon: {exc}",
            ) from exc
        self._reader = spawn_context_thread(self._read_messages, name="codex-daemon")
        self._reader.start()

    def close(self, timeout: float = 3.0) -> None:
        """Disconnect this client; the shared daemon and its other clients stay alive."""
        if self._closed:
            return
        self._closed = True
        self._fail_pending_requests("Codex daemon connection is closing")
        self._socket.close()
        self._reader.join(timeout=timeout)

    def is_alive(self) -> bool:
        return not self._closed and not self._disconnected.is_set()

    def stderr_tail(self, n: int = 20) -> list[str]:
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    def _send(self, obj: dict) -> None:
        if not self.is_alive():
            raise CodexAppServerTransportError(code=-32000, message="Codex daemon connection is closed")
        try:
            self._socket.send(json.dumps(obj))
        except (ConnectionClosed, OSError) as exc:
            raise CodexAppServerTransportError(code=-32000, message=f"Codex daemon connection lost: {exc}") from exc

    def _read_messages(self) -> None:
        try:
            for raw in self._socket:
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError("expected a JSON-RPC object")
                self._dispatch(message)
        except (ConnectionClosed, OSError, ValueError) as exc:
            self._append_stderr(f"Codex daemon connection lost: {exc}")
        finally:
            self._disconnected.set()
            self._fail_pending_requests("Codex daemon connection closed")
