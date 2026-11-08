"""Shared-daemon connections exchange real RPC without owning the server lifecycle."""

import json
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from websockets.sync.server import unix_serve

from agent.transports.codex_app_server import CodexAppServerTransportError
from hermes_cli.codex_runtime_switch import get_configured_codex_client


@contextmanager
def daemon_server(path):
    requests = []
    waiting = threading.Event()

    def handle(connection):
        for raw in connection:
            message = json.loads(raw)
            requests.append(message)
            method = message.get("method")
            if method == "test/wait":
                waiting.set()
                continue
            result = {
                "initialize": {"userAgent": "test-daemon"},
                "thread/start": {"thread": {"id": "test-thread"}},
                "test/ping": {"pong": True},
            }.get(method)
            if result is None:
                continue
            connection.send(json.dumps({"id": message["id"], "result": result}))

    with unix_serve(handle, str(path)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield requests, waiting
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.platforms("posix")
def test_configured_daemon_session_and_disconnect_leave_server_alive(tmp_path):
    from agent.transports.codex_app_server_session import CodexAppServerSession
    from hermes_cli.config import atomic_config_write, get_config_path, load_config

    # Keep AF_UNIX paths below macOS' 104-byte limit.
    with tempfile.TemporaryDirectory(prefix="hcd-", dir="/tmp") as socket_dir:
        path = Path(socket_dir) / "daemon.sock"
        atomic_config_write(get_config_path(), {"codex": {"transport": "daemon", "daemon_socket": str(path)}})
        factory = get_configured_codex_client(load_config())
        with daemon_server(path) as (requests, _):
            session = CodexAppServerSession(cwd=str(tmp_path), client_factory=factory)
            try:
                assert session.ensure_started() == "test-thread"
            finally:
                session.close()
            with factory() as client:
                client.initialize()
                assert client.request("test/ping") == {"pong": True}
        assert next(r for r in requests if r.get("method") == "thread/start")["params"]["cwd"] == str(tmp_path)


@pytest.mark.platforms("posix")
def test_close_unblocks_pending_rpc_and_missing_daemon_never_spawns(monkeypatch):
    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("An explicit socket connection must not spawn a Codex process")

    monkeypatch.setattr("subprocess.Popen", forbidden_spawn)
    with tempfile.TemporaryDirectory(prefix="hcd-", dir="/tmp") as socket_dir:
        path = Path(socket_dir) / "daemon.sock"
        factory = get_configured_codex_client({"codex": {"transport": "daemon", "daemon_socket": str(path)}})
        with daemon_server(path) as (_, waiting):
            client = factory()
            client.initialize()
            errors = []

            def blocked():
                try:
                    client.request("test/wait", timeout=30)
                except CodexAppServerTransportError as exc:
                    errors.append(exc)

            thread = threading.Thread(target=blocked)
            thread.start()
            assert waiting.wait(timeout=5)
            client.close()
            thread.join(timeout=5)
            assert not thread.is_alive()
            assert len(errors) == 1
        with pytest.raises(CodexAppServerTransportError, match="daemon"):
            factory()


def test_default_transport_and_dispatcher_workers_remain_isolated(monkeypatch):
    assert get_configured_codex_client({}) is None
    with pytest.raises(ValueError, match="codex.transport"):
        get_configured_codex_client({"codex": {"transport": "typo"}})
    monkeypatch.setenv("HERMES_KANBAN_TASK", "worker-test")
    assert get_configured_codex_client({"codex": {"transport": "daemon"}}) is None
