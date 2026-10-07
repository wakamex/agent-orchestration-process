"""Network egress for --no-web runs.

The sandbox gets its own network namespace, which has only a loopback
interface. Its one way out is a Unix socket to an HTTP CONNECT proxy that
runs in the controller and opens connections only to allowed endpoints.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.parse
from collections.abc import Iterable
from pathlib import Path

from .worktrees import AOPError

GUEST_SOCKET = "/run/aop/egress.sock"
GUEST_BRIDGE = "/run/aop/egress_bridge.py"
BRIDGE_SOURCE = Path(__file__).with_name("egress_bridge.py")
PROXY_PORT = 3128
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})

# Endpoints each harness needs for inference and for keeping its login valid.
HARNESS_ENDPOINTS: dict[str, tuple[str, ...]] = {
    # Inference, and the OAuth token endpoint named in Claude Code's TOKEN_URL.
    "claude": ("api.anthropic.com:443", "platform.claude.com:443"),
    # ChatGPT-login inference; AOP refreshes Codex tokens outside the sandbox.
    "codex": ("chatgpt.com:443",),
    # xAI login and token refresh, and Grok Build's subscription inference.
    "grok": ("auth.x.ai:443", "cli-chat-proxy.grok.com:443"),
    # Devin's API server: inference, login, and its model list.
    "devin": ("server.codeium.com:443",),
    # OpenCode Zen inference.
    "opencode": ("opencode.ai:443",),
    # DeepSeek Harness and Hermes endpoints depend on their inference provider.
    "dsh:deepseek-official": ("api.deepseek.com:443",),
    "hermes:xai-oauth": ("auth.x.ai:443", "api.x.ai:443"),
}


def endpoint(url: str) -> str:
    """Return the host:port an http(s) URL connects to."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise AOPError(f"allowed URL must be an http(s) URL: {url!r}")
    host = parsed.hostname.lower()
    if parsed.scheme == "http" and host not in LOOPBACK_HOSTS:
        raise AOPError(f"only loopback URLs may use plain http: {url!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{host}:{port}"


def allowed_endpoints(
    provider: str, urls: Iterable[str], inference_provider: str | None = None
) -> list[str]:
    key = f"{provider}:{inference_provider}" if inference_provider else provider
    return sorted({*HARNESS_ENDPOINTS.get(key, ()), *map(endpoint, urls)})


def forwarded_endpoints(endpoints: Iterable[str]) -> list[str]:
    """Loopback endpoints, which the sandbox reaches at their own address."""
    return [item for item in endpoints if item.rpartition(":")[0] in LOOPBACK_HOSTS]


class EgressProxy:
    """CONNECT proxy on a Unix socket that allows only listed host:port pairs."""

    def __init__(self, socket_path: Path, allowed: Iterable[str], log_path: Path):
        self.socket_path = socket_path
        self.allowed = frozenset(allowed)
        self.log_path = log_path
        self._log_lock = threading.Lock()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(os.fspath(socket_path))
        os.chmod(socket_path, 0o600)
        self._server.listen(64)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> EgressProxy:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self._server.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._server.close()
        self._thread.join(timeout=5)
        self.socket_path.unlink(missing_ok=True)

    def _serve(self) -> None:
        while True:
            try:
                client, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(client,), daemon=True).start()

    def _handle(self, client: socket.socket) -> None:
        with client:
            header = b""
            while b"\r\n\r\n" not in header and len(header) < 16384:
                chunk = client.recv(4096)
                if not chunk:
                    return
                header += chunk
            head, _, rest = header.partition(b"\r\n\r\n")
            parts = head.split(b"\r\n", 1)[0].decode("latin-1").split()
            target = parts[1].lower() if len(parts) == 3 else ""
            if len(parts) != 3 or parts[0] != "CONNECT" or target not in self.allowed:
                self._log(target or " ".join(parts[:2]), "denied")
                client.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                return
            host, _, port = target.rpartition(":")
            try:
                upstream = socket.create_connection((host, int(port)), timeout=15)
            except OSError as error:
                self._log(target, f"unreachable: {error}")
                client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                return
            self._log(target, "allowed")
            with upstream:
                upstream.settimeout(None)
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                if rest:
                    upstream.sendall(rest)
                _pipe(client, upstream)

    def _log(self, target: str, decision: str) -> None:
        line = json.dumps(
            {"time": time.time(), "endpoint": target, "decision": decision}
        )
        with self._log_lock, self.log_path.open("a") as stream:
            stream.write(line + "\n")


def _pipe(left: socket.socket, right: socket.socket) -> None:
    def copy(source: socket.socket, target: socket.socket) -> None:
        try:
            while data := source.recv(65536):
                target.sendall(data)
        except OSError:
            pass
        finally:
            try:
                target.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    other = threading.Thread(target=copy, args=(right, left), daemon=True)
    other.start()
    copy(left, right)
    other.join()
