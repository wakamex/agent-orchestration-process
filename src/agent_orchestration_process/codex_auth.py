"""Delegate shared ChatGPT credential refresh to a controller-side Codex process."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import queue
import shutil
import stat
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .worktrees import AOPError


def _claims(token: str) -> dict[str, Any]:
    try:
        value = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "==="))
        if isinstance(value, dict):
            return value
    except (ValueError, IndexError, TypeError):
        pass
    raise AOPError("Codex returned an invalid ChatGPT access token")


def _binding(account: str, subject: str) -> str:
    return hashlib.sha256(json.dumps([account, subject]).encode()).hexdigest()


@dataclass(repr=False)
class AccessToken:
    value: str
    account: str
    subject: str
    plan: str | None
    expires: float

    @classmethod
    def parse(cls, token: str) -> AccessToken:
        claims = _claims(token)
        auth = claims.get("https://api.openai.com/auth", {})
        account = auth.get("chatgpt_account_id") if isinstance(auth, dict) else None
        subject = claims.get("sub")
        expires = claims.get("exp")
        if (
            not isinstance(account, str)
            or not account
            or not isinstance(subject, str)
            or not subject
            or not isinstance(expires, (int, float))
            or isinstance(expires, bool)
        ):
            raise AOPError(
                "Codex ChatGPT credentials have no valid account binding or expiry"
            )
        plan = auth.get("chatgpt_plan_type")
        return cls(
            token, account, subject, plan if isinstance(plan, str) else None, expires
        )

    @property
    def binding(self) -> str:
        return _binding(self.account, self.subject)

    def parameters(self) -> dict[str, Any]:
        return {
            "accessToken": self.value,
            "chatgptAccountId": self.account,
            "chatgptPlanType": self.plan,
        }


def source_binding(home: Path | None) -> str | None:
    """Inspect only the configured source, never search task homes for credentials."""
    if home is None:
        return None
    try:
        value = json.loads((home / "auth.json").read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise AOPError(
            "Could not read the configured Codex authentication source"
        ) from None
    if not isinstance(value, dict):
        raise AOPError("Invalid Codex authentication source")
    mode = value.get("auth_mode")
    if mode not in (None, "chatgpt") or (mode is None and value.get("OPENAI_API_KEY")):
        return None
    tokens = value.get("tokens")
    if tokens is None and mode is None:
        return None
    if not isinstance(tokens, dict) or not isinstance(tokens.get("access_token"), str):
        raise AOPError("Codex ChatGPT source credentials are incomplete")
    token = AccessToken.parse(tokens["access_token"])
    if tokens.get("account_id") != token.account:
        raise AOPError(
            "Codex source credential account does not match its access token"
        )
    return token.binding


def retire_task_copy(home: Path, archive: Path) -> bool:
    """Preserve a legacy copy outside the task mounts, without trusting its contents."""
    path = home / "auth.json"
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return False
    except OSError:
        raise AOPError(
            "Cannot safely retire the task's old Codex credential copy"
        ) from None
    with os.fdopen(descriptor, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise AOPError("The old Codex credential copy must be a regular file")
        archive.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target = os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(target, "wb") as destination:
            shutil.copyfileobj(source, destination)
    path.unlink()
    return True


@contextmanager
def _refresh_lock(home: Path, deadline: float) -> Iterator[None]:
    # The canonical source owns this lock across repositories and AOP processes.
    path = home / ".aop-auth.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError:
        raise AOPError(
            "Codex authentication source is not writable for coordinated refresh"
        ) from None
    with os.fdopen(descriptor, "w") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise AOPError(
                        "Timed out waiting for Codex credential refresh"
                    ) from None
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class _NativeAuth:
    """A short-lived, no-thread native client; its responses are never logged."""

    def __init__(
        self, binary: str, home: Path, environment: dict[str, str], deadline: float
    ):
        self.deadline = deadline
        self.sequence = 0
        self.lines: queue.Queue[str | None] = queue.Queue()
        child_environment = {
            name: environment[name]
            for name in (
                "HOME",
                "PATH",
                "LANG",
                "LC_ALL",
                "DBUS_SESSION_BUS_ADDRESS",
                "XDG_RUNTIME_DIR",
                "XDG_CONFIG_HOME",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "HTTPS_PROXY",
                "HTTP_PROXY",
                "NO_PROXY",
                "https_proxy",
                "http_proxy",
                "no_proxy",
                "CODEX_REFRESH_TOKEN_URL_OVERRIDE",
            )
            if name in environment
        }
        child_environment["CODEX_HOME"] = os.fspath(home)
        try:
            self.process = subprocess.Popen(
                [
                    binary,
                    "-c",
                    'model_provider="openai"',
                    "app-server",
                    "--stdio",
                ],
                cwd=home,
                env=child_environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
        except OSError:
            raise AOPError("Could not start Codex credential refresh") from None
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self.lines.put(line)
        finally:
            self.lines.put(None)

    def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(
                json.dumps({"id": self.sequence, "method": method, "params": params})
                + "\n"
            )
            self.process.stdin.flush()
            while time.monotonic() < self.deadline:
                line = self.lines.get(timeout=max(0, self.deadline - time.monotonic()))
                if line is None:
                    break
                value = json.loads(line)
                if isinstance(value, dict) and value.get("id") == self.sequence:
                    result = value.get("result")
                    if "error" not in value and isinstance(result, dict):
                        return result
                    break
        except (OSError, ValueError, queue.Empty):
            pass
        raise AOPError(
            "Codex credential operation failed or timed out; no task was given refresh credentials"
        )

    def close(self) -> None:
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.reader.join(timeout=1)
        if self.process.stdout is not None:
            self.process.stdout.close()


@dataclass
class CredentialSource:
    binary: str
    home: Path
    environment: dict[str, str] = field(repr=False)
    binding: str

    def read(
        self, *, previous: AccessToken | None = None, deadline: float | None = None
    ) -> AccessToken:
        # Native external-auth callbacks have a ten-second timeout.
        deadline = min(deadline or float("inf"), time.monotonic() + 8)
        with _refresh_lock(self.home, deadline):
            if source_binding(self.home) != self.binding:
                raise AOPError(
                    "Codex source account changed; refusing credential refresh"
                )
            client = _NativeAuth(self.binary, self.home, self.environment, deadline)
            try:
                client.call(
                    "initialize",
                    {"clientInfo": {"name": "aop-auth", "version": "0.1.0"}},
                )
                assert client.process.stdin is not None
                client.process.stdin.write('{"method":"initialized","params":{}}\n')
                client.process.stdin.flush()
                token = self._read_token(client, refresh=False)
                # Another controller may already have refreshed while we waited.
                if token.expires <= time.time() + 30 or (
                    previous is not None and token.value == previous.value
                ):
                    refreshed = self._read_token(client, refresh=True)
                    if refreshed.value == token.value:
                        raise AOPError(
                            "Codex could not refresh the configured source credentials"
                        )
                    token = refreshed
                if token.expires <= time.time():
                    raise AOPError("Codex returned expired source credentials")
                return token
            finally:
                client.close()

    def _read_token(self, client: _NativeAuth, *, refresh: bool) -> AccessToken:
        result = client.call(
            "getAuthStatus", {"includeToken": True, "refreshToken": refresh}
        )
        if result.get("authMethod") != "chatgpt" or not isinstance(
            result.get("authToken"), str
        ):
            raise AOPError(
                "Codex could not provide valid ChatGPT source credentials; task credential copies are not promoted automatically"
            )
        token = AccessToken.parse(result["authToken"])
        if token.binding != self.binding:
            raise AOPError("Codex refreshed credentials for a different account")
        return token
