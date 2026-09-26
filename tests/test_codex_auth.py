from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import threading
import time

import pytest

from agent_orchestration_process import codex_auth
from agent_orchestration_process.runner import AgentRunner, CodexAdapter
from agent_orchestration_process.worktrees import AOPError, WorktreeManager


def token(account="account", subject="user", serial=0, expires=None):
    claims = {
        "sub": subject,
        "exp": expires or int(time.time()) + 3600,
        "serial": serial,
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account,
            "chatgpt_plan_type": "pro",
        },
    }
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return "e30." + encoded + ".synthetic-signature"


def write_auth(home, access=None):
    access = access or token()
    parsed = codex_auth.AccessToken.parse(access)
    (home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "OPENAI_API_KEY": None,
                "tokens": {
                    "id_token": access,
                    "access_token": access,
                    "refresh_token": "synthetic-refresh",
                    "account_id": parsed.account,
                },
                "last_refresh": datetime.now(UTC).isoformat(),
            }
        )
    )


@pytest.fixture
def auth_codex(fake_codex):
    source = Path(os.environ["AOP_CODEX_SOURCE_HOME"])
    write_auth(source)
    script = fake_codex.read_text()
    owner = """
if args[:1] == ["-c"]:
    home = pathlib.Path(os.environ["CODEX_HOME"])
    for line in sys.stdin:
        request = json.loads(line)
        method = request.get("method")
        if method == "initialize":
            result = {}
        elif method == "getAuthStatus":
            auth = json.loads((home / "auth.json").read_text())
            if request["params"]["refreshToken"]:
                if (home / "fail-refresh").exists():
                    print(json.dumps({"id": request["id"], "result": {"authMethod": "chatgpt", "authToken": None}}), flush=True)
                    continue
                marker = home / "refresh-count"
                count = int(marker.read_text()) if marker.exists() else 0
                time.sleep(0.1)
                marker.write_text(str(count + 1))
                auth["tokens"]["access_token"] = (home / "next-token").read_text()
                auth["tokens"]["refresh_token"] = "rotated-synthetic-refresh"
                (home / "auth.json").write_text(json.dumps(auth))
            result = {"authMethod": "chatgpt", "authToken": auth["tokens"]["access_token"]}
        elif method == "initialized":
            continue
        else:
            raise RuntimeError("auth owner may not start a model or tools")
        print(json.dumps({"id": request["id"], "result": result}), flush=True)
    raise SystemExit(0)
external_auth = False
"""
    script = script.replace("app_server = False\n", owner + "\napp_server = False\n")
    login = """        elif request.get("method") == "account/login/start":
            assert experimental_api
            assert request["params"]["type"] == "chatgptAuthTokens"
            assert set(request["params"]) == {"type", "accessToken", "chatgptAccountId", "chatgptPlanType"}
            assert not pathlib.Path(os.environ["CODEX_HOME"], "auth.json").exists()
            external_auth = True
            print(json.dumps({"id": request["id"], "result": {"type": "chatgptAuthTokens"}}), flush=True)
"""
    script = script.replace(
        '        elif request.get("method") == "model/list":',
        login + '        elif request.get("method") == "model/list":',
    )
    refresh = """            if "REFRESH" in prompt:
                account = "wrong-account" if "WRONG" in prompt else "account"
                print(json.dumps({"id": 1, "method": "account/chatgptAuthTokens/refresh", "params": {"reason": "unauthorized", "previousAccountId": account}}), flush=True)
                response = json.loads(next(sys.stdin))
                assert response["id"] == 1
                if "error" in response:
                    raise SystemExit(0)
                assert response["result"]["chatgptAccountId"] == "account"
"""
    script = script.replace(
        '            if prompt == "DIFFERENT_SESSION":',
        refresh + '            if prompt == "DIFFERENT_SESSION":',
    )
    script = script.replace(
        '([] if prompt.startswith("CHECK_ZAI_ROUTE") else ["auth.json"])',
        '([] if external_auth or prompt.startswith("CHECK_ZAI_ROUTE") else ["auth.json"])',
    )
    fake_codex.write_text(script)
    (source / "next-token").write_text(token(serial=1))
    return fake_codex, source


def owner(binary, home):
    return codex_auth.CredentialSource(
        str(binary), home, os.environ.copy(), codex_auth.source_binding(home)
    )


@pytest.mark.parametrize("profile", ["sealed", "edit", "review", "host"])
def test_run_resume_refresh_and_redaction(repository, auth_codex, profile):
    binary, source = auth_codex
    original_token = json.loads((source / "auth.json").read_text())["tokens"][
        "access_token"
    ]
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(str(binary))
    )
    first = runner.run(
        task="auth-test", profile=profile, prompt="REFRESH", timeout_seconds=15
    )
    assert first.succeeded, first.error
    assert (source / "refresh-count").read_text() == "1"
    request = runner.store.load_request(first.run_id)
    private = (
        Path(request.effective_policy["controller"]["provider_state"]) / "codex/home"
    )
    assert not (private / "auth.json").exists()
    # A model-written replacement is discarded, never promoted to the source.
    write_auth(private, token(account="forged"))
    resumed = runner.resume(run_id=first.run_id, prompt="again")
    assert resumed.succeeded, resumed.error
    assert resumed.session_id == first.session_id
    assert not (private / "auth.json").exists()
    assert (
        codex_auth.source_binding(source)
        == request.effective_policy["codex_auth"]["account_binding"]
    )
    assert first.billing.route == resumed.billing.route == "subscription"
    for result in (first, resumed):
        for artifact in runner.store.path(result.run_id).iterdir():
            if artifact.is_file():
                text = artifact.read_text()
                assert original_token not in text
                assert (source / "next-token").read_text() not in text
                assert "synthetic-refresh" not in text


def test_concurrent_refresh_uses_one_native_rotation(auth_codex):
    binary, source = auth_codex
    authority = owner(binary, source)
    previous = authority.read()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: owner(binary, source).read(previous=previous), range(4))
        )
    assert len({result.value for result in results}) == 1
    assert results[0].value != previous.value
    assert (source / "refresh-count").read_text() == "1"


def test_legacy_resume_replaces_copy_without_promoting_it(repository, auth_codex):
    binary, source = auth_codex
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(str(binary))
    )
    first = runner.run(task="legacy", profile="sealed", prompt="first")
    assert first.succeeded, first.error
    request = runner.store.load_request(first.run_id)
    private = (
        Path(request.effective_policy["controller"]["provider_state"]) / "codex/home"
    )
    write_auth(private, token(serial=999))
    retired = (private / "auth.json").read_bytes()
    request.effective_policy.pop("codex_auth")
    runner.store.write_json(
        runner.store.path(first.run_id) / "request.json", request.to_dict()
    )
    before = (source / "auth.json").read_bytes()
    resumed = runner.resume(run_id=first.run_id, prompt="second")
    assert resumed.succeeded, resumed.error
    assert (source / "auth.json").read_bytes() == before
    assert not (private / "auth.json").exists()
    resumed_request = runner.store.load_request(resumed.run_id)
    archive = Path(
        resumed_request.effective_policy["codex_auth"]["retired_credential_copy"]
    )
    assert archive.read_bytes() == retired
    assert archive.stat().st_mode & 0o777 == 0o600
    assert not archive.is_relative_to(private.parent.parent)


def test_resume_rejects_source_inside_task_state(repository, auth_codex, monkeypatch):
    binary, source = auth_codex
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(str(binary))
    )
    first = runner.run(task="self-source", profile="sealed", prompt="first")
    assert first.succeeded, first.error
    request = runner.store.load_request(first.run_id)
    private = (
        Path(request.effective_policy["controller"]["provider_state"]) / "codex/home"
    )
    write_auth(private)
    before = (private / "auth.json").read_bytes()
    monkeypatch.setenv("AOP_CODEX_SOURCE_HOME", str(private))
    with pytest.raises(AOPError, match="outside this task"):
        runner.resume(run_id=first.run_id, prompt="second")
    assert (private / "auth.json").read_bytes() == before


def test_retire_copy_rejects_symlink(tmp_path):
    home = tmp_path / "task"
    home.mkdir()
    original = tmp_path / "source-auth.json"
    original.write_text("private-source")
    (home / "auth.json").symlink_to(original)
    with pytest.raises(AOPError, match="safely retire"):
        codex_auth.retire_task_copy(home, tmp_path / "archive/auth.json")
    assert original.read_text() == "private-source"
    assert (home / "auth.json").is_symlink()


def test_new_run_cannot_fall_back_to_old_task_credentials(repository, auth_codex):
    binary, source = auth_codex
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(str(binary))
    )
    first = runner.run(task="old-copy", profile="edit", prompt="first")
    assert first.succeeded, first.error
    request = runner.store.load_request(first.run_id)
    private = (
        Path(request.effective_policy["controller"]["provider_state"]) / "codex/home"
    )
    write_auth(private)
    (source / "auth.json").unlink()
    with pytest.raises(AOPError, match="no current source login"):
        runner.run(task="old-copy", profile="edit", prompt="again")
    assert (private / "auth.json").exists()


def test_expired_source_fails_before_task_dispatch(repository, auth_codex):
    binary, source = auth_codex
    write_auth(source, token(expires=int(time.time()) - 1))
    (source / "fail-refresh").touch()
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(str(binary))
    )
    with pytest.raises(AOPError, match="source credentials"):
        runner.run(task="expired", profile="sealed", prompt="first")
    assert not list(
        runner.manager.sealed_runtime_dir.glob("provider-state/*/codex/home")
    )


def test_refresh_lock_has_a_deadline(auth_codex):
    binary, source = auth_codex
    with codex_auth._refresh_lock(source, time.monotonic() + 1):
        started = time.monotonic()
        with pytest.raises(AOPError, match="Timed out waiting"):
            owner(binary, source).read(deadline=started + 0.05)
        assert time.monotonic() - started < 1


@pytest.mark.parametrize("change", ["account", "subject"])
def test_resume_rejects_account_switch(repository, auth_codex, change):
    binary, source = auth_codex
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(str(binary))
    )
    first = runner.run(task="binding", profile="sealed", prompt="first")
    assert first.succeeded, first.error
    write_auth(source, token(**{change: "different"}))
    with pytest.raises(AOPError, match="original source account"):
        runner.resume(run_id=first.run_id, prompt="second")


@pytest.mark.parametrize("failure", ["WRONG REFRESH", "REFRESH"])
def test_refresh_failure_is_closed(repository, auth_codex, failure):
    binary, source = auth_codex
    (source / "fail-refresh").touch()
    before = (source / "auth.json").read_bytes()
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(str(binary))
    )
    result = runner.run(task="failure", profile="sealed", prompt=failure)
    assert not result.succeeded
    assert "account" in result.error or "credentials" in result.error
    assert (source / "auth.json").read_bytes() == before
    assert not (source / "refresh-count").exists()


def test_native_refresh_and_external_token_login(tmp_path):
    binary = shutil.which("codex")
    if binary is None:
        pytest.skip("Codex is not installed")
    source = tmp_path / "source"
    source.mkdir()
    write_auth(source)
    calls = []
    refreshed = token(serial=2)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"models": []}')

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            assert payload["refresh_token"] == "synthetic-refresh"
            assert payload["grant_type"] == "refresh_token"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "access_token": refreshed,
                        "id_token": refreshed,
                        "refresh_token": "native-rotated-refresh",
                    }
                ).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    authority = owner(binary, source)
    authority.environment["CODEX_REFRESH_TOKEN_URL_OVERRIDE"] = (
        f"http://127.0.0.1:{server.server_port}/token"
    )
    try:
        original = authority.read()
        replacement = authority.read(previous=original)
        assert replacement.value == refreshed
        assert len(calls) == 1
        assert (
            json.loads((source / "auth.json").read_text())["tokens"]["refresh_token"]
            == "native-rotated-refresh"
        )
        task = tmp_path / "task"
        task.mkdir()
        (task / "config.toml").write_text(
            f'chatgpt_base_url = "http://127.0.0.1:{server.server_port}"\n'
        )
        client = codex_auth._NativeAuth(
            binary, task, os.environ.copy(), time.monotonic() + 8
        )
        try:
            client.call(
                "initialize",
                {
                    "clientInfo": {"name": "aop-test", "version": "0.1"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            client.process.stdin.write('{"method":"initialized","params":{}}\n')
            client.process.stdin.flush()
            result = client.call(
                "account/login/start",
                {"type": "chatgptAuthTokens", **replacement.parameters()},
            )
            assert result["type"] == "chatgptAuthTokens"
            status = client.call(
                "getAuthStatus", {"includeToken": True, "refreshToken": False}
            )
            assert status["authMethod"] == "chatgptAuthTokens"
            assert status["authToken"] == refreshed
            assert not (task / "auth.json").exists()
        finally:
            client.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
