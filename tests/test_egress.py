from __future__ import annotations

import http.server
import json
import os
import threading
from pathlib import Path

import pytest

from agent_orchestration_process import egress
from agent_orchestration_process.runner import AgentRunner, ClaudeAdapter
from agent_orchestration_process.worktrees import AOPError, WorktreeManager

# Runs inside the sandbox before the fake Claude answers: tries each endpoint
# and writes what happened to /output/egress.json.
PROBE = """
import json, socket, urllib.request, urllib.error
outcome = {}
def attempt(name, url):
    try:
        urllib.request.urlopen(url, data=b"probe", timeout=5)
        outcome[name] = "ok"
    except urllib.error.HTTPError as error:
        outcome[name] = f"http {error.code}"
    except Exception as error:
        outcome[name] = f"failed: {type(error).__name__}"
attempt("allowed", ALLOWED + "/v1/messages")
attempt("other_loopback", OTHER + "/v1/messages")
attempt("remote", "https://example.com/")
try:
    socket.create_connection(("1.1.1.1", 443), timeout=3)
    outcome["direct"] = "connected"
except OSError as error:
    outcome["direct"] = f"failed: {error.errno}"
open("/output/egress.json", "w").write(json.dumps(outcome))
"""


def recorder() -> tuple[http.server.HTTPServer, list[bytes]]:
    received: list[bytes] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(400)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


def test_no_web_sandbox_reaches_only_allowed_endpoints(
    repository: Path, fake_claude: Path
) -> None:
    allowed, allowed_bodies = recorder()
    other, other_bodies = recorder()
    allowed_url = f"http://127.0.0.1:{allowed.server_port}"
    other_url = f"http://127.0.0.1:{other.server_port}"
    probe = PROBE.replace("ALLOWED", repr(allowed_url)).replace(
        "OTHER", repr(other_url)
    )
    script = fake_claude.read_text()
    fake_claude.write_text(
        script.replace(
            "prompt = sys.stdin.read()\n",
            "prompt = sys.stdin.read()\n" + f"exec({probe!r})\n",
            1,
        )
    )
    runner = AgentRunner(
        WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
    )
    try:
        result = runner.run(
            task="egress",
            prompt="hello",
            profile="sealed",
            no_web=True,
            allowed_urls=[allowed_url],
            artifacts=["egress.json"],
        )
    finally:
        allowed.shutdown()
        other.shutdown()

    assert result.succeeded, result.error
    run_dir = runner.store.path(result.run_id)
    outcome = json.loads((run_dir / "artifacts" / "egress.json").read_text())
    assert outcome["allowed"] == "http 400"
    assert outcome["other_loopback"].startswith("failed")
    assert outcome["remote"].startswith("failed")
    assert outcome["direct"].startswith("failed")
    assert allowed_bodies == [b"probe"]
    assert other_bodies == []
    log = [
        json.loads(line)
        for line in (run_dir / "network.jsonl").read_text().splitlines()
    ]
    assert {"endpoint": f"127.0.0.1:{allowed.server_port}", "decision": "allowed"} in [
        {key: entry[key] for key in ("endpoint", "decision")} for entry in log
    ]
    assert {"endpoint": "example.com:443", "decision": "denied"} in [
        {key: entry[key] for key in ("endpoint", "decision")} for entry in log
    ]
    request = runner.store.load_request(result.run_id)
    assert request.allowed_urls == (allowed_url,)
    network = request.effective_policy["network"]
    assert network["isolation"] == "network-namespace"
    assert f"127.0.0.1:{allowed.server_port}" in network["allowed_endpoints"]


def test_allowed_urls_need_no_web_and_valid_urls(
    repository: Path, fake_claude: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
    )

    with pytest.raises(AOPError, match="requires --no-web"):
        runner.run(task="a", prompt="x", allowed_urls=["https://example.com"])
    with pytest.raises(AOPError, match="only loopback URLs may use plain http"):
        runner.run(
            task="b", prompt="x", no_web=True, allowed_urls=["http://example.com"]
        )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://API.example.com/v1", "api.example.com:443"),
        ("http://127.0.0.1:18977", "127.0.0.1:18977"),
        ("https://example.com:8443/x", "example.com:8443"),
    ],
)
def test_endpoint_parsing(url: str, expected: str) -> None:
    assert egress.endpoint(url) == expected


def test_harness_endpoints_join_explicit_urls() -> None:
    assert egress.allowed_endpoints("claude", ["http://127.0.0.1:9"]) == [
        "127.0.0.1:9",
        "api.anthropic.com:443",
        "platform.claude.com:443",
    ]
    assert egress.allowed_endpoints("hermes", [], "xai-oauth") == [
        "api.x.ai:443",
        "auth.x.ai:443",
    ]
    # A provider without verified endpoints gets only what the run names.
    assert egress.allowed_endpoints("hermes", [], "nous") == []
