"""Opt-in local contract tests against the actual Zcode CLI, without paid inference.

Run with AOP_TEST_ZCODE_BIN pointing to the installed zcode.cjs bundle.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import shlex
from pathlib import Path
import threading

import pytest

from agent_orchestration_process.runner import AgentRunner, ZcodeAdapter
from agent_orchestration_process.worktrees import WorktreeManager


@pytest.fixture
def native_zcode():
    configured = os.environ.get("AOP_TEST_ZCODE_BIN")
    if not configured:
        pytest.skip("set AOP_TEST_ZCODE_BIN to test the installed native Zcode bundle")
    return Path(configured).resolve()


@pytest.mark.parametrize(
    "profile,exercise_tools,effort",
    [
        (profile, tools, "high")
        for profile in ("edit", "review", "sealed", "host")
        for tools in (False, True)
    ]
    + [("host", False, effort) for effort in (None, "low", "max", "medium")],
)
def test_native_run_and_resume(
    native_zcode, repository, tmp_path, monkeypatch, profile, exercise_tools, effort
):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"data":{}}')

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            assert self.path == "/v1/messages"
            assert payload["model"] == "glm-5.3-flash"
            assert self.headers.get("x-api-key") == "local-test-key"
            assert payload["stream"] is True
            assert payload["output_config"]["effort"] == (effort or "max")
            assert isinstance(payload["messages"], list)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            events = [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "model": "glm-5.3-flash",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 2,
                            "cache_creation_input_tokens": 0,
                        },
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "native test answer"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 4},
                },
                {"type": "message_stop"},
            ]
            if exercise_tools and len(calls) == 1:
                tools = {tool["name"]: tool for tool in payload["tools"]}
                assert "Bash" in tools, list(tools)
                texts = [
                    part["text"]
                    for message in payload["messages"]
                    for part in message["content"]
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                output_dir = re.search(
                    r"Write the following deliverables beneath (.+):", "\n".join(texts)
                ).group(1)
                script = (
                    "import json; from pathlib import Path\n"
                    "target = Path.cwd() / 'README.md'\n"
                    "evidence = {'workspace_readable': target.is_file()}\n"
                    "try:\n target.write_text('native tool edit\\n'); evidence['workspace_writable'] = True\n"
                    "except OSError:\n evidence['workspace_writable'] = False\n"
                    f"Path({output_dir!r}, 'proof.json').write_text(json.dumps(evidence))\n"
                )
                arguments = {
                    "command": "python3 -c " + shlex.quote(script),
                    "description": "Check workspace permissions and create the requested artifact",
                }
                assert (
                    set(tools["Bash"]["input_schema"].get("required", []))
                    <= arguments.keys()
                ), tools["Bash"]["input_schema"]
                events = [
                    events[0],
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "tool_native",
                            "name": "Bash",
                            "input": {},
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(arguments),
                        },
                    },
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                        "usage": {"output_tokens": 4},
                    },
                    {"type": "message_stop"},
                ]
            for event in events:
                self.wfile.write(
                    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    source = tmp_path / "native-source"
    (source / "cli").mkdir(parents=True)
    config = {
        "provider": {
            "test": {
                "kind": "anthropic",
                "options": {"baseURL": base_url, "apiKey": "local-test-key"},
                "models": {"glm-5.3-flash": {"contextWindow": 1000000}},
            }
        },
        "model": {"main": "test/glm-5.3-flash", "lite": "test/glm-5.3-flash"},
        "plugins": {"enabled": False},
        "features": {"mcp": False},
    }
    if profile == "sealed" and not exercise_tools:
        config["provider"]["test"]["options"].pop("apiKey")
        config["provider"]["test"]["options"]["apiKeyRequired"] = True
        monkeypatch.setenv("ANTHROPIC_API_KEY", "local-test-key")
    (source / "cli/config.json").write_text(json.dumps(config))
    monkeypatch.setenv("AOP_ZCODE_SOURCE_HOME", str(source))
    monkeypatch.setenv("ZCODE_BASE_URL", base_url)
    monkeypatch.delenv("ZCODE_MODEL", raising=False)
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(native_zcode))
    )
    try:
        first = runner.run(
            task="native-zcode",
            prompt="Reply with native test answer"
            + (" café synthetic\n" * 50000 if effort is None else ""),
            profile=profile,
            effort=effort,
            timeout_seconds=30,
            artifacts=["proof.json"] if exercise_tools else [],
        )
        if effort == "medium":
            assert not first.succeeded
            assert "effort" in first.error.lower() or "reasoning" in first.error.lower()
            assert calls == []
            return
        assert first.succeeded, first.error
        assert first.effort == (effort or "max")
        assert first.model == "test/glm-5.3-flash"
        assert first.final_message == "native test answer"
        turns = 2 if exercise_tools else 1
        assert first.usage.input_tokens == 12 * turns
        assert first.usage.cached_input_tokens == 2 * turns
        assert first.usage.output_tokens == 4 * turns
        if exercise_tools:
            assert len(first.artifacts) == 1
            proof_path = (
                runner.store.root / first.run_id / first.artifacts[0].archive_path
            )
            proof = json.loads(proof_path.read_text())
            assert proof["workspace_writable"] == (profile in {"edit", "host"})
            assert proof["workspace_readable"] == (profile != "sealed")
            assert (repository / "README.md").read_text() == "# Test project\n"
            if profile != "sealed":
                workspace = runner.manager.get("native-zcode").path
                expected = (
                    "native tool edit\n"
                    if profile in {"edit", "host"}
                    else "# Test project\n"
                )
                assert (workspace / "README.md").read_text() == expected
            tool_results = [
                part
                for message in calls[1]["messages"]
                for part in message["content"]
                if isinstance(part, dict) and part.get("type") == "tool_result"
            ]
            assert any(
                part["tool_use_id"] == "tool_native" and not part.get("is_error")
                for part in tool_results
            )

        first_call_count = len(calls)
        followup = "Reply again" + (
            " résumé followup\n" * 50000 if effort is None else ""
        )
        resumed = runner.resume(run_id=first.run_id, prompt=followup)
        assert resumed.effort == (effort or "max")
        assert resumed.succeeded, resumed.error
        assert resumed.session_id == first.session_id
        # Native automatic compaction can add an inference call on resume.
        resumed_call_count = len(calls) - first_call_count
        assert resumed_call_count >= 1
        assert resumed.usage.input_tokens == 12 * resumed_call_count
        assert resumed.usage.cached_input_tokens == 2 * resumed_call_count
        assert resumed.usage.output_tokens == 4 * resumed_call_count
        assert len(calls[-1]["messages"]) > len(calls[0]["messages"])
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
