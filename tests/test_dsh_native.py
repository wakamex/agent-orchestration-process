"""Opt-in contracts against installed dsh, using local inference without real keys."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import shlex
import threading

from support import AOPTestCase, parametrize, skip
import yaml

from agent_orchestration_process.runner import AgentRunner, DeepSeekHarnessAdapter
from agent_orchestration_process.worktrees import WorktreeManager


def _texts(message):
    """The text of one Messages API message, a string or a list of blocks."""
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    return [part.get("text", "") for part in content or [] if isinstance(part, dict)]


class DshNativeTest(AOPTestCase):
    @parametrize("exercise_tools", [False, True])
    @parametrize("profile", ["edit", "review", "sealed", "host"])
    def test_native_dsh_run_and_resume(
        self,
        repository, tmp_path, monkeypatch, profile, exercise_tools
    ):
        binary = os.environ.get("AOP_TEST_DSH_BIN")
        if not binary:
            skip("set AOP_TEST_DSH_BIN to test installed dsh")
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                # dsh 0.2 calls DeepSeek's Anthropic-compatible Messages API.
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(payload)
                assert self.path == "/v1/messages"
                assert payload["model"] == "deepseek-flash"
                assert payload.get("stream") is True
                assert self.headers.get("x-api-key") == "local-test-key"
                assert "dsh_plugin_packages" not in payload
                assert "dsh_session_log" not in payload
                block = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": "native answer"}
                stop = "end_turn"
                if exercise_tools and len(calls) == 1:
                    definitions = {tool["name"]: tool for tool in payload["tools"]}
                    assert "bash" in definitions
                    prompt = "\n".join(
                        text
                        for m in payload["messages"]
                        for text in _texts(m)
                    )
                    output_dir = re.search(
                        r"Write the following deliverables beneath (.+):", prompt
                    ).group(1)
                    script = (
                        "import json; from pathlib import Path\n"
                        "target = Path.cwd() / 'README.md'\n"
                        "evidence = {'readable': target.is_file()}\n"
                        "try:\n target.write_text('native edit\\n'); evidence['writable'] = True\n"
                        "except OSError:\n evidence['writable'] = False\n"
                        f"Path({output_dir!r}, 'proof.json').write_text(json.dumps(evidence))\n"
                    )
                    arguments = {
                        "command": "python3 -c " + shlex.quote(script),
                        "description": "Check workspace access and write proof",
                    }
                    assert (
                        set(definitions["bash"]["input_schema"].get("required", []))
                        <= arguments.keys()
                    )
                    block = {"type": "tool_use", "id": "native-tool", "name": "bash", "input": {}}
                    delta = {"type": "input_json_delta", "partial_json": json.dumps(arguments)}
                    stop = "tool_use"
                events = [
                    ("message_start", {
                        "type": "message_start",
                        "message": {
                            "id": f"msg-{len(calls)}",
                            "type": "message",
                            "role": "assistant",
                            "model": "deepseek-flash",
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {
                                "input_tokens": 10,
                                "cache_read_input_tokens": 2,
                                "cache_creation_input_tokens": 0,
                                "output_tokens": 0,
                            },
                        },
                    }),
                    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block}),
                    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    ("message_delta", {
                        "type": "message_delta",
                        "delta": {"stop_reason": stop, "stop_sequence": None},
                        "usage": {"output_tokens": 4},
                    }),
                    ("message_stop", {"type": "message_stop"}),
                ]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for name, event in events:
                    self.wfile.write(f"event: {name}\ndata: {json.dumps(event)}\n\n".encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        source = tmp_path / "dsh-source"
        source.mkdir()
        (source / "settings.yaml").write_text(
            yaml.safe_dump(
                {
                    "llm-deepseek": {
                        "apiKeyEnv": "AOP_NATIVE_TEST_KEY",
                        "baseURL": f"http://127.0.0.1:{server.server_port}",
                    }
                }
            )
        )
        monkeypatch.setenv("AOP_DSH_SOURCE_HOME", str(source))
        monkeypatch.setenv("AOP_NATIVE_TEST_KEY", "local-test-key")
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        runner = AgentRunner(
            WorktreeManager.discover(repository), DeepSeekHarnessAdapter(binary)
        )
        # Exceed Linux's per-argument limit with synthetic UTF-8 text. Verify exact
        # delivery for both turns, including punctuation and leading option syntax.
        first_prompt = "--first\n" + "synthetic café 'quoted' $value\n" * 24000
        resume_prompt = "--resume\n" + "different résumé `literal`\n" * 25000

        def contains_prompt(payload, prompt):
            return any(
                prompt in "".join(_texts(message))
                for message in payload["messages"]
                if message.get("role") == "user"
            )

        try:
            first = runner.run(
                task="native-dsh",
                prompt=first_prompt,
                effort="none",
                profile=profile,
                timeout_seconds=30,
                artifacts=["proof.json"] if exercise_tools else [],
            )
            assert first.succeeded, first.error
            assert first.final_message == "native answer"
            assert contains_prompt(calls[0], first_prompt)
            first_calls = 2 if exercise_tools else 1
            assert first.usage.input_tokens == 12 * first_calls
            assert first.usage.cached_input_tokens == 2 * first_calls
            assert first.usage.output_tokens == 4 * first_calls
            if exercise_tools:
                proof = json.loads(
                    (
                        runner.store.root / first.run_id / first.artifacts[0].archive_path
                    ).read_text()
                )
                assert proof == {
                    "readable": profile != "sealed",
                    "writable": profile in {"edit", "host"},
                }
                assert (repository / "README.md").read_text() == "# Test project\n"
                if profile != "sealed":
                    expected = (
                        "native edit\n"
                        if profile in {"edit", "host"}
                        else "# Test project\n"
                    )
                    assert (
                        runner.manager.get("native-dsh").path / "README.md"
                    ).read_text() == expected
                assert any(
                    part.get("type") == "tool_result"
                    and part.get("tool_use_id") == "native-tool"
                    for m in calls[1]["messages"]
                    if m.get("role") == "user" and isinstance(m.get("content"), list)
                    for part in m["content"]
                )
            resumed = runner.resume(run_id=first.run_id, prompt=resume_prompt)
            assert resumed.succeeded, resumed.error
            assert resumed.session_id == first.session_id
            assert contains_prompt(calls[-1], resume_prompt)
            assert resumed.usage.input_tokens == 12
            assert len(calls) == first_calls + 1
            assert len(calls[-1]["messages"]) > len(calls[0]["messages"])
        finally:
            (tmp_path / "requests.json").write_text(json.dumps(calls))
            server.shutdown()
            server.server_close()
            worker.join()


class DshNativeNoWebTest(AOPTestCase):
    def test_no_web_dsh_offers_no_web_shell_or_code_tools(self, tmp_path, monkeypatch):
        binary = os.environ.get("AOP_TEST_DSH_BIN")
        if not binary:
            skip("set AOP_TEST_DSH_BIN to test installed dsh")
        offered = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                offered.append(sorted(tool["name"] for tool in payload.get("tools", [])))
                usage = {"input_tokens": 1, "output_tokens": 0}
                events = [
                    {"type": "message_start", "message": {
                        "id": "msg", "type": "message", "role": "assistant",
                        "model": "deepseek-flash", "content": [], "stop_reason": None,
                        "stop_sequence": None, "usage": usage}},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                     "usage": {"output_tokens": 1}},
                    {"type": "message_stop"},
                ]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for event in events:
                    self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        source = tmp_path / "dsh-source"
        source.mkdir()
        (source / "settings.yaml").write_text(yaml.safe_dump({"llm-deepseek": {
            "apiKeyEnv": "AOP_NATIVE_TEST_KEY",
            "baseURL": f"http://127.0.0.1:{server.server_port}",
        }}))
        monkeypatch.setenv("AOP_DSH_SOURCE_HOME", str(source))
        monkeypatch.setenv("AOP_NATIVE_TEST_KEY", "local-test-key")
        monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
        manager = WorktreeManager.standalone(tmp_path / "controller")
        try:
            # The loopback model server is forwarded into the network sandbox.
            result = AgentRunner(manager, DeepSeekHarnessAdapter(binary)).run(
                task="no-web",
                prompt="answer",
                profile="sealed",
                no_web=True,
                allowed_urls=[f"http://127.0.0.1:{server.server_port}"],
                timeout_seconds=30,
            )
        finally:
            server.shutdown()
            server.server_close()
            worker.join()

        assert result.succeeded, result.error
        stderr = (manager.state_dir / "runs" / result.run_id / "stderr.log").read_text()
        assert "did not activate" not in stderr, stderr
        tools = set(offered[0])
        assert tools, "dsh offered no tools at all"
        forbidden = {"web_search", "web_fetch", "bash", "pwsh", "workflow", "jobs"}
        assert not tools & forbidden, sorted(tools & forbidden)
        assert not any("web" in name or "workflow" in name or "ptc" in name for name in tools), sorted(tools)
