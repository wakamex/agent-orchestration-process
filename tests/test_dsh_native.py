"""Opt-in contracts against installed dsh, using local inference without real keys."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import shlex
import threading

import pytest
import yaml

from agent_orchestration_process.runner import AgentRunner, DeepSeekHarnessAdapter
from agent_orchestration_process.worktrees import WorktreeManager


@pytest.mark.parametrize("exercise_tools", [False, True])
@pytest.mark.parametrize("profile", ["edit", "review", "sealed", "host"])
def test_native_dsh_run_and_resume(
    repository, tmp_path, monkeypatch, profile, exercise_tools
):
    binary = os.environ.get("AOP_TEST_DSH_BIN")
    if not binary:
        pytest.skip("set AOP_TEST_DSH_BIN to test installed dsh")
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(payload)
            assert self.path == "/chat/completions"
            assert payload["model"] == "deepseek-flash"
            assert self.headers.get("Authorization") == "Bearer local-test-key"
            assert "dsh_plugin_packages" not in payload
            assert "dsh_session_log" not in payload
            delta = {"role": "assistant", "content": "native answer"}
            finish = "stop"
            if exercise_tools and len(calls) == 1:
                definitions = {
                    tool["function"]["name"]: tool["function"]
                    for tool in payload["tools"]
                }
                assert "bash" in definitions
                prompt = "\n".join(
                    m["content"]
                    for m in payload["messages"]
                    if isinstance(m.get("content"), str)
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
                    set(definitions["bash"]["parameters"].get("required", []))
                    <= arguments.keys()
                )
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "native-tool",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
                finish = "tool_calls"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta, finish, usage in [
                (delta, None, None),
                (
                    {},
                    finish,
                    {
                        "prompt_tokens": 12,
                        "prompt_cache_hit_tokens": 2,
                        "prompt_cache_miss_tokens": 10,
                        "completion_tokens": 4,
                        "total_tokens": 16,
                    },
                ),
            ]:
                event = {
                    "id": f"chat-{len(calls)}",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "deepseek-flash",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                if usage:
                    event["usage"] = usage
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

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
    try:
        first = runner.run(
            task="native-dsh",
            prompt="Say native answer",
            effort="none",
            profile=profile,
            timeout_seconds=30,
            artifacts=["proof.json"] if exercise_tools else [],
        )
        assert first.succeeded, first.error
        assert first.final_message == "native answer"
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
                m.get("role") == "tool" and m.get("tool_call_id") == "native-tool"
                for m in calls[1]["messages"]
            )
        resumed = runner.resume(run_id=first.run_id, prompt="Say native answer again")
        assert resumed.succeeded, resumed.error
        assert resumed.session_id == first.session_id
        assert resumed.usage.input_tokens == 12
        assert len(calls) == first_calls + 1
        assert len(calls[-1]["messages"]) > len(calls[0]["messages"])
    finally:
        (tmp_path / "requests.json").write_text(json.dumps(calls))
        server.shutdown()
        server.server_close()
        worker.join()
