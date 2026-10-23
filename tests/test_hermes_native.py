"""Opt-in contract against installed Hermes, using a local model server without real keys."""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from support import AOPTestCase, parametrize, skip

from agent_orchestration_process.runner import AgentRunner, HermesAdapter
from agent_orchestration_process.worktrees import WorktreeManager


def _serve(calls: list[dict]) -> ThreadingHTTPServer:
    """An OpenAI-compatible chat server that answers every request the same way."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self._json({"object": "list", "data": [{"id": "test-model", "object": "model"}]})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.path.endswith("/chat/completions"):
                # Hermes probes local servers for Ollama and similar APIs.
                self.send_error(404)
                return
            calls.append(payload)
            assert self.headers.get("Authorization") == "Bearer local-test-key"
            usage = {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
            if not payload.get("stream"):
                self._json({
                    "id": "chat", "object": "chat.completion", "created": 1, "model": "test-model",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "native answer"}}],
                    "usage": usage,
                })
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta, finish, extra in (
                ({"role": "assistant", "content": "native answer"}, None, {}),
                ({}, "stop", {"usage": usage}),
            ):
                chunk = {
                    "id": "chat", "object": "chat.completion.chunk", "created": 1,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    **extra,
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def _json(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


class HermesNativeTest(AOPTestCase):
    @parametrize("profile", ["sealed", "host"])
    def test_native_hermes_run_and_resume(self, tmp_path, monkeypatch, profile):
        binary = os.environ.get("AOP_TEST_HERMES_BIN")
        if not binary:
            skip("set AOP_TEST_HERMES_BIN to test installed Hermes")
        installed = Path(os.environ.get("AOP_TEST_HERMES_HOME", Path.home() / ".hermes"))
        calls: list[dict] = []
        server = _serve(calls)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        # A Hermes home with only a local model route, beside the installed
        # dependency store and tools that every Hermes home shares.
        source = tmp_path / "hermes-source"
        source.mkdir()
        for name in ("installs", "tools"):
            (source / name).symlink_to(installed / name)
        (source / "config.yaml").write_text(
            "model:\n"
            "  provider: custom\n"
            "  default: test-model\n"
            f"  base_url: http://127.0.0.1:{server.server_port}/v1\n"
            "  api_key: local-test-key\n"
        )
        monkeypatch.setenv("HERMES_HOME", os.fspath(source))
        manager = (
            WorktreeManager.standalone(tmp_path / "controller")
            if profile == "sealed"
            else WorktreeManager.standalone(tmp_path / "project")
        )
        runner = AgentRunner(manager, HermesAdapter(binary))
        workdir = tmp_path / "work"
        workdir.mkdir()
        try:
            first = runner.run(
                task="native-hermes",
                prompt="first question",
                profile=profile,
                workdir=workdir if profile == "host" else None,
                timeout_seconds=120,
            )
            assert first.succeeded, first.error
            assert first.final_message == "native answer"
            assert any("first question" in json.dumps(call) for call in calls)

            resumed = runner.resume(run_id=first.run_id, prompt="second question")
            assert resumed.succeeded, resumed.error
            assert resumed.session_id == first.session_id
            assert resumed.final_message == "native answer"
            assert "first question" in json.dumps(calls[-1])
            assert "second question" in json.dumps(calls[-1])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()
