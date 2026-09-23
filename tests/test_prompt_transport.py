"""Prompt delivery contracts across harness transports."""

import hashlib
import json
import subprocess
import sys

import pytest

from agent_orchestration_process import prompt_transport
from agent_orchestration_process.runner import (
    AgentRunner,
    adapter_for,
    _capture_process,
)
from agent_orchestration_process.worktrees import WorktreeManager
from test_zcode import fake_zcode, zcode_config  # noqa: F401


@pytest.mark.parametrize("provider", list(prompt_transport.TRANSPORTS))
def test_large_prompt_run_and_resume(provider, request, repository, monkeypatch):
    if provider == "zcode":
        request.getfixturevalue("zcode_config")
    binary = request.getfixturevalue(f"fake_{provider}")
    monkeypatch.setenv(f"AOP_{provider.upper()}_BIN", str(binary))
    runner = AgentRunner(WorktreeManager.discover(repository), adapter_for(provider))
    first_prompt = "Synthetic first turn\n" + "café '$value' `literal`\n" * 30000
    followup = "Synthetic followup\n" + "résumé and different text\n" * 28000
    first = runner.run(
        task="large", profile="sealed", prompt=first_prompt, timeout_seconds=15
    )
    assert first.succeeded, first.error
    assert first_prompt.rstrip() in first.final_message
    second = runner.resume(run_id=first.run_id, prompt=followup, timeout_seconds=15)
    assert second.succeeded, second.error
    assert second.session_id == first.session_id
    assert followup.rstrip() in second.final_message
    # Persisted argv remains bounded, including bwrap's arguments.
    for result in (first, second):
        recorded = json.loads(
            (runner.store.root / result.run_id / "request.json").read_text()
        )
        assert (
            recorded["effective_policy"]["prompt_transport"]
            == prompt_transport.TRANSPORTS[provider]
        )
        assert all(len(arg.encode()) < 100000 for arg in result.command)


def test_runtime_loader_preserves_exact_utf8(tmp_path):
    text = "synthetic\n" + "café '$value' `literal`\r\n" * 30000
    path = prompt_transport.stage_prompt(tmp_path, "test", text)
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.read_bytes() == text.encode()
    entry = tmp_path / "cli.py"
    entry.write_text(
        "import hashlib, sys\nprint(hashlib.sha256(sys.argv[2].encode()).hexdigest())\n"
    )
    command = [
        sys.executable,
        "-c",
        prompt_transport.PYTHON_ARGV_LOADER,
        str(entry),
        "-q",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == hashlib.sha256(text.encode()).hexdigest()


def test_stdin_delivery_does_not_bypass_timeout(tmp_path):
    result = _capture_process(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        environment={},
        prompt="x" * 700000,
        timeout_seconds=0.1,
        is_response=lambda line: False,
    )
    assert result.timed_out
    assert result.duration_seconds < 5


def test_structured_stdin_preserves_one_turn():
    text = 'first\n{"type":"user"}\nlast'
    wire = prompt_transport.stdin_prompt("agy", text)
    assert len(wire.splitlines()) == 1
    assert json.loads(wire) == {
        "event": "user",
        "message": {"role": "user", "content": text},
    }
