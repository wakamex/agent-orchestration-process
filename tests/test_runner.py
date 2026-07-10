from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from aop.cli import main
from aop.runner import AgentRunner, CodexAdapter
from aop.worktrees import WorktreeManager


SESSION_ID = "019f4da1-342f-7670-8aac-25999973b294"


def test_run_persists_structured_codex_artifacts(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    result = runner.run(
        task="implement",
        prompt="make the change",
        model="gpt-5.6-sol",
        effort="high",
        timeout_seconds=5,
    )

    assert result.succeeded
    assert result.session_id == SESSION_ID
    assert result.model == "gpt-5.6-sol"
    assert result.effort == "high"
    assert result.final_message == "answer:make the change"
    assert result.usage.input_tokens == 1
    assert result.usage.output_tokens == 1
    assert result.usage.reasoning_output_tokens == 0
    assert result.api_equivalent_cost is not None
    assert result.api_equivalent_cost.amount_usd == 0.000035
    assert result.time_to_first_event_seconds is not None
    assert result.time_to_first_response_seconds is not None
    assert "--dangerously-bypass-approvals-and-sandbox" not in result.command
    assert result.command[-1] == "-"
    assert ["--sandbox", "workspace-write"] == result.command[5:7]

    run_dir = repository / ".aop" / "runs" / result.run_id
    request = json.loads((run_dir / "request.json").read_text())
    persisted_result = json.loads((run_dir / "result.json").read_text())
    events = (run_dir / "events.jsonl").read_text()

    assert request["prompt"] == "make the change"
    assert request["model"] == "gpt-5.6-sol"
    assert persisted_result["succeeded"] is True
    assert persisted_result["api_equivalent_cost"]["pricing_version"] == "2026-07-10"
    assert '"type": "thread.started"' in events
    assert (run_dir / "stderr.log").read_text() == ""


def test_resume_uses_recorded_session_and_links_runs(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))
    first = runner.run(
        task="review",
        prompt="first",
        model="gpt-5.6-terra",
        effort="medium",
        timeout_seconds=5,
    )

    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert resumed.succeeded
    assert resumed.session_id == SESSION_ID
    assert resumed.model == "gpt-5.6-terra"
    assert resumed.effort == "medium"
    assert resumed.api_equivalent_cost is not None
    assert resumed.final_message == "answer:second"
    resume_index = resumed.command.index("resume")
    assert resumed.command[resume_index + 1 : resume_index + 3] == [SESSION_ID, "-"]
    assert "--model" not in resumed.command

    request_path = repository / ".aop" / "runs" / resumed.run_id / "request.json"
    request = json.loads(request_path.read_text())
    assert request["parent_run_id"] == first.run_id
    assert request["session_id"] == SESSION_ID
    assert request["timeout_seconds"] == 5


def test_timeout_terminates_process_and_records_result(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    result = runner.run(task="slow", prompt="SLEEP", timeout_seconds=0.05)

    assert not result.succeeded
    assert result.timed_out
    assert result.error == "timed out after 0.05 seconds"
    result_path = repository / ".aop" / "runs" / result.run_id / "result.json"
    assert json.loads(result_path.read_text())["timed_out"] is True


def test_turn_failure_is_not_reported_as_success(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    result = runner.run(task="failure", prompt="FAIL")

    assert not result.succeeded
    assert result.exit_code == 1
    assert result.error == "synthetic failure"
    assert result.final_message is None


def test_cli_runs_codex_and_reports_resumable_ids(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))

    exit_code = main(["run", "cli-task", "--prompt", "from cli"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "answer:from cli\n"
    assert f"session_id={SESSION_ID}" in captured.err
    run_id = re.search(r"run_id=([0-9a-f-]+)", captured.err)
    assert run_id is not None
    assert (repository / ".aop" / "runs" / run_id.group(1) / "result.json").exists()
