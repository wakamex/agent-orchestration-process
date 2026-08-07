from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from aop import __version__
from aop.cli import build_parser, main
from aop.locks import exclusive_lock, task_lock_path
from aop.runner import AgentRunner, CodexAdapter
from aop.worktrees import AOPError, WorktreeManager


SESSION_ID = "019f4da1-342f-7670-8aac-25999973b294"


def test_cli_reports_version_and_provider_neutral_resume_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"aop {__version__}\n"
    help_text = parser.format_help()
    assert "resume an agent session from a run" in help_text
    assert "resume the Codex session" not in help_text
    args = parser.parse_args(
        [
            "run",
            "player",
            "--agent",
            "hermes",
            "--mode",
            "participant",
            "--prompt",
            "play",
        ]
    )
    assert args.mode == "participant"
    assert parser.parse_args(
        ["run", "worker", "--agent", "opencode", "--prompt", "fix"]
    ).agent == "opencode"


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
    assert result.provider_duration_seconds is None
    assert result.command[0] == "bwrap"
    assert "--dangerously-bypass-approvals-and-sandbox" in result.command
    assert result.command[-1] == "-"

    run_dir = repository / ".aop" / "runs" / result.run_id
    request = json.loads((run_dir / "request.json").read_text())
    persisted_result = json.loads((run_dir / "result.json").read_text())
    events = (run_dir / "events.jsonl").read_text()

    assert request["prompt"] == "make the change"
    assert request["model"] == "gpt-5.6-sol"
    assert request["artifacts"] == []
    assert persisted_result["succeeded"] is True
    assert persisted_result["artifacts"] == []
    assert persisted_result["api_equivalent_cost"]["pricing_version"] == "2026-07-10"
    assert '"type": "thread.started"' in events
    assert (run_dir / "stderr.log").read_text() == ""


def test_sandbox_can_hide_external_control_directories(
    repository: Path,
    fake_codex: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control"
    control.mkdir()
    monkeypatch.setenv("AOP_HIDE_PATHS", os.fspath(control))
    manager = WorktreeManager.discover(repository)
    result = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).run(
        task="hidden-control", prompt="make the change", timeout_seconds=5
    )

    index = result.command.index("--tmpfs")
    assert result.command[index + 1] == os.fspath(control)
    assert result.succeeded


def test_declared_artifact_is_validated_and_archived(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    result = runner.run(
        task="extract",
        prompt="WRITE_ARTIFACT",
        sandbox="scratch-write",
        artifacts=["paper.md"],
    )

    assert result.succeeded
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    content = b"# Extracted\n"
    assert artifact.logical_path == "paper.md"
    assert artifact.archive_path == "artifacts/paper.md"
    assert artifact.size_bytes == len(content)
    assert artifact.sha256 == hashlib.sha256(content).hexdigest()

    run_dir = manager.state_dir / "runs" / result.run_id
    assert (run_dir / artifact.archive_path).read_bytes() == content
    assert "narrating before writing" in (run_dir / "events.jsonl").read_text()
    request = json.loads((run_dir / "request.json").read_text())
    output_dir = manager.get("extract").path / "scratch" / "outputs" / result.run_id
    assert os.fspath(output_dir) in request["prompt"]
    assert request["artifacts"] == ["paper.md"]


def test_declared_artifact_directory_is_recursively_archived(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    result = runner.run(
        task="linked-extract",
        prompt="WRITE_ARTIFACT_TREE",
        sandbox="scratch-write",
        artifacts=["paper.md", "assets"],
    )

    assert result.succeeded
    assert [artifact.logical_path for artifact in result.artifacts] == [
        "paper.md",
        "assets/figure-1.png",
        "assets/nested/figure-2.svg",
    ]
    expected = {
        "paper.md": b"# Extracted\n\n![Figure](assets/figure-1.png)\n",
        "assets/figure-1.png": b"PNG figure 1",
        "assets/nested/figure-2.svg": b"<svg>figure 2</svg>\n",
    }
    run_dir = manager.state_dir / "runs" / result.run_id
    for artifact in result.artifacts:
        content = expected[artifact.logical_path]
        assert (run_dir / artifact.archive_path).read_bytes() == content
        assert artifact.size_bytes == len(content)
        assert artifact.sha256 == hashlib.sha256(content).hexdigest()


def test_declared_empty_artifact_directory_is_valid(
    repository: Path, fake_codex: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    result = runner.run(
        task="empty-assets",
        prompt="WRITE_EMPTY_ARTIFACT_DIRECTORY",
        sandbox="scratch-write",
        artifacts=["assets"],
    )

    assert result.succeeded
    assert result.artifacts == ()


@pytest.mark.parametrize(
    ("prompt", "diagnostic"),
    [
        ("WRITE_ARTIFACT_TREE_SYMLINK", "symlinks are not allowed"),
        ("WRITE_ARTIFACT_TREE_EMPTY_FILE", "empty"),
        ("WRITE_ARTIFACT_TREE_SPECIAL_FILE", "not a regular file"),
    ],
)
def test_unsafe_file_in_declared_artifact_directory_fails_the_run(
    repository: Path,
    fake_codex: Path,
    prompt: str,
    diagnostic: str,
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    result = runner.run(
        task=f"unsafe-tree-{diagnostic.split()[0]}",
        prompt=prompt,
        sandbox="scratch-write",
        artifacts=["assets"],
    )

    assert not result.succeeded
    assert result.error == f"artifact assets/figure-1.png: {diagnostic}"
    assert result.artifacts == ()


def test_overlapping_artifact_declarations_fail_before_launch(
    repository: Path, fake_codex: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    with pytest.raises(
        AOPError,
        match="overlapping artifact paths: assets and assets/figure-1.png",
    ):
        runner.run(
            task="overlapping-artifacts",
            prompt="WRITE_ARTIFACT_TREE",
            sandbox="scratch-write",
            artifacts=["assets", "assets/figure-1.png"],
        )


@pytest.mark.parametrize(
    ("prompt", "diagnostic"),
    [
        ("MISSING_ARTIFACT", "missing"),
        ("EMPTY_ARTIFACT", "empty"),
        ("SYMLINK_ARTIFACT", "symlinks are not allowed"),
    ],
)
def test_invalid_declared_artifact_fails(
    repository: Path,
    fake_codex: Path,
    prompt: str,
    diagnostic: str,
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    result = runner.run(
        task=f"invalid-{diagnostic.split()[0]}",
        prompt=prompt,
        sandbox="scratch-write",
        artifacts=["paper.md"],
    )

    assert not result.succeeded
    assert result.exit_code == 0
    assert result.error == f"artifact paper.md: {diagnostic}"
    assert result.artifacts == ()
    run_dir = repository / ".aop" / "runs" / result.run_id
    assert not (run_dir / "artifacts" / "paper.md").exists()


def test_previous_run_artifact_cannot_satisfy_a_new_run(
    repository: Path, fake_codex: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )
    first = runner.run(
        task="repeat-extract",
        prompt="WRITE_ARTIFACT",
        sandbox="scratch-write",
        artifacts=["paper.md"],
    )

    second = runner.run(
        task="repeat-extract",
        prompt="MISSING_ARTIFACT",
        sandbox="scratch-write",
        artifacts=["paper.md"],
    )

    assert first.succeeded
    assert not second.succeeded
    assert second.error == "artifact paper.md: missing"
    assert first.run_id != second.run_id


@pytest.mark.parametrize("path", ["../paper.md", "/tmp/paper.md", "."])
def test_unsafe_artifact_declarations_fail_before_launch(
    repository: Path, fake_codex: Path, path: str
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    with pytest.raises(AOPError, match="relative and contained"):
        runner.run(task="unsafe-path", prompt="unused", artifacts=[path])

    assert not (repository / ".aop" / "worktrees" / "unsafe-path").exists()


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


def test_resume_uses_a_fresh_explicit_artifact_contract(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))
    first = runner.run(
        task="artifact-resume",
        prompt="WRITE_ARTIFACT",
        sandbox="scratch-write",
        artifacts=["paper.md"],
    )

    resumed = runner.resume(run_id=first.run_id, prompt="MISSING_ARTIFACT")

    assert first.succeeded
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert resumed.artifacts == ()
    request = json.loads(
        (manager.state_dir / "runs" / resumed.run_id / "request.json").read_text()
    )
    assert request["artifacts"] == []
    assert request["parent_run_id"] == first.run_id


def test_resume_reuses_lock_held_by_aop_exec(
    repository: Path, fake_codex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))
    first = runner.run(task="nested-resume", prompt="first", timeout_seconds=5)
    monkeypatch.setenv("AOP_TASK_LOCK_HELD", "nested-resume")

    with exclusive_lock(
        task_lock_path(manager.state_dir, "nested-resume"), "task nested-resume"
    ):
        resumed = runner.resume(run_id=first.run_id, prompt="inside exec")

    assert resumed.succeeded
    assert resumed.final_message == "answer:inside exec"


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


def test_cli_prints_machine_readable_run_and_resume_results(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))

    assert main(["run", "json-task", "--prompt", "first", "--json"]) == 0
    first_output = capsys.readouterr()
    first = json.loads(first_output.out)

    assert first_output.err == ""
    assert first["succeeded"] is True
    assert first["task"] == "json-task"
    assert first["final_message"] == "answer:first"

    assert main(["resume", first["run_id"], "--prompt", "second", "--json"]) == 0
    resumed_output = capsys.readouterr()
    resumed = json.loads(resumed_output.out)

    assert resumed_output.err == ""
    assert resumed["succeeded"] is True
    assert resumed["session_id"] == first["session_id"]
    assert resumed["final_message"] == "answer:second"


def test_cli_cleanup_discards_a_run_worktree_but_keeps_run_records(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))
    manager = WorktreeManager.discover(repository)
    result = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).run(
        task="disposable", prompt="first"
    )
    run_dir = manager.state_dir / "runs" / result.run_id

    assert main(["cleanup", result.run_id]) == 0
    assert capsys.readouterr().out == "disposable\n"
    assert not any(item.task == "disposable" for item in manager.list())
    assert (run_dir / "request.json").exists()
    assert (run_dir / "result.json").exists()

    assert main(["cleanup", result.run_id]) == 0
    assert capsys.readouterr().out == "disposable\n"


def test_cli_accepts_artifact_declarations(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))

    exit_code = main(
        [
            "run",
            "cli-artifact",
            "--sandbox",
            "scratch-write",
            "--artifact",
            "paper.md",
            "--prompt",
            "WRITE_ARTIFACT",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    run_id = re.search(r"run_id=([0-9a-f-]+)", captured.err)
    assert run_id is not None
    result = json.loads(
        (repository / ".aop" / "runs" / run_id.group(1) / "result.json").read_text()
    )
    assert result["artifacts"][0]["logical_path"] == "paper.md"
