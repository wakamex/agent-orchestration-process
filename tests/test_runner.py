from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from agent_orchestration_process import __version__
from agent_orchestration_process.cli import build_parser, main
from agent_orchestration_process.locks import exclusive_lock, task_lock_path
from agent_orchestration_process.runner import AgentRunner, CodexAdapter
from agent_orchestration_process.worktrees import AOPError, WorktreeManager


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
    read_args = parser.parse_args(
        [
            "run",
            "reader",
            "--read",
            "/sources/one",
            "--read",
            "/sources/two",
            "--prompt",
            "inspect",
        ]
    )
    assert read_args.read_paths == ["/sources/one", "/sources/two"]

    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(["resume", "--help"])
    assert help_exit.value.code == 0
    resume_help = capsys.readouterr().out
    assert "replace inherited read paths" in resume_help


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
    assert result.billing.route == "subscription"
    assert result.billing.credential_source == "chatgpt-oauth"
    assert result.billing.detected_by == "codex login status"
    assert not result.billing.actual_cost_known
    assert result.time_to_first_event_seconds is not None
    assert result.time_to_first_response_seconds is not None
    assert result.provider_duration_seconds is None
    assert result.command[0] == "bwrap"
    assert "--dangerously-bypass-approvals-and-sandbox" in result.command
    assert result.command[-1] == "-"

    run_dir = repository / ".aop" / "runs" / result.run_id
    assert (repository / ".aop" / "runs").stat().st_mode & 0o777 == 0o700
    assert run_dir.stat().st_mode & 0o777 == 0o700
    request = json.loads((run_dir / "request.json").read_text())
    persisted_result = json.loads((run_dir / "result.json").read_text())
    events = (run_dir / "events.jsonl").read_text()

    assert request["prompt"] == "make the change"
    assert request["model"] == "gpt-5.6-sol"
    assert request["artifacts"] == []
    assert request["read_paths"] == []
    assert persisted_result["succeeded"] is True
    assert persisted_result["artifacts"] == []
    assert persisted_result["read_paths"] == []
    assert not (run_dir / "input-manifest.json").exists()
    assert persisted_result["billing"] == {
        "route": "subscription",
        "credential_source": "chatgpt-oauth",
        "detected_by": "codex login status",
        "actual_cost_known": False,
    }
    assert persisted_result["api_equivalent_cost"]["pricing_version"].startswith(
        "models-dev-"
    )
    assert '"type": "thread.started"' in events
    assert (run_dir / "stderr.log").read_text() == ""


@pytest.mark.parametrize("sandbox", ["scratch-write", "workspace-write"])
def test_declared_read_paths_are_hashed_mounted_twice_and_recorded(
    repository: Path,
    fake_codex: Path,
    tmp_path: Path,
    sandbox: str,
) -> None:
    sources = tmp_path / "sources"
    transcripts = sources / "transcripts"
    transcripts.mkdir(parents=True)
    day_four = transcripts / "day-4.md"
    day_four.write_text("Day four\n")
    nested = transcripts / "nested"
    nested.mkdir()
    day_five = nested / "day-5.md"
    day_five.write_text("Day five\n")
    ledger = sources / "ledger.json"
    ledger.write_text('{"status":"verified"}\n')
    manager = WorktreeManager.discover(repository)

    result = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).run(
        task="reader",
        prompt="CHECK_READ_PATHS",
        sandbox=sandbox,
        read_paths=[transcripts, ledger],
    )

    assert result.succeeded
    assert [Path(item.mounted_path).name for item in result.read_paths] == [
        "transcripts",
        "ledger.json",
    ]
    transcript_input, ledger_input = result.read_paths
    assert transcript_input.source_path == os.fspath(transcripts.resolve())
    assert transcript_input.kind == "directory"
    assert transcript_input.size_bytes == len(b"Day four\nDay five\n")
    assert [item.relative_path for item in transcript_input.files] == [
        "day-4.md",
        "nested/day-5.md",
    ]
    assert transcript_input.files[0].sha256 == hashlib.sha256(b"Day four\n").hexdigest()
    assert ledger_input.kind == "file"
    assert ledger_input.sha256 == hashlib.sha256(ledger.read_bytes()).hexdigest()
    assert ledger_input.files[0].relative_path == "ledger.json"
    assert not (transcripts / "forbidden").exists()
    assert ledger.read_text() == '{"status":"verified"}\n'

    for item in result.read_paths:
        assert ["--ro-bind", item.source_path, item.source_path] in [
            result.command[index : index + 3]
            for index in range(len(result.command) - 2)
        ]
        assert ["--ro-bind", item.source_path, item.mounted_path] in [
            result.command[index : index + 3]
            for index in range(len(result.command) - 2)
        ]

    run_dir = manager.state_dir / "runs" / result.run_id
    request = json.loads((run_dir / "request.json").read_text())
    persisted_result = json.loads((run_dir / "result.json").read_text())
    manifest = json.loads((run_dir / "input-manifest.json").read_text())
    assert request["read_paths"] == persisted_result["read_paths"]
    assert manifest == {"schema_version": 1, "read_paths": request["read_paths"]}
    assert transcript_input.mounted_path in request["prompt"]
    assert transcript_input.source_path in request["prompt"]
    assert list(Path(transcript_input.mounted_path).iterdir()) == []
    assert Path(ledger_input.mounted_path).read_bytes() == b""


def test_resume_inherits_read_paths_and_refreshes_their_hashes(
    repository: Path,
    fake_codex: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.md"
    source.write_text("first\n")
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    first = runner.run(task="reader-resume", prompt="first", read_paths=[source])
    source.write_text("second\n")
    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert first.succeeded
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert resumed.read_paths[0].source_path == first.read_paths[0].source_path
    assert resumed.read_paths[0].sha256 == hashlib.sha256(b"second\n").hexdigest()
    assert resumed.read_paths[0].sha256 != first.read_paths[0].sha256
    assert resumed.read_paths[0].mounted_path != first.read_paths[0].mounted_path


def test_declared_read_paths_reject_unsafe_or_ambiguous_sources(
    repository: Path,
    fake_codex: Path,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "source.md"
    second = tmp_path / "second" / "source.md"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("one\n")
    second.write_text("two\n")
    linked = tmp_path / "linked.md"
    linked.symlink_to(first)
    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "linked.md").symlink_to(first)
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    with pytest.raises(AOPError, match="same basename"):
        runner.run(task="duplicate-read", prompt="unused", read_paths=[first, second])
    with pytest.raises(AOPError, match="may not be a symlink"):
        runner.run(task="linked-read", prompt="unused", read_paths=[linked])
    with pytest.raises(AOPError, match="contains a symlink"):
        runner.run(task="nested-linked-read", prompt="unused", read_paths=[directory])
    with pytest.raises(AOPError, match="requires workspace-write or scratch-write"):
        runner.run(
            task="danger-read",
            prompt="unused",
            sandbox="danger-full-access",
            read_paths=[first],
        )


@pytest.mark.parametrize(
    "sandbox", ["scratch-write", "workspace-write", "danger-full-access"]
)
def test_codex_uses_private_runtime_state_with_read_only_global_profile(
    repository: Path,
    fake_codex: Path,
    sandbox: str,
) -> None:
    source_home = Path(os.environ["AOP_CODEX_SOURCE_HOME"])
    for path in source_home.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source_home.chmod(0o555)
    task = f"managed-codex-{sandbox}"
    manager = WorktreeManager.discover(repository)

    try:
        runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))
        first = runner.run(task=task, prompt="first", sandbox=sandbox)
        resumed = runner.resume(run_id=first.run_id, prompt="second")
    finally:
        source_home.chmod(0o755)
        for path in source_home.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)

    private_home = manager.state_dir / "provider-state" / task / "codex" / "home"
    worktree = manager.get(task).path
    assert first.succeeded
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert private_home.joinpath("auth.json").is_file()
    assert private_home.joinpath("config.toml").is_file()
    assert private_home.joinpath("models_cache.json").is_file()
    assert private_home.joinpath("rules", "default.rules").is_file()
    assert private_home.joinpath("skills", "user-skill", "SKILL.md").is_file()
    assert not private_home.joinpath("skills", ".system", "generated").exists()
    assert private_home.joinpath("sessions", SESSION_ID).is_file()
    assert private_home.joinpath("history.jsonl").read_text() == "first\nsecond\n"
    assert source_home.joinpath("history.jsonl").read_text() == "global history\n"
    assert source_home.joinpath("state_5.sqlite").read_text() == "global database\n"
    assert source_home.joinpath("sessions", "global-session").is_file()
    assert not list(worktree.rglob("auth.json"))

    manager.remove(task)
    assert not private_home.exists()
    assert (manager.state_dir / "runs" / first.run_id / "result.json").is_file()
    assert (manager.state_dir / "runs" / resumed.run_id / "result.json").is_file()


def test_codex_records_metered_api_authentication_without_credentials(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOP_FAKE_CODEX_AUTH", "api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-persisted")
    result = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    ).run(task="codex-api-billing", prompt="test")

    assert result.succeeded
    assert result.billing.route == "metered-api"
    assert result.billing.credential_source == "openai-api-key"
    assert not result.billing.actual_cost_known
    persisted = repository / ".aop" / "runs" / result.run_id / "result.json"
    assert "must-not-be-persisted" not in persisted.read_text()


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


def test_workspace_sandbox_mounts_shared_git_metadata_read_only(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    result = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).run(
        task="read-only-git", prompt="make the change", timeout_seconds=5
    )
    worktree = manager.get("read-only-git").path
    common = Path(
        subprocess.run(
            (
                "git",
                "-C",
                os.fspath(worktree),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    index = result.command.index(os.fspath(common))
    assert result.command[index - 1] == "--ro-bind"
    assert result.command[index + 1] == os.fspath(common)
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


def test_codex_requires_a_terminal_completion_event(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    result = runner.run(task="incomplete", prompt="INCOMPLETE")

    assert not result.succeeded
    assert result.exit_code == 0
    assert result.session_id == SESSION_ID
    assert result.final_message == "answer:INCOMPLETE"
    assert result.error == "Codex did not emit a terminal turn.completed event"


def test_codex_rejects_a_changed_resume_thread(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))
    first = runner.run(task="codex-resume-identity", prompt="first")

    resumed = runner.resume(run_id=first.run_id, prompt="DIFFERENT_SESSION")

    assert not resumed.succeeded
    assert resumed.exit_code == 0
    assert resumed.session_id is None
    assert resumed.error == (
        "Codex resumed as thread different-codex-thread instead of the requested "
        f"thread {first.session_id}"
    )


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
