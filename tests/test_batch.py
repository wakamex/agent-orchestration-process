from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_orchestration_process.batch import BatchRunner, load_manifest
from agent_orchestration_process.cli import main
from agent_orchestration_process.worktrees import AOPError, WorktreeManager


def test_manifest_resolves_prompt_files_and_options(tmp_path: Path) -> None:
    (tmp_path / "prompt.md").write_text("Implement the parser.\n")
    (tmp_path / "sources").mkdir()
    (tmp_path / "ledger.md").write_text("verified\n")
    manifest = tmp_path / "tasks.toml"
    manifest.write_text(
        """
[[tasks]]
id = "parser"
agent = "agy"
prompt_file = "prompt.md"
model = "test-model"
mode = "agent"
effort = "xhigh"
profile = "edit"
timeout = 30
artifacts = ["paper.md"]
inputs = ["sources", "ledger.md"]

[[tasks]]
id = "tests"
prompt = "Add tests"
"""
    )

    tasks = load_manifest(manifest)

    assert [task.id for task in tasks] == ["parser", "tests"]
    assert tasks[0].prompt == "Implement the parser.\n"
    assert tasks[0].prompt_source == os.fspath(tmp_path / "prompt.md")
    assert tasks[0].agent == "agy"
    assert tasks[0].model == "test-model"
    assert tasks[0].mode == "agent"
    assert tasks[0].effort == "xhigh"
    assert tasks[0].profile == "edit"
    assert tasks[0].timeout_seconds == 30
    assert tasks[0].artifacts == ("paper.md",)
    assert tasks[0].input_paths == (
        os.fspath(tmp_path / "sources"),
        os.fspath(tmp_path / "ledger.md"),
    )
    assert tasks[1].prompt_source == "inline"


def test_batch_runs_four_tasks_and_persists_summary(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))
    manifest = repository / "tasks.toml"
    manifest.write_text(
        """
[[tasks]]
id = "alpha"
prompt = "one"
model = "gpt-5.6-sol"
effort = "high"

[[tasks]]
id = "beta"
prompt = "two"

[[tasks]]
id = "gamma"
prompt = "three"

[[tasks]]
id = "delta"
prompt = "four"
"""
    )
    messages: list[str] = []
    manager = WorktreeManager.discover(repository)

    result = BatchRunner(manager, jobs=4).run(manifest, messages.append)

    assert result.succeeded
    assert result.jobs == 4
    assert [task.task for task in result.tasks] == ["alpha", "beta", "gamma", "delta"]
    assert all(task.status == "succeeded" for task in result.tasks)
    assert all(task.run_id for task in result.tasks)
    assert result.tasks[0].input_tokens == 1
    assert result.tasks[0].output_tokens == 1
    assert result.tasks[0].api_equivalent_cost_usd == 0.000035
    assert result.tasks[0].billing_route == "subscription"
    assert {item.task for item in manager.list()} == {"alpha", "beta", "gamma", "delta"}
    assert sum(message.endswith("started") for message in messages) == 4

    summary_path = repository / ".aop" / "batches" / f"{result.batch_id}.json"
    summary = json.loads(summary_path.read_text())
    assert summary["succeeded"] is True
    assert len(summary["tasks"]) == 4
    assert summary["tasks"][0]["model"] == "gpt-5.6-sol"
    assert summary["tasks"][0]["billing_route"] == "subscription"


def test_batch_archives_declared_artifacts(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))
    manifest = repository / "artifacts.toml"
    manifest.write_text(
        """
[[tasks]]
id = "extract"
prompt = "WRITE_ARTIFACT"
profile = "review"
artifacts = ["paper.md"]
"""
    )
    manager = WorktreeManager.discover(repository)

    result = BatchRunner(manager).run(manifest)

    assert result.succeeded
    run_id = result.tasks[0].run_id
    assert run_id is not None
    run_result = json.loads(
        (manager.state_dir / "runs" / run_id / "result.json").read_text()
    )
    assert run_result["artifacts"][0]["logical_path"] == "paper.md"
    assert (
        manager.state_dir / "runs" / run_id / run_result["artifacts"][0]["archive_path"]
    ).read_text() == "# Extracted\n"


def test_batch_snapshots_declared_inputs(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))
    sources = repository.parent / "review-sources"
    transcripts = sources / "transcripts"
    transcripts.mkdir(parents=True)
    (transcripts / "day.md").write_text("transcript\n")
    ledger = sources / "ledger.md"
    ledger.write_text("ledger\n")
    manifest = repository / "read-paths.toml"
    manifest.write_text(
        f'''
[[tasks]]
id = "reader"
prompt = "CHECK_READ_PATHS"
profile = "review"
inputs = ["{transcripts}", "{ledger}"]
'''
    )
    manager = WorktreeManager.discover(repository)

    result = BatchRunner(manager).run(manifest)

    assert result.succeeded
    run_id = result.tasks[0].run_id
    assert run_id is not None
    request = json.loads(
        (manager.state_dir / "runs" / run_id / "request.json").read_text()
    )
    assert [Path(item["source_path"]).name for item in request["inputs"]] == [
        "transcripts",
        "ledger.md",
    ]


def test_cli_batch_returns_failure_and_keeps_other_results(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))
    manifest = repository / "tasks.toml"
    manifest.write_text(
        """
[[tasks]]
id = "passes"
prompt = "OK"

[[tasks]]
id = "fails"
prompt = "FAIL"
"""
    )

    exit_code = main(["batch", os.fspath(manifest), "--jobs", "2"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[passes] succeeded" in captured.err
    assert "[fails] failed" in captured.err
    assert "succeeded=1/2" in captured.err

    summaries = list((repository / ".aop" / "batches").glob("*.json"))
    summary = json.loads(summaries[0].read_text())
    assert [task["status"] for task in summary["tasks"]] == ["succeeded", "failed"]
    assert all(task["run_id"] for task in summary["tasks"])


def test_batch_can_mix_all_non_codex_providers(
    repository: Path,
    fake_claude: Path,
    fake_cursor: Path,
    fake_devin: Path,
    fake_opencode: Path,
    fake_agy: Path,
    fake_hermes: Path,
    fake_dsh: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOP_CLAUDE_BIN", os.fspath(fake_claude))
    monkeypatch.setenv("AOP_CURSOR_BIN", os.fspath(fake_cursor))
    monkeypatch.setenv("AOP_DEVIN_BIN", os.fspath(fake_devin))
    monkeypatch.setenv("AOP_OPENCODE_BIN", os.fspath(fake_opencode))
    monkeypatch.setenv("AOP_AGY_BIN", os.fspath(fake_agy))
    monkeypatch.setenv("AOP_HERMES_BIN", os.fspath(fake_hermes))
    monkeypatch.setenv("AOP_DSH_BIN", os.fspath(fake_dsh))
    manifest = repository / "providers.toml"
    manifest.write_text(
        """
[[tasks]]
id = "claude-task"
agent = "claude"
model = "sonnet"
effort = "high"
prompt = "one"

[[tasks]]
id = "cursor-task"
agent = "cursor"
prompt = "cursor"

[[tasks]]
id = "devin-task"
agent = "devin"
prompt = "devin"

[[tasks]]
id = "agy-task"
agent = "agy"
model = "gemini-3.1-pro"
effort = "low"
prompt = "two"

[[tasks]]
id = "opencode-task"
agent = "opencode"
model = "opencode/deepseek-v4-flash"
effort = "low"
prompt = "opencode"

[[tasks]]
id = "hermes-task"
agent = "hermes"
model = "deepseek/deepseek-v4-flash-0731"
provider = "nous"
effort = "high"
prompt = "three"

[[tasks]]
id = "dsh-task"
agent = "dsh"
model = "deepseek-v4-pro"
effort = "max"
prompt = "four"
"""
    )

    result = BatchRunner(WorktreeManager.discover(repository), jobs=2).run(manifest)

    assert result.succeeded
    assert [task.agent for task in result.tasks] == [
        "claude",
        "cursor",
        "devin",
        "agy",
        "opencode",
        "hermes",
        "dsh",
    ]
    assert result.tasks[0].model == "claude-test-model"
    assert result.tasks[1].model == "composer-2.5"
    assert result.tasks[2].model == "swe-1-7"
    assert result.tasks[3].model == "gemini-3.1-pro"
    assert result.tasks[4].model == "opencode/deepseek-v4-flash"
    assert result.tasks[5].model == "deepseek/deepseek-v4-flash-0731"
    assert result.tasks[5].inference_provider == "nous"
    assert result.tasks[6].model == "deepseek-v4-pro"


def test_batch_runs_hermes_participant_mode_and_records_provenance(
    repository: Path,
    fake_hermes: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOP_HERMES_BIN", os.fspath(fake_hermes))
    manifest = repository / "participant.toml"
    manifest.write_text(
        """
[[tasks]]
id = "player"
agent = "hermes"
mode = "participant"
prompt = "CHECK_PARTICIPANT batch"
profile = "review"
"""
    )
    manager = WorktreeManager.discover(repository)

    result = BatchRunner(manager).run(manifest)

    assert result.succeeded
    assert result.tasks[0].mode == "participant"
    run_id = result.tasks[0].run_id
    assert run_id is not None
    request = json.loads(
        (manager.state_dir / "runs" / run_id / "request.json").read_text()
    )
    persisted = json.loads(
        (manager.state_dir / "runs" / run_id / "result.json").read_text()
    )
    assert request["mode"] == "participant"
    assert persisted["mode"] == "participant"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "at least one"),
        (
            """
[[tasks]]
id = "same"
prompt = "one"
[[tasks]]
id = "same"
prompt = "two"
""",
            "duplicate",
        ),
        (
            """
[[tasks]]
id = "bad"
prompt = "one"
typo = true
""",
            "unknown field",
        ),
        (
            """
[[tasks]]
id = "bad"
prompt = "one"
timeout = 0
""",
            "greater than zero",
        ),
    ],
)
def test_invalid_manifests_fail_before_launch(
    tmp_path: Path, content: str, message: str
) -> None:
    manifest = tmp_path / "tasks.toml"
    manifest.write_text(content)

    with pytest.raises(AOPError, match=message):
        load_manifest(manifest)
