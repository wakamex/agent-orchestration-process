from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aop.batch import BatchRunner, load_manifest
from aop.cli import main
from aop.worktrees import AOPError, WorktreeManager


def test_manifest_resolves_prompt_files_and_options(tmp_path: Path) -> None:
    (tmp_path / "prompt.md").write_text("Implement the parser.\n")
    manifest = tmp_path / "tasks.toml"
    manifest.write_text(
        """
[[tasks]]
id = "parser"
prompt_file = "prompt.md"
model = "test-model"
effort = "xhigh"
sandbox = "read-only"
timeout = 30

[[tasks]]
id = "tests"
prompt = "Add tests"
"""
    )

    tasks = load_manifest(manifest)

    assert [task.id for task in tasks] == ["parser", "tests"]
    assert tasks[0].prompt == "Implement the parser.\n"
    assert tasks[0].prompt_source == os.fspath(tmp_path / "prompt.md")
    assert tasks[0].model == "test-model"
    assert tasks[0].effort == "xhigh"
    assert tasks[0].sandbox == "read-only"
    assert tasks[0].timeout_seconds == 30
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
    assert {item.task for item in manager.list()} == {"alpha", "beta", "gamma", "delta"}
    assert sum(message.endswith("started") for message in messages) == 4

    summary_path = repository / ".aop" / "batches" / f"{result.batch_id}.json"
    summary = json.loads(summary_path.read_text())
    assert summary["succeeded"] is True
    assert len(summary["tasks"]) == 4
    assert summary["tasks"][0]["model"] == "gpt-5.6-sol"


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
