from __future__ import annotations

import os
from pathlib import Path

import pytest

from aop.runner import AgyAdapter, AgentRunner, ClaudeAdapter
from aop.worktrees import AOPError, WorktreeManager


def test_claude_run_and_exact_resume(repository: Path, fake_claude: Path) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, ClaudeAdapter(os.fspath(fake_claude)))

    first = runner.run(
        task="claude-task",
        prompt="first",
        model="opus",
        effort="high",
        timeout_seconds=5,
    )
    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert first.succeeded
    assert first.provider == "claude"
    assert first.model == "claude-test-model"
    assert first.usage.input_tokens == 60
    assert first.usage.cached_input_tokens == 20
    assert first.usage.output_tokens == 40
    assert first.api_equivalent_cost is not None
    assert first.api_equivalent_cost.amount_usd == 0.0123
    assert first.command[0] == "bwrap"
    assert "--dangerously-skip-permissions" in first.command
    assert ["--model", "opus"] == first.command[
        first.command.index("--model") : first.command.index("--model") + 2
    ]
    assert ["--effort", "high"] == first.command[
        first.command.index("--effort") : first.command.index("--effort") + 2
    ]
    assert resumed.succeeded
    assert ["--resume", first.session_id] == resumed.command[-2:]
    assert "--model" not in resumed.command


def test_agy_translates_model_effort_and_resumes_exact_conversation(
    repository: Path, fake_agy: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, AgyAdapter(os.fspath(fake_agy)))

    first = runner.run(
        task="agy-task",
        prompt="first",
        model="gemini-3.5-flash",
        effort="low",
        timeout_seconds=5,
    )
    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert first.succeeded
    assert first.provider == "agy"
    assert first.model == "gemini-3.5-flash-low"
    assert first.effort == "low"
    assert ["--model", "gemini-3.5-flash-low"] == first.command[
        first.command.index("--model") : first.command.index("--model") + 2
    ]
    assert first.usage.total_tokens == 0
    assert first.api_equivalent_cost is None
    assert resumed.succeeded
    assert ["--conversation", first.session_id] == resumed.command[
        resumed.command.index("--conversation") : resumed.command.index(
            "--conversation"
        )
        + 2
    ]
    assert "--model" not in resumed.command


def test_agy_rejects_an_unavailable_effort(repository: Path, fake_agy: Path) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), AgyAdapter(os.fspath(fake_agy))
    )

    with pytest.raises(AOPError, match="supports effort: low, high"):
        runner.run(
            task="bad-agy",
            prompt="test",
            model="gemini-3.1-pro",
            effort="medium",
        )


def test_agy_defaults_to_gemini_35_flash_medium(
    repository: Path, fake_agy: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), AgyAdapter(os.fspath(fake_agy))
    )

    result = runner.run(task="default-agy", prompt="test", timeout_seconds=5)

    assert result.succeeded
    assert result.model == "gemini-3.5-flash-medium"
    assert result.effort == "medium"


def test_agy_supports_the_gemini_36_flash_alias(
    repository: Path, fake_agy: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), AgyAdapter(os.fspath(fake_agy))
    )

    result = runner.run(
        task="agy-36",
        prompt="test",
        model="gemini-3.6-flash",
        effort="high",
    )

    assert result.succeeded
    assert result.model == "gemini-3.6-flash-high"
    assert result.effort == "high"


def test_agy_can_produce_a_declared_artifact(repository: Path, fake_agy: Path) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, AgyAdapter(os.fspath(fake_agy)))

    result = runner.run(
        task="agy-artifact",
        prompt="WRITE_ARTIFACT",
        sandbox="scratch-write",
        artifacts=["paper.md"],
    )

    assert result.succeeded
    artifact = result.artifacts[0]
    assert (
        manager.state_dir / "runs" / result.run_id / artifact.archive_path
    ).read_text() == "# Extracted by agy\n"


def test_claude_workspace_uses_bwrap_to_protect_main_and_git_metadata(
    repository: Path, fake_claude: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, ClaudeAdapter(os.fspath(fake_claude)))

    result = runner.run(task="sandboxed", prompt="CHECK_SANDBOX", timeout_seconds=5)
    worktree = manager.get("sandboxed")

    assert result.succeeded
    assert (worktree.path / "agent-write.txt").read_text() == "allowed"
    assert (manager.cache_dir / "provider-cache.txt").read_text() == "shared"
    assert not (repository / "main-write.txt").exists()
    assert (worktree.path / ".git").read_text().startswith("gitdir:")


def test_scratch_write_only_mounts_the_task_scratch_directory(
    repository: Path, fake_claude: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, ClaudeAdapter(os.fspath(fake_claude)))

    result = runner.run(
        task="proposal",
        prompt="CHECK_SCRATCH",
        sandbox="scratch-write",
        timeout_seconds=5,
    )
    worktree = manager.get("proposal")

    assert result.succeeded
    assert (worktree.path / "scratch" / "analysis.txt").read_text() == "allowed"
    assert not (worktree.path / "agent-write.txt").exists()
    assert not (repository / "main-write.txt").exists()
