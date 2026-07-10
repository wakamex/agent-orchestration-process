from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest

from aop.integration import CheckpointManager, IntegrationManager
from aop.locks import task_lock_path
from aop.runner import AgentRunner, CodexAdapter
from aop.worktrees import AOPError, WorktreeManager

from conftest import git


def commit_aop_ignore(repository: Path) -> None:
    git(repository, "add", ".gitignore")
    git(repository, "commit", "-m", "Ignore AOP runtime state")


def author_runner(manager: WorktreeManager, fake_codex: Path, task: str) -> AgentRunner:
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))
    result = runner.run(task=task, prompt="Author the task", timeout_seconds=5)
    assert result.succeeded
    return runner


def test_checkpoint_and_integrate_linear_task(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("feature")
    runner = author_runner(manager, fake_codex, "feature")
    commit_aop_ignore(repository)
    (worktree.path / "feature.txt").write_text("done\n")

    checkpoint = CheckpointManager(manager).checkpoint("feature", "Add feature")
    result = IntegrationManager(manager, runner).integrate("feature")

    assert git(worktree.path, "status", "--porcelain") == ""
    assert checkpoint.commit == result.source_commits[0]
    assert len(result.integrated_commits) == 1
    assert (repository / "feature.txt").read_text() == "done\n"
    record = json.loads(result.record_path.read_text())
    assert record["task"] == "feature"
    assert record["source_commits"] == [checkpoint.commit]
    assert record["validation_run_id"] == result.author_run.run_id
    assert result.author_run.command[
        result.author_run.command.index("--sandbox") + 1
    ] == ("workspace-write")
    assert result.resolution_run_ids == []
    assert manager.metadata("feature").base_commit == result.integrated_head


def test_integration_can_remove_worktree(repository: Path, fake_codex: Path) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("remove-me")
    runner = author_runner(manager, fake_codex, "remove-me")
    commit_aop_ignore(repository)
    (worktree.path / "change.txt").write_text("change\n")
    CheckpointManager(manager).checkpoint("remove-me", "Make change")

    IntegrationManager(manager, runner).integrate("remove-me", remove_worktree=True)

    assert not worktree.path.exists()
    assert manager.list() == []


def test_integration_refuses_dirty_main_without_moving_head(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("feature")
    runner = author_runner(manager, fake_codex, "feature")
    commit_aop_ignore(repository)
    (worktree.path / "feature.txt").write_text("done\n")
    CheckpointManager(manager).checkpoint("feature", "Add feature")
    before = git(repository, "rev-parse", "HEAD")
    (repository / "local.txt").write_text("not committed\n")

    with pytest.raises(AOPError, match="main worktree must be clean"):
        IntegrationManager(manager, runner).integrate("feature")

    assert git(repository, "rev-parse", "HEAD") == before


def test_integration_refuses_dirty_task(repository: Path, fake_codex: Path) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("feature")
    runner = author_runner(manager, fake_codex, "feature")
    commit_aop_ignore(repository)
    (worktree.path / "feature.txt").write_text("done\n")
    CheckpointManager(manager).checkpoint("feature", "Add feature")
    (worktree.path / "later.txt").write_text("not committed\n")

    with pytest.raises(AOPError, match="task worktree feature must be clean"):
        IntegrationManager(manager, runner).integrate("feature")


def test_authoring_agent_resolves_rebase_conflict(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("conflict")
    runner = author_runner(manager, fake_codex, "conflict")
    commit_aop_ignore(repository)
    (worktree.path / "README.md").write_text("task version\n")
    CheckpointManager(manager).checkpoint("conflict", "Edit from task")

    (repository / "README.md").write_text("main version\n")
    git(repository, "add", "README.md")
    git(repository, "commit", "-m", "Edit on main")
    result = IntegrationManager(manager, runner).integrate("conflict")

    assert git(repository, "rev-parse", "HEAD") == result.integrated_head
    assert git(worktree.path, "rev-parse", "HEAD") == result.integrated_head
    assert git(repository, "status", "--porcelain") == ""
    assert (repository / "README.md").read_text() == "main version\ntask version\n"
    assert len(result.resolution_run_ids) == 1
    resolution_result = json.loads(
        (
            repository / ".aop" / "runs" / result.resolution_run_ids[0] / "result.json"
        ).read_text()
    )
    command = resolution_result["command"]
    assert command[command.index("--sandbox") + 1] == "workspace-write"


def test_checkpoint_refuses_an_active_task(repository: Path) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("busy")
    (worktree.path / "feature.txt").write_text("done\n")
    lock_path = task_lock_path(manager.state_dir, "busy")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(AOPError, match="task busy is busy"):
            CheckpointManager(manager).checkpoint("busy", "Add feature")
