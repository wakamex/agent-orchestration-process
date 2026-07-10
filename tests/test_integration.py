from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

from aop.integration import CheckpointManager, IntegrationManager
from aop.locks import task_lock_path
from aop.worktrees import AOPError, WorktreeManager

from conftest import git


def commit_aop_ignore(repository: Path) -> None:
    git(repository, "add", ".gitignore")
    git(repository, "commit", "-m", "Ignore AOP runtime state")


def test_checkpoint_and_integrate_linear_task(repository: Path) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("feature")
    commit_aop_ignore(repository)
    (worktree.path / "feature.txt").write_text("done\n")

    checkpoint = CheckpointManager(manager).checkpoint("feature", "Add feature")
    result = IntegrationManager(manager).integrate("feature")

    assert git(worktree.path, "status", "--porcelain") == ""
    assert checkpoint.commit == result.source_commits[0]
    assert len(result.integrated_commits) == 1
    assert (repository / "feature.txt").read_text() == "done\n"
    record = json.loads(result.record_path.read_text())
    assert record["task"] == "feature"
    assert record["source_commits"] == [checkpoint.commit]


def test_integration_can_remove_worktree(repository: Path) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("remove-me")
    commit_aop_ignore(repository)
    (worktree.path / "change.txt").write_text("change\n")
    CheckpointManager(manager).checkpoint("remove-me", "Make change")

    IntegrationManager(manager).integrate("remove-me", remove_worktree=True)

    assert not worktree.path.exists()
    assert manager.list() == []


def test_integration_refuses_dirty_main_without_moving_head(repository: Path) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("feature")
    commit_aop_ignore(repository)
    (worktree.path / "feature.txt").write_text("done\n")
    CheckpointManager(manager).checkpoint("feature", "Add feature")
    before = git(repository, "rev-parse", "HEAD")
    (repository / "local.txt").write_text("not committed\n")

    with pytest.raises(AOPError, match="main worktree must be clean"):
        IntegrationManager(manager).integrate("feature")

    assert git(repository, "rev-parse", "HEAD") == before


def test_integration_refuses_dirty_task(repository: Path) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("feature")
    commit_aop_ignore(repository)
    (worktree.path / "feature.txt").write_text("done\n")
    CheckpointManager(manager).checkpoint("feature", "Add feature")
    (worktree.path / "later.txt").write_text("not committed\n")

    with pytest.raises(AOPError, match="task worktree feature must be clean"):
        IntegrationManager(manager).integrate("feature")


def test_conflict_preflight_leaves_main_unchanged(repository: Path) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("conflict")
    commit_aop_ignore(repository)
    (worktree.path / "README.md").write_text("task version\n")
    CheckpointManager(manager).checkpoint("conflict", "Edit from task")

    (repository / "README.md").write_text("main version\n")
    git(repository, "add", "README.md")
    git(repository, "commit", "-m", "Edit on main")
    before = git(repository, "rev-parse", "HEAD")

    with pytest.raises(AOPError, match="conflicts with the current branch"):
        IntegrationManager(manager).integrate("conflict")

    assert git(repository, "rev-parse", "HEAD") == before
    assert git(repository, "status", "--porcelain") == ""


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
