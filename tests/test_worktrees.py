from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agent_orchestration_process import worktrees
from agent_orchestration_process.worktrees import AOPError, WorktreeManager


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", os.fspath(cwd), *args],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def test_worktree_lifecycle(repository: Path) -> None:
    manager = WorktreeManager.discover(repository)
    manager.state_dir.mkdir(mode=0o755)
    manager.initialize()
    manager.initialize()

    assert manager.state_dir.stat().st_mode & 0o777 == 0o700
    assert manager.worktrees_dir.stat().st_mode & 0o777 == 0o700
    assert manager.cache_dir.stat().st_mode & 0o777 == 0o700
    assert (manager.state_dir / "tasks").stat().st_mode & 0o777 == 0o700

    created = manager.create("task-one")

    assert created.path == repository / ".aop" / "worktrees" / "task-one"
    assert git(created.path, "rev-parse", "HEAD") == git(
        repository, "rev-parse", "HEAD"
    )
    assert manager.get("task-one").path == created.path
    metadata = manager.metadata("task-one")
    assert metadata.base_commit == created.head
    assert metadata.path == os.fspath(created.path)
    assert [item.task for item in manager.list()] == ["task-one"]
    assert (repository / ".gitignore").read_text().count("/.aop/") == 1
    provider_state = repository / ".aop" / "provider-state" / "task-one"
    provider_state.mkdir(parents=True)
    (provider_state / "state.db").write_text("task state\n")

    manager.remove("task-one")
    assert manager.list() == []
    assert not created.path.exists()
    assert not (repository / ".aop" / "tasks" / "task-one.json").exists()
    assert not provider_state.exists()


def test_explicit_state_root_anchors_aop_to_a_linked_worktree(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    research = tmp_path / "research"
    git(repository, "worktree", "add", "-b", "research", os.fspath(research), "HEAD")
    monkeypatch.setenv("AOP_STATE_ROOT", os.fspath(research))

    manager = WorktreeManager.discover(research)
    created = manager.create("experiment-one")

    assert manager.root == research
    assert created.path == research / ".aop" / "worktrees" / "experiment-one"
    assert "/.aop/" in (research / ".gitignore").read_text()
    primary_ignore = repository / ".gitignore"
    assert not primary_ignore.exists() or "/.aop/" not in primary_ignore.read_text()
    manager.remove("experiment-one")


def test_dirty_worktree_requires_force(repository: Path) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("dirty-task")
    (worktree.path / "changed.txt").write_text("uncommitted\n")

    with pytest.raises(AOPError, match="contains modified or untracked files"):
        manager.remove("dirty-task")

    manager.remove("dirty-task", force=True)
    assert not worktree.path.exists()


def test_exec_exposes_task_environment(repository: Path) -> None:
    manager = WorktreeManager.discover(repository)
    worktree = manager.create("exec-task")

    result = manager.run(
        "exec-task",
        [
            "python3",
            "-c",
            (
                "import os, pathlib; "
                "pathlib.Path('environment.txt').write_text("
                "os.environ['AOP_TASK'] + '\\n' + os.environ['AOP_CACHE_DIR'] + "
                "'\\n' + os.environ['AOP_TASK_LOCK_HELD'])"
            ),
        ],
    )

    assert result == 0
    lines = (worktree.path / "environment.txt").read_text().splitlines()
    assert lines == [
        "exec-task",
        os.fspath(repository / ".aop" / "cache"),
        "exec-task",
    ]

    manager.remove("exec-task", force=True)


def test_sealed_runtime_falls_back_when_user_runtime_is_read_only(
    repository: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_runtime = tmp_path / "user-runtime"
    user_runtime.mkdir()
    monkeypatch.setattr(worktrees, "_user_runtime_dir", lambda: user_runtime)
    monkeypatch.setattr(worktrees.os, "access", lambda path, mode: False)

    runtime = WorktreeManager(repository).sealed_runtime_dir

    assert runtime.parent == Path("/tmp") / f"aop-sealed-{os.getuid()}"


def test_exec_overlay_is_private_and_persists_until_task_removal(
    repository: Path,
) -> None:
    manager = WorktreeManager.discover(repository)
    (repository / "build").mkdir()
    (repository / "build" / "shared.txt").write_text("shared\n")
    manager.create("overlay-task")

    result = manager.run(
        "overlay-task",
        [
            "python3",
            "-c",
            (
                "from pathlib import Path; "
                "assert Path('build/shared.txt').read_text() == 'shared\\n'; "
                "Path('build/private.txt').write_text('private\\n')"
            ),
        ],
        overlays=["build"],
    )

    assert result == 0
    assert not (repository / "build" / "private.txt").exists()
    upper = repository / ".aop" / "overlays" / "overlay-task" / "build" / "upper"
    assert (upper / "private.txt").read_text() == "private\n"

    manager.remove("overlay-task")
    assert not upper.exists()


@pytest.mark.parametrize("path", ["", ".", "../build", "/tmp/build"])
def test_exec_rejects_unsafe_overlay_paths(repository: Path, path: str) -> None:
    manager = WorktreeManager.discover(repository)
    manager.create("unsafe-overlay")

    with pytest.raises(AOPError, match="repository-relative"):
        manager.run("unsafe-overlay", ["true"], overlays=[path])


def test_parallel_worktree_creation_is_serialized(repository: Path) -> None:
    tasks = ["parallel-a", "parallel-b", "parallel-c", "parallel-d"]

    def create(task: str) -> Path:
        return WorktreeManager.discover(repository).create(task).path

    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(create, tasks))

    manager = WorktreeManager.discover(repository)
    assert [item.task for item in manager.list()] == tasks
    assert all(path.exists() for path in paths)

    for task in tasks:
        manager.remove(task)


@pytest.mark.parametrize("task", ["", "../escape", "has/slash", "has space", "x" * 65])
def test_invalid_task_ids_are_rejected(repository: Path, task: str) -> None:
    manager = WorktreeManager.discover(repository)

    with pytest.raises(AOPError, match="task id"):
        manager.create(task)
