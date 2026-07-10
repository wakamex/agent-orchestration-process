"""Explicit checkpoints and conservative task integration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .locks import exclusive_lock, task_lock_path
from .worktrees import AOPError, WorktreeManager, git


@dataclass(frozen=True)
class CheckpointResult:
    task: str
    commit: str
    record_path: Path


@dataclass(frozen=True)
class IntegrationResult:
    task: str
    previous_head: str
    integrated_head: str
    source_commits: list[str]
    integrated_commits: list[str]
    record_path: Path


class CheckpointManager:
    """Turn all current changes in one task worktree into a named commit."""

    def __init__(self, worktrees: WorktreeManager):
        self.worktrees = worktrees

    def checkpoint(self, task: str, message: str) -> CheckpointResult:
        if not message.strip():
            raise AOPError("checkpoint message cannot be empty")

        with exclusive_lock(
            task_lock_path(self.worktrees.state_dir, task),
            f"task {task}",
        ):
            worktree = self.worktrees.get(task)
            metadata = self.worktrees.metadata(task)
            _verify_metadata_path(worktree.path, metadata.path)
            _require_no_unmerged_files(worktree.path)
            if not _is_dirty(worktree.path):
                raise AOPError(f"task worktree has no changes to checkpoint: {task}")

            git(worktree.path, "var", "GIT_AUTHOR_IDENT")
            git(worktree.path, "diff", "--check")
            parent = git(worktree.path, "rev-parse", "HEAD").stdout.strip()
            staged = False
            try:
                git(worktree.path, "add", "--all")
                staged = True
                git(worktree.path, "diff", "--cached", "--check")
                git(worktree.path, "commit", "-m", message)
            except BaseException:
                if staged:
                    git(worktree.path, "restore", "--staged", ":/", check=False)
                raise

            commit = git(worktree.path, "rev-parse", "HEAD").stdout.strip()
            record = {
                "task": task,
                "base_commit": metadata.base_commit,
                "parent_commit": parent,
                "commit": commit,
                "message": message,
                "source_run_ids": _source_run_ids(self.worktrees.state_dir, task),
                "created_at": datetime.now(UTC).isoformat(),
            }
            record_path = (
                self.worktrees.state_dir / "checkpoints" / task / f"{commit}.json"
            )
            _write_json(record_path, record)
            return CheckpointResult(task=task, commit=commit, record_path=record_path)


class IntegrationManager:
    """Replay one clean task's commits onto a clean current main branch."""

    def __init__(self, worktrees: WorktreeManager):
        self.worktrees = worktrees

    def integrate(
        self, task: str, *, remove_worktree: bool = False
    ) -> IntegrationResult:
        with (
            exclusive_lock(
                self.worktrees.state_dir / "integration.lock",
                "integration",
            ),
            exclusive_lock(
                task_lock_path(self.worktrees.state_dir, task),
                f"task {task}",
            ),
        ):
            result = self._integrate(task)
        if remove_worktree:
            self.worktrees.remove(task)
        return result

    def _integrate(self, task: str) -> IntegrationResult:
        worktree = self.worktrees.get(task)
        metadata = self.worktrees.metadata(task)
        _verify_metadata_path(worktree.path, metadata.path)
        _require_clean(self.worktrees.root, "main worktree")
        _require_clean(worktree.path, f"task worktree {task}")

        branch = git(
            self.worktrees.root,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            check=False,
        )
        if branch.returncode:
            raise AOPError(
                "main worktree must be on an attached branch before integration"
            )

        previous_head = git(self.worktrees.root, "rev-parse", "HEAD").stdout.strip()
        task_head = git(worktree.path, "rev-parse", "HEAD").stdout.strip()
        _require_ancestor(
            self.worktrees.root,
            metadata.base_commit,
            task_head,
            "task history no longer descends from its recorded base",
        )
        _require_ancestor(
            self.worktrees.root,
            metadata.base_commit,
            previous_head,
            "current branch no longer descends from the task's recorded base",
        )

        source_commits = _commits_between(
            self.worktrees.root, metadata.base_commit, task_head
        )
        if not source_commits:
            raise AOPError(f"task has no commits to integrate: {task}")
        merge_commits = git(
            self.worktrees.root,
            "rev-list",
            "--merges",
            f"{metadata.base_commit}..{task_head}",
        ).stdout.splitlines()
        if merge_commits:
            raise AOPError(
                "task history contains merge commits; integration requires linear history"
            )

        preflight = git(
            self.worktrees.root,
            "merge-tree",
            "--write-tree",
            previous_head,
            task_head,
            check=False,
        )
        if preflight.returncode:
            detail = preflight.stdout.strip() or preflight.stderr.strip()
            suffix = f": {detail}" if detail else ""
            raise AOPError(f"task conflicts with the current branch{suffix}")

        cherry_pick = git(
            self.worktrees.root, "cherry-pick", *source_commits, check=False
        )
        if cherry_pick.returncode:
            failure = cherry_pick.stderr.strip() or cherry_pick.stdout.strip()
            aborted = git(self.worktrees.root, "cherry-pick", "--abort", check=False)
            current_head = git(self.worktrees.root, "rev-parse", "HEAD").stdout.strip()
            if aborted.returncode or current_head != previous_head:
                raise AOPError(
                    "integration failed and automatic cherry-pick rollback was incomplete; "
                    f"inspect the repository immediately: {failure}"
                )
            raise AOPError(f"integration failed and was rolled back: {failure}")

        integrated_head = git(self.worktrees.root, "rev-parse", "HEAD").stdout.strip()
        integrated_commits = _commits_between(
            self.worktrees.root, previous_head, integrated_head
        )
        record_id = (
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex[:8]}"
        )
        record_path = self.worktrees.state_dir / "integrations" / f"{record_id}.json"
        _write_json(
            record_path,
            {
                "task": task,
                "branch": branch.stdout.strip(),
                "base_commit": metadata.base_commit,
                "task_head": task_head,
                "previous_head": previous_head,
                "integrated_head": integrated_head,
                "source_commits": source_commits,
                "integrated_commits": integrated_commits,
                "source_run_ids": _source_run_ids(self.worktrees.state_dir, task),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        return IntegrationResult(
            task=task,
            previous_head=previous_head,
            integrated_head=integrated_head,
            source_commits=source_commits,
            integrated_commits=integrated_commits,
            record_path=record_path,
        )


def _is_dirty(path: Path) -> bool:
    return bool(git(path, "status", "--porcelain").stdout)


def _require_clean(path: Path, description: str) -> None:
    if _is_dirty(path):
        raise AOPError(f"{description} must be clean before integration")


def _require_no_unmerged_files(path: Path) -> None:
    names = git(path, "diff", "--name-only", "--diff-filter=U").stdout.strip()
    if names:
        raise AOPError(f"task worktree contains unresolved conflicts: {names}")


def _require_ancestor(path: Path, ancestor: str, descendant: str, message: str) -> None:
    result = git(path, "merge-base", "--is-ancestor", ancestor, descendant, check=False)
    if result.returncode:
        raise AOPError(message)


def _commits_between(path: Path, base: str, head: str) -> list[str]:
    return git(path, "rev-list", "--reverse", f"{base}..{head}").stdout.splitlines()


def _verify_metadata_path(actual: Path, recorded: str) -> None:
    if actual.resolve() != Path(recorded).resolve():
        raise AOPError("task metadata path does not match its registered worktree")


def _source_run_ids(state_dir: Path, task: str) -> list[str]:
    runs: list[tuple[str, str]] = []
    for request_path in (state_dir / "runs").glob("*/request.json"):
        try:
            request: dict[str, Any] = json.loads(request_path.read_text())
            result: dict[str, Any] = json.loads(
                (request_path.parent / "result.json").read_text()
            )
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if request.get("task") == task and result.get("succeeded") is True:
            runs.append((str(request.get("created_at", "")), request_path.parent.name))
    return [run_id for _, run_id in sorted(runs)]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(f"{json.dumps(value, indent=2, sort_keys=True)}\n")
    os.replace(temporary, path)
