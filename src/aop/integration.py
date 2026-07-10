"""Explicit checkpoints and author-owned task integration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .locks import exclusive_lock, task_lock_path
from .models import RunResult
from .runner import AgentRunner
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
    author_run: RunResult
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
    """Ask the task's authoring session to rebase and fast-forward main."""

    def __init__(
        self,
        worktrees: WorktreeManager,
        runner: AgentRunner | None = None,
    ):
        self.worktrees = worktrees
        self.runner = runner or AgentRunner(worktrees)

    def integrate(
        self,
        task: str,
        *,
        remove_worktree: bool = False,
        timeout_seconds: float | None = None,
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
            result = self._integrate(task, timeout_seconds)
        if remove_worktree:
            self.worktrees.remove(task)
        return result

    def _integrate(self, task: str, timeout_seconds: float | None) -> IntegrationResult:
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
        if git(
            self.worktrees.root,
            "rev-list",
            "--merges",
            f"{metadata.base_commit}..{task_head}",
        ).stdout:
            raise AOPError(
                "task history contains merge commits; integration requires linear history"
            )

        author_run_id = _latest_resumable_run_id(self.worktrees.state_dir, task)
        prompt = _integration_prompt(
            task=task,
            root=self.worktrees.root,
            worktree=worktree.path,
            branch=branch.stdout.strip(),
            base_commit=metadata.base_commit,
            task_head=task_head,
            main_head=previous_head,
        )
        author_run = self.runner.resume(
            run_id=author_run_id,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            sandbox="danger-full-access",
            _task_lock_held=True,
        )

        _require_clean(self.worktrees.root, "main worktree after author integration")
        _require_clean(worktree.path, f"task worktree {task} after author integration")
        final_branch = git(
            self.worktrees.root,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            check=False,
        ).stdout.strip()
        if final_branch != branch.stdout.strip():
            raise AOPError(
                f"authoring agent changed main branch from {branch.stdout.strip()} "
                f"to {final_branch or '(detached)'}; run_id={author_run.run_id}"
            )
        integrated_head = git(self.worktrees.root, "rev-parse", "HEAD").stdout.strip()
        final_task_head = git(worktree.path, "rev-parse", "HEAD").stdout.strip()
        if integrated_head != final_task_head:
            raise AOPError(
                "authoring agent did not fast-forward main to the rebased task head; "
                f"run_id={author_run.run_id}"
            )
        if integrated_head == previous_head:
            raise AOPError(
                f"authoring agent left main unchanged; run_id={author_run.run_id}"
            )
        _require_ancestor(
            self.worktrees.root,
            previous_head,
            integrated_head,
            "authoring agent did not preserve main as an ancestor of the integrated result",
        )

        integrated_commits = _commits_between(
            self.worktrees.root, previous_head, integrated_head
        )
        self.worktrees.update_base(task, branch.stdout.strip(), integrated_head)
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
                "original_task_head": task_head,
                "previous_head": previous_head,
                "integrated_head": integrated_head,
                "source_commits": source_commits,
                "integrated_commits": integrated_commits,
                "author_run_id": author_run.run_id,
                "author_session_id": author_run.session_id,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        return IntegrationResult(
            task=task,
            previous_head=previous_head,
            integrated_head=integrated_head,
            source_commits=source_commits,
            integrated_commits=integrated_commits,
            author_run=author_run,
            record_path=record_path,
        )


def _integration_prompt(
    *,
    task: str,
    root: Path,
    worktree: Path,
    branch: str,
    base_commit: str,
    task_head: str,
    main_head: str,
) -> str:
    return f"""AOP integration assignment for task {task}.

You authored this task and are responsible for completing its integration. Work persistently until
the rebase, conflict resolution, relevant tests, and final fast-forward all succeed. You have
explicit permission to update the task's Git history and the main worktree for this assignment.

Main worktree: {root}
Main branch: {branch}
Current main commit: {main_head}
Task worktree: {worktree}
Recorded task base: {base_commit}
Original task head: {task_head}

In the task worktree, rebase exactly the task commits onto current main with:

    git rebase --onto {main_head} {base_commit} {task_head}

Resolve any conflicts yourself, continue the rebase, and run the relevant tests. Do not return
while a rebase or conflict is unfinished. Before promotion, verify that main is still clean and at
{main_head}. Then, from the main worktree, fast-forward {branch} to the rebased task HEAD with
`git merge --ff-only <rebased-task-head>`. Finish only after both worktrees are clean and both HEADs
name the same integrated commit. Do not force-update, reset, merge non-fast-forward, or discard
either side's changes.
"""


def _is_dirty(path: Path) -> bool:
    return bool(git(path, "status", "--porcelain").stdout)


def _require_clean(path: Path, description: str) -> None:
    if _is_dirty(path):
        raise AOPError(f"{description} must be clean")


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


def _run_artifacts(state_dir: Path, task: str) -> list[tuple[str, str, dict[str, Any]]]:
    runs: list[tuple[str, str, dict[str, Any]]] = []
    for request_path in (state_dir / "runs").glob("*/request.json"):
        try:
            request: dict[str, Any] = json.loads(request_path.read_text())
            result: dict[str, Any] = json.loads(
                (request_path.parent / "result.json").read_text()
            )
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if request.get("task") == task:
            runs.append(
                (str(request.get("created_at", "")), request_path.parent.name, result)
            )
    return sorted(runs)


def _source_run_ids(state_dir: Path, task: str) -> list[str]:
    return [
        run_id
        for _, run_id, result in _run_artifacts(state_dir, task)
        if result.get("succeeded") is True
    ]


def _latest_resumable_run_id(state_dir: Path, task: str) -> str:
    resumable = [
        run_id
        for _, run_id, result in _run_artifacts(state_dir, task)
        if isinstance(result.get("session_id"), str)
    ]
    if not resumable:
        raise AOPError(
            f"task has no authoring agent session to resume for integration: {task}"
        )
    return resumable[-1]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(f"{json.dumps(value, indent=2, sort_keys=True)}\n")
    os.replace(temporary, path)
