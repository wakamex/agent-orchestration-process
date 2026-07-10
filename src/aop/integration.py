"""Explicit checkpoints and sandboxed author-assisted task integration."""

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
    resolution_run_ids: list[str]
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
    """Rebase mechanically while the sandboxed author resolves content."""

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

        branch = _attached_branch(self.worktrees.root)
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
        rebase = git(
            worktree.path,
            "rebase",
            "--onto",
            previous_head,
            metadata.base_commit,
            task_head,
            check=False,
        )
        resolution_run_ids: list[str] = []
        while rebase.returncode:
            if not _rebase_in_progress(worktree.path):
                detail = rebase.stderr.strip() or rebase.stdout.strip()
                raise AOPError(f"could not start or continue task rebase: {detail}")

            conflicts = _unmerged_files(worktree.path)
            if not conflicts:
                detail = rebase.stderr.strip() or rebase.stdout.strip()
                raise AOPError(
                    f"task rebase stopped without conflicted files: {detail}"
                )
            resolution = self.runner.resume(
                run_id=author_run_id,
                prompt=_conflict_prompt(
                    task=task,
                    root=self.worktrees.root,
                    worktree=worktree.path,
                    branch=branch,
                    main_head=previous_head,
                    conflicts=conflicts,
                ),
                timeout_seconds=timeout_seconds,
                _task_lock_held=True,
            )
            resolution_run_ids.append(resolution.run_id)
            author_run_id = resolution.run_id
            if not resolution.succeeded:
                raise AOPError(
                    "authoring agent failed while resolving rebase conflicts; "
                    f"run_id={resolution.run_id}"
                )

            git(worktree.path, "diff", "--check")
            git(worktree.path, "add", "--all")
            if _unmerged_files(worktree.path):
                raise AOPError(
                    "authoring agent left unresolved conflict entries; "
                    f"run_id={resolution.run_id}"
                )
            git(worktree.path, "diff", "--cached", "--check")
            rebase = git(
                worktree.path,
                "-c",
                "core.editor=true",
                "rebase",
                "--continue",
                check=False,
            )

        rebased_head = git(worktree.path, "rev-parse", "HEAD").stdout.strip()
        _require_ancestor(
            self.worktrees.root,
            previous_head,
            rebased_head,
            "rebased task does not descend from current main",
        )
        self.worktrees.update_base(task, branch, previous_head)

        validation = self.runner.resume(
            run_id=author_run_id,
            prompt=_validation_prompt(
                task=task,
                root=self.worktrees.root,
                worktree=worktree.path,
                branch=branch,
                main_head=previous_head,
                rebased_head=rebased_head,
            ),
            timeout_seconds=timeout_seconds,
            _task_lock_held=True,
        )
        if not validation.succeeded:
            raise AOPError(
                "authoring agent failed while validating the rebased task; "
                f"run_id={validation.run_id}"
            )

        if _is_dirty(worktree.path):
            _require_no_unmerged_files(worktree.path)
            git(worktree.path, "diff", "--check")
            git(worktree.path, "add", "--all")
            git(worktree.path, "diff", "--cached", "--check")
            git(
                worktree.path,
                "commit",
                "-m",
                f"Address integration validation for {task}",
            )

        _require_clean(worktree.path, f"task worktree {task} after validation")
        _require_clean(self.worktrees.root, "main worktree before fast-forward")
        if _attached_branch(self.worktrees.root) != branch:
            raise AOPError(f"main worktree is no longer on expected branch {branch}")
        current_main = git(self.worktrees.root, "rev-parse", "HEAD").stdout.strip()
        if current_main != previous_head:
            raise AOPError(
                "main advanced during integration; retry against its new head"
            )

        final_task_head = git(worktree.path, "rev-parse", "HEAD").stdout.strip()
        git(self.worktrees.root, "merge", "--ff-only", final_task_head)
        integrated_head = git(self.worktrees.root, "rev-parse", "HEAD").stdout.strip()
        if integrated_head != final_task_head:
            raise AOPError("fast-forward did not leave main at the task head")
        _require_clean(self.worktrees.root, "main worktree after fast-forward")
        self.worktrees.update_base(task, branch, integrated_head)

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
                "branch": branch,
                "base_commit": metadata.base_commit,
                "original_task_head": task_head,
                "previous_head": previous_head,
                "integrated_head": integrated_head,
                "source_commits": source_commits,
                "integrated_commits": integrated_commits,
                "resolution_run_ids": resolution_run_ids,
                "validation_run_id": validation.run_id,
                "author_session_id": validation.session_id,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        return IntegrationResult(
            task=task,
            previous_head=previous_head,
            integrated_head=integrated_head,
            source_commits=source_commits,
            integrated_commits=integrated_commits,
            author_run=validation,
            resolution_run_ids=resolution_run_ids,
            record_path=record_path,
        )


def _conflict_prompt(
    *,
    task: str,
    root: Path,
    worktree: Path,
    branch: str,
    main_head: str,
    conflicts: list[str],
) -> str:
    listed = "\n".join(f"- {name}" for name in conflicts)
    return f"""AOP conflict resolution for task {task}.

AOP is rebasing your task onto {branch} at {main_head}. Resolve the content conflicts in your
isolated task worktree and run relevant tests where possible:

{listed}

Task worktree: {worktree}
Main worktree (read-only context): {root}

You are running with your original sandbox. Edit the conflicted files until they contain the final
intended result and contain no conflict markers. Do not run git add, git commit, git rebase, git
merge, or modify the main worktree; AOP owns those privileged Git operations. Do not finish until
the file conflicts are resolved.
"""


def _validation_prompt(
    *,
    task: str,
    root: Path,
    worktree: Path,
    branch: str,
    main_head: str,
    rebased_head: str,
) -> str:
    return f"""AOP rebased-task validation for task {task}.

AOP rebased your task onto {branch} at {main_head}. The rebased task head is {rebased_head}.
Inspect the result in {worktree}, run the relevant tests, and fix any content problems you find.
The main worktree at {root} is read-only context.

You are running with your original sandbox. You may edit task files, but do not run git add, git
commit, git rebase, git merge, or modify main; AOP will capture your final edits and perform the
fast-forward. Finish only when the rebased task is ready to integrate.
"""


def _is_dirty(path: Path) -> bool:
    return bool(git(path, "status", "--porcelain").stdout)


def _require_clean(path: Path, description: str) -> None:
    if _is_dirty(path):
        raise AOPError(f"{description} must be clean")


def _unmerged_files(path: Path) -> list[str]:
    return git(path, "diff", "--name-only", "--diff-filter=U").stdout.splitlines()


def _require_no_unmerged_files(path: Path) -> None:
    names = _unmerged_files(path)
    if names:
        raise AOPError(
            f"task worktree contains unresolved conflicts: {', '.join(names)}"
        )


def _rebase_in_progress(path: Path) -> bool:
    for name in ("rebase-merge", "rebase-apply"):
        value = git(path, "rev-parse", "--git-path", name).stdout.strip()
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = path / candidate
        if candidate.exists():
            return True
    return False


def _attached_branch(path: Path) -> str:
    branch = git(
        path, "symbolic-ref", "--quiet", "--short", "HEAD", check=False
    ).stdout.strip()
    if not branch:
        raise AOPError("main worktree must be on an attached branch before integration")
    return branch


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
