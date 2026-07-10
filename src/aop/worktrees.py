"""Creation and lifecycle management for isolated task worktrees."""

from __future__ import annotations

import fcntl
import os
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Sequence


TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
IGNORE_ENTRY = "/.aop/"


class AOPError(RuntimeError):
    """A user-actionable orchestration error."""


@dataclass(frozen=True)
class Worktree:
    task: str
    path: Path
    head: str
    created_at: str | None = None


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        ["git", "-C", os.fspath(cwd), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and process.returncode:
        detail = (
            process.stderr.strip() or process.stdout.strip() or "git command failed"
        )
        raise AOPError(detail)
    return process


class WorktreeManager:
    """Manage AOP-owned Git worktrees beneath the main worktree."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.state_dir = self.root / ".aop"
        self.worktrees_dir = self.state_dir / "worktrees"
        self.cache_dir = self.state_dir / "cache"
        self.lock_file = self.state_dir / "worktrees.lock"

    @classmethod
    def discover(cls, start: Path | None = None) -> WorktreeManager:
        cwd = (start or Path.cwd()).resolve()
        listing = _git(cwd, "worktree", "list", "--porcelain").stdout
        first = next(
            (line for line in listing.splitlines() if line.startswith("worktree ")),
            None,
        )
        if first is None:
            raise AOPError("could not locate the main Git worktree")
        return cls(Path(first.removeprefix("worktree ")))

    def initialize(self) -> None:
        self.worktrees_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_ignored()

    def create(self, task: str, base: str = "HEAD") -> Worktree:
        self._validate_task(task)
        self.initialize()
        path = self.worktrees_dir / task

        with self._lock():
            if path.exists() or any(item.task == task for item in self.list()):
                raise AOPError(f"task worktree already exists: {task}")

            head = _git(
                self.root, "rev-parse", "--verify", f"{base}^{{commit}}"
            ).stdout.strip()
            _git(self.root, "worktree", "add", "--detach", os.fspath(path), head)

        return Worktree(
            task=task,
            path=path,
            head=head,
            created_at=datetime.now(UTC).isoformat(),
        )

    def list(self) -> list[Worktree]:
        listing = _git(self.root, "worktree", "list", "--porcelain").stdout
        worktrees: list[Worktree] = []

        for block in listing.strip().split("\n\n"):
            fields: dict[str, str] = {}
            for line in block.splitlines():
                key, _, value = line.partition(" ")
                fields[key] = value
            if "worktree" not in fields:
                continue

            path = Path(fields["worktree"]).resolve()
            try:
                relative = path.relative_to(self.worktrees_dir)
            except ValueError:
                continue
            if len(relative.parts) != 1:
                continue
            worktrees.append(
                Worktree(task=relative.name, path=path, head=fields.get("HEAD", ""))
            )

        return sorted(worktrees, key=lambda item: item.task)

    def get(self, task: str) -> Worktree:
        self._validate_task(task)
        for worktree in self.list():
            if worktree.task == task:
                return worktree
        raise AOPError(f"unknown task worktree: {task}")

    def remove(self, task: str, *, force: bool = False) -> None:
        worktree = self.get(task)
        with self._lock():
            args = ["worktree", "remove"]
            if force:
                args.append("--force")
            args.append(os.fspath(worktree.path))
            _git(self.root, *args)

    def run(self, task: str, command: Sequence[str]) -> int:
        if not command:
            raise AOPError("a command is required after --")
        worktree = self.get(task)
        environment = os.environ.copy()
        environment.update(
            {
                "AOP_ROOT": os.fspath(self.root),
                "AOP_TASK": task,
                "AOP_WORKTREE": os.fspath(worktree.path),
                "AOP_CACHE_DIR": os.fspath(self.cache_dir),
            }
        )
        return subprocess.run(
            command, cwd=worktree.path, env=environment, check=False
        ).returncode

    def _ensure_ignored(self) -> None:
        ignore_file = self.root / ".gitignore"
        content = ignore_file.read_text() if ignore_file.exists() else ""
        if IGNORE_ENTRY in content.splitlines():
            return
        prefix = "" if not content or content.endswith("\n") else "\n"
        ignore_file.write_text(f"{content}{prefix}{IGNORE_ENTRY}\n")

    @staticmethod
    def _validate_task(task: str) -> None:
        if not TASK_ID.fullmatch(task):
            raise AOPError(
                "task id must be 1-64 characters using letters, numbers, '.', '_', or '-'"
            )

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
