"""Creation and lifecycle management for isolated task worktrees."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Sequence


TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
IGNORE_ENTRY = "/.aop/"


def _user_runtime_dir() -> Path:
    return Path("/run/user") / str(os.getuid())


def _user_state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    return Path(base) / "aop"


def state_roots_file() -> Path:
    """List of every AOP state directory that has recorded a run."""
    return _user_state_dir() / "state-roots"


def results_archive_dir() -> Path:
    """A copy of every run result, kept when its state directory is deleted."""
    return _user_state_dir() / "results"


@contextmanager
def _locked_registry() -> Iterator[Path]:
    path = state_roots_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / "state-roots.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield path


def register_state_root(state_dir: Path) -> None:
    entry = os.fspath(state_dir.resolve())
    # The lock makes the check and the append one step for concurrent runs.
    with _locked_registry() as path:
        if entry in _registered_entries(path):
            return
        with path.open("a") as stream:
            stream.write(f"{entry}\n")


def compact_state_roots() -> tuple[int, int]:
    """Drop duplicate and deleted roots; return the entries kept and removed."""
    with _locked_registry() as path:
        try:
            lines = [line for line in path.read_text().splitlines() if line]
        except FileNotFoundError:
            return 0, 0
        kept = [line for line in dict.fromkeys(lines) if Path(line).is_dir()]
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text("".join(f"{line}\n" for line in kept))
        os.replace(temporary, path)
        return len(kept), len(lines) - len(kept)


def registered_state_roots() -> list[Path]:
    return [Path(line) for line in _registered_entries(state_roots_file())]


def _registered_entries(path: Path) -> list[str]:
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    return list(dict.fromkeys(line for line in lines if line))


class AOPError(RuntimeError):
    """A user-actionable orchestration error."""


@dataclass(frozen=True)
class Worktree:
    task: str
    path: Path
    head: str
    created_at: str | None = None


@dataclass(frozen=True)
class TaskMetadata:
    task: str
    path: str
    base_ref: str
    base_commit: str
    created_at: str


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
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

    @property
    def sealed_runtime_identity(self) -> str:
        return hashlib.sha256(os.fsencode(self.root)).hexdigest()[:24]

    @property
    def sealed_runtime_dir(self) -> Path:
        identity = self.sealed_runtime_identity
        configured = os.environ.get("AOP_SEALED_RUNTIME_ROOT")
        if configured:
            base = Path(configured)
            if not base.is_absolute():
                raise AOPError("AOP_SEALED_RUNTIME_ROOT must be absolute")
            base.mkdir(parents=True, exist_ok=True, mode=0o700)
            stat = base.stat()
            if stat.st_uid != os.getuid() or stat.st_mode & 0o077:
                raise AOPError(
                    "AOP_SEALED_RUNTIME_ROOT must be private and owned by the current user"
                )
            return base.resolve() / f"aop-sealed-{os.getuid()}" / identity
        user_runtime = _user_runtime_dir()
        base = (
            user_runtime
            if user_runtime.is_dir()
            and os.access(user_runtime, os.W_OK | os.X_OK)
            else Path("/tmp")
        )
        return base / f"aop-sealed-{os.getuid()}" / identity

    @classmethod
    def discover(cls, start: Path | None = None) -> WorktreeManager:
        cwd = (start or Path.cwd()).resolve()
        listing = git(cwd, "worktree", "list", "--porcelain").stdout
        configured_root = os.environ.get("AOP_STATE_ROOT")
        if configured_root:
            requested = Path(configured_root).resolve()
            worktree_paths = {
                Path(line.removeprefix("worktree ")).resolve()
                for line in listing.splitlines()
                if line.startswith("worktree ")
            }
            if requested not in worktree_paths:
                raise AOPError(
                    f"AOP_STATE_ROOT is not a registered Git worktree: {requested}"
                )
            return cls(requested)
        first = next(
            (line for line in listing.splitlines() if line.startswith("worktree ")),
            None,
        )
        if first is None:
            raise AOPError("could not locate the main Git worktree")
        return cls(Path(first.removeprefix("worktree ")))

    @classmethod
    def standalone(cls, root: Path | None = None) -> WorktreeManager:
        """Create a state manager for execution profiles that do not require Git."""
        manager = cls((root or Path.cwd()).resolve())
        manager._initialize_directories()
        return manager

    def initialize(self) -> None:
        self._initialize_directories()
        self._ensure_ignored()

    def _initialize_directories(self) -> None:
        for directory in (
            self.state_dir,
            self.worktrees_dir,
            self.cache_dir,
            self.state_dir / "tasks",
        ):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory.chmod(0o700)

    def create(self, task: str, base: str = "HEAD") -> Worktree:
        self._validate_task(task)
        self.initialize()
        path = self.worktrees_dir / task

        with self._lock():
            if path.exists() or any(item.task == task for item in self.list()):
                raise AOPError(f"task worktree already exists: {task}")

            head = git(
                self.root, "rev-parse", "--verify", f"{base}^{{commit}}"
            ).stdout.strip()
            git(self.root, "worktree", "add", "--detach", os.fspath(path), head)
            created_at = datetime.now(UTC).isoformat()
            self._write_metadata(
                TaskMetadata(
                    task=task,
                    path=os.fspath(path),
                    base_ref=base,
                    base_commit=head,
                    created_at=created_at,
                )
            )

        return Worktree(
            task=task,
            path=path,
            head=head,
            created_at=created_at,
        )

    def list(self) -> list[Worktree]:
        listing = git(self.root, "worktree", "list", "--porcelain").stdout
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

    def remove(self, task: str, *, force: bool = False) -> str | None:
        """Remove a task's worktree and private state.

        A worktree whose commit no branch or tag contains would lose it, so the
        commit is first kept at refs/aop-archive/<task>/<commit>, whose name is
        returned.
        """
        from .locks import exclusive_lock, task_lock_path

        with exclusive_lock(task_lock_path(self.state_dir, task), f"task {task}"):
            worktree = self.get(task)
            archived = None
            head = worktree.head
            reachable = git(
                self.root, "for-each-ref", "--count=1", "--contains", head,
                "refs/heads", "refs/tags", "refs/aop-archive",
            ).stdout.strip()
            if not reachable:
                archived = f"refs/aop-archive/{task}/{head[:12]}"
                git(self.root, "update-ref", archived, head)
            with self._lock():
                args = ["worktree", "remove"]
                if force:
                    args.append("--force")
                args.append(os.fspath(worktree.path))
                git(self.root, *args)
                metadata_path = self._metadata_path(task)
                if metadata_path.exists():
                    metadata_path.unlink()
                shutil.rmtree(self.state_dir / "overlays" / task, ignore_errors=True)
                shutil.rmtree(
                    self.state_dir / "provider-state" / task, ignore_errors=True
                )
                shutil.rmtree(self.state_dir / "scratch" / task, ignore_errors=True)
            return archived

    def metadata(self, task: str) -> TaskMetadata:
        self._validate_task(task)
        path = self._metadata_path(task)
        try:
            value = json.loads(path.read_text())
        except FileNotFoundError as error:
            raise AOPError(
                f"task metadata is missing for {task}; recreate the worktree with AOP"
            ) from error
        except json.JSONDecodeError as error:
            raise AOPError(f"invalid task metadata {path}: {error}") from error
        try:
            return TaskMetadata(**value)
        except TypeError as error:
            raise AOPError(f"invalid task metadata {path}: {error}") from error

    def update_base(self, task: str, base_ref: str, base_commit: str) -> TaskMetadata:
        metadata = self.metadata(task)
        resolved = git(
            self.root, "rev-parse", "--verify", f"{base_commit}^{{commit}}"
        ).stdout.strip()
        updated = TaskMetadata(
            task=metadata.task,
            path=metadata.path,
            base_ref=base_ref,
            base_commit=resolved,
            created_at=metadata.created_at,
        )
        self._write_metadata(updated)
        return updated

    def run(
        self,
        task: str,
        command: Sequence[str],
        *,
        overlays: Sequence[str] = (),
    ) -> int:
        from .locks import exclusive_lock, task_lock_path

        if not command:
            raise AOPError("a command is required after --")
        worktree = self.get(task)
        environment = os.environ.copy()
        environment.update(
            {
                "AOP_ROOT": os.fspath(self.root),
                "AOP_TASK": task,
                "AOP_TASK_LOCK_HELD": task,
                "AOP_WORKTREE": os.fspath(worktree.path),
                "AOP_CACHE_DIR": os.fspath(self.cache_dir),
            }
        )
        with exclusive_lock(task_lock_path(self.state_dir, task), f"task {task}"):
            with self._overlays(worktree, overlays):
                return subprocess.run(
                    command, cwd=worktree.path, env=environment, check=False
                ).returncode

    @contextmanager
    def _overlays(self, worktree: Worktree, paths: Sequence[str]) -> Iterator[None]:
        mounted: list[tuple[Path, bool]] = []
        try:
            for value in paths:
                relative = self._validate_overlay(value)
                lower = self.root / relative
                mountpoint = worktree.path / relative
                if not lower.is_dir():
                    raise AOPError(f"overlay source is not a directory: {relative}")
                if mountpoint.is_symlink():
                    raise AOPError(f"overlay target is a symlink: {relative}")
                if mountpoint.exists() and any(mountpoint.iterdir()):
                    raise AOPError(f"overlay target is not empty: {relative}")

                created = not mountpoint.exists()
                mountpoint.mkdir(parents=True, exist_ok=True)
                overlay_root = self.state_dir / "overlays" / worktree.task / relative
                upper = overlay_root / "upper"
                overlay_work = overlay_root / "work"
                upper.mkdir(parents=True, exist_ok=True)
                overlay_work.mkdir(parents=True, exist_ok=True)
                command = [
                    os.environ.get("AOP_FUSE_OVERLAYFS_BIN", "fuse-overlayfs"),
                    "-o",
                    f"lowerdir={lower},upperdir={upper},workdir={overlay_work}",
                    os.fspath(mountpoint),
                ]
                process = subprocess.run(command, text=True, capture_output=True)
                if process.returncode:
                    detail = process.stderr.strip() or "fuse-overlayfs failed"
                    raise AOPError(f"could not mount overlay {relative}: {detail}")
                mounted.append((mountpoint, created))
            yield
        finally:
            for mountpoint, created in reversed(mounted):
                subprocess.run(
                    [
                        os.environ.get("AOP_FUSERMOUNT_BIN", "fusermount3"),
                        "-u",
                        os.fspath(mountpoint),
                    ],
                    check=False,
                    capture_output=True,
                )
                if created:
                    try:
                        mountpoint.rmdir()
                    except OSError:
                        pass

    @staticmethod
    def _validate_overlay(value: str) -> Path:
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts or path == Path("."):
            raise AOPError(
                "overlay paths must be non-empty repository-relative directories"
            )
        return path

    def _ensure_ignored(self) -> None:
        ignore_file = self.root / ".gitignore"
        content = ignore_file.read_text() if ignore_file.exists() else ""
        if IGNORE_ENTRY in content.splitlines():
            return
        prefix = "" if not content or content.endswith("\n") else "\n"
        ignore_file.write_text(f"{content}{prefix}{IGNORE_ENTRY}\n")

    def _metadata_path(self, task: str) -> Path:
        return self.state_dir / "tasks" / f"{task}.json"

    def _write_metadata(self, metadata: TaskMetadata) -> None:
        path = self._metadata_path(metadata.task)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            f"{json.dumps(metadata.__dict__, indent=2, sort_keys=True)}\n"
        )
        os.replace(temporary, path)

    @staticmethod
    def _validate_task(task: str) -> None:
        if not TASK_ID.fullmatch(task):
            raise AOPError(
                "task id must be 1-64 characters using letters, numbers, '.', '_', or '-'"
            )

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_dir.chmod(0o700)
        with self.lock_file.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
