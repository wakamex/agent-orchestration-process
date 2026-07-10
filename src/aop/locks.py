"""Small filesystem locks for task and integration ownership."""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

from .worktrees import AOPError


@contextmanager
def exclusive_lock(
    path: Path, description: str, *, blocking: bool = False
) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        operation = fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(handle, operation)
        except BlockingIOError as error:
            raise AOPError(f"{description} is busy") from error
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def task_lock_path(state_dir: Path, task: str) -> Path:
    return state_dir / "locks" / f"task-{task}.lock"
