"""Bounded parallel execution of task manifests."""

from __future__ import annotations

import json
import os
import time
import tomllib
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from .runner import AgentRunner
from .worktrees import AOPError, TASK_ID, WorktreeManager


EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}
SANDBOXES = {"read-only", "workspace-write", "danger-full-access"}
TASK_FIELDS = {
    "id",
    "agent",
    "prompt",
    "prompt_file",
    "base",
    "model",
    "effort",
    "sandbox",
    "timeout",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class BatchTask:
    id: str
    prompt: str
    prompt_source: str
    base: str = "HEAD"
    model: str | None = None
    effort: str | None = None
    sandbox: str = "workspace-write"
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class BatchTaskResult:
    task: str
    status: str
    run_id: str | None
    session_id: str | None
    duration_seconds: float | None
    exit_code: int | None
    error: str | None


@dataclass(frozen=True)
class BatchResult:
    batch_id: str
    manifest: str
    jobs: int
    started_at: str
    finished_at: str
    duration_seconds: float
    interrupted: bool
    tasks: list[BatchTaskResult]

    @property
    def succeeded(self) -> bool:
        return not self.interrupted and all(
            task.status == "succeeded" for task in self.tasks
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "succeeded": self.succeeded,
        }


class BatchRunner:
    def __init__(self, manager: WorktreeManager, jobs: int = 4):
        if jobs < 1:
            raise AOPError("jobs must be greater than zero")
        self.manager = manager
        self.jobs = jobs

    def run(
        self,
        manifest_path: Path,
        progress: Callable[[str], None] | None = None,
    ) -> BatchResult:
        report = progress or (lambda _message: None)
        manifest = manifest_path.resolve()
        tasks = load_manifest(manifest)
        batch_id = str(uuid.uuid4())
        started_at = _now()
        started = time.monotonic()
        outcomes: dict[str, BatchTaskResult] = {}
        active: dict[Future[BatchTaskResult], BatchTask] = {}
        next_task = 0
        interrupted = False

        executor = ThreadPoolExecutor(max_workers=self.jobs, thread_name_prefix="aop")

        def submit_available() -> None:
            nonlocal next_task
            while len(active) < self.jobs and next_task < len(tasks):
                task = tasks[next_task]
                next_task += 1
                report(f"[{task.id}] started")
                active[executor.submit(self._execute_task, task)] = task

        def collect(future: Future[BatchTaskResult], task: BatchTask) -> None:
            outcome = future.result()
            outcomes[task.id] = outcome
            detail = f" run_id={outcome.run_id}" if outcome.run_id else ""
            report(f"[{task.id}] {outcome.status}{detail}")

        try:
            submit_available()
            while active:
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    task = active.pop(future)
                    collect(future, task)
                submit_available()
        except KeyboardInterrupt:
            interrupted = True
            report(
                "batch interrupted; waiting for active tasks and launching no new work"
            )
            for future, task in list(active.items()):
                collect(future, task)
            active.clear()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        for task in tasks:
            if task.id not in outcomes:
                outcomes[task.id] = BatchTaskResult(
                    task=task.id,
                    status="not_started",
                    run_id=None,
                    session_id=None,
                    duration_seconds=None,
                    exit_code=None,
                    error="batch interrupted before launch",
                )

        result = BatchResult(
            batch_id=batch_id,
            manifest=os.fspath(manifest),
            jobs=self.jobs,
            started_at=started_at,
            finished_at=_now(),
            duration_seconds=round(time.monotonic() - started, 6),
            interrupted=interrupted,
            tasks=[outcomes[task.id] for task in tasks],
        )
        self._write_result(result)
        return result

    def _execute_task(self, task: BatchTask) -> BatchTaskResult:
        try:
            result = AgentRunner(self.manager).run(
                task=task.id,
                prompt=task.prompt,
                base=task.base,
                model=task.model,
                effort=task.effort,
                sandbox=task.sandbox,
                timeout_seconds=task.timeout_seconds,
            )
        except Exception as error:
            return BatchTaskResult(
                task=task.id,
                status="error",
                run_id=None,
                session_id=None,
                duration_seconds=None,
                exit_code=None,
                error=str(error),
            )
        return BatchTaskResult(
            task=task.id,
            status="succeeded" if result.succeeded else "failed",
            run_id=result.run_id,
            session_id=result.session_id,
            duration_seconds=result.duration_seconds,
            exit_code=result.exit_code,
            error=result.error,
        )

    def _write_result(self, result: BatchResult) -> None:
        directory = self.manager.state_dir / "batches"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{result.batch_id}.json"
        temporary = directory / f".{result.batch_id}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(
            f"{json.dumps(result.to_dict(), indent=2, sort_keys=True)}\n"
        )
        os.replace(temporary, destination)


def load_manifest(path: Path) -> list[BatchTask]:
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError as error:
        raise AOPError(f"could not read batch manifest {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise AOPError(f"invalid batch manifest {path}: {error}") from error

    unknown_top_level = set(document) - {"tasks"}
    if unknown_top_level:
        names = ", ".join(sorted(unknown_top_level))
        raise AOPError(f"unknown batch manifest field(s): {names}")
    raw_tasks = document.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise AOPError("batch manifest must contain at least one [[tasks]] entry")

    tasks = [
        _parse_task(value, index, path.parent) for index, value in enumerate(raw_tasks)
    ]
    identifiers = [task.id for task in tasks]
    duplicates = sorted({task for task in identifiers if identifiers.count(task) > 1})
    if duplicates:
        raise AOPError(f"duplicate batch task id(s): {', '.join(duplicates)}")
    return tasks


def _parse_task(value: object, index: int, manifest_dir: Path) -> BatchTask:
    label = f"tasks[{index}]"
    if not isinstance(value, dict):
        raise AOPError(f"{label} must be a table")
    unknown = set(value) - TASK_FIELDS
    if unknown:
        raise AOPError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}")

    task_id = _required_string(value, "id", label)
    if not TASK_ID.fullmatch(task_id):
        raise AOPError(f"{label}.id is not a valid task id: {task_id}")
    agent = value.get("agent", "codex")
    if agent != "codex":
        raise AOPError(f"{label}.agent must be 'codex'")

    prompt = value.get("prompt")
    prompt_file = value.get("prompt_file")
    if (prompt is None) == (prompt_file is None):
        raise AOPError(f"{label} must define exactly one of prompt or prompt_file")
    if prompt is not None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise AOPError(f"{label}.prompt must be a non-empty string")
        prompt_text = prompt
        prompt_source = "inline"
    else:
        if not isinstance(prompt_file, str) or not prompt_file:
            raise AOPError(f"{label}.prompt_file must be a non-empty string")
        prompt_path = (manifest_dir / prompt_file).resolve()
        try:
            prompt_text = prompt_path.read_text()
        except OSError as error:
            raise AOPError(
                f"could not read {label}.prompt_file {prompt_path}: {error}"
            ) from error
        if not prompt_text.strip():
            raise AOPError(f"{label}.prompt_file is empty: {prompt_path}")
        prompt_source = os.fspath(prompt_path)

    base = _optional_string(value, "base", label) or "HEAD"
    model = _optional_string(value, "model", label)
    effort = _optional_string(value, "effort", label)
    if effort is not None and effort not in EFFORTS:
        raise AOPError(f"{label}.effort must be one of: {', '.join(sorted(EFFORTS))}")
    sandbox = _optional_string(value, "sandbox", label) or "workspace-write"
    if sandbox not in SANDBOXES:
        raise AOPError(
            f"{label}.sandbox must be one of: {', '.join(sorted(SANDBOXES))}"
        )

    timeout = value.get("timeout")
    if timeout is not None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise AOPError(f"{label}.timeout must be a number greater than zero")
        timeout = float(timeout)

    return BatchTask(
        id=task_id,
        prompt=prompt_text,
        prompt_source=prompt_source,
        base=base,
        model=model,
        effort=effort,
        sandbox=sandbox,
        timeout_seconds=timeout,
    )


def _required_string(value: dict[str, object], field: str, label: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise AOPError(f"{label}.{field} must be a non-empty string")
    return result


def _optional_string(value: dict[str, object], field: str, label: str) -> str | None:
    result = value.get(field)
    if result is not None and (not isinstance(result, str) or not result):
        raise AOPError(f"{label}.{field} must be a non-empty string")
    return result
