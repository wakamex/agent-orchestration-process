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

from .model_listing import AGENTS
from .isolation import PROFILES
from .isolation import resolve_policy
from .runner import AgentRunner, adapter_for, normalize_artifacts
from .worktrees import AOPError, TASK_ID, WorktreeManager


EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
MODES = {"agent", "participant"}
TASK_FIELDS = {
    "id",
    "agent",
    "prompt",
    "prompt_file",
    "base",
    "model",
    "provider",
    "mode",
    "effort",
    "profile",
    "no_web",
    "timeout",
    "artifacts",
    "inputs",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class BatchTask:
    id: str
    prompt: str
    prompt_source: str
    agent: str = "codex"
    base: str = "HEAD"
    model: str | None = None
    inference_provider: str | None = None
    effort: str | None = None
    mode: str = "agent"
    profile: str = "edit"
    no_web: bool = False
    timeout_seconds: float | None = None
    artifacts: tuple[str, ...] = ()
    input_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class BatchTaskResult:
    task: str
    agent: str
    status: str
    mode: str
    model: str | None
    inference_provider: str | None
    effort: str | None
    run_id: str | None
    session_id: str | None
    duration_seconds: float | None
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_output_tokens: int | None
    calculated_cost_usd: float | None
    provider_reported_cost_usd: float | None
    billing_route: str | None
    exit_code: int | None
    error: str | None
    execution_completed: bool
    response_available: bool
    provider_status: str | None
    provider_error: str | None


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

    @property
    def clean_successes(self) -> int:
        return sum(task.status == "succeeded" for task in self.tasks)

    @property
    def responses_with_provider_errors(self) -> int:
        return sum(
            task.status == "response_available_with_provider_error"
            for task in self.tasks
        )

    @property
    def runs_without_response(self) -> int:
        return sum(not task.response_available for task in self.tasks)

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "clean_successes": self.clean_successes,
            "responses_with_provider_errors": self.responses_with_provider_errors,
            "runs_without_response": self.runs_without_response,
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
        for execution_profile, agent, no_web in sorted(
            {(task.profile, task.agent, task.no_web) for task in tasks}
        ):
            policy = resolve_policy(execution_profile, provider=agent, no_web=no_web)
            report(
                f"policy {execution_profile}/{agent}: "
                f"workspace={policy.workspace['access']} "
                f"repository={policy.repository['access']} "
                f"host={policy.host['access']} "
                f"network={policy.network}"
            )
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
                    agent=task.agent,
                    status="not_started",
                    mode=task.mode,
                    model=task.model,
                    inference_provider=task.inference_provider,
                    effort=task.effort,
                    run_id=None,
                    session_id=None,
                    duration_seconds=None,
                    input_tokens=None,
                    cached_input_tokens=None,
                    output_tokens=None,
                    reasoning_output_tokens=None,
                    calculated_cost_usd=None,
                    provider_reported_cost_usd=None,
                    billing_route=None,
                    exit_code=None,
                    error="batch interrupted before launch",
                    execution_completed=False,
                    response_available=False,
                    provider_status=None,
                    provider_error=None,
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
            result = AgentRunner(self.manager, adapter_for(task.agent)).run(
                task=task.id,
                prompt=task.prompt,
                base=task.base,
                model=task.model,
                inference_provider=task.inference_provider,
                effort=task.effort,
                mode=task.mode,
                profile=task.profile,
                no_web=task.no_web,
                timeout_seconds=task.timeout_seconds,
                artifacts=task.artifacts,
                input_paths=task.input_paths,
            )
        except Exception as error:
            return BatchTaskResult(
                task=task.id,
                agent=task.agent,
                status="error",
                mode=task.mode,
                model=task.model,
                inference_provider=task.inference_provider,
                effort=task.effort,
                run_id=None,
                session_id=None,
                duration_seconds=None,
                input_tokens=None,
                cached_input_tokens=None,
                output_tokens=None,
                reasoning_output_tokens=None,
                calculated_cost_usd=None,
                provider_reported_cost_usd=None,
                billing_route=None,
                exit_code=None,
                error=str(error),
                execution_completed=False,
                response_available=False,
                provider_status=None,
                provider_error=None,
            )
        return BatchTaskResult(
            task=task.id,
            agent=task.agent,
            status=result.status,
            mode=result.mode,
            model=result.model,
            inference_provider=result.inference_provider,
            effort=result.effort,
            run_id=result.run_id,
            session_id=result.session_id,
            duration_seconds=result.duration_seconds,
            input_tokens=result.usage.input_tokens,
            cached_input_tokens=result.usage.cached_input_tokens,
            output_tokens=result.usage.output_tokens,
            reasoning_output_tokens=result.usage.reasoning_output_tokens,
            calculated_cost_usd=(
                result.calculated_cost.amount_usd if result.calculated_cost else None
            ),
            provider_reported_cost_usd=(
                result.provider_reported_cost.amount_usd
                if result.provider_reported_cost
                else None
            ),
            billing_route=result.billing.route,
            exit_code=result.exit_code,
            error=result.error,
            execution_completed=result.execution_completed,
            response_available=result.response_available,
            provider_status=result.provider_status,
            provider_error=result.provider_error,
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
    if not isinstance(agent, str) or agent not in AGENTS:
        raise AOPError(f"{label}.agent must be one of: {', '.join(AGENTS)}")

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
    inference_provider = _optional_string(value, "provider", label)
    mode = _optional_string(value, "mode", label) or "agent"
    if mode not in MODES:
        raise AOPError(f"{label}.mode must be one of: {', '.join(sorted(MODES))}")
    effort = _optional_string(value, "effort", label)
    if effort is not None and effort not in EFFORTS:
        raise AOPError(f"{label}.effort must be one of: {', '.join(sorted(EFFORTS))}")
    execution_profile = _optional_string(value, "profile", label) or "edit"
    if execution_profile not in PROFILES:
        raise AOPError(f"{label}.profile must be one of: {', '.join(PROFILES)}")
    no_web = value.get("no_web", False)
    if not isinstance(no_web, bool):
        raise AOPError(f"{label}.no_web must be a boolean")

    timeout = value.get("timeout")
    if timeout is not None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise AOPError(f"{label}.timeout must be a number greater than zero")
        timeout = float(timeout)

    artifacts = value.get("artifacts", [])
    if not isinstance(artifacts, list) or not all(
        isinstance(artifact, str) and artifact for artifact in artifacts
    ):
        raise AOPError(f"{label}.artifacts must be an array of non-empty strings")

    raw_input_paths = value.get("inputs", [])
    if not isinstance(raw_input_paths, list) or not all(
        isinstance(path, str) and path for path in raw_input_paths
    ):
        raise AOPError(f"{label}.inputs must be an array of non-empty strings")
    resolved_input_paths: list[str] = []
    for item in raw_input_paths:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = manifest_dir / path
        resolved_input_paths.append(os.fspath(path.resolve()))

    return BatchTask(
        id=task_id,
        prompt=prompt_text,
        prompt_source=prompt_source,
        agent=agent,
        base=base,
        model=model,
        inference_provider=inference_provider,
        effort=effort,
        mode=mode,
        profile=execution_profile,
        no_web=no_web,
        timeout_seconds=timeout,
        artifacts=normalize_artifacts(artifacts),
        input_paths=tuple(resolved_input_paths),
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
