"""Structured execution and persistence for coding-agent runs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .models import RunRequest, RunResult
from .worktrees import AOPError, Worktree, WorktreeManager


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


class RunStore:
    """Store immutable inputs and terminal results for each invocation."""

    def __init__(self, root: Path):
        self.root = root

    def create(self, request: RunRequest) -> Path:
        run_dir = self.root / request.run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise AOPError(f"run already exists: {request.run_id}") from error
        self.write_json(run_dir / "request.json", request.to_dict())
        return run_dir

    def load_request(self, run_id: str) -> RunRequest:
        return RunRequest.from_dict(self._load_json(run_id, "request.json"))

    def load_result(self, run_id: str) -> RunResult:
        return RunResult.from_dict(self._load_json(run_id, "result.json"))

    def path(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self.root / run_id

    @staticmethod
    def write_json(path: Path, value: dict[str, object]) -> None:
        _atomic_write(path, f"{json.dumps(value, indent=2, sort_keys=True)}\n")

    def _load_json(self, run_id: str, name: str) -> dict[str, object]:
        path = self.path(run_id) / name
        try:
            value = json.loads(path.read_text())
        except FileNotFoundError as error:
            raise AOPError(f"run artifact not found: {path}") from error
        except json.JSONDecodeError as error:
            raise AOPError(f"invalid run artifact: {path}: {error}") from error
        if not isinstance(value, dict):
            raise AOPError(f"invalid run artifact: {path}: expected an object")
        return value

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        try:
            parsed = uuid.UUID(run_id)
        except ValueError as error:
            raise AOPError(f"invalid run id: {run_id}") from error
        if str(parsed) != run_id:
            raise AOPError(f"invalid run id: {run_id}")


class AgentAdapter(Protocol):
    provider: str

    def execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        run_dir: Path,
        environment: dict[str, str],
    ) -> RunResult: ...


class CodexAdapter:
    provider = "codex"

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("AOP_CODEX_BIN", "codex")

    def execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        run_dir: Path,
        environment: dict[str, str],
    ) -> RunResult:
        last_message_path = run_dir / "last-message.txt"
        command = self._command(request, worktree, last_message_path)
        started_at = _now()
        started = time.monotonic()
        timed_out = False

        try:
            process = subprocess.Popen(
                command,
                cwd=worktree.path,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
        except FileNotFoundError:
            stdout = ""
            stderr = f"command not found: {self.binary}\n"
            exit_code = 127
        else:
            try:
                stdout, stderr = process.communicate(
                    request.prompt, timeout=request.timeout_seconds
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    stdout, stderr = process.communicate()
            exit_code = process.returncode

        duration = time.monotonic() - started
        _atomic_write(run_dir / "events.jsonl", stdout)
        _atomic_write(run_dir / "stderr.log", stderr)

        session_id, event_error = self._parse_events(stdout)
        if timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif event_error:
            error = event_error
        elif exit_code != 0:
            error = stderr.strip() or f"Codex exited with status {exit_code}"
        elif session_id is None:
            error = "Codex did not emit a thread.started event"
        else:
            error = None

        final_message = None
        if last_message_path.exists():
            final_message = last_message_path.read_text()

        return RunResult(
            run_id=request.run_id,
            provider=request.provider,
            task=request.task,
            session_id=session_id or request.session_id,
            command=command,
            started_at=started_at,
            finished_at=_now(),
            duration_seconds=round(duration, 6),
            exit_code=exit_code,
            timed_out=timed_out,
            error=error,
            final_message=final_message,
        )

    def _command(
        self, request: RunRequest, worktree: Worktree, last_message_path: Path
    ) -> list[str]:
        command = [
            self.binary,
            "exec",
            "--json",
            "--color",
            "never",
            "--sandbox",
            request.sandbox,
            "--output-last-message",
            os.fspath(last_message_path),
            "-C",
            os.fspath(worktree.path),
        ]
        if request.model:
            command.extend(["--model", request.model])
        if request.effort:
            command.extend(["--config", f"model_reasoning_effort={request.effort}"])
        if request.session_id:
            command.extend(["resume", request.session_id, "-"])
        else:
            command.append("-")
        return command

    @staticmethod
    def _parse_events(output: str) -> tuple[str | None, str | None]:
        session_id = None
        error = None
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "thread.started" and isinstance(
                event.get("thread_id"), str
            ):
                session_id = event["thread_id"]
            elif event_type == "turn.failed":
                payload = event.get("error")
                if isinstance(payload, dict) and isinstance(
                    payload.get("message"), str
                ):
                    error = payload["message"]
                else:
                    error = "Codex turn failed"
            elif event_type == "error" and isinstance(event.get("message"), str):
                error = event["message"]
        return session_id, error


class AgentRunner:
    """Coordinate worktree selection, adapter execution, and durable records."""

    def __init__(
        self,
        manager: WorktreeManager,
        adapter: AgentAdapter | None = None,
    ):
        self.manager = manager
        self.adapter = adapter or CodexAdapter()
        self.store = RunStore(manager.state_dir / "runs")

    def run(
        self,
        *,
        task: str,
        prompt: str,
        base: str = "HEAD",
        model: str | None = None,
        effort: str | None = None,
        sandbox: str = "workspace-write",
        timeout_seconds: float | None = None,
    ) -> RunResult:
        worktree = self._get_or_create_worktree(task, base)
        request = self._request(
            task=task,
            prompt=prompt,
            base=base,
            model=model,
            effort=effort,
            sandbox=sandbox,
            timeout_seconds=timeout_seconds,
        )
        return self._execute(request, worktree)

    def resume(
        self,
        *,
        run_id: str,
        prompt: str,
        timeout_seconds: float | None = None,
    ) -> RunResult:
        parent_request = self.store.load_request(run_id)
        parent_result = self.store.load_result(run_id)
        if not parent_result.session_id:
            raise AOPError(f"run has no resumable Codex session: {run_id}")
        worktree = self.manager.get(parent_request.task)
        request = self._request(
            task=parent_request.task,
            prompt=prompt,
            base=parent_request.base,
            model=None,
            effort=None,
            sandbox=parent_request.sandbox,
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else parent_request.timeout_seconds
            ),
            session_id=parent_result.session_id,
            parent_run_id=run_id,
        )
        return self._execute(request, worktree)

    def _execute(self, request: RunRequest, worktree: Worktree) -> RunResult:
        run_dir = self.store.create(request)
        environment = os.environ.copy()
        environment.update(
            {
                "AOP_ROOT": os.fspath(self.manager.root),
                "AOP_TASK": request.task,
                "AOP_WORKTREE": os.fspath(worktree.path),
                "AOP_CACHE_DIR": os.fspath(self.manager.cache_dir),
                "AOP_RUN_ID": request.run_id,
            }
        )
        result = self.adapter.execute(request, worktree, run_dir, environment)
        self.store.write_json(run_dir / "result.json", result.to_dict())
        return result

    def _request(
        self,
        *,
        task: str,
        prompt: str,
        base: str,
        model: str | None,
        effort: str | None,
        sandbox: str,
        timeout_seconds: float | None,
        session_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> RunRequest:
        return RunRequest(
            run_id=str(uuid.uuid4()),
            provider=self.adapter.provider,
            task=task,
            prompt=prompt,
            base=base,
            model=model,
            effort=effort,
            sandbox=sandbox,
            timeout_seconds=timeout_seconds,
            session_id=session_id,
            parent_run_id=parent_run_id,
            created_at=_now(),
        )

    def _get_or_create_worktree(self, task: str, base: str) -> Worktree:
        for worktree in self.manager.list():
            if worktree.task == task:
                return worktree
        return self.manager.create(task, base)
