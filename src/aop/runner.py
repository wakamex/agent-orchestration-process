"""Structured execution and persistence for coding-agent runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Callable, Protocol, Sequence

from .models import RunArtifact, RunRequest, RunResult
from .locks import exclusive_lock, task_lock_path
from .pricing import EstimatedCost, TokenUsage, estimate_api_cost
from .worktrees import AOPError, Worktree, WorktreeManager


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


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

    def normalize_options(
        self, model: str | None, effort: str | None
    ) -> tuple[str | None, str | None]: ...

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

    def normalize_options(
        self, model: str | None, effort: str | None
    ) -> tuple[str | None, str | None]:
        return model, effort

    def execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        run_dir: Path,
        environment: dict[str, str],
    ) -> RunResult:
        last_message_path = (
            Path(tempfile.gettempdir()) / f"aop-codex-{request.run_id}.txt"
        )
        command = _provider_command(
            self._command(request, worktree, last_message_path),
            request,
            worktree,
            environment,
        )
        started_at = _now()
        started = time.monotonic()
        timed_out = False
        first_event_seconds = None
        first_response_seconds = None

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
            stdout_parts: list[str] = []
            stderr_parts: list[str] = []

            def read_stdout() -> None:
                nonlocal first_event_seconds, first_response_seconds
                assert process.stdout is not None
                for line in process.stdout:
                    elapsed = time.monotonic() - started
                    if first_event_seconds is None:
                        first_event_seconds = elapsed
                    if first_response_seconds is None and self._is_agent_message(line):
                        first_response_seconds = elapsed
                    stdout_parts.append(line)

            def read_stderr() -> None:
                assert process.stderr is not None
                stderr_parts.extend(process.stderr)

            stdout_reader = threading.Thread(target=read_stdout, daemon=True)
            stderr_reader = threading.Thread(target=read_stderr, daemon=True)
            stdout_reader.start()
            stderr_reader.start()

            try:
                assert process.stdin is not None
                try:
                    process.stdin.write(request.prompt)
                    process.stdin.flush()
                except BrokenPipeError:
                    pass
                finally:
                    process.stdin.close()
                process.wait(timeout=request.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._signal_process_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._signal_process_group(process, signal.SIGKILL)
                    process.wait()
            except KeyboardInterrupt:
                self._signal_process_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._signal_process_group(process, signal.SIGKILL)
                    process.wait()
                raise
            finally:
                stdout_reader.join(timeout=5)
                stderr_reader.join(timeout=5)
            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)
            exit_code = process.returncode

        duration = time.monotonic() - started
        _atomic_write(run_dir / "events.jsonl", stdout)
        _atomic_write(run_dir / "stderr.log", stderr)

        session_id, event_error, usage = self._parse_events(stdout)
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
            _atomic_write(run_dir / "last-message.txt", final_message)
            last_message_path.unlink()

        return RunResult(
            run_id=request.run_id,
            provider=request.provider,
            task=request.task,
            model=request.model,
            effort=request.effort,
            session_id=session_id or request.session_id,
            command=command,
            started_at=started_at,
            finished_at=_now(),
            duration_seconds=round(duration, 6),
            time_to_first_event_seconds=_rounded(first_event_seconds),
            time_to_first_response_seconds=_rounded(first_response_seconds),
            exit_code=exit_code,
            timed_out=timed_out,
            error=error,
            final_message=final_message,
            usage=usage,
            api_equivalent_cost=estimate_api_cost(request.model, usage),
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
            "--dangerously-bypass-approvals-and-sandbox",
            "--output-last-message",
            os.fspath(last_message_path),
            "-C",
            os.fspath(worktree.path),
        ]
        if request.model and not request.session_id:
            command.extend(["--model", request.model])
        if request.effort and not request.session_id:
            command.extend(["--config", f"model_reasoning_effort={request.effort}"])
        if request.session_id:
            command.extend(["resume", request.session_id, "-"])
        else:
            command.append("-")
        return command

    @staticmethod
    def _parse_events(
        output: str,
    ) -> tuple[str | None, str | None, TokenUsage]:
        session_id = None
        error = None
        usage = TokenUsage()
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
            elif event_type == "turn.completed" and isinstance(
                event.get("usage"), dict
            ):
                current = TokenUsage.from_dict(event["usage"])
                usage = TokenUsage(
                    input_tokens=usage.input_tokens + current.input_tokens,
                    cached_input_tokens=(
                        usage.cached_input_tokens + current.cached_input_tokens
                    ),
                    output_tokens=usage.output_tokens + current.output_tokens,
                    reasoning_output_tokens=(
                        usage.reasoning_output_tokens + current.reasoning_output_tokens
                    ),
                )
        return session_id, error, usage

    @staticmethod
    def _is_agent_message(line: str) -> bool:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        return (
            isinstance(event, dict)
            and event.get("type") in {"item.started", "item.updated", "item.completed"}
            and isinstance(event.get("item"), dict)
            and event["item"].get("type") == "agent_message"
        )

    @staticmethod
    def _signal_process_group(
        process: subprocess.Popen[str], signal_number: signal.Signals
    ) -> None:
        try:
            os.killpg(process.pid, signal_number)
        except ProcessLookupError:
            pass


@dataclass(frozen=True)
class _ProcessCapture:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_seconds: float
    first_event_seconds: float | None
    first_response_seconds: float | None


def _capture_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    prompt: str | None,
    timeout_seconds: float | None,
    is_response: Callable[[str], bool],
) -> _ProcessCapture:
    started = time.monotonic()
    timed_out = False
    first_event_seconds = None
    first_response_seconds = None
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE if prompt is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except FileNotFoundError:
        return _ProcessCapture(
            stdout="",
            stderr=f"command not found: {command[0]}\n",
            exit_code=127,
            timed_out=False,
            duration_seconds=round(time.monotonic() - started, 6),
            first_event_seconds=None,
            first_response_seconds=None,
        )

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def read_stdout() -> None:
        nonlocal first_event_seconds, first_response_seconds
        assert process.stdout is not None
        for line in process.stdout:
            elapsed = time.monotonic() - started
            if first_event_seconds is None:
                first_event_seconds = elapsed
            if first_response_seconds is None and is_response(line):
                first_response_seconds = elapsed
            stdout_parts.append(line)

    def read_stderr() -> None:
        assert process.stderr is not None
        stderr_parts.extend(process.stderr)

    stdout_reader = threading.Thread(target=read_stdout, daemon=True)
    stderr_reader = threading.Thread(target=read_stderr, daemon=True)
    stdout_reader.start()
    stderr_reader.start()
    try:
        if prompt is not None:
            assert process.stdin is not None
            try:
                process.stdin.write(prompt)
                process.stdin.flush()
            except BrokenPipeError:
                pass
            finally:
                process.stdin.close()
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        CodexAdapter._signal_process_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            CodexAdapter._signal_process_group(process, signal.SIGKILL)
            process.wait()
    except KeyboardInterrupt:
        CodexAdapter._signal_process_group(process, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            CodexAdapter._signal_process_group(process, signal.SIGKILL)
            process.wait()
        raise
    finally:
        stdout_reader.join(timeout=5)
        stderr_reader.join(timeout=5)
    return _ProcessCapture(
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        exit_code=process.returncode,
        timed_out=timed_out,
        duration_seconds=round(time.monotonic() - started, 6),
        first_event_seconds=_rounded(first_event_seconds),
        first_response_seconds=_rounded(first_response_seconds),
    )


class ClaudeAdapter:
    provider = "claude"
    EFFORTS = {"low", "medium", "high", "xhigh", "max"}

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("AOP_CLAUDE_BIN", "claude")

    def normalize_options(
        self, model: str | None, effort: str | None
    ) -> tuple[str | None, str | None]:
        if effort is not None and effort not in self.EFFORTS:
            raise AOPError(
                f"Claude effort must be one of: {', '.join(sorted(self.EFFORTS))}"
            )
        return model, effort

    def execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        run_dir: Path,
        environment: dict[str, str],
    ) -> RunResult:
        command = _provider_command(
            self._command(request, worktree), request, worktree, environment
        )
        started_at = _now()
        capture = _capture_process(
            command,
            cwd=worktree.path,
            environment=environment,
            prompt=request.prompt,
            timeout_seconds=request.timeout_seconds,
            is_response=self._is_response,
        )
        _atomic_write(run_dir / "events.jsonl", capture.stdout)
        _atomic_write(run_dir / "stderr.log", capture.stderr)
        parsed = self._parse(capture.stdout)
        final_message = parsed["final_message"]
        if final_message is not None:
            _atomic_write(run_dir / "last-message.txt", final_message)
        if capture.timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif parsed["error"]:
            error = parsed["error"]
        elif capture.exit_code:
            error = (
                capture.stderr.strip()
                or f"Claude exited with status {capture.exit_code}"
            )
        elif parsed["session_id"] is None:
            error = "Claude did not report a session ID"
        else:
            error = None
        return RunResult(
            run_id=request.run_id,
            provider=self.provider,
            task=request.task,
            model=parsed["model"] or request.model,
            effort=request.effort,
            session_id=parsed["session_id"] or request.session_id,
            command=command,
            started_at=started_at,
            finished_at=_now(),
            duration_seconds=capture.duration_seconds,
            time_to_first_event_seconds=capture.first_event_seconds,
            time_to_first_response_seconds=capture.first_response_seconds,
            exit_code=capture.exit_code,
            timed_out=capture.timed_out,
            error=error,
            final_message=final_message,
            usage=parsed["usage"],
            api_equivalent_cost=parsed["cost"],
        )

    def _command(self, request: RunRequest, worktree: Worktree) -> list[str]:
        command = [
            self.binary,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
            "--dangerously-skip-permissions",
            "--add-dir",
            os.fspath(worktree.path),
        ]
        if request.session_id:
            command.extend(["--resume", request.session_id])
        else:
            command.extend(["--session-id", str(uuid.uuid4())])
            if request.model:
                command.extend(["--model", request.model])
            if request.effort:
                command.extend(["--effort", request.effort])
        return command

    @staticmethod
    def _is_response(line: str) -> bool:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        if event.get("type") != "assistant":
            return False
        content = event.get("message", {}).get("content", [])
        return any(
            item.get("type") == "text" for item in content if isinstance(item, dict)
        )

    @staticmethod
    def _parse(output: str) -> dict[str, object]:
        session_id = None
        model = None
        final_message = None
        error = None
        usage = TokenUsage()
        cost = None
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            session_id = event.get("session_id") or session_id
            if event.get("type") == "system" and event.get("subtype") == "init":
                model = event.get("model") or model
            if event.get("type") != "result":
                continue
            final_message = (
                event.get("result") if isinstance(event.get("result"), str) else None
            )
            if event.get("is_error"):
                error = final_message or "Claude turn failed"
            raw_usage = (
                event.get("usage") if isinstance(event.get("usage"), dict) else {}
            )
            uncached = int(raw_usage.get("input_tokens", 0) or 0)
            cached = int(raw_usage.get("cache_read_input_tokens", 0) or 0)
            created = int(raw_usage.get("cache_creation_input_tokens", 0) or 0)
            usage = TokenUsage(
                input_tokens=uncached + cached + created,
                cached_input_tokens=cached,
                output_tokens=int(raw_usage.get("output_tokens", 0) or 0),
            )
            amount = event.get("total_cost_usd")
            if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                priced_as = model or "claude"
                cost = EstimatedCost(
                    amount_usd=round(float(amount), 8),
                    currency="USD",
                    estimated=False,
                    model=priced_as,
                    priced_as=priced_as,
                    pricing_version="claude-cli-reported",
                    pricing_source="Claude Code result event",
                    long_context_pricing=False,
                )
        return {
            "session_id": session_id,
            "model": model,
            "final_message": final_message,
            "error": error,
            "usage": usage,
            "cost": cost,
        }


class AgyAdapter:
    provider = "agy"
    EFFORTS = {"low", "medium", "high"}

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("AOP_AGY_BIN", "agy")

    def normalize_options(
        self, model: str | None, effort: str | None
    ) -> tuple[str | None, str | None]:
        selected = model or "gemini-3.5-flash"
        level = effort if effort is not None else ("medium" if model is None else None)
        if level is not None and level not in self.EFFORTS:
            raise AOPError(
                f"agy effort must be one of: {', '.join(sorted(self.EFFORTS))}"
            )
        return selected, level

    def execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        run_dir: Path,
        environment: dict[str, str],
    ) -> RunResult:
        command = _provider_command(
            self._command(request), request, worktree, environment
        )
        started_at = _now()
        capture = _capture_process(
            command,
            cwd=worktree.path,
            environment=environment,
            prompt=None,
            timeout_seconds=request.timeout_seconds,
            is_response=self._is_response,
        )
        _atomic_write(run_dir / "events.jsonl", capture.stdout)
        _atomic_write(run_dir / "stderr.log", capture.stderr)
        parsed = self._parse(capture.stdout)
        final_message = parsed["final_message"]
        if final_message is not None:
            _atomic_write(run_dir / "last-message.txt", final_message)
        session_id = parsed["session_id"] or request.session_id
        if capture.timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif parsed["error"]:
            error = parsed["error"]
        elif capture.exit_code:
            error = (
                capture.stderr.strip() or f"agy exited with status {capture.exit_code}"
            )
        elif not parsed["has_result"]:
            error = "agy did not emit a terminal result"
        elif session_id is None:
            error = "agy did not report a conversation ID"
        else:
            error = None
        return RunResult(
            run_id=request.run_id,
            provider=self.provider,
            task=request.task,
            model=parsed["model"] or request.model,
            effort=request.effort,
            session_id=session_id,
            command=command,
            started_at=started_at,
            finished_at=_now(),
            duration_seconds=capture.duration_seconds,
            time_to_first_event_seconds=capture.first_event_seconds,
            time_to_first_response_seconds=capture.first_response_seconds,
            exit_code=capture.exit_code,
            timed_out=capture.timed_out,
            error=error,
            final_message=final_message,
            usage=parsed["usage"],
            api_equivalent_cost=None,
            provider_duration_seconds=parsed["duration_seconds"],
        )

    def _command(self, request: RunRequest) -> list[str]:
        command = [
            self.binary,
            "--mode",
            "accept-edits",
            "--dangerously-skip-permissions",
            "--output-format",
            "stream-json",
            "--print-timeout",
            "24h",
        ]
        if request.session_id:
            command.extend(["--conversation", request.session_id])
        else:
            if request.model:
                command.extend(["--model", request.model])
            if request.effort:
                command.extend(["--effort", request.effort])
        command.extend(["-p", request.prompt])
        return command

    @staticmethod
    def _is_response(line: str) -> bool:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(event, dict):
            return False
        event_type = event.get("event")
        payload = event.get(event_type)
        if not isinstance(payload, dict):
            return False
        if event_type == "result":
            return bool(payload.get("response"))
        return (
            event_type == "step_update"
            and payload.get("step_type") == "agent_response"
            and bool(payload.get("text_delta"))
        )

    @staticmethod
    def _parse(output: str) -> dict[str, object]:
        session_id = None
        model = None
        final_message = None
        error = None
        duration_seconds = None
        usage = TokenUsage()
        has_result = False
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("event")
            payload = event.get(event_type)
            if not isinstance(payload, dict):
                continue
            conversation_id = payload.get("conversation_id")
            if isinstance(conversation_id, str):
                session_id = conversation_id
            if event_type == "init" and isinstance(payload.get("model"), str):
                model = payload["model"]
            if event_type != "result":
                continue
            has_result = True
            response = payload.get("response")
            final_message = response if isinstance(response, str) else None
            status = payload.get("status")
            if status != "SUCCESS":
                detail = payload.get("error")
                error = (
                    detail
                    if isinstance(detail, str) and detail
                    else f"agy ended with status {status or 'unknown'}"
                )
            duration = payload.get("duration_seconds")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                duration_seconds = round(float(duration), 6)
            raw_usage = (
                payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            )
            usage = TokenUsage.from_dict(
                {
                    "input_tokens": raw_usage.get("input_tokens"),
                    "cached_input_tokens": raw_usage.get("cache_read_tokens"),
                    "output_tokens": raw_usage.get("output_tokens"),
                    "reasoning_output_tokens": raw_usage.get("thinking_tokens"),
                }
            )
        return {
            "session_id": session_id,
            "model": model,
            "final_message": final_message,
            "error": error,
            "duration_seconds": duration_seconds,
            "usage": usage,
            "has_result": has_result,
        }


@dataclass(frozen=True)
class _HermesSession:
    model: str | None
    final_message: str | None
    last_assistant_id: str | None
    usage: TokenUsage
    cost_usd: float | None
    cost_estimated: bool
    cost_source: str
    pricing_version: str


class HermesAdapter:
    provider = "hermes"
    DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
    EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("AOP_HERMES_BIN", "hermes")

    def normalize_options(
        self, model: str | None, effort: str | None
    ) -> tuple[str | None, str | None]:
        if effort is not None and effort not in self.EFFORTS:
            raise AOPError(
                f"Hermes effort must be one of: {', '.join(sorted(self.EFFORTS))}"
            )
        return model or self.DEFAULT_MODEL, effort

    def execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        run_dir: Path,
        environment: dict[str, str],
    ) -> RunResult:
        before = self._session(request.session_id, worktree.path, environment)
        command = _provider_command(
            self._command(request), request, worktree, environment
        )
        started_at = _now()
        capture = _capture_process(
            command,
            cwd=worktree.path,
            environment=environment,
            prompt=None,
            timeout_seconds=request.timeout_seconds,
            is_response=lambda line: bool(line.strip()),
        )
        _atomic_write(run_dir / "events.jsonl", capture.stdout)
        _atomic_write(run_dir / "stderr.log", capture.stderr)
        reported_session_id = self._session_id(capture.stderr)
        session_id = reported_session_id or request.session_id
        after = self._session(session_id, worktree.path, environment)
        final_message = self._final_message(before, after, capture.stdout)
        if final_message is not None:
            _atomic_write(run_dir / "last-message.txt", final_message)
        baseline = before if request.session_id == session_id else None
        usage = self._usage_delta(baseline, after)
        cost = self._cost_delta(baseline, after, request.model)

        if capture.timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif capture.exit_code:
            error = self._exit_error(capture.stderr, capture.exit_code)
        elif session_id is None:
            error = "Hermes did not report a session ID"
        elif final_message is None:
            error = "Hermes did not emit a final response"
        else:
            error = None
        return RunResult(
            run_id=request.run_id,
            provider=self.provider,
            task=request.task,
            model=(after.model if after else None) or request.model,
            effort=request.effort,
            session_id=session_id,
            command=command,
            started_at=started_at,
            finished_at=_now(),
            duration_seconds=capture.duration_seconds,
            time_to_first_event_seconds=capture.first_event_seconds,
            time_to_first_response_seconds=capture.first_response_seconds,
            exit_code=capture.exit_code,
            timed_out=capture.timed_out,
            error=error,
            final_message=final_message,
            usage=usage,
            api_equivalent_cost=cost,
        )

    def _command(self, request: RunRequest) -> list[str]:
        command = [
            self.binary,
            "chat",
            "-Q",
            "--yolo",
            "--accept-hooks",
            "--source",
            "tool",
        ]
        if request.session_id:
            command.extend(["--resume", request.session_id, "--no-restore-cwd"])
        command.extend(["--provider", "nous"])
        if request.model:
            command.extend(["--model", request.model])
        if request.effort:
            command.extend(["--reasoning", request.effort])
        command.extend(["-q", request.prompt])
        return command

    def _session(
        self,
        session_id: str | None,
        cwd: Path,
        environment: dict[str, str],
    ) -> _HermesSession | None:
        if session_id is None:
            return None
        try:
            exported = subprocess.run(
                [
                    self.binary,
                    "sessions",
                    "export",
                    "-",
                    "--format",
                    "jsonl",
                    "--session-id",
                    session_id,
                ],
                cwd=cwd,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if exported.returncode:
            return None
        try:
            raw = json.loads(exported.stdout)
        except json.JSONDecodeError:
            return None
        if not isinstance(raw, dict) or raw.get("id") != session_id:
            return None

        uncached_input = self._integer(raw.get("input_tokens"))
        cached_input = self._integer(raw.get("cache_read_tokens"))
        cache_write = self._integer(raw.get("cache_write_tokens"))
        visible_output = self._integer(raw.get("output_tokens"))
        reasoning_output = self._integer(raw.get("reasoning_tokens"))
        actual_cost = self._number(raw.get("actual_cost_usd"))
        estimated_cost = self._number(raw.get("estimated_cost_usd"))
        source = raw.get("cost_source")
        version = raw.get("pricing_version")
        model = raw.get("model")
        final_message = None
        last_assistant_id = None
        messages = raw.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    final_message = content.rstrip()
                    message_id = message.get("id")
                    last_assistant_id = (
                        message_id if isinstance(message_id, str) else None
                    )
                    break
        return _HermesSession(
            model=model if isinstance(model, str) else None,
            final_message=final_message,
            last_assistant_id=last_assistant_id,
            usage=TokenUsage(
                input_tokens=uncached_input + cached_input + cache_write,
                cached_input_tokens=cached_input,
                output_tokens=visible_output + reasoning_output,
                reasoning_output_tokens=reasoning_output,
            ),
            cost_usd=actual_cost if actual_cost is not None else estimated_cost,
            cost_estimated=actual_cost is None,
            cost_source=source if isinstance(source, str) else "session_accounting",
            pricing_version=(
                version if isinstance(version, str) else "hermes-cli-reported"
            ),
        )

    @staticmethod
    def _final_message(
        before: _HermesSession | None,
        after: _HermesSession | None,
        stdout: str,
    ) -> str | None:
        if after and after.final_message:
            is_new = before is None or (
                after.last_assistant_id is None
                or after.last_assistant_id != before.last_assistant_id
            )
            if is_new:
                return after.final_message
        return stdout.rstrip() or None

    @staticmethod
    def _session_id(stderr: str) -> str | None:
        for line in reversed(stderr.splitlines()):
            key, separator, value = line.strip().partition(":")
            if separator and key == "session_id" and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _exit_error(stderr: str, exit_code: int) -> str:
        detail = "\n".join(
            line
            for line in stderr.splitlines()
            if line.strip() and not line.strip().startswith("session_id:")
        ).strip()
        return detail or f"Hermes exited with status {exit_code}"

    @staticmethod
    def _usage_delta(
        before: _HermesSession | None, after: _HermesSession | None
    ) -> TokenUsage:
        if after is None:
            return TokenUsage()
        previous = before.usage if before else TokenUsage()
        return TokenUsage(
            input_tokens=max(after.usage.input_tokens - previous.input_tokens, 0),
            cached_input_tokens=max(
                after.usage.cached_input_tokens - previous.cached_input_tokens, 0
            ),
            output_tokens=max(after.usage.output_tokens - previous.output_tokens, 0),
            reasoning_output_tokens=max(
                after.usage.reasoning_output_tokens - previous.reasoning_output_tokens,
                0,
            ),
        )

    @staticmethod
    def _cost_delta(
        before: _HermesSession | None,
        after: _HermesSession | None,
        requested_model: str | None,
    ) -> EstimatedCost | None:
        if after is None or after.cost_usd is None:
            return None
        previous = before.cost_usd if before and before.cost_usd is not None else 0.0
        amount = max(after.cost_usd - previous, 0.0)
        model = after.model or requested_model or "hermes"
        return EstimatedCost(
            amount_usd=round(amount, 8),
            currency="USD",
            estimated=after.cost_estimated,
            model=model,
            priced_as=model,
            pricing_version=after.pricing_version,
            pricing_source=f"Hermes CLI session accounting ({after.cost_source})",
            long_context_pricing=False,
        )

    @staticmethod
    def _integer(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            return 0
        return max(value, 0)

    @staticmethod
    def _number(value: object) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return max(float(value), 0.0)


def _provider_command(
    command: list[str],
    request: RunRequest,
    worktree: Worktree,
    environment: dict[str, str],
) -> list[str]:
    if request.sandbox == "danger-full-access":
        return command
    root = Path(environment["AOP_ROOT"])
    cache = Path(environment["AOP_CACHE_DIR"])
    bwrap = os.environ.get("AOP_BWRAP_BIN", "bwrap")
    scratch = Path(environment["AOP_SCRATCH_DIR"])
    wrapped = [
        bwrap,
        "--die-with-parent",
        "--dev-bind",
        "/",
        "/",
        "--ro-bind",
        os.fspath(root),
        os.fspath(root),
    ]
    if request.sandbox == "workspace-write":
        wrapped.extend(["--bind", os.fspath(worktree.path), os.fspath(worktree.path)])
        git_marker = worktree.path / ".git"
        if git_marker.exists():
            wrapped.extend(["--ro-bind", os.fspath(git_marker), os.fspath(git_marker)])
    else:
        wrapped.extend(["--bind", os.fspath(scratch), os.fspath(scratch)])
    wrapped.extend(["--bind", os.fspath(cache), os.fspath(cache)])
    wrapped.extend(["--chdir", os.fspath(worktree.path), "--", *command])
    return wrapped


def adapter_for(agent: str) -> AgentAdapter:
    if agent == "codex":
        return CodexAdapter()
    if agent == "claude":
        return ClaudeAdapter()
    if agent == "agy":
        return AgyAdapter()
    if agent == "hermes":
        return HermesAdapter()
    raise AOPError(f"unknown agent: {agent}")


class AgentRunner:
    """Coordinate worktree selection, adapter execution, and durable records."""

    def __init__(
        self,
        manager: WorktreeManager,
        adapter: AgentAdapter | None = None,
    ):
        self.manager = manager
        self.adapter = adapter or CodexAdapter()
        self._adapter_was_explicit = adapter is not None
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
        artifacts: Sequence[str] = (),
    ) -> RunResult:
        model, effort = self.adapter.normalize_options(model, effort)
        request = self._request(
            task=task,
            prompt=prompt,
            base=base,
            model=model,
            effort=effort,
            sandbox=sandbox,
            timeout_seconds=timeout_seconds,
            artifacts=artifacts,
        )
        worktree = self._get_or_create_worktree(task, base)
        return self._execute(request, worktree)

    def resume(
        self,
        *,
        run_id: str,
        prompt: str,
        timeout_seconds: float | None = None,
        artifacts: Sequence[str] = (),
        _task_lock_held: bool = False,
    ) -> RunResult:
        parent_request = self.store.load_request(run_id)
        parent_result = self.store.load_result(run_id)
        if parent_request.provider != self.adapter.provider:
            if self._adapter_was_explicit:
                raise AOPError(
                    f"run uses {parent_request.provider}, not {self.adapter.provider}"
                )
            self.adapter = adapter_for(parent_request.provider)
        if not parent_result.session_id:
            raise AOPError(f"run has no resumable agent session: {run_id}")
        _task_lock_held = _task_lock_held or (
            os.environ.get("AOP_TASK_LOCK_HELD") == parent_request.task
        )
        worktree = self.manager.get(parent_request.task)
        request = self._request(
            task=parent_request.task,
            prompt=prompt,
            base=parent_request.base,
            model=parent_request.model,
            effort=parent_request.effort,
            sandbox=parent_request.sandbox,
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else parent_request.timeout_seconds
            ),
            session_id=parent_result.session_id,
            parent_run_id=run_id,
            artifacts=artifacts,
        )
        return self._execute(request, worktree, task_lock_held=_task_lock_held)

    def _execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        *,
        task_lock_held: bool = False,
    ) -> RunResult:
        if task_lock_held:
            return self._execute_unlocked(request, worktree)
        with exclusive_lock(
            task_lock_path(self.manager.state_dir, request.task),
            f"task {request.task}",
        ):
            return self._execute_unlocked(request, worktree)

    def _execute_unlocked(self, request: RunRequest, worktree: Worktree) -> RunResult:
        scratch_dir = worktree.path / "scratch"
        scratch_dir.mkdir(exist_ok=True)
        output_dir = scratch_dir / "outputs" / request.run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        request = replace(
            request,
            prompt=_artifact_prompt(request.prompt, request.artifacts, output_dir),
        )
        run_dir = self.store.create(request)
        environment = os.environ.copy()
        environment.update(
            {
                "AOP_ROOT": os.fspath(self.manager.root),
                "AOP_TASK": request.task,
                "AOP_WORKTREE": os.fspath(worktree.path),
                "AOP_CACHE_DIR": os.fspath(self.manager.cache_dir),
                "AOP_SCRATCH_DIR": os.fspath(scratch_dir),
                "AOP_OUTPUT_DIR": os.fspath(output_dir),
                "AOP_RUN_ID": request.run_id,
            }
        )
        result = self.adapter.execute(request, worktree, run_dir, environment)
        if result.succeeded and request.artifacts:
            result = _archive_artifacts(result, request.artifacts, output_dir, run_dir)
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
        artifacts: Sequence[str] = (),
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
            artifacts=normalize_artifacts(artifacts),
            created_at=_now(),
        )

    def _get_or_create_worktree(self, task: str, base: str) -> Worktree:
        for worktree in self.manager.list():
            if worktree.task == task:
                return worktree
        return self.manager.create(task, base)


def normalize_artifacts(artifacts: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in artifacts:
        if not value:
            raise AOPError("artifact path must not be empty")
        if "\0" in value:
            raise AOPError("artifact path must not contain a null byte")
        path = PurePosixPath(value)
        if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
            raise AOPError(f"artifact path must be relative and contained: {value}")
        logical_path = path.as_posix()
        if logical_path in normalized:
            raise AOPError(f"duplicate artifact path: {logical_path}")
        for existing in map(PurePosixPath, normalized):
            if path in existing.parents or existing in path.parents:
                raise AOPError(
                    f"overlapping artifact paths: {existing.as_posix()} and {logical_path}"
                )
        normalized.append(logical_path)
    return tuple(normalized)


def _artifact_prompt(prompt: str, artifacts: tuple[str, ...], output_dir: Path) -> str:
    if not artifacts:
        return prompt
    paths = "\n".join(f"- {artifact}" for artifact in artifacts)
    return (
        f"{prompt.rstrip()}\n\n"
        "AOP artifact contract:\n"
        f"Write the following deliverables beneath {output_dir}:\n"
        f"{paths}\n"
        "Declared paths may be files or directories; directories are collected recursively. "
        "Stdout and stderr are logs, not deliverables. AOP will reject missing, empty, or "
        "unsafe files."
    )


def _archive_artifacts(
    result: RunResult,
    declarations: tuple[str, ...],
    output_dir: Path,
    run_dir: Path,
) -> RunResult:
    try:
        sources: list[tuple[str, Path]] = []
        for declaration in declarations:
            sources.extend(_collect_artifact(declaration, output_dir))
        artifacts = tuple(
            _archive_artifact(logical_path, source, run_dir)
            for logical_path, source in sources
        )
    except (AOPError, OSError) as error:
        shutil.rmtree(run_dir / "artifacts", ignore_errors=True)
        diagnostic = (
            str(error)
            if isinstance(error, AOPError)
            else f"could not archive artifacts: {error}"
        )
        return replace(result, error=diagnostic, artifacts=())
    return replace(result, artifacts=artifacts)


def _collect_artifact(
    logical_path: str, output_dir: Path
) -> tuple[tuple[str, Path], ...]:
    scratch_dir = output_dir.parent.parent
    for directory in (scratch_dir, output_dir.parent, output_dir):
        try:
            status = directory.lstat()
        except FileNotFoundError as error:
            raise AOPError(
                f"artifact {logical_path}: output directory is missing"
            ) from error
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise AOPError(f"artifact {logical_path}: unsafe output directory")

    source = output_dir.joinpath(*PurePosixPath(logical_path).parts)
    current = output_dir
    for part in PurePosixPath(logical_path).parts:
        current /= part
        try:
            status = current.lstat()
        except (FileNotFoundError, NotADirectoryError) as error:
            raise AOPError(f"artifact {logical_path}: missing") from error
        if stat.S_ISLNK(status.st_mode):
            raise AOPError(f"artifact {logical_path}: symlinks are not allowed")
    if stat.S_ISDIR(status.st_mode):
        return _collect_artifact_directory(
            source, PurePosixPath(logical_path), output_dir
        )
    return (
        (
            logical_path,
            _validate_artifact_file(logical_path, source, status, output_dir),
        ),
    )


def _collect_artifact_directory(
    directory: Path,
    logical_directory: PurePosixPath,
    output_dir: Path,
) -> tuple[tuple[str, Path], ...]:
    collected: list[tuple[str, Path]] = []
    for child in sorted(directory.iterdir(), key=lambda path: path.name):
        logical_path = (logical_directory / child.name).as_posix()
        status = child.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise AOPError(f"artifact {logical_path}: symlinks are not allowed")
        if stat.S_ISDIR(status.st_mode):
            collected.extend(
                _collect_artifact_directory(
                    child, PurePosixPath(logical_path), output_dir
                )
            )
            continue
        collected.append(
            (
                logical_path,
                _validate_artifact_file(logical_path, child, status, output_dir),
            )
        )
    return tuple(collected)


def _validate_artifact_file(
    logical_path: str,
    source: Path,
    status: os.stat_result,
    output_dir: Path,
) -> Path:
    if not stat.S_ISREG(status.st_mode):
        raise AOPError(f"artifact {logical_path}: not a regular file")
    if status.st_size == 0:
        raise AOPError(f"artifact {logical_path}: empty")
    try:
        source.resolve(strict=True).relative_to(output_dir.resolve(strict=True))
    except ValueError as error:
        raise AOPError(
            f"artifact {logical_path}: escapes the output directory"
        ) from error
    return source


def _archive_artifact(logical_path: str, source: Path, run_dir: Path) -> RunArtifact:
    archive_path = PurePosixPath("artifacts") / logical_path
    destination = run_dir.joinpath(*archive_path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    size_bytes = destination.stat().st_size
    with destination.open("rb") as artifact:
        digest = hashlib.file_digest(artifact, "sha256").hexdigest()
    return RunArtifact(
        logical_path=logical_path,
        archive_path=archive_path.as_posix(),
        size_bytes=size_bytes,
        sha256=digest,
    )
