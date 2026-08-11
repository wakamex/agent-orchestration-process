"""Structured execution and persistence for coding-agent runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Protocol, Sequence

from .models import (
    BillingProvenance,
    ReadPath,
    ReadPathFile,
    RunArtifact,
    RunRequest,
    RunResult,
)
from .locks import exclusive_lock, task_lock_path
from .pricing import EstimatedCost, TokenUsage, estimate_api_cost
from .worktrees import AOPError, Worktree, WorktreeManager


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _billing_probe(
    command: list[str], cwd: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str] | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return result if result.returncode == 0 else None


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


class RunStore:
    """Store immutable inputs and terminal results for each invocation."""

    def __init__(self, root: Path):
        self.root = root

    def create(self, request: RunRequest) -> Path:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        run_dir = self.root / request.run_id
        try:
            run_dir.mkdir(mode=0o700)
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
    modes: frozenset[str]

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
    modes = frozenset({"agent"})

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
        _prepare_codex_environment(environment)
        last_message_path = (
            Path(environment["AOP_SCRATCH_DIR"])
            / f".codex-last-message-{request.run_id}.txt"
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

        reported_session_id, event_error, usage, completed = self._parse_events(stdout)
        session_id = reported_session_id
        resume_error = None
        if request.session_id and reported_session_id != request.session_id:
            session_id = None
            if reported_session_id is None:
                resume_error = (
                    "Codex resume result did not report the requested thread ID "
                    f"{request.session_id}"
                )
            else:
                resume_error = (
                    f"Codex resumed as thread {reported_session_id} instead of "
                    f"the requested thread {request.session_id}"
                )
        if timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif event_error:
            error = event_error
        elif exit_code != 0:
            error = stderr.strip() or f"Codex exited with status {exit_code}"
        elif resume_error:
            error = resume_error
        elif reported_session_id is None:
            error = "Codex did not emit a thread.started event"
        elif not completed:
            error = "Codex did not emit a terminal turn.completed event"
        else:
            error = None

        final_message = None
        if last_message_path.exists():
            final_message = last_message_path.read_text()
            _atomic_write(run_dir / "last-message.txt", final_message)
            last_message_path.unlink()
        billing = self._billing_provenance(worktree.path, environment)

        return RunResult(
            run_id=request.run_id,
            provider=request.provider,
            mode=request.mode,
            task=request.task,
            model=request.model,
            effort=request.effort,
            session_id=session_id,
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
            billing=billing,
        )

    def _billing_provenance(
        self, cwd: Path, environment: dict[str, str]
    ) -> BillingProvenance:
        status = _billing_probe([self.binary, "login", "status"], cwd, environment)
        if status is None:
            return BillingProvenance()
        output = f"{status.stdout}\n{status.stderr}"
        if "Logged in using ChatGPT" in output:
            return BillingProvenance(
                route="subscription",
                credential_source="chatgpt-oauth",
                detected_by="codex login status",
            )
        if "Logged in using an API key" in output:
            return BillingProvenance(
                route="metered-api",
                credential_source="openai-api-key",
                detected_by="codex login status",
            )
        return BillingProvenance(detected_by="codex login status")

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
    ) -> tuple[str | None, str | None, TokenUsage, bool]:
        session_id = None
        error = None
        usage = TokenUsage()
        completed = False
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
            elif event_type == "turn.completed":
                completed = True
                if isinstance(event.get("usage"), dict):
                    current = TokenUsage.from_dict(event["usage"])
                    usage = TokenUsage(
                        input_tokens=usage.input_tokens + current.input_tokens,
                        cached_input_tokens=(
                            usage.cached_input_tokens + current.cached_input_tokens
                        ),
                        output_tokens=usage.output_tokens + current.output_tokens,
                        reasoning_output_tokens=(
                            usage.reasoning_output_tokens
                            + current.reasoning_output_tokens
                        ),
                    )
        return session_id, error, usage, completed

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
    modes = frozenset({"agent"})
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
        reported_session_id = parsed["session_id"]
        session_id = reported_session_id
        resume_error = None
        if request.session_id and reported_session_id != request.session_id:
            session_id = None
            if reported_session_id is None:
                resume_error = (
                    "Claude resume result did not report the requested session ID "
                    f"{request.session_id}"
                )
            else:
                resume_error = (
                    f"Claude resumed as session {reported_session_id} instead of "
                    f"the requested session {request.session_id}"
                )
        if capture.timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif parsed["error"]:
            error = parsed["error"]
        elif capture.exit_code:
            error = (
                capture.stderr.strip()
                or f"Claude exited with status {capture.exit_code}"
            )
        elif resume_error:
            error = resume_error
        elif reported_session_id is None:
            error = "Claude did not report a session ID"
        elif not parsed["has_result"]:
            error = "Claude did not emit a terminal result event"
        else:
            error = None
        billing = self._billing_provenance(worktree.path, environment)
        return RunResult(
            run_id=request.run_id,
            provider=self.provider,
            mode=request.mode,
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
            api_equivalent_cost=parsed["cost"],
            billing=billing,
        )

    def _billing_provenance(
        self, cwd: Path, environment: dict[str, str]
    ) -> BillingProvenance:
        status = _billing_probe(
            [self.binary, "auth", "status", "--json"], cwd, environment
        )
        if status is not None:
            try:
                value = json.loads(status.stdout)
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict):
                method = value.get("authMethod")
                provider = value.get("apiProvider")
                normalized = (
                    method.lower().replace("_", "").replace(".", "")
                    if isinstance(method, str)
                    else ""
                )
                if normalized == "claudeai":
                    return BillingProvenance(
                        route="subscription",
                        credential_source="claude-oauth",
                        detected_by="claude auth status",
                    )
                if normalized in {"apikey", "console"}:
                    return BillingProvenance(
                        route="metered-api",
                        credential_source=(
                            "anthropic-api-key"
                            if normalized == "apikey"
                            else "anthropic-console"
                        ),
                        detected_by="claude auth status",
                    )
                if isinstance(provider, str) and provider.lower() in {
                    "bedrock",
                    "vertex",
                }:
                    return BillingProvenance(
                        route="metered-api",
                        credential_source=provider.lower(),
                        detected_by="claude auth status",
                    )
                return BillingProvenance(detected_by="claude auth status")
        if environment.get("CLAUDE_CODE_USE_BEDROCK"):
            return BillingProvenance(
                route="metered-api",
                credential_source="bedrock",
                detected_by="environment",
            )
        if environment.get("CLAUDE_CODE_USE_VERTEX"):
            return BillingProvenance(
                route="metered-api",
                credential_source="vertex",
                detected_by="environment",
            )
        if environment.get("ANTHROPIC_API_KEY"):
            return BillingProvenance(
                route="metered-api",
                credential_source="anthropic-api-key",
                detected_by="environment",
            )
        return BillingProvenance()

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
        has_result = False
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
            has_result = True
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
            "has_result": has_result,
        }


class CursorAdapter:
    provider = "cursor"
    modes = frozenset({"agent"})
    DEFAULT_MODEL = "composer-2.5"

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("AOP_CURSOR_BIN", "agent")

    def normalize_options(
        self, model: str | None, effort: str | None
    ) -> tuple[str | None, str | None]:
        if effort is not None:
            raise AOPError(
                "Cursor Agent does not accept a separate effort; choose a model ID "
                "with the desired effort"
            )
        return model or self.DEFAULT_MODEL, None

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
            prompt=None,
            timeout_seconds=request.timeout_seconds,
            is_response=self._is_response,
        )
        _atomic_write(run_dir / "events.jsonl", capture.stdout)
        _atomic_write(run_dir / "stderr.log", capture.stderr)
        parsed = self._parse(capture.stdout)
        reported_session_id = parsed["session_id"]
        session_id = reported_session_id
        final_message = parsed["final_message"]
        if final_message is not None:
            _atomic_write(run_dir / "last-message.txt", final_message)

        if capture.timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif capture.exit_code:
            error = (
                capture.stderr.strip()
                or f"Cursor Agent exited with status {capture.exit_code}"
            )
        elif not parsed["has_result"]:
            error = "Cursor Agent did not emit a terminal result"
        elif request.session_id and reported_session_id != request.session_id:
            session_id = None
            if reported_session_id is None:
                error = (
                    "Cursor Agent resume result did not report the requested chat ID "
                    f"{request.session_id}"
                )
            else:
                error = (
                    f"Cursor Agent resumed as chat {reported_session_id} instead of "
                    f"the requested chat {request.session_id}"
                )
        elif reported_session_id is None:
            error = "Cursor Agent result did not report a chat ID"
        elif parsed["error"]:
            error = parsed["error"]
        elif final_message is None:
            error = "Cursor Agent did not emit a final response"
        else:
            error = None

        return RunResult(
            run_id=request.run_id,
            provider=self.provider,
            mode=request.mode,
            task=request.task,
            model=request.model,
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
            billing=BillingProvenance(
                route="provider-credits",
                credential_source=(
                    "cursor-api-key"
                    if environment.get("CURSOR_API_KEY")
                    else "cursor-account"
                ),
                detected_by="Cursor Agent billing contract",
            ),
            provider_duration_seconds=parsed["duration_seconds"],
        )

    def _command(self, request: RunRequest, worktree: Worktree) -> list[str]:
        command = [
            self.binary,
            "-p",
            "--output-format",
            "stream-json",
            "--force",
            "--sandbox",
            "disabled",
            "--trust",
            "--workspace",
            os.fspath(worktree.path),
        ]
        if request.session_id:
            command.extend(["--resume", request.session_id])
        elif request.model:
            command.extend(["--model", request.model])
        command.extend(["--", request.prompt])
        return command

    @staticmethod
    def _is_response(line: str) -> bool:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(event, dict) or event.get("type") != "assistant":
            return False
        content = event.get("message", {}).get("content", [])
        return any(
            item.get("type") == "text" for item in content if isinstance(item, dict)
        )

    @staticmethod
    def _parse(output: str) -> dict[str, object]:
        session_id = None
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
            event_session_id = event.get("session_id")
            if isinstance(event_session_id, str) and event_session_id:
                session_id = event_session_id
            if event.get("type") != "result":
                continue
            has_result = True
            result = event.get("result")
            final_message = result if isinstance(result, str) else None
            if event.get("is_error") or event.get("subtype") != "success":
                detail = event.get("error")
                error = (
                    detail
                    if isinstance(detail, str) and detail
                    else final_message or "Cursor Agent turn failed"
                )
            raw_usage = (
                event.get("usage") if isinstance(event.get("usage"), dict) else {}
            )
            usage = TokenUsage.from_dict(
                {
                    "input_tokens": raw_usage.get("inputTokens"),
                    "cached_input_tokens": raw_usage.get("cacheReadTokens"),
                    "output_tokens": raw_usage.get("outputTokens"),
                }
            )
            duration_ms = event.get("duration_api_ms")
            if isinstance(duration_ms, (int, float)) and not isinstance(
                duration_ms, bool
            ):
                duration_seconds = round(float(duration_ms) / 1000, 6)
        return {
            "session_id": session_id,
            "final_message": final_message,
            "error": error,
            "duration_seconds": duration_seconds,
            "usage": usage,
            "has_result": has_result,
        }


class DevinAdapter:
    provider = "devin"
    modes = frozenset({"agent"})
    DEFAULT_MODEL = "swe-1-7"

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("AOP_DEVIN_BIN", "devin")

    def normalize_options(
        self, model: str | None, effort: str | None
    ) -> tuple[str | None, str | None]:
        if effort is not None:
            raise AOPError(
                "Devin does not accept a separate effort; choose a model ID with "
                "the desired reasoning level"
            )
        return model or self.DEFAULT_MODEL, None

    def execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        run_dir: Path,
        environment: dict[str, str],
    ) -> RunResult:
        _prepare_devin_environment(environment)
        export_path = (
            Path(environment["AOP_SCRATCH_DIR"])
            / f".devin-export-{request.run_id}.json"
        )
        command = _provider_command(
            self._command(request, export_path), request, worktree, environment
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
        parsed = self._parse_export(export_path, request.prompt)
        if export_path.is_file():
            _atomic_write(run_dir / "provider-result.json", export_path.read_text())
            export_path.unlink()
        final_message = parsed["final_message"]
        if final_message is not None:
            _atomic_write(run_dir / "last-message.txt", final_message)
        reported_session_id = parsed["session_id"]
        session_id = reported_session_id

        if capture.timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif capture.exit_code:
            error = (
                capture.stderr.strip()
                or f"Devin exited with status {capture.exit_code}"
            )
        elif parsed["error"]:
            error = parsed["error"]
        elif request.session_id and reported_session_id != request.session_id:
            session_id = None
            if reported_session_id is None:
                error = (
                    "Devin resume result did not report the requested session ID "
                    f"{request.session_id}"
                )
            else:
                error = (
                    f"Devin resumed as session {reported_session_id} instead of "
                    f"the requested session {request.session_id}"
                )
        elif reported_session_id is None:
            error = "Devin did not report a session ID"
        elif final_message is None:
            error = "Devin did not emit a final response"
        else:
            error = None

        return RunResult(
            run_id=request.run_id,
            provider=self.provider,
            mode=request.mode,
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
            billing=BillingProvenance(
                route="provider-credits",
                credential_source="devin-account",
                detected_by="Devin authenticated account",
            ),
        )

    def _command(self, request: RunRequest, export_path: Path) -> list[str]:
        command = [
            self.binary,
            "--permission-mode",
            "dangerous",
            "--respect-workspace-trust",
            "false",
            "--export",
            os.fspath(export_path),
        ]
        if request.session_id:
            command.extend(["--resume", request.session_id])
        elif request.model:
            command.extend(["--model", request.model])
        command.extend(["-p", request.prompt])
        return command

    @staticmethod
    def _parse_export(path: Path, prompt: str) -> dict[str, object]:
        try:
            value = json.loads(path.read_text())
        except FileNotFoundError:
            return DevinAdapter._invalid_export(
                "Devin did not write a trajectory export"
            )
        except (OSError, json.JSONDecodeError) as error:
            return DevinAdapter._invalid_export(
                f"Devin wrote an invalid trajectory export: {error}"
            )
        if not isinstance(value, dict) or not str(
            value.get("schema_version", "")
        ).startswith("ATIF-"):
            return DevinAdapter._invalid_export(
                "Devin wrote an invalid trajectory export"
            )
        steps = value.get("steps")
        if not isinstance(steps, list):
            return DevinAdapter._invalid_export(
                "Devin trajectory export did not contain steps"
            )
        turn_start = None
        for index, step in enumerate(steps):
            if (
                isinstance(step, dict)
                and step.get("source") == "user"
                and step.get("message") == prompt
            ):
                turn_start = index
        if turn_start is None:
            return DevinAdapter._invalid_export(
                "Devin trajectory export did not contain the current prompt"
            )
        agent_steps = [
            step
            for step in steps[turn_start + 1 :]
            if isinstance(step, dict) and step.get("source") == "agent"
        ]
        final_message = next(
            (
                step["message"]
                for step in reversed(agent_steps)
                if isinstance(step.get("message"), str) and step["message"]
            ),
            None,
        )
        model = next(
            (
                extra["generation_model"]
                for step in reversed(agent_steps)
                if isinstance((extra := step.get("extra")), dict)
                and isinstance(extra.get("generation_model"), str)
            ),
            None,
        )
        usage = TokenUsage(
            input_tokens=sum(
                DevinAdapter._metric(step, "prompt_tokens") for step in agent_steps
            ),
            cached_input_tokens=sum(
                DevinAdapter._metric(step, "cached_tokens") for step in agent_steps
            ),
            output_tokens=sum(
                DevinAdapter._metric(step, "completion_tokens") for step in agent_steps
            ),
        )
        session_id = value.get("session_id")
        return {
            "session_id": session_id
            if isinstance(session_id, str) and session_id
            else None,
            "model": model,
            "final_message": final_message,
            "usage": usage,
            "error": None,
        }

    @staticmethod
    def _metric(step: dict[str, object], name: str) -> int:
        metrics = step.get("metrics")
        value = metrics.get(name) if isinstance(metrics, dict) else None
        if isinstance(value, bool) or not isinstance(value, int):
            return 0
        return max(value, 0)

    @staticmethod
    def _invalid_export(error: str) -> dict[str, object]:
        return {
            "session_id": None,
            "model": None,
            "final_message": None,
            "usage": TokenUsage(),
            "error": error,
        }


class OpenCodeAdapter:
    provider = "opencode"
    modes = frozenset({"agent"})
    DEFAULT_MODEL = "opencode/deepseek-v4-flash"

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("AOP_OPENCODE_BIN", "opencode")

    def normalize_options(
        self, model: str | None, effort: str | None
    ) -> tuple[str | None, str | None]:
        selected = model or self.DEFAULT_MODEL
        if "/" not in selected:
            selected = f"opencode/{selected}"
        return selected, effort

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
            prompt=None,
            timeout_seconds=request.timeout_seconds,
            is_response=self._is_response,
        )
        _atomic_write(run_dir / "events.jsonl", capture.stdout)
        _atomic_write(run_dir / "stderr.log", capture.stderr)
        parsed = self._parse(capture.stdout, request.model)
        final_message = parsed["final_message"]
        if final_message is not None:
            _atomic_write(run_dir / "last-message.txt", final_message)
        reported_session_id = parsed["session_id"]
        session_id = reported_session_id
        resume_error = None
        if request.session_id and reported_session_id != request.session_id:
            session_id = None
            if reported_session_id is None:
                resume_error = (
                    "OpenCode resume result did not report the requested session ID "
                    f"{request.session_id}"
                )
            else:
                resume_error = (
                    f"OpenCode resumed as session {reported_session_id} instead of "
                    f"the requested session {request.session_id}"
                )

        if capture.timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif capture.exit_code:
            error = (
                capture.stderr.strip()
                or parsed["error"]
                or f"OpenCode exited with status {capture.exit_code}"
            )
        elif resume_error:
            error = resume_error
        elif parsed["error"]:
            error = parsed["error"]
        elif not parsed["has_finish"]:
            error = "OpenCode did not emit a terminal step_finish event"
        elif reported_session_id is None:
            error = "OpenCode result did not report a session ID"
        elif final_message is None:
            error = "OpenCode did not emit a final response"
        else:
            error = None
        billing = self._billing_provenance(request.model, environment, parsed["cost"])

        return RunResult(
            run_id=request.run_id,
            provider=self.provider,
            mode=request.mode,
            task=request.task,
            model=request.model,
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
            api_equivalent_cost=parsed["cost"],
            billing=billing,
            provider_duration_seconds=parsed["duration_seconds"],
        )

    @staticmethod
    def _billing_provenance(
        model: str | None,
        environment: dict[str, str],
        cost: object,
    ) -> BillingProvenance:
        provider = model.partition("/")[0] if model else ""
        route = "unknown"
        credential_source = None
        detected_by = None
        if provider == "opencode":
            route = "provider-credits"
            credential_source = "opencode-api-key"
            detected_by = "OpenCode model provider"
        elif provider in {"ollama", "lmstudio"}:
            route = "local"
            credential_source = provider
            detected_by = "OpenCode model provider"
        else:
            source_data = _opencode_source_data(environment)
            auth_path = source_data / "auth.json" if source_data else None
            try:
                auth = json.loads(auth_path.read_text()) if auth_path else None
            except (OSError, json.JSONDecodeError):
                auth = None
            entry = auth.get(provider) if isinstance(auth, dict) else None
            auth_type = entry.get("type") if isinstance(entry, dict) else None
            if auth_type == "oauth" and provider in {"openai", "xai"}:
                route = "subscription"
                credential_source = f"{provider}-oauth"
                detected_by = "OpenCode authentication metadata"
            elif auth_type in {"api", "api-key"}:
                route = "metered-api"
                credential_source = f"{provider}-api-key"
                detected_by = "OpenCode authentication metadata"
        actual_cost_known = (
            isinstance(cost, EstimatedCost)
            and not cost.estimated
            and route in {"provider-credits", "metered-api"}
        )
        return BillingProvenance(
            route=route,
            credential_source=credential_source,
            detected_by=detected_by,
            actual_cost_known=actual_cost_known,
        )

    def _command(self, request: RunRequest, worktree: Worktree) -> list[str]:
        command = [
            self.binary,
            "run",
            "--format",
            "json",
            "--auto",
            "--dir",
            os.fspath(worktree.path),
        ]
        if request.model:
            command.extend(["--model", request.model])
        if request.effort:
            command.extend(["--variant", request.effort])
        if request.session_id:
            command.extend(["--session", request.session_id])
        command.extend(["--", request.prompt])
        return command

    @staticmethod
    def _is_response(line: str) -> bool:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(event, dict) or event.get("type") != "text":
            return False
        part = event.get("part")
        return isinstance(part, dict) and bool(part.get("text"))

    @staticmethod
    def _parse(output: str, model: str | None) -> dict[str, object]:
        session_id = None
        final_message = None
        current_message_parts: list[str] = []
        error = None
        has_finish = False
        input_tokens = 0
        cached_input_tokens = 0
        output_tokens = 0
        reasoning_output_tokens = 0
        cost_usd = 0.0
        first_timestamp = None
        last_timestamp = None

        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_session_id = event.get("sessionID")
            if isinstance(event_session_id, str) and event_session_id:
                session_id = event_session_id
            timestamp = event.get("timestamp")
            if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                first_timestamp = (
                    float(timestamp)
                    if first_timestamp is None
                    else min(first_timestamp, float(timestamp))
                )
                last_timestamp = (
                    float(timestamp)
                    if last_timestamp is None
                    else max(last_timestamp, float(timestamp))
                )
            event_type = event.get("type")
            part = event.get("part")
            if event_type == "step_start":
                current_message_parts = []
            elif event_type == "text" and isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    current_message_parts.append(text)
            elif event_type == "error":
                error = OpenCodeAdapter._error_message(event.get("error"))
            elif event_type == "step_finish" and isinstance(part, dict):
                has_finish = True
                final_message = "".join(current_message_parts) or None
                tokens = part.get("tokens")
                if isinstance(tokens, dict):
                    cache = tokens.get("cache")
                    cache = cache if isinstance(cache, dict) else {}
                    cache_read = OpenCodeAdapter._integer(cache.get("read"))
                    cache_write = OpenCodeAdapter._integer(cache.get("write"))
                    input_tokens += (
                        OpenCodeAdapter._integer(tokens.get("input"))
                        + cache_read
                        + cache_write
                    )
                    cached_input_tokens += cache_read
                    output_tokens += OpenCodeAdapter._integer(tokens.get("output"))
                    reasoning_output_tokens += OpenCodeAdapter._integer(
                        tokens.get("reasoning")
                    )
                amount = part.get("cost")
                if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                    cost_usd += max(float(amount), 0.0)

        usage = TokenUsage(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
        )
        cost = None
        if has_finish:
            priced_as = model or "opencode"
            cost = EstimatedCost(
                amount_usd=round(cost_usd, 8),
                currency="USD",
                estimated=False,
                model=priced_as,
                priced_as=priced_as,
                pricing_version="opencode-cli-reported",
                pricing_source="OpenCode step_finish events",
                long_context_pricing=False,
            )
        duration_seconds = None
        if first_timestamp is not None and last_timestamp is not None:
            duration_seconds = round((last_timestamp - first_timestamp) / 1000, 6)
        return {
            "session_id": session_id,
            "final_message": final_message,
            "error": error,
            "has_finish": has_finish,
            "usage": usage,
            "cost": cost,
            "duration_seconds": duration_seconds,
        }

    @staticmethod
    def _error_message(value: object) -> str:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            data = value.get("data")
            for candidate in (
                value.get("message"),
                data.get("message") if isinstance(data, dict) else None,
                value.get("name"),
            ):
                if isinstance(candidate, str) and candidate:
                    return candidate
            return json.dumps(value, sort_keys=True)
        return "OpenCode turn failed"

    @staticmethod
    def _integer(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            return 0
        return max(value, 0)


class AgyAdapter:
    provider = "agy"
    modes = frozenset({"agent"})
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
        gemini_dir = Path(environment["AOP_PROVIDER_STATE_DIR"]) / "agy" / "gemini"
        command = _provider_command(
            self._command(request, gemini_dir), request, worktree, environment
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
        reported_session_id = parsed["session_id"]
        session_id = reported_session_id
        if capture.timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif capture.exit_code:
            error = (
                capture.stderr.strip() or f"agy exited with status {capture.exit_code}"
            )
        elif not parsed["has_result"]:
            error = "agy did not emit a terminal result"
        elif request.session_id and reported_session_id != request.session_id:
            session_id = None
            if reported_session_id is None:
                error = (
                    "agy resume result did not report the requested conversation ID "
                    f"{request.session_id}"
                )
            else:
                error = (
                    f"agy resumed as conversation {reported_session_id} instead of "
                    f"the requested conversation {request.session_id}"
                )
        elif reported_session_id is None:
            error = "agy result did not report a conversation ID"
        elif parsed["error"]:
            error = parsed["error"]
        else:
            error = None
        model = parsed["model"] or request.model
        return RunResult(
            run_id=request.run_id,
            provider=self.provider,
            mode=request.mode,
            task=request.task,
            model=model,
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
            api_equivalent_cost=estimate_api_cost(
                model,
                parsed["usage"],
                providers=("google",),
                additive_cached_input=True,
                catalog_model=_agy_catalog_model(model),
            ),
            billing=BillingProvenance(
                route="subscription",
                credential_source="google-oauth",
                detected_by="Antigravity authenticated profile",
            ),
            provider_duration_seconds=parsed["duration_seconds"],
        )

    def _command(self, request: RunRequest, gemini_dir: Path) -> list[str]:
        command = [
            self.binary,
            "--gemini_dir",
            os.fspath(gemini_dir),
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
        result_session_id = None
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
            if event_type == "init" and isinstance(payload.get("model"), str):
                model = payload["model"]
            if event_type != "result":
                continue
            has_result = True
            conversation_id = payload.get("conversation_id")
            if isinstance(conversation_id, str) and conversation_id:
                result_session_id = conversation_id
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
            "session_id": result_session_id,
            "model": model,
            "final_message": final_message,
            "error": error,
            "duration_seconds": duration_seconds,
            "usage": usage,
            "has_result": has_result,
        }


def _agy_catalog_model(model: str | None) -> str | None:
    if model is None:
        return None
    for suffix in ("-low", "-medium", "-high"):
        if model.endswith(suffix):
            return model.removesuffix(suffix)
    return model


@dataclass(frozen=True)
class _HermesSession:
    model: str | None
    billing_provider: str | None
    final_message: str | None
    last_assistant_id: str | None
    usage: TokenUsage
    cost_usd: float | None
    cost_estimated: bool
    cost_source: str
    pricing_version: str


class HermesAdapter:
    provider = "hermes"
    modes = frozenset({"agent", "participant"})
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
        with _hermes_runtime_environment(environment) as runtime_environment:
            return self._execute(request, worktree, run_dir, runtime_environment)

    def _execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        run_dir: Path,
        environment: dict[str, str],
    ) -> RunResult:
        before = self._session(request.session_id, request, worktree, environment)
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
        after = self._session(session_id, request, worktree, environment)
        final_message = self._final_message(before, after)
        if final_message is not None:
            _atomic_write(run_dir / "last-message.txt", final_message)
        baseline = before if request.session_id == session_id else None
        usage = self._usage_delta(baseline, after)
        cost = self._cost_delta(baseline, after, request.model, usage)
        billing = self._billing_provenance(after, cost)

        if capture.timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif capture.exit_code:
            error = self._exit_error(
                capture.stdout, capture.stderr, capture.exit_code
            )
        elif session_id is None:
            error = "Hermes did not report a session ID"
        elif final_message is None:
            error = "Hermes did not emit a final response"
        else:
            error = None
        return RunResult(
            run_id=request.run_id,
            provider=self.provider,
            mode=request.mode,
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
            billing=billing,
        )

    @staticmethod
    def _billing_provenance(
        session: _HermesSession | None, cost: EstimatedCost | None
    ) -> BillingProvenance:
        provider = session.billing_provider if session else None
        routes = {
            "nous": ("subscription", "nous-oauth"),
            "xai-oauth": ("subscription", "xai-oauth"),
            "openai-codex": ("subscription", "chatgpt-oauth"),
            "copilot": ("subscription", "github-copilot"),
            "copilot-acp": ("subscription", "github-copilot"),
            "ollama": ("local", "ollama"),
            "vllm": ("local", "vllm"),
            "openai": ("metered-api", "openai-api-key"),
            "xai": ("metered-api", "xai-api-key"),
            "openrouter": ("metered-api", "openrouter-api-key"),
        }
        route, credential_source = routes.get(provider, ("unknown", None))
        return BillingProvenance(
            route=route,
            credential_source=credential_source,
            detected_by="Hermes session billing provider" if provider else None,
            actual_cost_known=(
                cost is not None
                and session is not None
                and session.cost_usd is not None
                and not session.cost_estimated
                and route in {"metered-api", "provider-credits"}
            ),
        )

    def _command(self, request: RunRequest) -> list[str]:
        command = [self.binary, "chat", "-Q"]
        if request.mode == "participant":
            command.extend(
                [
                    "--safe-mode",
                    "--toolsets",
                    "__aop_no_tools__",
                    "--max-turns",
                    "1",
                ]
            )
        else:
            command.extend(["--yolo", "--accept-hooks"])
        command.extend(["--source", "tool"])
        if request.session_id:
            command.extend(["--resume", request.session_id, "--no-restore-cwd"])
        if request.inference_provider:
            command.extend(["--provider", request.inference_provider])
        if request.model:
            command.extend(["--model", request.model])
        if request.effort:
            command.extend(["--reasoning", request.effort])
        command.extend(["-q", request.prompt])
        return command

    def _session(
        self,
        session_id: str | None,
        request: RunRequest,
        worktree: Worktree,
        environment: dict[str, str],
    ) -> _HermesSession | None:
        if session_id is None:
            return None
        try:
            command = _provider_command(
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
                request,
                worktree,
                environment,
            )
            exported = subprocess.run(
                command,
                cwd=worktree.path,
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
        billing_provider = raw.get("billing_provider")
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
            billing_provider=(
                billing_provider if isinstance(billing_provider, str) else None
            ),
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
    ) -> str | None:
        if after is None or after.final_message is None:
            return None
        if before is None:
            return after.final_message
        if before.last_assistant_id and after.last_assistant_id:
            is_new = before.last_assistant_id != after.last_assistant_id
        else:
            is_new = before.final_message != after.final_message
        return after.final_message if is_new else None

    @staticmethod
    def _session_id(stderr: str) -> str | None:
        for line in reversed(stderr.splitlines()):
            key, separator, value = line.strip().partition(":")
            if separator and key == "session_id" and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _exit_error(stdout: str, stderr: str, exit_code: int) -> str:
        provider_error_prefixes = (
            "API call failed",
            "Authentication failed",
            "Error:",
            "No access token",
            "✗",
        )
        for line in reversed(stdout.splitlines()):
            detail = line.strip()
            if detail.startswith(provider_error_prefixes):
                return detail
        detail = "\n".join(
            line
            for line in stderr.splitlines()
            if line.strip()
            and not line.strip().startswith(("session_id:", "↻ Resumed session "))
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
        usage: TokenUsage,
    ) -> EstimatedCost | None:
        if after is None:
            return None
        model = after.model or requested_model or "hermes"
        if after.cost_source == "none":
            provider = HermesAdapter._catalog_provider(after.billing_provider)
            if provider is None or usage.total_tokens == 0:
                return None
            return estimate_api_cost(model, usage, providers=(provider,))
        if after.cost_usd is None:
            return None
        previous = before.cost_usd if before and before.cost_usd is not None else 0.0
        amount = max(after.cost_usd - previous, 0.0)
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
    def _catalog_provider(provider: str | None) -> str | None:
        return {
            "anthropic": "anthropic",
            "gemini": "google",
            "google": "google",
            "openai": "openai",
            "openai-codex": "openai",
            "xai": "xai",
            "xai-oauth": "xai",
        }.get(provider or "")

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
    provider_state = Path(environment["AOP_PROVIDER_STATE_DIR"])
    bwrap = os.environ.get("AOP_BWRAP_BIN", "bwrap")
    scratch = Path(environment["AOP_SCRATCH_DIR"])
    wrapped = [
        bwrap,
        "--die-with-parent",
        "--dev-bind",
        "/",
        "/",
    ]
    hidden_paths = environment.get("AOP_HIDE_PATHS", "")
    for raw_path in hidden_paths.split(os.pathsep):
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise AOPError(f"AOP hidden path must be an existing absolute directory: {path}")
        resolved = path.resolve(strict=True)
        if resolved in {Path("/"), root, worktree.path, cache, provider_state, scratch}:
            raise AOPError(f"AOP refuses to hide a required runtime path: {resolved}")
        wrapped.extend(["--tmpfs", os.fspath(resolved)])
    wrapped.extend(["--ro-bind", os.fspath(root), os.fspath(root)])
    if request.sandbox == "workspace-write":
        wrapped.extend(["--bind", os.fspath(worktree.path), os.fspath(worktree.path)])
        git_marker = worktree.path / ".git"
        if git_marker.exists():
            wrapped.extend(["--ro-bind", os.fspath(git_marker), os.fspath(git_marker)])
        common_git_directory = _git_common_directory(worktree.path)
        wrapped.extend(
            [
                "--ro-bind",
                os.fspath(common_git_directory),
                os.fspath(common_git_directory),
            ]
        )
    else:
        wrapped.extend(["--bind", os.fspath(scratch), os.fspath(scratch)])
    wrapped.extend(["--bind", os.fspath(cache), os.fspath(cache)])
    if request.provider in {
        "agy",
        "codex",
        "cursor",
        "devin",
        "hermes",
        "opencode",
    }:
        wrapped.extend(["--bind", os.fspath(provider_state), os.fspath(provider_state)])
    if request.provider == "agy":
        source_dir = _agy_source_dir(environment)
        wrapped.extend(["--ro-bind", os.fspath(source_dir), os.fspath(source_dir)])
    if request.provider == "cursor":
        source_home = _cursor_source_home(environment)
        source_auth = _cursor_source_auth(environment)
        isolated_state = provider_state / "cursor"
        _prepare_cursor_state(source_home, source_auth, isolated_state)
        cursor_cache = cache / "cursor"
        cursor_cache.mkdir(parents=True, exist_ok=True)
        wrapped.extend(
            [
                "--bind",
                os.fspath(isolated_state / "home"),
                os.fspath(source_home),
                "--setenv",
                "XDG_CONFIG_HOME",
                os.fspath(isolated_state / "config"),
                "--setenv",
                "XDG_CACHE_HOME",
                os.fspath(cursor_cache),
            ]
        )
    if request.provider == "hermes":
        isolated_home = Path(environment["HERMES_HOME"])
        if isolated_home != provider_state / "hermes" / "home":
            raise AOPError("Hermes runtime home is outside its task-private state")
    if request.provider == "opencode":
        source_config = _opencode_source_config(environment)
        source_data = _opencode_source_data(environment)
        home_config = Path(environment["HOME"]) / ".opencode"
        isolated_state = provider_state / "opencode"
        _prepare_opencode_state(source_config, source_data, isolated_state)
        opencode_cache = cache / "opencode"
        opencode_cache.mkdir(parents=True, exist_ok=True)
        wrapped.extend(
            [
                "--setenv",
                "XDG_CONFIG_HOME",
                os.fspath(isolated_state / "config"),
                "--setenv",
                "XDG_DATA_HOME",
                os.fspath(isolated_state / "data"),
                "--setenv",
                "XDG_STATE_HOME",
                os.fspath(isolated_state / "state"),
                "--setenv",
                "XDG_CACHE_HOME",
                os.fspath(cache),
                "--setenv",
                "OPENCODE_DISABLE_AUTOUPDATE",
                "1",
            ]
        )
        dependencies = source_config / "node_modules" if source_config else None
        private_dependencies = isolated_state / "config" / "opencode" / "node_modules"
        if dependencies is not None and dependencies.is_dir():
            wrapped.extend(
                [
                    "--ro-bind",
                    os.fspath(dependencies),
                    os.fspath(private_dependencies),
                ]
            )
        if home_config.is_dir():
            wrapped.extend(
                ["--ro-bind", os.fspath(home_config), os.fspath(home_config)]
            )
    for read_path in request.read_paths:
        wrapped.extend(
            ["--ro-bind", read_path.source_path, read_path.source_path]
        )
        wrapped.extend(
            ["--ro-bind", read_path.source_path, read_path.mounted_path]
        )
    wrapped.extend(["--chdir", os.fspath(worktree.path), "--", *command])
    return wrapped


def _git_common_directory(worktree: Path) -> Path:
    result = subprocess.run(
        (
            "git",
            "-C",
            os.fspath(worktree),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "git rev-parse failed"
        raise AOPError(f"Could not resolve candidate Git metadata: {detail}")
    path = Path(result.stdout.strip())
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise AOPError("Candidate Git common directory is invalid")
    return path.resolve(strict=True)


_AGY_SEED_DIRECTORIES = {"config"}
_AGY_NESTED_SEED_FILES = {
    "antigravity": {
        "browserAllowlist.txt",
        "mcp_config.json",
        "user_settings.pb",
    },
    "antigravity-cli": {
        "antigravity-oauth-token",
        "settings.json",
    },
}

_CODEX_SEED_DIRECTORIES = {"rules", "skills"}
_CODEX_RUNTIME_FILES = {
    ".personality_migration",
    "history.jsonl",
    "installation_id",
    "log",
    "session_index.jsonl",
    "usage-limits.json",
    "version.json",
}

_CURSOR_RUNTIME_DIRECTORIES = {
    "ai-tracking",
    "chats",
    "extensions",
    "projects",
    "worktrees",
}

_OPENCODE_CONFIG_RUNTIME_NAMES = {
    ".gitignore",
    "bun.lock",
    "node_modules",
    "package-lock.json",
    "package.json",
}
_OPENCODE_DATA_SEED_FILES = {"auth.json", "mcp-auth.json"}
_DEVIN_DATA_RUNTIME_NAMES = {
    "cli",
    "sessions.db",
    "sessions.db-shm",
    "sessions.db-wal",
}


def _prepare_codex_environment(environment: dict[str, str]) -> None:
    source = _codex_source_home(environment)
    destination = Path(environment["AOP_PROVIDER_STATE_DIR"]) / "codex" / "home"
    _prepare_codex_state(source, destination)
    environment["CODEX_HOME"] = os.fspath(destination)


def _codex_source_home(environment: dict[str, str]) -> Path | None:
    configured = environment.get("AOP_CODEX_SOURCE_HOME")
    inherited = environment.get("CODEX_HOME")
    source = Path(configured or inherited or Path(environment["HOME"]) / ".codex")
    source = source.expanduser()
    if source.is_dir():
        return source.resolve()
    if configured or inherited:
        raise AOPError(f"Codex home does not exist: {source}")
    return None


def _prepare_codex_state(source: Path | None, destination: Path) -> None:
    if destination.is_dir():
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, mode=0o700)
    try:
        if source:
            for entry in source.iterdir():
                target = temporary / entry.name
                if entry.name in _CODEX_SEED_DIRECTORIES and entry.is_dir():
                    ignore = (
                        shutil.ignore_patterns(".system")
                        if entry.name == "skills"
                        else None
                    )
                    shutil.copytree(entry, target, ignore=ignore)
                elif entry.is_file() and not _codex_runtime_file(entry.name):
                    shutil.copy2(entry, target)
        _make_tree_user_writable(temporary)
        os.replace(temporary, destination)
    except OSError as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise AOPError(f"could not prepare isolated Codex state: {error}") from error


def _codex_runtime_file(name: str) -> bool:
    return name in _CODEX_RUNTIME_FILES or ".sqlite" in name


def _cursor_source_home(environment: dict[str, str]) -> Path:
    configured = environment.get("AOP_CURSOR_HOME")
    source = (
        Path(configured).expanduser()
        if configured
        else Path(environment["HOME"]) / ".cursor"
    )
    try:
        return source.resolve(strict=True)
    except FileNotFoundError as error:
        raise AOPError(
            f"Cursor home does not exist: {source}; authenticate Cursor Agent first"
        ) from error


def _cursor_source_auth(environment: dict[str, str]) -> Path | None:
    configured = environment.get("AOP_CURSOR_CONFIG_DIR")
    source = (
        Path(configured).expanduser()
        if configured
        else Path(environment["HOME"]) / ".config" / "cursor"
    )
    try:
        return source.resolve(strict=True)
    except FileNotFoundError as error:
        if environment.get("CURSOR_API_KEY"):
            return None
        raise AOPError(
            f"Cursor authentication does not exist: {source}; authenticate Cursor Agent first"
        ) from error


def _prepare_cursor_state(
    source_home: Path, source_auth: Path | None, destination: Path
) -> None:
    if destination.is_dir():
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    private_home = temporary / "home"
    private_config = temporary / "config" / "cursor"
    private_home.mkdir(parents=True, mode=0o700)
    try:
        for entry in source_home.iterdir():
            if entry.name in _CURSOR_RUNTIME_DIRECTORIES:
                continue
            target = private_home / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, symlinks=True)
            elif entry.is_file():
                shutil.copy2(entry, target)
        for name in _CURSOR_RUNTIME_DIRECTORIES:
            if name != "extensions":
                private_home.joinpath(name).mkdir()
        if source_auth is None:
            private_config.mkdir(parents=True)
        else:
            shutil.copytree(source_auth, private_config, symlinks=True)
        _make_tree_user_writable(temporary)
        os.replace(temporary, destination)
    except OSError as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise AOPError(f"could not prepare isolated Cursor state: {error}") from error


def _prepare_devin_environment(environment: dict[str, str]) -> None:
    source_data = _devin_source_data(environment)
    source_config = _devin_source_config(environment)
    destination = Path(environment["AOP_PROVIDER_STATE_DIR"]) / "devin"
    _prepare_devin_state(source_data, source_config, destination)
    cache = Path(environment["AOP_CACHE_DIR"]) / "devin"
    cache.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "XDG_DATA_HOME": os.fspath(destination / "data"),
            "XDG_CONFIG_HOME": os.fspath(destination / "config"),
            "XDG_STATE_HOME": os.fspath(destination / "state"),
            "XDG_CACHE_HOME": os.fspath(cache),
        }
    )


def _devin_source_data(environment: dict[str, str]) -> Path:
    configured = environment.get("AOP_DEVIN_DATA_DIR")
    base = Path(
        environment.get("XDG_DATA_HOME", Path(environment["HOME"]) / ".local" / "share")
    )
    source = Path(configured).expanduser() if configured else base / "devin"
    if not (source / "credentials.toml").is_file():
        raise AOPError(
            f"Devin authentication does not exist: {source}; authenticate Devin first"
        )
    return source.resolve()


def _devin_source_config(environment: dict[str, str]) -> Path | None:
    configured = environment.get("AOP_DEVIN_CONFIG_DIR")
    base = Path(
        environment.get("XDG_CONFIG_HOME", Path(environment["HOME"]) / ".config")
    )
    source = Path(configured).expanduser() if configured else base / "devin"
    if source.is_dir():
        return source.resolve()
    if configured:
        raise AOPError(f"Devin config directory does not exist: {source}")
    return None


def _prepare_devin_state(
    source_data: Path, source_config: Path | None, destination: Path
) -> None:
    if destination.is_dir():
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    private_data = temporary / "data" / "devin"
    private_config = temporary / "config" / "devin"
    private_state = temporary / "state"
    private_data.mkdir(parents=True, mode=0o700)
    private_state.mkdir(parents=True, mode=0o700)
    try:
        for entry in source_data.iterdir():
            if entry.name in _DEVIN_DATA_RUNTIME_NAMES:
                continue
            target = private_data / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, symlinks=True)
            elif entry.is_file():
                shutil.copy2(entry, target)
        if source_config is None:
            private_config.mkdir(parents=True)
        else:
            shutil.copytree(source_config, private_config, symlinks=True)
        _make_tree_user_writable(temporary)
        os.replace(temporary, destination)
    except OSError as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise AOPError(f"could not prepare isolated Devin state: {error}") from error


def _make_tree_user_writable(root: Path) -> None:
    for directory, names, files in os.walk(root):
        path = Path(directory)
        path.chmod(path.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        for name in [*names, *files]:
            entry = path / name
            if entry.is_symlink():
                continue
            permissions = stat.S_IRUSR | stat.S_IWUSR
            if entry.is_dir():
                permissions |= stat.S_IXUSR
            entry.chmod(entry.stat().st_mode | permissions)


def _opencode_source_config(environment: dict[str, str]) -> Path | None:
    configured = environment.get("AOP_OPENCODE_CONFIG_DIR")
    base = Path(environment.get("XDG_CONFIG_HOME", Path(environment["HOME"]) / ".config"))
    source = Path(configured).expanduser() if configured else base / "opencode"
    if source.is_dir():
        return source.resolve()
    if configured:
        raise AOPError(f"OpenCode config directory does not exist: {source}")
    return None


def _opencode_source_data(environment: dict[str, str]) -> Path | None:
    configured = environment.get("AOP_OPENCODE_DATA_DIR")
    base = Path(
        environment.get(
            "XDG_DATA_HOME", Path(environment["HOME"]) / ".local" / "share"
        )
    )
    source = Path(configured).expanduser() if configured else base / "opencode"
    if source.is_dir():
        return source.resolve()
    if configured:
        raise AOPError(f"OpenCode data directory does not exist: {source}")
    return None


def _prepare_opencode_state(
    source_config: Path | None, source_data: Path | None, destination: Path
) -> None:
    if destination.is_dir():
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    private_config = temporary / "config" / "opencode"
    private_data = temporary / "data" / "opencode"
    private_state = temporary / "state"
    private_config.mkdir(parents=True, mode=0o700)
    private_data.mkdir(parents=True, mode=0o700)
    private_state.mkdir(parents=True, mode=0o700)
    try:
        if source_config:
            for entry in source_config.iterdir():
                if entry.name in _OPENCODE_CONFIG_RUNTIME_NAMES:
                    continue
                target = private_config / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, target, symlinks=True)
                elif entry.is_file():
                    shutil.copy2(entry, target)
        if source_data:
            for name in _OPENCODE_DATA_SEED_FILES:
                source = source_data / name
                if source.is_file():
                    shutil.copy2(source, private_data / name)
        (private_config / "node_modules").mkdir()
        _make_tree_user_writable(temporary)
        os.replace(temporary, destination)
    except OSError as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise AOPError(f"could not prepare isolated OpenCode state: {error}") from error


def _agy_source_dir(environment: dict[str, str]) -> Path:
    configured = environment.get("AOP_AGY_SOURCE_DIR")
    source = (
        Path(configured).expanduser()
        if configured
        else Path(environment["HOME"]) / ".gemini"
    )
    try:
        return source.resolve(strict=True)
    except FileNotFoundError as error:
        raise AOPError(
            f"Agy profile does not exist: {source}; authenticate Agy first"
        ) from error


def _prepare_agy_dir(source: Path, destination: Path) -> None:
    if destination.is_dir():
        return
    if not source.is_dir():
        raise AOPError(f"Agy profile is not a directory: {source}")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, mode=0o700)
    try:
        for entry in source.iterdir():
            target = temporary / entry.name
            if entry.name in _AGY_SEED_DIRECTORIES and entry.is_dir():
                shutil.copytree(entry, target)
            elif entry.is_file():
                shutil.copy2(entry, target)
        for directory, names in _AGY_NESTED_SEED_FILES.items():
            source_directory = source / directory
            for name in names:
                entry = source_directory / name
                if not entry.is_file():
                    continue
                target = temporary / directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entry, target)
        os.replace(temporary, destination)
    except OSError as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise AOPError(f"could not prepare isolated Agy profile: {error}") from error


_HERMES_SEED_DIRECTORIES = {
    "cron",
    "hooks",
    "local",
    "memories",
    "optional-mcps",
    "optional-skills",
    "plugins",
    "shared",
    "skills",
}
_HERMES_RUNTIME_FILES = {
    ".hermes_history",
    ".update_check",
    "active_profile",
    "auth.lock",
    "gateway.pid",
    "gateway_state.json",
    "hermes_state.db",
    "hermes_state.db-shm",
    "hermes_state.db-wal",
    "processes.json",
    "response_store.db",
    "response_store.db-shm",
    "response_store.db-wal",
    "state.db",
    "state.db-shm",
    "state.db-wal",
}


def _hermes_source_home(environment: dict[str, str]) -> Path:
    configured = environment.get("HERMES_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    default = Path(environment["HOME"]) / ".hermes"
    active_profile = default / "active_profile"
    try:
        active = active_profile.read_text().strip()
    except OSError:
        active = ""
    if active and active != "default":
        profile = default / "profiles" / active
        if profile.is_dir():
            return profile
    return default


def _prepare_hermes_home(source: Path, destination: Path) -> None:
    if destination.is_dir():
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, mode=0o700)
    try:
        for entry in source.iterdir():
            target = temporary / entry.name
            if entry.name in _HERMES_SEED_DIRECTORIES and entry.is_dir():
                shutil.copytree(entry, target, symlinks=True)
            elif entry.is_file() and entry.name not in _HERMES_RUNTIME_FILES:
                shutil.copy2(entry, target)
        os.replace(temporary, destination)
    except OSError as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise AOPError(f"could not prepare isolated Hermes home: {error}") from error


def _parse_hermes_timestamp(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = float(value)
        return timestamp / 1000 if timestamp > 10_000_000_000 else timestamp
    if not isinstance(value, str) or not value.strip():
        return 0.0
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _hermes_value_freshness(
    value: object, store_updated_at: float
) -> tuple[float, float]:
    # A failed request can update the store after copying an already-consumed
    # refresh token. Token lifecycle timestamps must therefore win first.
    refresh_times: list[float] = []
    obtained_times: list[float] = []

    def visit(item: object) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if key == "last_refresh":
                    refresh_times.append(_parse_hermes_timestamp(nested))
                elif key in {"obtained_at", "agent_key_obtained_at"}:
                    obtained_times.append(_parse_hermes_timestamp(nested))
                elif isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    lifecycle_time = max(refresh_times or obtained_times or [0.0])
    return lifecycle_time, store_updated_at


def _hermes_entry_identity(entry: dict[str, object]) -> tuple[str, str, str]:
    identifier = entry.get("id")
    if isinstance(identifier, str) and identifier:
        return "id", identifier, ""
    source = entry.get("source")
    label = entry.get("label")
    return "source-label", str(source or ""), str(label or "")


def _merge_hermes_auth_stores(
    stores: Sequence[dict[str, object]],
) -> dict[str, object] | None:
    if not stores:
        return None
    ordered = sorted(
        stores,
        key=lambda store: _parse_hermes_timestamp(store.get("updated_at")),
    )
    merged: dict[str, object] = {}
    provider_freshness: dict[str, tuple[float, float]] = {}
    entry_freshness: dict[tuple[str, tuple[str, str, str]], tuple[float, float]] = {}

    for store in ordered:
        store_updated_at = _parse_hermes_timestamp(store.get("updated_at"))
        for key, value in store.items():
            if key not in {"providers", "credential_pool"}:
                merged[key] = value

        providers = store.get("providers")
        if isinstance(providers, dict):
            merged_providers = merged.setdefault("providers", {})
            if isinstance(merged_providers, dict):
                for provider, state in providers.items():
                    freshness = _hermes_value_freshness(state, store_updated_at)
                    if freshness >= provider_freshness.get(provider, (0.0, 0.0)):
                        merged_providers[provider] = state
                        provider_freshness[provider] = freshness

        pool = store.get("credential_pool")
        if not isinstance(pool, dict):
            continue
        merged_pool = merged.setdefault("credential_pool", {})
        if not isinstance(merged_pool, dict):
            continue
        for provider, entries in pool.items():
            if not isinstance(entries, list):
                continue
            merged_entries = merged_pool.setdefault(provider, [])
            if not isinstance(merged_entries, list):
                continue
            positions = {
                _hermes_entry_identity(entry): index
                for index, entry in enumerate(merged_entries)
                if isinstance(entry, dict)
            }
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                identity = _hermes_entry_identity(entry)
                key = provider, identity
                freshness = _hermes_value_freshness(entry, store_updated_at)
                position = positions.get(identity)
                if position is None:
                    positions[identity] = len(merged_entries)
                    merged_entries.append(entry)
                    entry_freshness[key] = freshness
                elif freshness >= entry_freshness.get(key, (0.0, 0.0)):
                    merged_entries[position] = entry
                    entry_freshness[key] = freshness
    return merged


def _load_hermes_auth(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_hermes_auth(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(f"{json.dumps(value, indent=2, sort_keys=True)}\n")
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _reconcile_hermes_credentials(
    source_home: Path,
    task_home: Path,
    state_dir: Path,
) -> Path:
    shared_auth = state_dir / "shared-provider-state" / "hermes" / "auth.json"
    candidates = [source_home / "auth.json", shared_auth]
    # Recover a rotation even if AOP was interrupted before committing it.
    candidates.extend(state_dir.glob("provider-state/*/hermes/home/auth.json"))
    stores = []
    for path in candidates:
        store = _load_hermes_auth(path)
        if store is not None:
            stores.append(store)
    merged = _merge_hermes_auth_stores(stores)
    if merged is not None:
        _write_hermes_auth(shared_auth, merged)
        _write_hermes_auth(task_home / "auth.json", merged)
    return shared_auth


@contextmanager
def _hermes_runtime_environment(
    environment: dict[str, str],
) -> Iterator[dict[str, str]]:
    source_home = _hermes_source_home(environment)
    if not source_home.is_dir():
        raise AOPError(
            f"Hermes home does not exist: {source_home}; authenticate Hermes first"
        )
    provider_state = Path(environment["AOP_PROVIDER_STATE_DIR"])
    task_home = provider_state / "hermes" / "home"
    _prepare_hermes_home(source_home, task_home)
    state_dir = provider_state.parent.parent
    lock_path = state_dir / "locks" / "hermes-credentials.lock"
    runtime_environment = environment.copy()
    runtime_environment["HERMES_HOME"] = os.fspath(task_home)

    # Hermes has no separate auth path. Hold the lock for the complete turn so
    # two task-private homes cannot consume the same single-use refresh token.
    with exclusive_lock(lock_path, "Hermes credential state", blocking=True):
        shared_auth = _reconcile_hermes_credentials(source_home, task_home, state_dir)
        try:
            yield runtime_environment
        finally:
            task_auth = _load_hermes_auth(task_home / "auth.json")
            if task_auth is not None:
                _write_hermes_auth(shared_auth, task_auth)


def adapter_for(agent: str) -> AgentAdapter:
    if agent == "codex":
        return CodexAdapter()
    if agent == "claude":
        return ClaudeAdapter()
    if agent == "cursor":
        return CursorAdapter()
    if agent == "devin":
        return DevinAdapter()
    if agent == "opencode":
        return OpenCodeAdapter()
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
        inference_provider: str | None = None,
        effort: str | None = None,
        mode: str = "agent",
        sandbox: str = "workspace-write",
        timeout_seconds: float | None = None,
        artifacts: Sequence[str] = (),
        read_paths: Sequence[str | os.PathLike[str]] = (),
    ) -> RunResult:
        self._validate_mode(mode)
        self._validate_inference_provider(inference_provider, model)
        if read_paths and sandbox == "danger-full-access":
            raise AOPError("--read requires workspace-write or scratch-write")
        model, effort = self.adapter.normalize_options(model, effort)
        request = self._request(
            task=task,
            prompt=prompt,
            base=base,
            model=model,
            inference_provider=inference_provider,
            effort=effort,
            mode=mode,
            sandbox=sandbox,
            timeout_seconds=timeout_seconds,
            artifacts=artifacts,
        )
        worktree = self._get_or_create_worktree(task, base)
        return self._execute(request, worktree, read_paths=read_paths)

    def resume(
        self,
        *,
        run_id: str,
        prompt: str,
        timeout_seconds: float | None = None,
        artifacts: Sequence[str] = (),
        read_paths: Sequence[str | os.PathLike[str]] | None = None,
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
        self._validate_mode(parent_request.mode)
        if not parent_result.session_id:
            raise AOPError(f"run has no resumable agent session: {run_id}")
        _task_lock_held = _task_lock_held or (
            os.environ.get("AOP_TASK_LOCK_HELD") == parent_request.task
        )
        worktree = self.manager.get(parent_request.task)
        inherited_read_paths = tuple(
            item.source_path for item in parent_request.read_paths
        )
        selected_read_paths = (
            inherited_read_paths if read_paths is None else tuple(read_paths)
        )
        if selected_read_paths and parent_request.sandbox == "danger-full-access":
            raise AOPError("--read requires workspace-write or scratch-write")
        request = self._request(
            task=parent_request.task,
            prompt=prompt,
            base=parent_request.base,
            model=parent_request.model,
            inference_provider=parent_request.inference_provider,
            effort=parent_request.effort,
            mode=parent_request.mode,
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
        return self._execute(
            request,
            worktree,
            read_paths=selected_read_paths,
            task_lock_held=_task_lock_held,
        )

    def _execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        *,
        read_paths: Sequence[str | os.PathLike[str]],
        task_lock_held: bool = False,
    ) -> RunResult:
        if task_lock_held:
            return self._execute_unlocked(request, worktree, read_paths)
        with exclusive_lock(
            task_lock_path(self.manager.state_dir, request.task),
            f"task {request.task}",
        ):
            return self._execute_unlocked(request, worktree, read_paths)

    def _execute_unlocked(
        self,
        request: RunRequest,
        worktree: Worktree,
        read_paths: Sequence[str | os.PathLike[str]],
    ) -> RunResult:
        scratch_dir = worktree.path / "scratch"
        scratch_dir.mkdir(exist_ok=True)
        input_dir = scratch_dir / "inputs" / request.run_id
        input_dir.mkdir(parents=True, exist_ok=False)
        output_dir = scratch_dir / "outputs" / request.run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        provider_state = self.manager.state_dir / "provider-state" / request.task
        provider_state.mkdir(parents=True, exist_ok=True, mode=0o700)
        provider_state.chmod(0o700)
        declared_read_paths = _prepare_read_paths(read_paths, input_dir)
        request = replace(
            request,
            prompt=_run_prompt(
                request.prompt,
                declared_read_paths,
                request.artifacts,
                output_dir,
            ),
            read_paths=declared_read_paths,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "AOP_ROOT": os.fspath(self.manager.root),
                "AOP_TASK": request.task,
                "AOP_WORKTREE": os.fspath(worktree.path),
                "AOP_CACHE_DIR": os.fspath(self.manager.cache_dir),
                "AOP_PROVIDER_STATE_DIR": os.fspath(provider_state),
                "AOP_SCRATCH_DIR": os.fspath(scratch_dir),
                "AOP_INPUT_DIR": os.fspath(input_dir),
                "AOP_OUTPUT_DIR": os.fspath(output_dir),
                "AOP_RUN_ID": request.run_id,
            }
        )
        if request.provider == "hermes":
            environment.pop("HERMES_NO_TOOLS", None)
            if request.mode == "participant":
                environment.pop("HERMES_KANBAN_TASK", None)
        if request.provider == "agy":
            _prepare_agy_dir(
                _agy_source_dir(environment), provider_state / "agy" / "gemini"
            )
        run_dir = self.store.create(request)
        if request.read_paths:
            input_manifest = run_dir / "input-manifest.json"
            self.store.write_json(
                input_manifest,
                {"schema_version": 1, "read_paths": request.to_dict()["read_paths"]},
            )
            environment["AOP_INPUT_MANIFEST"] = os.fspath(input_manifest)
        result = self.adapter.execute(request, worktree, run_dir, environment)
        result = replace(
            result,
            inference_provider=request.inference_provider,
            read_paths=request.read_paths,
        )
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
        inference_provider: str | None,
        effort: str | None,
        mode: str,
        sandbox: str,
        timeout_seconds: float | None,
        session_id: str | None = None,
        parent_run_id: str | None = None,
        artifacts: Sequence[str] = (),
    ) -> RunRequest:
        return RunRequest(
            run_id=str(uuid.uuid4()),
            provider=self.adapter.provider,
            mode=mode,
            task=task,
            prompt=prompt,
            base=base,
            model=model,
            inference_provider=inference_provider,
            effort=effort,
            sandbox=sandbox,
            timeout_seconds=timeout_seconds,
            session_id=session_id,
            parent_run_id=parent_run_id,
            artifacts=normalize_artifacts(artifacts),
            read_paths=(),
            created_at=_now(),
        )

    def _validate_mode(self, mode: str) -> None:
        if mode not in {"agent", "participant"}:
            raise AOPError(f"unknown agent mode: {mode}")
        if mode not in self.adapter.modes:
            raise AOPError(f"{self.adapter.provider} does not support {mode} mode")

    def _validate_inference_provider(
        self, inference_provider: str | None, model: str | None
    ) -> None:
        if inference_provider is None:
            return
        if self.adapter.provider != "hermes":
            raise AOPError(
                f"{self.adapter.provider} does not support --provider overrides"
            )
        if model is None:
            raise AOPError("Hermes --provider requires an explicit --model")

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


def _prepare_read_paths(
    values: Sequence[str | os.PathLike[str]], input_dir: Path
) -> tuple[ReadPath, ...]:
    prepared: list[ReadPath] = []
    aliases: set[str] = set()
    for value in values:
        raw = os.fspath(value)
        if not raw or "\0" in raw:
            raise AOPError("read path must be a non-empty filesystem path")
        candidate = Path(raw).expanduser()
        try:
            initial_status = candidate.lstat()
            source = candidate.resolve(strict=True)
        except OSError as error:
            raise AOPError(f"could not inspect read path {candidate}: {error}") from error
        if stat.S_ISLNK(initial_status.st_mode):
            raise AOPError(f"read path may not be a symlink: {candidate}")
        alias = source.name
        if not alias:
            raise AOPError(f"read path must have a basename: {source}")
        if alias in aliases:
            raise AOPError(f"read paths have the same basename: {alias}")
        aliases.add(alias)

        kind, files = _inspect_read_path(source)
        mounted = input_dir / alias
        if kind == "directory":
            mounted.mkdir()
        else:
            mounted.touch()
        size_bytes = sum(item.size_bytes for item in files)
        prepared.append(
            ReadPath(
                source_path=os.fspath(source),
                mounted_path=os.fspath(mounted),
                kind=kind,
                size_bytes=size_bytes,
                sha256=_read_path_digest(kind, files),
                files=files,
            )
        )
    return tuple(prepared)


def _inspect_read_path(source: Path) -> tuple[str, tuple[ReadPathFile, ...]]:
    status = source.lstat()
    if stat.S_ISLNK(status.st_mode):
        raise AOPError(f"read path may not be a symlink: {source}")
    if stat.S_ISREG(status.st_mode):
        return "file", (_read_path_file(source, source.name),)
    if not stat.S_ISDIR(status.st_mode):
        raise AOPError(f"read path is not a regular file or directory: {source}")

    files: list[ReadPathFile] = []

    def collect(directory: Path, relative: PurePosixPath) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError as error:
            raise AOPError(f"could not read input directory {directory}: {error}") from error
        for child in children:
            logical = relative / child.name
            child_status = child.lstat()
            if stat.S_ISLNK(child_status.st_mode):
                raise AOPError(f"read path contains a symlink: {child}")
            if stat.S_ISDIR(child_status.st_mode):
                collect(child, logical)
            elif stat.S_ISREG(child_status.st_mode):
                files.append(_read_path_file(child, logical.as_posix()))
            else:
                raise AOPError(f"read path contains a special file: {child}")

    collect(source, PurePosixPath())
    return "directory", tuple(files)


def _read_path_file(source: Path, relative_path: str) -> ReadPathFile:
    try:
        size_bytes = source.stat().st_size
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as error:
        raise AOPError(f"could not hash read path file {source}: {error}") from error
    return ReadPathFile(
        relative_path=relative_path,
        size_bytes=size_bytes,
        sha256=digest,
    )


def _read_path_digest(kind: str, files: tuple[ReadPathFile, ...]) -> str:
    if kind == "file":
        return files[0].sha256
    digest = hashlib.sha256(b"aop-read-directory-v1\0")
    for item in files:
        path = item.relative_path.encode()
        digest.update(len(path).to_bytes(8, "big"))
        digest.update(path)
        digest.update(item.size_bytes.to_bytes(8, "big"))
        digest.update(bytes.fromhex(item.sha256))
    return digest.hexdigest()


def _run_prompt(
    prompt: str,
    read_paths: tuple[ReadPath, ...],
    artifacts: tuple[str, ...],
    output_dir: Path,
) -> str:
    if read_paths:
        paths = "\n".join(
            f"- Preferred: {json.dumps(item.mounted_path)}\n"
            f"  Original: {json.dumps(item.source_path)}\n"
            f"  SHA-256: {item.sha256} ({item.size_bytes} bytes, {item.kind})"
            for item in read_paths
        )
        prompt = (
            f"{prompt.rstrip()}\n\n"
            "AOP declared read-only paths:\n"
            f"{paths}\n"
            "Both locations are read-only. Prefer the task-local path when the provider "
            "limits file access to the workspace."
        )
    return _artifact_prompt(prompt, artifacts, output_dir)


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
