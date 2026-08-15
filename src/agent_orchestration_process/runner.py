"""Structured execution and persistence for coding-agent runs."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
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

import yaml
from yaml.constructor import ConstructorError

from .models import (
    BillingProvenance,
    InputFile,
    InputSnapshot,
    RunArtifact,
    RunRequest,
    RunResult,
    ProviderReportedCost,
)
from .locks import exclusive_lock, task_lock_path
from .isolation import PROFILES, resolve_policy
from .pricing import CalculatedCost, TokenUsage, estimate_api_cost
from .worktrees import AOPError, Worktree, WorktreeManager


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _token_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


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


def _atomic_write_private(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
                env=_launch_environment(command, environment),
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
            command=_recorded_command(command),
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
            calculated_cost=estimate_api_cost(request.model, usage),
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
            env=_launch_environment(command, environment),
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


def _launch_environment(
    command: list[str], environment: dict[str, str]
) -> dict[str, str]:
    if "--clearenv" not in command or "--unshare-pid" not in command:
        return environment
    selected = {
        key: environment[key]
        for key in ("LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ")
        if key in environment
    }
    selected["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    return selected


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
        _prepare_claude_environment(environment)
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
            command=_recorded_command(command),
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
            calculated_cost=parsed["cost"],
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
                cost = CalculatedCost(
                    amount_usd=round(float(amount), 8),
                    currency="USD",
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


class GrokAdapter:
    provider = "grok"
    modes = frozenset({"agent"})
    DEFAULT_MODEL = "grok-build"
    EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("AOP_GROK_BIN", "grok")

    def normalize_options(
        self, model: str | None, effort: str | None
    ) -> tuple[str | None, str | None]:
        if effort is not None and effort not in self.EFFORTS:
            raise AOPError(
                f"Grok effort must be one of: {', '.join(sorted(self.EFFORTS))}"
            )
        return model or self.DEFAULT_MODEL, effort

    def execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        run_dir: Path,
        environment: dict[str, str],
    ) -> RunResult:
        _prepare_grok_environment(environment)
        prompt_path = (
            Path(environment["AOP_SCRATCH_DIR"]) / f".grok-prompt-{request.run_id}.txt"
        )
        _atomic_write_private(prompt_path, request.prompt)
        command = _provider_command(
            self._command(request, worktree, prompt_path),
            request,
            worktree,
            environment,
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
                    "Grok resume result did not report the requested session ID "
                    f"{request.session_id}"
                )
            else:
                resume_error = (
                    f"Grok resumed as session {reported_session_id} instead of "
                    f"the requested session {request.session_id}"
                )

        if capture.timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif parsed["error"]:
            error = parsed["error"]
        elif capture.exit_code:
            error = (
                capture.stderr.strip() or f"Grok exited with status {capture.exit_code}"
            )
        elif resume_error:
            error = resume_error
        elif reported_session_id is None:
            error = "Grok did not report a session ID"
        elif not parsed["completed"]:
            error = "Grok did not emit a terminal end event"
        elif final_message is None:
            error = "Grok did not emit a final response"
        else:
            error = None

        model = parsed["model"] or request.model
        usage = parsed["usage"]
        billing = self._billing_provenance(environment)
        reported_cost = (
            parsed["reported_cost"]
            if billing.route in {"metered-api", "provider-credits"}
            else None
        )
        return RunResult(
            run_id=request.run_id,
            provider=self.provider,
            mode=request.mode,
            task=request.task,
            model=model,
            effort=request.effort,
            session_id=session_id,
            command=_recorded_command(command),
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
            calculated_cost=estimate_api_cost(model, usage, providers=("xai",)),
            provider_reported_cost=reported_cost,
            billing=billing,
        )

    def _command(
        self,
        request: RunRequest,
        worktree: Worktree,
        prompt_path: Path,
    ) -> list[str]:
        command = [
            self.binary,
            "--cwd",
            os.fspath(worktree.path),
            "--output-format",
            "streaming-json",
            "--always-approve",
            "--no-plan",
            "--no-leader",
            "--no-auto-update",
            "--verbatim",
            "--sandbox",
            "off",
            "--prompt-file",
            os.fspath(prompt_path),
        ]
        if request.model:
            command.extend(["--model", request.model])
        if request.effort:
            command.extend(["--reasoning-effort", request.effort])
        if request.session_id:
            command.extend(["--resume", request.session_id])
        return command

    @staticmethod
    def _is_response(line: str) -> bool:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        return (
            isinstance(event, dict)
            and event.get("type") == "text"
            and isinstance(event.get("data"), str)
            and bool(event["data"])
        )

    @staticmethod
    def _parse(output: str, requested_model: str | None) -> dict[str, object]:
        session_id = None
        final_message = None
        response_parts: list[str] = []
        error = None
        completed = False
        usage = TokenUsage()
        reported_cost = None
        model = None

        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "text" and isinstance(event.get("data"), str):
                response_parts.append(event["data"])
                continue
            if event_type == "usage":
                if response_parts:
                    final_message = "".join(response_parts)
                    response_parts = []
                continue
            if event_type not in {"end", "error"}:
                continue

            raw_usage = event.get("usage")
            if isinstance(raw_usage, dict):
                uncached = GrokAdapter._integer(raw_usage.get("input_tokens"))
                cached = GrokAdapter._integer(raw_usage.get("cache_read_input_tokens"))
                created = GrokAdapter._integer(
                    raw_usage.get("cache_creation_input_tokens")
                )
                total_output = GrokAdapter._integer(raw_usage.get("output_tokens"))
                reasoning = GrokAdapter._integer(raw_usage.get("reasoning_tokens"))
                usage = TokenUsage(
                    input_tokens=uncached + cached + created,
                    cached_input_tokens=cached,
                    output_tokens=total_output,
                    reasoning_output_tokens=reasoning,
                )
            event_session_id = event.get("sessionId")
            if isinstance(event_session_id, str) and event_session_id:
                session_id = event_session_id
            model_usage = event.get("modelUsage")
            if isinstance(model_usage, dict):
                if requested_model in model_usage:
                    model = requested_model
                elif len(model_usage) == 1:
                    only_model = next(iter(model_usage))
                    if isinstance(only_model, str) and only_model:
                        model = only_model
            amount = GrokAdapter._number(event.get("total_cost_usd"))
            if amount is not None:
                reported_cost = ProviderReportedCost(
                    amount_usd=round(amount, 8),
                    currency="USD",
                    source="Grok headless terminal usage",
                )
            if event_type == "end":
                completed = True
                stop_reason = event.get("stopReason")
                if stop_reason != "end_turn":
                    error = (
                        f"Grok stopped with reason {stop_reason}"
                        if isinstance(stop_reason, str) and stop_reason
                        else "Grok terminal event did not report a stop reason"
                    )
            elif isinstance(event.get("message"), str):
                error = event["message"]
            else:
                error = "Grok turn failed"

        if response_parts:
            final_message = "".join(response_parts)
        return {
            "session_id": session_id,
            "model": model,
            "final_message": final_message,
            "usage": usage,
            "reported_cost": reported_cost,
            "completed": completed,
            "error": error,
        }

    @staticmethod
    def _billing_provenance(environment: dict[str, str]) -> BillingProvenance:
        auth_path = Path(environment["GROK_HOME"]) / "auth.json"
        try:
            raw_auth = json.loads(auth_path.read_text())
        except (OSError, json.JSONDecodeError):
            raw_auth = None
        modes = (
            {
                value.get("auth_mode")
                for value in raw_auth.values()
                if isinstance(value, dict) and isinstance(value.get("auth_mode"), str)
            }
            if isinstance(raw_auth, dict)
            else set()
        )
        if modes and "api_key" not in modes and modes <= {"oidc", "web_login"}:
            return BillingProvenance(
                route="subscription",
                credential_source="xai-oauth",
                detected_by="Grok authentication metadata",
            )
        if modes == {"api_key"} or (not modes and environment.get("XAI_API_KEY")):
            return BillingProvenance(
                route="metered-api",
                credential_source="xai-api-key",
                detected_by=(
                    "Grok authentication metadata"
                    if modes == {"api_key"}
                    else "environment"
                ),
            )
        return BillingProvenance(
            credential_source="external-auth" if modes == {"external"} else None,
            detected_by="Grok authentication metadata" if modes else None,
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
            command=_recorded_command(command),
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
            calculated_cost=None,
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
                    "input_tokens": (
                        _token_count(raw_usage.get("inputTokens"))
                        + _token_count(raw_usage.get("cacheReadTokens"))
                        + _token_count(raw_usage.get("cacheWriteTokens"))
                    ),
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
            command=_recorded_command(command),
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
            calculated_cost=None,
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
        billing = self._billing_provenance(request.model, environment)
        usage = parsed["usage"]
        reported_cost = (
            parsed["reported_cost"]
            if billing.route in {"metered-api", "provider-credits"}
            else None
        )

        return RunResult(
            run_id=request.run_id,
            provider=self.provider,
            mode=request.mode,
            task=request.task,
            model=request.model,
            effort=request.effort,
            session_id=session_id,
            command=_recorded_command(command),
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
            calculated_cost=self._calculated_cost(request.model, usage),
            provider_reported_cost=reported_cost,
            billing=billing,
            provider_duration_seconds=parsed["duration_seconds"],
        )

    @staticmethod
    def _billing_provenance(
        model: str | None,
        environment: dict[str, str],
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
        return BillingProvenance(
            route=route,
            credential_source=credential_source,
            detected_by=detected_by,
        )

    @staticmethod
    def _calculated_cost(model: str | None, usage: TokenUsage) -> CalculatedCost | None:
        provider = model.partition("/")[0] if model else ""
        if provider in {"ollama", "lmstudio"}:
            return None
        return estimate_api_cost(model, usage, providers=(provider,))

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
        has_reported_cost = False
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
                    visible_output = OpenCodeAdapter._integer(tokens.get("output"))
                    reasoning_output = OpenCodeAdapter._integer(tokens.get("reasoning"))
                    output_tokens += visible_output + reasoning_output
                    reasoning_output_tokens += reasoning_output
                amount = part.get("cost")
                if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                    cost_usd += max(float(amount), 0.0)
                    has_reported_cost = True

        usage = TokenUsage(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
        )
        reported_cost = None
        if has_finish and has_reported_cost:
            reported_cost = ProviderReportedCost(
                amount_usd=round(cost_usd, 8),
                currency="USD",
                source="OpenCode step_finish events",
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
            "reported_cost": reported_cost,
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
            command=_recorded_command(command),
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
            calculated_cost=estimate_api_cost(
                model,
                parsed["usage"],
                providers=("google",),
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
                    "input_tokens": (
                        _token_count(raw_usage.get("input_tokens"))
                        + _token_count(raw_usage.get("cache_read_tokens"))
                    ),
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
        calculated_cost, reported_cost = self._cost_delta(
            baseline, after, request.model, usage
        )
        billing = self._billing_provenance(after)
        if billing.route not in {"metered-api", "provider-credits"}:
            reported_cost = None

        if capture.timed_out:
            error = f"timed out after {request.timeout_seconds:g} seconds"
        elif capture.exit_code:
            error = self._exit_error(capture.stdout, capture.stderr, capture.exit_code)
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
            command=_recorded_command(command),
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
            calculated_cost=calculated_cost,
            provider_reported_cost=reported_cost,
            billing=billing,
        )

    @staticmethod
    def _billing_provenance(session: _HermesSession | None) -> BillingProvenance:
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
    ) -> tuple[CalculatedCost | None, ProviderReportedCost | None]:
        if after is None:
            return None, None
        model = after.model or requested_model or "hermes"
        provider = HermesAdapter._catalog_provider(after.billing_provider)
        if after.cost_source == "none":
            if provider is None or usage.total_tokens == 0:
                return None, None
            return estimate_api_cost(model, usage, providers=(provider,)), None
        if after.cost_usd is None:
            return None, None
        previous = before.cost_usd if before and before.cost_usd is not None else 0.0
        amount = max(after.cost_usd - previous, 0.0)
        source = f"Hermes CLI session accounting ({after.cost_source})"
        if after.cost_estimated:
            return (
                CalculatedCost(
                    amount_usd=round(amount, 8),
                    currency="USD",
                    model=model,
                    priced_as=model,
                    pricing_version=after.pricing_version,
                    pricing_source=source,
                    long_context_pricing=False,
                ),
                None,
            )
        calculated = (
            estimate_api_cost(model, usage, providers=(provider,))
            if provider is not None
            else None
        )
        return calculated, ProviderReportedCost(
            amount_usd=round(amount, 8),
            currency="USD",
            source=source,
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


class DeepSeekHarnessAdapter:
    provider = "dsh"
    modes = frozenset({"agent"})
    DEFAULT_MODEL = "deepseek-v4-flash"
    EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}

    def __init__(self, binary: str | None = None):
        self.binary = binary or os.environ.get("AOP_DSH_BIN", "dsh")

    def normalize_options(
        self, model: str | None, effort: str | None
    ) -> tuple[str | None, str | None]:
        if effort is not None and effort not in self.EFFORTS:
            raise AOPError(
                "dsh effort must be one of: " + ", ".join(sorted(self.EFFORTS))
            )
        return model or self.DEFAULT_MODEL, effort

    def execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        run_dir: Path,
        environment: dict[str, str],
    ) -> RunResult:
        patch = _prepare_dsh_environment(request, environment)
        command = _provider_command(
            [
                self.binary,
                "--profile",
                "headless",
                "--patch",
                os.fspath(patch),
                request.prompt,
            ],
            request,
            worktree,
            environment,
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
        elif request.session_id and reported_session_id != request.session_id:
            session_id = None
            if reported_session_id is None:
                error = (
                    "dsh resume result did not report the requested session ID "
                    f"{request.session_id}"
                )
            else:
                error = (
                    f"dsh resumed as session {reported_session_id} instead of the "
                    f"requested session {request.session_id}"
                )
        elif parsed["error"]:
            error = parsed["error"]
        elif capture.exit_code:
            error = (
                capture.stderr.strip() or f"dsh exited with status {capture.exit_code}"
            )
        elif reported_session_id is None:
            error = "dsh did not report a session ID"
        elif not parsed["completed"]:
            error = "dsh did not emit a terminal completed result"
        elif final_message is None:
            error = "dsh did not emit a final response"
        else:
            error = None
        model = parsed["model"] or request.model
        usage = parsed["usage"]
        return RunResult(
            run_id=request.run_id,
            provider=self.provider,
            mode=request.mode,
            task=request.task,
            model=model,
            effort=request.effort,
            session_id=session_id,
            command=_recorded_command(
                command,
                secret_names={environment.get("AOP_DSH_CREDENTIAL_REF", "")},
            ),
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
            calculated_cost=self._estimate_cost(request, model, usage),
            billing=BillingProvenance(
                route="metered-api",
                credential_source=(
                    environment["AOP_DSH_CREDENTIAL_REF"].lower().replace("_", "-")
                    if "AOP_DSH_CREDENTIAL_REF" in environment
                    else "provider-native"
                ),
                detected_by="DeepSeek Harness provider contract",
            ),
        )

    @staticmethod
    def _estimate_cost(
        request: RunRequest, model: str | None, usage: TokenUsage
    ) -> CalculatedCost | None:
        provider = request.inference_provider or "deepseek-official"
        catalog_provider = {
            "deepseek-official": "deepseek",
            "anthropic": "anthropic",
            "google": "google",
            "gemini": "google",
            "openai": "openai",
            "xai": "xai",
        }.get(provider)
        if catalog_provider is None:
            return None
        return estimate_api_cost(
            model,
            usage,
            providers=(catalog_provider,),
        )

    @staticmethod
    def _is_response(line: str) -> bool:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return False
        return isinstance(event, dict) and event.get("type") == "aop.dsh.result"

    @staticmethod
    def _parse(output: str) -> dict[str, object]:
        result: dict[str, object] | None = None
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "aop.dsh.result":
                result = event
        if result is None:
            return {
                "session_id": None,
                "model": None,
                "final_message": None,
                "usage": TokenUsage(),
                "completed": False,
                "error": None,
            }
        raw_usage = result.get("usage")
        raw_usage = raw_usage if isinstance(raw_usage, dict) else {}
        uncached_input = _token_count(raw_usage.get("input_tokens"))
        cached_input = _token_count(raw_usage.get("cached_input_tokens"))
        cache_write = _token_count(raw_usage.get("cache_write_input_tokens"))
        usage = TokenUsage(
            input_tokens=uncached_input + cached_input + cache_write,
            cached_input_tokens=cached_input,
            output_tokens=_token_count(raw_usage.get("output_tokens")),
            reasoning_output_tokens=_token_count(
                raw_usage.get("reasoning_output_tokens")
            ),
        )
        return {
            "session_id": (
                result.get("session_id")
                if isinstance(result.get("session_id"), str)
                else None
            ),
            "model": result.get("model")
            if isinstance(result.get("model"), str)
            else None,
            "final_message": (
                result.get("final_message")
                if isinstance(result.get("final_message"), str)
                else None
            ),
            "usage": usage,
            "completed": result.get("completed") is True,
            "error": result.get("error")
            if isinstance(result.get("error"), str)
            else None,
        }


def _provider_command(
    command: list[str],
    request: RunRequest,
    worktree: Worktree,
    environment: dict[str, str],
) -> list[str]:
    if request.profile == "host":
        return command
    root = Path(environment["AOP_ROOT"])
    cache = Path(environment["AOP_CACHE_DIR"])
    provider_state = Path(environment["AOP_PROVIDER_STATE_DIR"])
    bwrap = os.environ.get("AOP_BWRAP_BIN", "bwrap")
    scratch = Path(environment["AOP_SCRATCH_DIR"])
    output = Path(environment["AOP_OUTPUT_DIR"])
    input_root = Path(environment["AOP_INPUT_DIR"])
    wrapped = _isolated_root_command(bwrap)
    if request.profile in {"edit", "review"}:
        wrapped.extend(["--ro-bind", os.fspath(root), "/repository"])
        if (root / ".aop").is_dir():
            wrapped.extend(["--tmpfs", "/repository/.aop"])
        git_directory = root / ".git"
        _add_guest_parent_directories(wrapped, git_directory)
        wrapped.extend(
            ["--ro-bind", os.fspath(git_directory), os.fspath(git_directory)]
        )
    if request.profile == "edit":
        wrapped.extend(["--bind", os.fspath(worktree.path), "/workspace"])
        git_marker = worktree.path / ".git"
        if git_marker.exists():
            wrapped.extend(["--ro-bind", os.fspath(git_marker), "/workspace/.git"])
        common_git_directory = _git_common_directory(worktree.path)
        wrapped.extend(["--ro-bind", os.fspath(common_git_directory), "/git"])
    elif request.profile == "review":
        wrapped.extend(["--ro-bind", os.fspath(worktree.path), "/workspace"])
    else:
        wrapped.extend(["--ro-bind", os.fspath(worktree.path), "/workspace"])
    wrapped.extend(
        [
            "--bind",
            os.fspath(scratch),
            "/scratch",
            "--bind",
            os.fspath(cache),
            "/cache",
            "--bind",
            os.fspath(output),
            "/output",
            "--ro-bind",
            os.fspath(input_root),
            "/inputs",
        ]
    )
    if request.provider in {
        "agy",
        "claude",
        "codex",
        "cursor",
        "devin",
        "dsh",
        "grok",
        "hermes",
        "opencode",
    }:
        wrapped.extend(["--bind", os.fspath(provider_state), "/state"])
    if request.provider == "cursor":
        source_home = _cursor_source_home(environment)
        source_auth = _cursor_source_auth(environment)
        isolated_state = provider_state / "cursor"
        _prepare_cursor_state(
            source_home, source_auth, isolated_state, sealed=_sealed(environment)
        )
        cursor_cache = cache / "cursor"
        cursor_cache.mkdir(parents=True, exist_ok=True)
        environment["HOME"] = os.fspath(isolated_state / "home")
        environment["XDG_CONFIG_HOME"] = os.fspath(isolated_state / "config")
        environment["XDG_CACHE_HOME"] = os.fspath(cursor_cache)
    if request.provider == "hermes":
        isolated_home = Path(environment["HERMES_HOME"])
        if isolated_home != provider_state / "hermes" / "home":
            raise AOPError("Hermes runtime home is outside its task-private state")
        environment["HERMES_REAL_HOME"] = os.fspath(isolated_home)
    if request.provider == "opencode":
        source_config = _opencode_source_config(environment)
        source_data = _opencode_source_data(environment)
        isolated_state = provider_state / "opencode"
        _prepare_opencode_state(
            source_config,
            source_data,
            isolated_state,
            sealed=_sealed(environment),
        )
        opencode_cache = cache / "opencode"
        opencode_cache.mkdir(parents=True, exist_ok=True)
        environment["XDG_CONFIG_HOME"] = os.fspath(isolated_state / "config")
        environment["XDG_DATA_HOME"] = os.fspath(isolated_state / "data")
        environment["XDG_STATE_HOME"] = os.fspath(isolated_state / "state")
        environment["XDG_CACHE_HOME"] = os.fspath(cache)
        environment["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
        dependencies = source_config / "node_modules" if source_config else None
        private_dependencies = isolated_state / "config" / "opencode" / "node_modules"
        if dependencies is not None and dependencies.is_dir():
            shutil.copytree(
                dependencies,
                private_dependencies,
                symlinks=True,
                dirs_exist_ok=True,
            )
            _make_tree_user_writable(private_dependencies)
    mappings = (
        (output, Path("/output")),
        (input_root, Path("/inputs")),
        (provider_state, Path("/state")),
        (cache, Path("/cache")),
        (scratch, Path("/scratch")),
        (worktree.path, Path("/workspace")),
        (root, Path("/repository")),
    )
    command = [_guest_path(argument, mappings) for argument in command]
    command, runtime_mounts = _provider_runtime(command, provider=request.provider)
    for source, destination in runtime_mounts:
        _add_guest_parent_directories(wrapped, Path(destination))
        wrapped.extend(["--ro-bind", source, destination])
    for key, value in _guest_environment(environment, mappings).items():
        wrapped.extend(["--setenv", key, value])
    wrapped.extend(["--chdir", "/workspace", "--", *command])
    return wrapped


def _isolated_root_command(bwrap: str) -> list[str]:
    command = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--cap-drop",
        "ALL",
        "--unshare-pid",
        "--as-pid-1",
        "--unshare-ipc",
        "--unshare-uts",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for path in ("/usr", "/bin", "/lib", "/lib64"):
        source = Path(path)
        if source.exists():
            command.extend(["--ro-bind", path, path])
    for path in (
        "/etc/alternatives",
        "/etc/ca-certificates",
        "/etc/crypto-policies",
        "/etc/hosts",
        "/etc/localtime",
        "/etc/nsswitch.conf",
        "/etc/pki",
        "/etc/resolv.conf",
        "/etc/ssl",
    ):
        source = Path(path)
        if source.exists():
            command.extend(["--ro-bind", path, path])
    return command


def _add_guest_parent_directories(command: list[str], path: Path) -> None:
    parents = list(path.parents)[:-1]
    for parent in reversed(parents):
        if parent != Path("/"):
            command.extend(["--dir", os.fspath(parent)])


def _provider_runtime(
    command: list[str], *, provider: str | None = None
) -> tuple[list[str], list[tuple[str, str]]]:
    executable = shutil.which(command[0])
    if executable is None:
        return command, []
    resolved = Path(executable).resolve()
    if resolved.is_relative_to(Path("/usr")) or resolved.is_relative_to(Path("/bin")):
        return command, []
    brew = Path("/home/linuxbrew/.linuxbrew")
    if resolved.is_relative_to(brew):
        command[0] = os.fspath(resolved)
        return command, [(os.fspath(brew), os.fspath(brew))]
    if provider == "hermes":
        hermes_runtime = _hermes_wrapper_runtime(resolved, command[1:])
        if hermes_runtime is not None:
            return hermes_runtime
    if provider == "dsh":
        node_modules = next(
            (parent for parent in resolved.parents if parent.name == "node_modules"),
            None,
        )
        if node_modules is not None:
            command[0] = os.fspath(resolved)
            return command, [(os.fspath(node_modules), os.fspath(node_modules))]
    guest = Path("/runtime/provider")
    provider_guest = guest / resolved.name
    command[0] = os.fspath(provider_guest)
    mounts = [(os.fspath(resolved), os.fspath(provider_guest))]
    try:
        first_line = resolved.open("rb").readline(4096).decode(errors="ignore").strip()
    except OSError:
        first_line = ""
    if first_line.startswith("#!"):
        interpreter_name = first_line[2:].split(maxsplit=1)[0]
        interpreter = Path(interpreter_name)
        if interpreter.is_absolute() and not interpreter.is_relative_to(Path("/usr")):
            resolved_interpreter = interpreter.resolve()
            runtime_root = resolved_interpreter.parent.parent
            interpreter_guest = Path("/runtime/interpreter")
            mounts.append((os.fspath(runtime_root), os.fspath(interpreter_guest)))
            command = [
                os.fspath(interpreter_guest / "bin" / resolved_interpreter.name),
                os.fspath(provider_guest),
                *command[1:],
            ]
    return command, mounts


def _hermes_wrapper_runtime(
    wrapper: Path, arguments: list[str]
) -> tuple[list[str], list[tuple[str, str]]] | None:
    try:
        contents = wrapper.read_text()
    except (OSError, UnicodeDecodeError):
        return None
    match = re.search(
        r'^exec\s+"([^"\n]+/venv/bin/python)"\s+"([^"\n]+/hermes)"\s+"\$@"\s*$',
        contents,
        flags=re.MULTILINE,
    )
    if match is None:
        return None
    python = Path(match.group(1))
    entrypoint = Path(match.group(2))
    agent_root = entrypoint.parent.resolve()
    venv = python.parent.parent
    if not python.exists() or not entrypoint.is_file() or not venv.is_dir():
        raise AOPError("Hermes launcher references an incomplete private runtime")
    resolved_python = python.resolve()
    python_runtime = resolved_python.parent.parent
    if not python_runtime.is_dir():
        raise AOPError("Hermes launcher Python runtime is unavailable")

    finders = sorted(venv.glob("lib/python*/site-packages/__editable__*_finder.py"))
    if len(finders) != 1:
        raise AOPError("Hermes private runtime has no unique editable package map")
    try:
        tree = ast.parse(finders[0].read_text())
    except (OSError, SyntaxError) as error:
        raise AOPError(f"could not inspect Hermes private runtime: {error}") from error
    mapping: dict[str, str] | None = None
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name) or node.target.id != "MAPPING":
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError) as error:
            raise AOPError("Hermes editable package map is not literal") from error
        if isinstance(value, dict) and all(
            isinstance(key, str) and isinstance(path, str)
            for key, path in value.items()
        ):
            mapping = value
        break
    if mapping is None:
        raise AOPError("Hermes private runtime has no valid editable package map")

    sources = {entrypoint, venv, python_runtime}
    for raw_path in mapping.values():
        candidate = Path(raw_path)
        if not candidate.exists():
            candidate = candidate.with_suffix(".py")
        source = candidate.resolve()
        if not source.is_relative_to(agent_root) or not source.exists():
            raise AOPError("Hermes editable package map escapes its installation")
        sources.add(source)
    mounts = [(os.fspath(source), os.fspath(source)) for source in sorted(sources)]
    return [os.fspath(python), os.fspath(entrypoint), *arguments], mounts


def _guest_path(value: str, mappings: tuple[tuple[Path, Path], ...]) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    for host, guest in mappings:
        try:
            relative = candidate.relative_to(host)
        except ValueError:
            continue
        return os.fspath(guest / relative)
    return value


def _guest_environment(
    environment: dict[str, str], mappings: tuple[tuple[Path, Path], ...]
) -> dict[str, str]:
    selected = _filtered_environment(environment)
    allowed_dsh_auth = environment.get("AOP_DSH_ALLOWED_AUTH_ENV")
    if allowed_dsh_auth is not None:
        allowed = set(allowed_dsh_auth.split(",")) if allowed_dsh_auth else set()
        for name in _DSH_AUTH_ENV_NAMES - allowed:
            selected.pop(name, None)
    credential_ref = environment.get("AOP_DSH_CREDENTIAL_REF")
    if credential_ref and credential_ref in environment:
        selected[credential_ref] = environment[credential_ref]
    for key in (
        "AOP_CACHE_DIR",
        "AOP_DSH_RESUME",
        "AOP_DSH_SESSION_ID",
        "AOP_PROVIDER_STATE_DIR",
        "AOP_SCRATCH_DIR",
        "AOP_INPUT_DIR",
        "AOP_OUTPUT_DIR",
        "CODEX_HOME",
        "DSH_HOME",
        "GROK_HOME",
        "HERMES_HOME",
        "HERMES_REAL_HOME",
        "AOP_INPUT_MANIFEST",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_CACHE_HOME",
    ):
        if key in environment:
            selected[key] = _guest_path(environment[key], mappings)
    selected["HOME"] = _guest_path(
        environment.get(
            "HOME", os.fspath(Path(environment["AOP_PROVIDER_STATE_DIR"]) / "home")
        ),
        mappings,
    )
    selected["PWD"] = "/workspace"
    selected["PATH"] = "/home/linuxbrew/.linuxbrew/bin:/usr/local/bin:/usr/bin:/bin"
    selected.pop("OLDPWD", None)
    return selected


_DSH_AMBIENT_AUTH_ENV = {
    "amazon-bedrock": {
        "AWS_ACCESS_KEY_ID",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
        "AWS_ROLE_ARN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    },
    "anthropic": {"ANTHROPIC_API_KEY"},
    "deepseek": {"DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"},
    "gemini": {"GEMINI_API_KEY", "GOOGLE_API_KEY"},
    "google": {"GEMINI_API_KEY", "GOOGLE_API_KEY"},
    "openai": {"OPENAI_API_KEY", "OPENAI_BASE_URL"},
    "xai": {"XAI_API_KEY"},
}

_DSH_AUTH_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_DEFAULT_REGION",
    "AWS_PROFILE",
    "AWS_REGION",
    "AWS_ROLE_ARN",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "XAI_API_KEY",
}


def _filtered_environment(environment: dict[str, str]) -> dict[str, str]:
    names = {
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_ROLE_ARN",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CURSOR_API_KEY",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_SEARCH_BASE_URL",
        "DSH_TELEMETRY_DISABLED",
        "DEVIN_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GROK_STORAGE_MODE",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "NOUS_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENCODE_API_KEY",
        "PATH",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TZ",
        "NODE_USE_ENV_PROXY",
        "XAI_API_KEY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
    return {key: value for key, value in environment.items() if key in names}


def _recorded_command(
    command: list[str], *, secret_names: set[str] | None = None
) -> list[str]:
    recorded = list(command)
    exact_secret_names = {name.upper() for name in secret_names or set() if name}
    for index in range(len(recorded) - 2):
        if recorded[index] != "--setenv":
            continue
        name = recorded[index + 1].upper()
        if name in exact_secret_names or any(
            marker in name
            for marker in (
                "KEY",
                "TOKEN",
                "SECRET",
                "PASSWORD",
                "CREDENTIAL",
                "PROXY",
            )
        ):
            recorded[index + 2] = "<redacted>"
    return recorded


def _sealed(environment: dict[str, str]) -> bool:
    return environment.get("AOP_PROFILE") == "sealed"


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
        detail = (
            result.stderr.strip() or result.stdout.strip() or "git rev-parse failed"
        )
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
_AGY_SEALED_SEED_FILES = {"google_accounts.json", "oauth_creds.json"}
_AGY_SEALED_NESTED_SEED_FILES = {
    "antigravity-cli": {"antigravity-oauth-token"},
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

_GROK_RUNTIME_NAMES = {
    ".metadata_version",
    "auth.json.lock",
    "bin",
    "bundled",
    "cache",
    "campaigns_state.json",
    "claude_import_state.json",
    "crash",
    "last-copy.txt",
    "leader.log",
    "leader.sock",
    "logs",
    "marketplace-cache",
    "memory",
    "models_cache.json",
    "plugin-data",
    "sandbox-events.jsonl",
    "sessions",
    "trace-exports",
    "version.json",
    "worktrees",
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
    _prepare_codex_state(source, destination, sealed=_sealed(environment))
    environment["CODEX_HOME"] = os.fspath(destination)


def _prepare_grok_environment(environment: dict[str, str]) -> None:
    source = _grok_source_home(environment)
    destination = Path(environment["AOP_PROVIDER_STATE_DIR"]) / "grok" / "home"
    _prepare_grok_state(source, destination, sealed=_sealed(environment))
    environment["GROK_HOME"] = os.fspath(destination)


def _prepare_dsh_environment(request: RunRequest, environment: dict[str, str]) -> Path:
    provider_root = Path(environment["AOP_PROVIDER_STATE_DIR"]) / "dsh"
    dsh_home = provider_root / "home"
    dsh_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    dsh_home.chmod(0o700)
    source_home = Path(
        environment.get(
            "AOP_DSH_SOURCE_HOME",
            environment.get("DSH_HOME", Path.home() / ".dsh"),
        )
    ).expanduser()
    provider = request.inference_provider or "deepseek-official"
    settings = _read_yaml_mapping(source_home / "settings.yaml", "dsh settings")
    provider_settings, credential_ref = _project_dsh_provider_settings(
        settings, provider
    )
    if provider_settings:
        _atomic_write_private(
            dsh_home / "settings.yaml",
            yaml.safe_dump(
                provider_settings,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            ),
        )

    if credential_ref is not None:
        credentials = _read_yaml_mapping(
            source_home / ".credentials.yaml", "dsh managed credentials"
        )
        _project_dsh_credential(
            credentials, credential_ref, dsh_home / ".credentials.yaml"
        )
        environment["AOP_DSH_CREDENTIAL_REF"] = credential_ref
        environment["AOP_DSH_ALLOWED_AUTH_ENV"] = credential_ref
    else:
        environment.pop("AOP_DSH_CREDENTIAL_REF", None)
        environment["AOP_DSH_ALLOWED_AUTH_ENV"] = ",".join(
            sorted(_DSH_AMBIENT_AUTH_ENV.get(provider, set()))
        )

    scratch = Path(environment["AOP_SCRATCH_DIR"])
    runner = dsh_home / "profiles" / "headless" / "aop-dsh-runner.mjs"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner_source = Path(__file__).with_name("dsh_runner.mjs")
    _atomic_write(runner, runner_source.read_text())
    runner.chmod(0o600)
    runner_uri = (
        runner.as_uri()
        if request.profile == "host"
        else "file:///state/dsh/home/profiles/headless/aop-dsh-runner.mjs"
    )
    effort = "off" if request.effort == "none" else request.effort
    rows = [
        "- id: agent-default-model",
        "  config:",
        f"    provider: {json.dumps(provider)}",
        f"    model: {json.dumps(request.model)}",
    ]
    if effort is not None:
        rows.append(f"    reasoningEffort: {json.dumps(effort)}")
    rows.extend(
        [
            "- id: session-title-llm",
            "  disabled: true",
            "- id: headless-runner",
            "  disabled: true",
            "- insert:",
            "    - id: aop-headless-runner",
            f"      name: {json.dumps(runner_uri)}",
            "      inject: [headlessStartup]",
        ]
    )
    patch = scratch / f"aop-dsh-{request.run_id}.cordis.yml"
    _atomic_write(patch, "\n".join(rows) + "\n")
    patch.chmod(0o600)

    environment["DSH_HOME"] = os.fspath(dsh_home)
    environment["DSH_TELEMETRY_DISABLED"] = "1"
    environment["AOP_DSH_SESSION_ID"] = (
        request.session_id or f"session-{request.run_id}"
    )
    if request.session_id:
        environment["AOP_DSH_RESUME"] = "1"
    else:
        environment.pop("AOP_DSH_RESUME", None)
    return patch


def _read_yaml_mapping(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = yaml.load(path.read_text(), Loader=_UniqueKeyLoader)
    except OSError as error:
        raise AOPError(f"could not read {label}") from error
    except (UnicodeError, yaml.YAMLError) as error:
        raise AOPError(f"{label} are invalid YAML") from error
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AOPError(f"{label} must be a YAML mapping")
    return value


def _project_dsh_provider_settings(
    settings: dict[str, object], provider: str
) -> tuple[dict[str, object], str | None]:
    if provider == "deepseek-official":
        profile = settings.get("llm-deepseek", {})
        if not isinstance(profile, dict):
            raise AOPError("dsh llm-deepseek settings must be a mapping")
        credential_ref = profile.get("apiKeyEnv", "DEEPSEEK_API_KEY")
        projected = {"llm-deepseek": profile} if profile else {}
    else:
        pi_ai = settings.get("llm-pi-ai", {})
        if not isinstance(pi_ai, dict):
            raise AOPError("dsh llm-pi-ai settings must be a mapping")
        providers = pi_ai.get("providers", {})
        if not isinstance(providers, dict):
            raise AOPError("dsh llm-pi-ai providers must be a mapping")
        profile = providers.get(provider)
        if not isinstance(profile, dict):
            raise AOPError(
                f'dsh provider "{provider}" is not configured in llm-pi-ai.providers'
            )
        credential_ref = profile.get("apiKeyEnv")
        projected = {"llm-pi-ai": {"providers": {provider: profile}}}
    if credential_ref is not None and (
        not isinstance(credential_ref, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", credential_ref) is None
    ):
        raise AOPError(f'dsh provider "{provider}" has an invalid apiKeyEnv')
    return projected, credential_ref


def _project_dsh_credential(
    credentials: dict[str, object], credential_ref: str, destination: Path
) -> None:
    credential = credentials.get(credential_ref)
    if credential is None:
        return
    if not isinstance(credential, str) or not credential:
        raise AOPError(f"dsh managed {credential_ref} must be a nonempty string")
    _atomic_write_private(
        destination,
        yaml.safe_dump(
            {credential_ref: credential},
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ),
    )


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


def _prepare_codex_state(
    source: Path | None, destination: Path, *, sealed: bool = False
) -> None:
    if destination.is_dir():
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, mode=0o700)
    try:
        if source:
            for entry in source.iterdir():
                target = temporary / entry.name
                if sealed and entry.name != "auth.json":
                    continue
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


def _grok_source_home(environment: dict[str, str]) -> Path | None:
    configured = environment.get("AOP_GROK_SOURCE_HOME")
    inherited = environment.get("GROK_HOME")
    source = Path(configured or inherited or Path(environment["HOME"]) / ".grok")
    source = source.expanduser()
    if source.is_dir():
        return source.resolve()
    if configured or inherited:
        raise AOPError(f"Grok home does not exist: {source}")
    return None


def _prepare_grok_state(
    source: Path | None, destination: Path, *, sealed: bool = False
) -> None:
    if destination.is_dir():
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, mode=0o700)
    try:
        if source:
            for entry in source.iterdir():
                if entry.name in _GROK_RUNTIME_NAMES:
                    continue
                if sealed and entry.name != "auth.json":
                    continue
                target = temporary / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, target, symlinks=True)
                elif entry.is_file():
                    shutil.copy2(entry, target)
        _make_tree_user_writable(temporary)
        os.replace(temporary, destination)
    except OSError as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise AOPError(f"could not prepare isolated Grok state: {error}") from error


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
    source_home: Path,
    source_auth: Path | None,
    destination: Path,
    *,
    sealed: bool = False,
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
            if sealed and entry.name != "auth.json":
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
        elif sealed:
            private_config.mkdir(parents=True)
            for entry in source_auth.iterdir():
                if entry.is_file() and any(
                    marker in entry.name.lower()
                    for marker in ("auth", "credential", "token")
                ):
                    shutil.copy2(entry, private_config / entry.name)
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
    _prepare_devin_state(
        source_data,
        source_config,
        destination,
        sealed=_sealed(environment),
    )
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


def _prepare_claude_environment(environment: dict[str, str]) -> None:
    source_home = Path(environment["HOME"])
    destination = Path(environment["AOP_PROVIDER_STATE_DIR"]) / "claude" / "home"
    if not destination.exists():
        destination.mkdir(parents=True, mode=0o700)
        source_config = source_home / ".claude"
        private_config = destination / ".claude"
        if _sealed(environment):
            private_config.mkdir()
            credentials = source_config / ".credentials.json"
            if credentials.is_file():
                shutil.copy2(credentials, private_config / credentials.name)
        elif source_config.is_dir():
            shutil.copytree(
                source_config,
                private_config,
                ignore=shutil.ignore_patterns(
                    "debug",
                    "history.jsonl",
                    "projects",
                    "session-env",
                    "shell-snapshots",
                    "statsig",
                    "todos",
                ),
            )
        source_settings = source_home / ".claude.json"
        if source_settings.is_file() and not _sealed(environment):
            shutil.copy2(source_settings, destination / source_settings.name)
        _make_tree_user_writable(destination)
    environment["HOME"] = os.fspath(destination)


def _prepare_devin_state(
    source_data: Path,
    source_config: Path | None,
    destination: Path,
    *,
    sealed: bool = False,
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
            if sealed and not (
                entry.is_file()
                and any(
                    marker in entry.name.lower()
                    for marker in ("auth", "credential", "token")
                )
            ):
                continue
            target = private_data / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, symlinks=True)
            elif entry.is_file():
                shutil.copy2(entry, target)
        if source_config is None:
            private_config.mkdir(parents=True)
        elif sealed:
            private_config.mkdir(parents=True)
            source = source_config / "config.json"
            if source.is_file():
                try:
                    value = json.loads(source.read_text())
                except (OSError, json.JSONDecodeError) as error:
                    raise AOPError(
                        f"could not sanitize Devin configuration: {error}"
                    ) from error
                private_config.joinpath("config.json").write_text(
                    f"{json.dumps(_without_instruction_sources(value), sort_keys=True)}\n"
                )
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


def _without_instruction_sources(value: object) -> object:
    forbidden = ("instruction", "mcp", "plugin", "hook", "skill", "rule", "memory")
    if isinstance(value, dict):
        return {
            key: _without_instruction_sources(nested)
            for key, nested in value.items()
            if not any(marker in key.lower() for marker in forbidden)
        }
    if isinstance(value, list):
        return [_without_instruction_sources(item) for item in value]
    return value


def _opencode_source_config(environment: dict[str, str]) -> Path | None:
    configured = environment.get("AOP_OPENCODE_CONFIG_DIR")
    base = Path(
        environment.get("XDG_CONFIG_HOME", Path(environment["HOME"]) / ".config")
    )
    source = Path(configured).expanduser() if configured else base / "opencode"
    if source.is_dir():
        return source.resolve()
    if configured:
        raise AOPError(f"OpenCode config directory does not exist: {source}")
    return None


def _opencode_source_data(environment: dict[str, str]) -> Path | None:
    configured = environment.get("AOP_OPENCODE_DATA_DIR")
    base = Path(
        environment.get("XDG_DATA_HOME", Path(environment["HOME"]) / ".local" / "share")
    )
    source = Path(configured).expanduser() if configured else base / "opencode"
    if source.is_dir():
        return source.resolve()
    if configured:
        raise AOPError(f"OpenCode data directory does not exist: {source}")
    return None


def _prepare_opencode_state(
    source_config: Path | None,
    source_data: Path | None,
    destination: Path,
    *,
    sealed: bool = False,
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
        if source_config and not sealed:
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


def _prepare_agy_dir(source: Path, destination: Path, *, sealed: bool = False) -> None:
    if destination.is_dir():
        return
    if not source.is_dir():
        raise AOPError(f"Agy profile is not a directory: {source}")
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, mode=0o700)
    try:
        for entry in source.iterdir():
            target = temporary / entry.name
            if not sealed and entry.name in _AGY_SEED_DIRECTORIES and entry.is_dir():
                shutil.copytree(entry, target)
            elif entry.is_file() and (
                not sealed or entry.name in _AGY_SEALED_SEED_FILES
            ):
                shutil.copy2(entry, target)
        nested_files = (
            _AGY_SEALED_NESTED_SEED_FILES if sealed else _AGY_NESTED_SEED_FILES
        )
        for directory, names in nested_files.items():
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


def _prepare_hermes_home(
    source: Path, destination: Path, *, sealed: bool = False
) -> None:
    if destination.is_dir():
        return
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, mode=0o700)
    try:
        for entry in source.iterdir():
            target = temporary / entry.name
            if not sealed and entry.name in _HERMES_SEED_DIRECTORIES and entry.is_dir():
                shutil.copytree(entry, target, symlinks=True)
            elif (
                entry.is_file()
                and entry.name not in _HERMES_RUNTIME_FILES
                and (not sealed or entry.name == "auth.json")
            ):
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
    _prepare_hermes_home(source_home, task_home, sealed=_sealed(environment))
    configured_state = environment.get("AOP_HERMES_SHARED_STATE_DIR")
    state_dir = (
        Path(configured_state) if configured_state else provider_state.parent.parent
    )
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
    if agent == "dsh":
        return DeepSeekHarnessAdapter()
    if agent == "grok":
        return GrokAdapter()
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
        profile: str = "edit",
        timeout_seconds: float | None = None,
        artifacts: Sequence[str] = (),
        input_paths: Sequence[str | os.PathLike[str]] = (),
    ) -> RunResult:
        self._validate_mode(mode)
        self._validate_inference_provider(inference_provider, model)
        if profile not in PROFILES:
            raise AOPError(f"unknown execution profile: {profile}")
        model, effort = self.adapter.normalize_options(model, effort)
        request = self._request(
            task=task,
            prompt=prompt,
            base=base,
            model=model,
            inference_provider=inference_provider,
            effort=effort,
            mode=mode,
            profile=profile,
            timeout_seconds=timeout_seconds,
            artifacts=artifacts,
        )
        worktree = (
            self._create_sealed_workspace(request.run_id)
            if profile == "sealed"
            else self._get_or_create_worktree(task, base)
        )
        return self._execute(request, worktree, input_paths=input_paths)

    def resume(
        self,
        *,
        run_id: str,
        prompt: str,
        timeout_seconds: float | None = None,
        artifacts: Sequence[str] = (),
        input_paths: Sequence[str | os.PathLike[str]] | None = None,
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
        if parent_request.profile == "sealed":
            workspace_record = parent_request.effective_policy.get("workspace", {})
            workspace_path = workspace_record.get("controller_path")
            if not isinstance(workspace_path, str):
                raise AOPError("sealed run is missing its controller workspace path")
            path = Path(workspace_path).resolve()
            sealed_root = (self.manager.sealed_runtime_dir / "sealed").resolve()
            if not path.is_relative_to(sealed_root) or not path.is_dir():
                raise AOPError("sealed run workspace is missing or outside AOP state")
            worktree = Worktree(task=parent_request.run_id, path=path, head="")
        else:
            worktree = self.manager.get(parent_request.task)
        if parent_request.profile == "sealed" and input_paths is None:
            snapshot_root = parent_request.effective_policy.get("controller", {}).get(
                "input_snapshot"
            )
            if not isinstance(snapshot_root, str):
                raise AOPError("sealed run is missing its controller input snapshot")
            resolved_snapshot = Path(snapshot_root).resolve()
            snapshots_root = (self.manager.sealed_runtime_dir / "snapshots").resolve()
            if (
                not resolved_snapshot.is_relative_to(snapshots_root)
                or not resolved_snapshot.is_dir()
            ):
                raise AOPError(
                    "sealed run input snapshot is missing or outside AOP state"
                )
            inherited_inputs = tuple(
                os.fspath(resolved_snapshot / Path(item.mounted_path).name)
                for item in parent_request.inputs
            )
            for item, inherited in zip(
                parent_request.inputs, inherited_inputs, strict=True
            ):
                kind, files = _inspect_read_path(Path(inherited))
                if kind != item.kind or _read_path_digest(kind, files) != item.sha256:
                    raise AOPError(
                        f"sealed input snapshot changed since run {parent_request.run_id}"
                    )
            input_provenance = tuple(item.source_path for item in parent_request.inputs)
        else:
            inherited_inputs = tuple(item.source_path for item in parent_request.inputs)
            input_provenance = None
        selected_inputs = (
            inherited_inputs if input_paths is None else tuple(input_paths)
        )
        request = self._request(
            task=parent_request.task,
            prompt=prompt,
            base=parent_request.base,
            model=parent_request.model,
            inference_provider=parent_request.inference_provider,
            effort=parent_request.effort,
            mode=parent_request.mode,
            profile=parent_request.profile,
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
            input_paths=selected_inputs,
            input_provenance=input_provenance,
            task_lock_held=_task_lock_held,
        )

    def _execute(
        self,
        request: RunRequest,
        worktree: Worktree,
        *,
        input_paths: Sequence[str | os.PathLike[str]],
        input_provenance: Sequence[str] | None = None,
        task_lock_held: bool = False,
    ) -> RunResult:
        if task_lock_held:
            return self._execute_unlocked(
                request, worktree, input_paths, input_provenance=input_provenance
            )
        with exclusive_lock(
            task_lock_path(self.manager.state_dir, request.task),
            f"task {request.task}",
        ):
            return self._execute_unlocked(
                request, worktree, input_paths, input_provenance=input_provenance
            )

    def _execute_unlocked(
        self,
        request: RunRequest,
        worktree: Worktree,
        input_paths: Sequence[str | os.PathLike[str]],
        *,
        input_provenance: Sequence[str] | None = None,
    ) -> RunResult:
        parent_controller: dict[str, object] = {}
        if request.profile == "sealed" and request.parent_run_id:
            parent = self.store.load_request(request.parent_run_id)
            parent_controller = parent.effective_policy.get("controller", {})
        state_key = request.run_id if request.profile == "sealed" else request.task
        runtime_dir = (
            self.manager.sealed_runtime_dir
            if request.profile == "sealed"
            else self.manager.state_dir
        )
        runtime_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        runtime_dir.parent.chmod(0o700)
        runtime_dir.chmod(0o700)
        recorded_scratch = parent_controller.get("scratch")
        scratch_dir = (
            Path(recorded_scratch)
            if isinstance(recorded_scratch, str)
            else runtime_dir / "scratch" / state_key
        )
        scratch_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        scratch_dir.parent.chmod(0o700)
        scratch_dir.chmod(0o700)
        input_dir = runtime_dir / "snapshots" / request.run_id
        input_dir.mkdir(parents=True, exist_ok=False)
        input_dir.parent.chmod(0o700)
        input_dir.chmod(0o700)
        output_dir = scratch_dir / "outputs" / request.run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        if parent_controller:
            recorded_state = parent_controller.get("provider_state")
            if not isinstance(recorded_state, str):
                raise AOPError("sealed run is missing its controller state path")
            provider_state = Path(recorded_state)
        else:
            provider_state = runtime_dir / "provider-state" / state_key
        provider_state.mkdir(parents=True, exist_ok=True, mode=0o700)
        provider_state.chmod(0o700)
        (provider_state / "home").mkdir(exist_ok=True, mode=0o700)
        recorded_cache = parent_controller.get("cache")
        if request.profile == "sealed":
            cache_dir = (
                Path(recorded_cache)
                if isinstance(recorded_cache, str)
                else runtime_dir / "sealed-cache" / state_key
            )
            cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            cache_dir.parent.chmod(0o700)
            cache_dir.chmod(0o700)
        else:
            cache_dir = self.manager.cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        declared_inputs = _prepare_inputs(
            input_paths,
            input_dir,
            guest_paths=request.profile != "host",
            provenance=input_provenance,
        )
        effective_policy = resolve_policy(
            request.profile,
            workspace_host_path=(
                worktree.path if request.profile in {"edit", "review", "host"} else None
            ),
            input_names=tuple(Path(item.mounted_path).name for item in declared_inputs),
        ).to_dict()
        effective_policy["workspace"]["controller_path"] = os.fspath(worktree.path)
        effective_policy["controller"] = {
            "provider_state": os.fspath(provider_state),
            "scratch": os.fspath(scratch_dir),
            "cache": os.fspath(cache_dir),
            "input_snapshot": os.fspath(input_dir),
        }
        effective_policy["provider_runtime"] = provider_runtime_record(self.adapter)
        effective_policy["instruction_sources"] = _instruction_sources(
            request, worktree, os.environ
        )
        effective_policy["environment"]["inherited_names"] = sorted(
            (
                os.environ.keys()
                if request.profile == "host"
                else _filtered_environment(os.environ).keys()
            )
        )
        request = replace(
            request,
            prompt=_run_prompt(
                request.prompt,
                declared_inputs,
                request.artifacts,
                Path("/output") if request.profile != "host" else output_dir,
            ),
            inputs=declared_inputs,
            effective_policy=effective_policy,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "AOP_ROOT": os.fspath(self.manager.root),
                "AOP_TASK": (
                    request.run_id if request.profile == "sealed" else request.task
                ),
                "AOP_WORKTREE": os.fspath(worktree.path),
                "AOP_CACHE_DIR": os.fspath(cache_dir),
                "AOP_PROVIDER_STATE_DIR": os.fspath(provider_state),
                "AOP_SCRATCH_DIR": os.fspath(scratch_dir),
                "AOP_INPUT_DIR": os.fspath(input_dir),
                "AOP_OUTPUT_DIR": os.fspath(output_dir),
                "AOP_RUN_ID": request.run_id,
                "AOP_PROFILE": request.profile,
            }
        )
        if request.provider == "hermes":
            environment["AOP_HERMES_SHARED_STATE_DIR"] = os.fspath(
                self.manager.state_dir
            )
            environment.pop("HERMES_NO_TOOLS", None)
            if request.mode == "participant":
                environment.pop("HERMES_KANBAN_TASK", None)
        if request.provider == "agy":
            _prepare_agy_dir(
                _agy_source_dir(environment),
                provider_state / "agy" / "gemini",
                sealed=request.profile == "sealed",
            )
        run_dir = self.store.create(request)
        if request.inputs:
            input_manifest = input_dir / "manifest.json"
            guest_manifest = {
                "schema_version": 1,
                "inputs": [
                    {key: value for key, value in item.items() if key != "source_path"}
                    for item in request.to_dict()["inputs"]
                ],
            }
            self.store.write_json(
                run_dir / "input-manifest.json",
                {"schema_version": 1, "inputs": request.to_dict()["inputs"]},
            )
            self.store.write_json(input_manifest, guest_manifest)
            environment["AOP_INPUT_MANIFEST"] = os.fspath(input_manifest)
        _make_snapshot_read_only(input_dir)
        result = self.adapter.execute(request, worktree, run_dir, environment)
        result = replace(
            result,
            inference_provider=request.inference_provider,
            inputs=request.inputs,
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
        profile: str,
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
            profile=profile,
            effective_policy={},
            timeout_seconds=timeout_seconds,
            session_id=session_id,
            parent_run_id=parent_run_id,
            artifacts=normalize_artifacts(artifacts),
            inputs=(),
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
        if self.adapter.provider not in {"hermes", "dsh"}:
            raise AOPError(
                f"{self.adapter.provider} does not support --provider overrides"
            )
        if model is None:
            raise AOPError(
                f"{self.adapter.provider} --provider requires an explicit --model"
            )

    def _get_or_create_worktree(self, task: str, base: str) -> Worktree:
        for worktree in self.manager.list():
            if worktree.task == task:
                return worktree
        return self.manager.create(task, base)

    def _create_sealed_workspace(self, run_id: str) -> Worktree:
        path = self.manager.sealed_runtime_dir / "sealed" / run_id / "workspace"
        path.mkdir(parents=True, mode=0o700)
        self.manager.sealed_runtime_dir.chmod(0o700)
        (self.manager.sealed_runtime_dir / "sealed").chmod(0o700)
        path.parent.chmod(0o700)
        return Worktree(task=run_id, path=path, head="")


def provider_runtime_record(adapter: AgentAdapter) -> dict[str, object]:
    binary = getattr(adapter, "binary", None)
    if not isinstance(binary, str):
        return {"executable": None, "sha256": None}
    resolved_name = shutil.which(binary)
    if resolved_name is None:
        return {"executable": binary, "sha256": None}
    executable = Path(resolved_name).resolve()
    try:
        with executable.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:
        digest = None
    return {
        "executable": os.fspath(executable),
        "guest_root": "/runtime/provider",
        "sha256": digest,
    }


def _instruction_sources(
    request: RunRequest,
    worktree: Worktree,
    environment: dict[str, str],
) -> list[dict[str, object]]:
    if request.profile == "sealed":
        return []
    candidates: list[Path] = []
    if request.provider == "codex":
        source = _codex_source_home(environment)
        if source:
            candidates.extend(
                source / name for name in ("config.toml", "rules", "skills")
            )
    elif request.provider == "claude":
        home = Path(environment["HOME"])
        candidates.extend([home / ".claude.json", home / ".claude" / "settings.json"])
    elif request.provider == "cursor":
        home = _cursor_source_home(environment)
        candidates.extend(
            home / name for name in ("skills", "skills-cursor", "plugins", "policies")
        )
    elif request.provider == "devin":
        config = _devin_source_config(environment)
        if config:
            candidates.append(config)
    elif request.provider == "opencode":
        config = _opencode_source_config(environment)
        if config:
            candidates.extend(
                entry
                for entry in config.iterdir()
                if entry.name not in _OPENCODE_CONFIG_RUNTIME_NAMES
            )
    elif request.provider == "agy":
        candidates.append(_agy_source_dir(environment) / "config")
    elif request.provider == "grok":
        source = _grok_source_home(environment)
        if source:
            candidates.extend(
                source / name
                for name in (
                    "AGENTS.md",
                    "agents",
                    "commands",
                    "config.toml",
                    "hooks",
                    "personas",
                    "plugins",
                    "rules",
                    "settings.json",
                    "skills",
                    "workflows",
                )
            )
    elif request.provider == "hermes":
        home = _hermes_source_home(environment)
        candidates.extend(home / name for name in sorted(_HERMES_SEED_DIRECTORIES))
    candidates.extend(worktree.path / name for name in ("AGENTS.md", "CLAUDE.md"))
    records = []
    for candidate in candidates:
        record = _path_provenance(candidate)
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda record: str(record["source_path"]))


def _path_provenance(path: Path) -> dict[str, object] | None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(status.st_mode):
        target = os.readlink(path).encode()
        return {
            "source_path": os.fspath(path),
            "kind": "symlink",
            "sha256": hashlib.sha256(target).hexdigest(),
        }
    if stat.S_ISREG(status.st_mode):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        return {"source_path": os.fspath(path), "kind": "file", "sha256": digest}
    if not stat.S_ISDIR(status.st_mode):
        return None
    digest = hashlib.sha256(b"aop-instruction-directory-v1\0")
    files = 0
    for directory, names, filenames in os.walk(path):
        names[:] = sorted(
            name for name in names if not (Path(directory) / name).is_symlink()
        )
        for name in sorted(filenames):
            source = Path(directory) / name
            if source.is_symlink() or not source.is_file():
                continue
            relative = source.relative_to(path).as_posix().encode()
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with source.open("rb") as handle:
                digest.update(hashlib.file_digest(handle, "sha256").digest())
            files += 1
    return {
        "source_path": os.fspath(path),
        "kind": "directory",
        "files": files,
        "sha256": digest.hexdigest(),
    }


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


def _prepare_inputs(
    values: Sequence[str | os.PathLike[str]],
    input_dir: Path,
    *,
    guest_paths: bool = True,
    provenance: Sequence[str] | None = None,
) -> tuple[InputSnapshot, ...]:
    if provenance is not None and len(provenance) != len(values):
        raise AOPError("input provenance does not match the input snapshot set")
    prepared: list[InputSnapshot] = []
    aliases: set[str] = set()
    for index, value in enumerate(values):
        raw = os.fspath(value)
        if not raw or "\0" in raw:
            raise AOPError("input path must be a non-empty filesystem path")
        candidate = Path(raw).expanduser()
        try:
            initial_status = candidate.lstat()
            source = candidate.resolve(strict=True)
        except OSError as error:
            raise AOPError(
                f"could not inspect input path {candidate}: {error}"
            ) from error
        if stat.S_ISLNK(initial_status.st_mode):
            raise AOPError(f"input path may not be a symlink: {candidate}")
        alias = source.name
        if not alias:
            raise AOPError(f"input path must have a basename: {source}")
        if alias in aliases:
            raise AOPError(f"input paths have the same basename: {alias}")
        aliases.add(alias)

        mounted = input_dir / alias
        if source.is_dir():
            _copy_input_directory(source, mounted)
        else:
            shutil.copyfile(source, mounted)
        kind, files = _inspect_read_path(mounted)
        size_bytes = sum(item.size_bytes for item in files)
        prepared.append(
            InputSnapshot(
                source_path=(
                    provenance[index] if provenance is not None else os.fspath(source)
                ),
                mounted_path=(
                    f"/inputs/{alias}" if guest_paths else os.fspath(mounted)
                ),
                kind=kind,
                size_bytes=size_bytes,
                sha256=_read_path_digest(kind, files),
                files=files,
            )
        )
    return tuple(prepared)


def _copy_input_directory(source: Path, destination: Path) -> None:
    destination.mkdir()
    for child in sorted(source.iterdir(), key=lambda item: item.name):
        status = child.lstat()
        target = destination / child.name
        if stat.S_ISLNK(status.st_mode):
            raise AOPError(f"read path contains a symlink: {child}")
        if stat.S_ISDIR(status.st_mode):
            _copy_input_directory(child, target)
        elif stat.S_ISREG(status.st_mode):
            shutil.copyfile(child, target)
        else:
            raise AOPError(f"read path contains a special file: {child}")


def _make_snapshot_read_only(root: Path) -> None:
    for directory, names, files in os.walk(root, topdown=False):
        path = Path(directory)
        for name in files:
            entry = path / name
            entry.chmod(stat.S_IRUSR)
        for name in names:
            entry = path / name
            if entry.is_dir() and not entry.is_symlink():
                entry.chmod(stat.S_IRUSR | stat.S_IXUSR)
        path.chmod(stat.S_IRUSR | stat.S_IXUSR)


def _inspect_read_path(source: Path) -> tuple[str, tuple[InputFile, ...]]:
    status = source.lstat()
    if stat.S_ISLNK(status.st_mode):
        raise AOPError(f"read path may not be a symlink: {source}")
    if stat.S_ISREG(status.st_mode):
        return "file", (_read_path_file(source, source.name),)
    if not stat.S_ISDIR(status.st_mode):
        raise AOPError(f"read path is not a regular file or directory: {source}")

    files: list[InputFile] = []

    def collect(directory: Path, relative: PurePosixPath) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError as error:
            raise AOPError(
                f"could not read input directory {directory}: {error}"
            ) from error
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


def _read_path_file(source: Path, relative_path: str) -> InputFile:
    try:
        size_bytes = source.stat().st_size
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as error:
        raise AOPError(f"could not hash read path file {source}: {error}") from error
    return InputFile(
        relative_path=relative_path,
        size_bytes=size_bytes,
        sha256=digest,
    )


def _read_path_digest(kind: str, files: tuple[InputFile, ...]) -> str:
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
    inputs: tuple[InputSnapshot, ...],
    artifacts: tuple[str, ...],
    output_dir: Path,
) -> str:
    if inputs:
        paths = "\n".join(
            f"- Path: {json.dumps(item.mounted_path)}\n"
            f"  SHA-256: {item.sha256} ({item.size_bytes} bytes, {item.kind})"
            for item in inputs
        )
        prompt = (
            f"{prompt.rstrip()}\n\n"
            "AOP snapshotted read-only inputs:\n"
            f"{paths}\n"
            "The paths above are immutable snapshots. Source host paths are not exposed."
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
