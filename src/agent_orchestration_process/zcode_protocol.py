"""Zcode's native stdio protocol, with controller-owned deadlines and selection."""

from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
import uuid
from typing import Callable

from .worktrees import AOPError


@dataclass
class Capture:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False
    duration_seconds: float = 0
    first_event_seconds: float | None = None
    first_response_seconds: float | None = None
    effort: str | None = None
    error: str | None = None


def run(
    command: list[str],
    *,
    cwd: Path,
    workspace: str,
    environment: dict[str, str],
    prompt: str,
    model: str,
    effort: str | None,
    session_id: str | None,
    timeout_seconds: float | None,
    on_selection: Callable[[str | None], None] | None = None,
) -> Capture:
    """Use v4 commands for mutations and session snapshots for verification.

    All pipe IO is nonblocking, including large prompt writes. The same deadline
    covers startup, selection, input delivery, and inference.
    """
    capture = Capture()
    started = time.monotonic()
    deadline = started + timeout_seconds if timeout_seconds is not None else None
    output, errors = [], []
    pending = bytearray()
    buffer = bytearray()
    responses = {}
    events = deque()
    counter = 0
    with (
        subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        ) as process,
        selectors.DefaultSelector() as selector,
    ):
        for stream in (process.stdin, process.stdout, process.stderr):
            os.set_blocking(stream.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ)
        selector.register(process.stderr, selectors.EVENT_READ)

        def send(value):
            if not pending:
                selector.register(process.stdin, selectors.EVENT_WRITE)
            pending.extend((json.dumps(value) + "\n").encode())

        def receive():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError
            ready = selector.select(remaining)
            if not ready:
                raise TimeoutError
            for key, _ in ready:
                stream = key.fileobj
                if stream is process.stdin:
                    count = os.write(stream.fileno(), pending[:65536])
                    del pending[:count]
                    if not pending:
                        selector.unregister(stream)
                    continue
                chunk = os.read(stream.fileno(), 65536)
                if not chunk:
                    selector.unregister(stream)
                    if stream is process.stdout:
                        raise AOPError(
                            "Zcode app-server closed before completing the protocol"
                        )
                    continue
                if stream is process.stderr:
                    errors.append(chunk)
                    continue
                output.append(chunk)
                buffer.extend(chunk)
                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer[:] = rest
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise AOPError("Zcode returned invalid protocol data")
                    if capture.first_event_seconds is None:
                        capture.first_event_seconds = time.monotonic() - started
                    if "id" in value and "method" in value:
                        if value["method"] != "session/requestRuntimePreferences":
                            raise AOPError(
                                f"Zcode requested unsupported interaction: {value['method']}"
                            )
                        send(
                            {
                                "id": value["id"],
                                "result": {
                                    "nativeSearchEnhancementsEnabled": False,
                                    "memoryEnabled": False,
                                    "askUserQuestionAutoResolutionEnabled": False,
                                },
                            }
                        )
                    elif "id" in value:
                        responses[value["id"]] = value
                    elif value.get("method") == "session/event":
                        event = value["params"]
                        events.append(event)
                        payload = event.get("payload", {})
                        if capture.first_response_seconds is None and (
                            (
                                payload.get("kind") == "text_delta"
                                and payload.get("delta")
                            )
                            or (
                                event.get("type") == "turn.completed"
                                and payload.get("response")
                            )
                        ):
                            capture.first_response_seconds = time.monotonic() - started

        def rpc(method, params):
            nonlocal counter
            counter += 1
            request_id = counter
            send({"id": request_id, "method": method, "params": params})
            while request_id not in responses:
                receive()
            response = responses.pop(request_id)
            if "error" in response:
                raise AOPError(
                    f"Zcode {method}: {response['error'].get('message', 'request failed')}"
                )
            return response["result"]

        def command_rpc(kind, payload, sid):
            command_id = str(uuid.uuid4())
            ack = rpc(
                "v4/command",
                {
                    "commandId": command_id,
                    "clientId": "aop",
                    "sessionId": sid,
                    "type": kind,
                    "payload": payload,
                    "issuedAt": int(time.time() * 1000),
                },
            )
            if ack.get("commandId") != command_id or ack.get("status") != "accepted":
                raise AOPError(
                    f"Zcode {kind}: {ack.get('message') or ack.get('reasonCode') or ack.get('status')}"
                )
            return command_id, ack.get("result", {})

        provider, model_id = model.split("/", 1)
        selection = {"providerId": provider, "modelId": model_id}
        if effort is not None:
            selection["options"] = {"reasoningLevel": effort}

        def verify(snapshot):
            if snapshot.get("session", {}).get("sessionId") != sid:
                raise AOPError("Zcode did not resume the requested session ID")
            settings = snapshot.get("settings", {})
            current = settings.get("model", {}).get("current", {})
            if any(
                current.get(key) != selection[key] for key in ("providerId", "modelId")
            ):
                raise AOPError("Zcode did not confirm the selected model identity")
            thought = settings.get("thoughtLevel", {})
            available = [item["value"] for item in thought.get("available", [])]
            actual = current.get("options", {}).get(
                "reasoningLevel", thought.get("current")
            )
            if effort is not None and (effort not in available or actual != effort):
                raise AOPError(
                    f"Zcode did not confirm effort {effort!r}; supported levels: {', '.join(available) or 'none'}"
                )
            if capture.effort is not None and actual != capture.effort:
                raise AOPError("Zcode changed its reasoning effort during the turn")
            if settings.get("mode", {}).get("current") != "yolo":
                raise AOPError("Zcode did not confirm the selected permission mode")
            capture.effort = actual

        try:
            if session_id:
                sid = session_id
                snapshot = rpc("session/resume", {"sessionId": sid, "mcpServers": []})
            else:
                config = {
                    "modelSelection": selection,
                    "mode": "yolo",
                    "planEnabled": False,
                }
                if effort is not None:
                    config["thought"] = effort
                _, created = command_rpc(
                    "createSession",
                    {"workspaceId": workspace, "config": config, "mcpServers": []},
                    None,
                )
                sid = created["sessionId"]
                snapshot = rpc("session/read", {"sessionId": sid})
            verify(snapshot)
            # Send the resolved native selection, including its default effort.
            # Omitting options here would overwrite the session selection.
            selection = snapshot["settings"]["model"]["current"]
            if on_selection is not None:
                on_selection(capture.effort)
            rpc(
                "session/subscribe",
                {
                    "sessionId": sid,
                    "deliveryKind": "desktop-continuous",
                    "afterSeq": snapshot["runtime"]["eventSeq"],
                    "includeSnapshot": False,
                },
            )
            events.clear()
            input_id, _ = command_rpc(
                "sendText",
                {
                    "text": prompt,
                    "requestedDelivery": "startNow",
                    "modelSelection": selection,
                    "mode": "yolo",
                },
                sid,
            )
            terminal = None
            while terminal is None:
                while events:
                    event = events.popleft()
                    if event.get("sessionId") != sid:
                        raise AOPError(
                            "Zcode stream contains conflicting session identities"
                        )
                    if (
                        event.get("type") in ("turn.completed", "turn.failed")
                        and event.get("payload", {}).get("inputId") == input_id
                    ):
                        terminal = event
                if terminal is None:
                    receive()
            verify(rpc("session/read", {"sessionId": sid}))
        except TimeoutError:
            capture.timed_out = True
        except (AOPError, OSError, ValueError, KeyError, TypeError) as error:
            capture.error = str(error)
        finally:
            natural_exit = process.poll()
            if natural_exit is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            capture.exit_code = (
                natural_exit
                if natural_exit is not None
                else (0 if capture.error is None and not capture.timed_out else 1)
            )
    capture.stdout = b"".join(output).decode("utf-8", errors="replace")
    capture.stderr = b"".join(errors).decode("utf-8", errors="replace")
    capture.duration_seconds = time.monotonic() - started
    return capture
