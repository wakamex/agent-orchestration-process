from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


SESSION_ID = "019f4da1-342f-7670-8aac-25999973b294"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", os.fspath(cwd), *args],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "AOP Test")
    git(root, "config", "user.email", "aop@example.invalid")
    (root / "README.md").write_text("# Test project\n")
    git(root, "add", "README.md")
    git(root, "commit", "-m", "Initial commit")
    return root


@pytest.fixture
def fake_codex(tmp_path: Path) -> Path:
    executable = tmp_path / "codex"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import subprocess
import sys
import time

args = sys.argv[1:]
prompt = sys.stdin.read()
output_path = pathlib.Path(args[args.index("--output-last-message") + 1])

if prompt == "SLEEP":
    time.sleep(10)

output_dir = pathlib.Path(os.environ["AOP_OUTPUT_DIR"])
if prompt.startswith("WRITE_ARTIFACT_TREE_SYMLINK"):
    output_dir.joinpath("assets").mkdir()
    target = pathlib.Path(os.environ["AOP_CACHE_DIR"]) / "escaped.png"
    target.write_bytes(b"escaped")
    output_dir.joinpath("assets", "figure-1.png").symlink_to(target)
elif prompt.startswith("WRITE_ARTIFACT_TREE_EMPTY_FILE"):
    output_dir.joinpath("assets").mkdir()
    output_dir.joinpath("assets", "figure-1.png").write_bytes(b"")
elif prompt.startswith("WRITE_ARTIFACT_TREE_SPECIAL_FILE"):
    output_dir.joinpath("assets").mkdir()
    os.mkfifo(output_dir.joinpath("assets", "figure-1.png"))
elif prompt.startswith("WRITE_ARTIFACT_TREE"):
    print("narrating before writing linked deliverables", flush=True)
    output_dir.joinpath("paper.md").write_text(
        "# Extracted\\n\\n![Figure](assets/figure-1.png)\\n"
    )
    output_dir.joinpath("assets", "nested").mkdir(parents=True)
    output_dir.joinpath("assets", "figure-1.png").write_bytes(b"PNG figure 1")
    output_dir.joinpath("assets", "nested", "figure-2.svg").write_text(
        "<svg>figure 2</svg>\\n"
    )
elif prompt.startswith("WRITE_EMPTY_ARTIFACT_DIRECTORY"):
    output_dir.joinpath("assets").mkdir()
elif prompt.startswith("WRITE_ARTIFACT"):
    print("narrating before writing the deliverable", flush=True)
    output_dir.joinpath("paper.md").write_text("# Extracted\\n")
elif prompt.startswith("EMPTY_ARTIFACT"):
    output_dir.joinpath("paper.md").write_text("")
elif prompt.startswith("SYMLINK_ARTIFACT"):
    target = pathlib.Path(os.environ["AOP_CACHE_DIR"]) / "escaped.md"
    target.write_text("escaped")
    output_dir.joinpath("paper.md").symlink_to(target)

session_id = {SESSION_ID!r}
if "resume" in args:
    session_id = args[args.index("resume") + 1]

if prompt.startswith("AOP conflict resolution"):
    conflicts = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    for name in conflicts:
        main_content = subprocess.run(
            ["git", "show", f":2:{{name}}"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.rstrip()
        task_content = subprocess.run(
            ["git", "show", f":3:{{name}}"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.rstrip()
        pathlib.Path(name).write_text(f"{{main_content}}\\n{{task_content}}\\n")

print(json.dumps({{"type": "thread.started", "thread_id": session_id}}), flush=True)
print(json.dumps({{"type": "turn.started"}}), flush=True)

if prompt == "FAIL":
    print(json.dumps({{
        "type": "turn.failed",
        "error": {{"message": "synthetic failure"}},
    }}), flush=True)
    raise SystemExit(1)

message = f"answer:{{prompt}}"
output_path.write_text(message)
print(json.dumps({{
    "type": "item.completed",
    "item": {{"id": "item_0", "type": "agent_message", "text": message}},
}}), flush=True)
print(json.dumps({{
    "type": "turn.completed",
    "usage": {{
        "input_tokens": 1,
        "cached_input_tokens": 0,
        "output_tokens": 1,
        "reasoning_output_tokens": 0,
    }},
}}), flush=True)
"""
    )
    executable.chmod(0o755)
    return executable


@pytest.fixture
def fake_claude(tmp_path: Path) -> Path:
    executable = tmp_path / "claude"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys

args = sys.argv[1:]
prompt = sys.stdin.read()
if prompt == "CHECK_SANDBOX":
    pathlib.Path("agent-write.txt").write_text("allowed")
    (pathlib.Path(os.environ["AOP_CACHE_DIR"]) / "provider-cache.txt").write_text(
        "shared"
    )
    for protected in [
        pathlib.Path(os.environ["AOP_ROOT"]) / "main-write.txt",
        pathlib.Path(".git"),
    ]:
        try:
            protected.write_text("forbidden")
        except OSError:
            pass
        else:
            raise RuntimeError(f"sandbox allowed write to {{protected}}")
if prompt == "CHECK_SCRATCH":
    pathlib.Path(os.environ["AOP_SCRATCH_DIR"]).joinpath("analysis.txt").write_text(
        "allowed"
    )
    for protected in [
        pathlib.Path("agent-write.txt"),
        pathlib.Path(os.environ["AOP_ROOT"]) / "main-write.txt",
        pathlib.Path(".git"),
    ]:
        try:
            protected.write_text("forbidden")
        except OSError:
            pass
        else:
            raise RuntimeError(f"scratch sandbox allowed write to {{protected}}")
session_id = (
    args[args.index("--resume") + 1]
    if "--resume" in args
    else args[args.index("--session-id") + 1]
)
print(json.dumps({{
    "type": "system",
    "subtype": "init",
    "session_id": session_id,
    "model": "claude-test-model",
}}), flush=True)
print(json.dumps({{
    "type": "assistant",
    "session_id": session_id,
    "message": {{"content": [{{"type": "text", "text": "working"}}]}},
}}), flush=True)
print(json.dumps({{
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "session_id": session_id,
    "result": f"answer:{{prompt}}",
    "total_cost_usd": 0.0123,
    "usage": {{
        "input_tokens": 10,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 30,
        "output_tokens": 40,
    }},
}}), flush=True)
"""
    )
    executable.chmod(0o755)
    return executable


@pytest.fixture
def fake_cursor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    executable = tmp_path / "agent"
    cursor_home = tmp_path / "cursor-home"
    cursor_home.mkdir()
    (cursor_home / "cli-config.json").write_text('{"model": "composer-2.5"}\n')
    (cursor_home / "skills-cursor").mkdir()
    (cursor_home / "skills-cursor" / "test.md").write_text("test skill\n")
    cursor_config = tmp_path / "cursor-config"
    cursor_config.mkdir()
    (cursor_config / "auth.json").write_text('{"token": "test"}\n')
    monkeypatch.setenv("AOP_CURSOR_HOME", os.fspath(cursor_home))
    monkeypatch.setenv("AOP_CURSOR_CONFIG_DIR", os.fspath(cursor_config))
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys

args = sys.argv[1:]
prompt = args[args.index("--") + 1]
cursor_home = pathlib.Path(os.environ["AOP_CURSOR_HOME"])
cursor_config = pathlib.Path(os.environ["XDG_CONFIG_HOME"]) / "cursor"
auth_path = cursor_config / "auth.json"
if not auth_path.is_file():
    raise RuntimeError("missing private Cursor authentication")
auth_path.write_text('{{"token": "refreshed"}}\\n')
session_id = (
    args[args.index("--resume") + 1]
    if "--resume" in args
    else "b5b6dbdd-0d68-452f-9d5c-20238c970169"
)
if prompt == "CURSOR_DIFFERENT_SESSION":
    session_id = "0478966c-19cc-4ade-b3b3-83b59dd670ba"
project_dir = cursor_home / "projects" / "test-project"
project_dir.mkdir(parents=True, exist_ok=True)
project_dir.joinpath("repo.json").write_text("{{}}\\n")
chat_dir = cursor_home / "chats" / session_id
chat_dir.mkdir(parents=True, exist_ok=True)
chat_dir.joinpath("state.json").write_text(prompt)
if prompt.startswith("WRITE_ARTIFACT"):
    pathlib.Path(os.environ["AOP_OUTPUT_DIR"]).joinpath("report.md").write_text(
        "# Cursor artifact\\n"
    )
if prompt == "CHECK_CURSOR_SANDBOX":
    pathlib.Path("agent-write.txt").write_text("allowed")
    for protected in [
        pathlib.Path(os.environ["AOP_ROOT"]) / "main-write.txt",
        pathlib.Path(".git"),
    ]:
        try:
            protected.write_text("forbidden")
        except OSError:
            pass
        else:
            raise RuntimeError(f"sandbox allowed write to {{protected}}")

print(json.dumps({{
    "type": "system",
    "subtype": "init",
    "session_id": session_id,
    "model": "Composer 2.5",
}}), flush=True)
print(json.dumps({{
    "type": "assistant",
    "message": {{"content": [{{"type": "text", "text": "working"}}]}},
    "session_id": session_id,
}}), flush=True)
print(json.dumps({{
    "type": "result",
    "subtype": "success",
    "duration_ms": 1500,
    "duration_api_ms": 1250,
    "is_error": False,
    "result": f"answer:{{prompt}}",
    "session_id": session_id,
    "usage": {{
        "inputTokens": 100,
        "outputTokens": 20,
        "cacheReadTokens": 30,
        "cacheWriteTokens": 0,
    }},
}}), flush=True)
"""
    )
    executable.chmod(0o755)
    return executable


@pytest.fixture
def fake_agy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    executable = tmp_path / "agy"
    source_dir = tmp_path / "agy-source"
    source_dir.mkdir()
    (source_dir / "oauth_creds.json").write_text('{"token": "test"}\n')
    (source_dir / "config").mkdir()
    (source_dir / "config" / "config.json").write_text("{}\n")
    (source_dir / "antigravity-cli").mkdir()
    (source_dir / "antigravity-cli" / "antigravity-oauth-token").write_text(
        "test-token\n"
    )
    (source_dir / "antigravity-cli" / "conversations").mkdir()
    (source_dir / "antigravity-cli" / "conversations" / "stale.json").write_text(
        "stale\n"
    )
    monkeypatch.setenv("AOP_AGY_SOURCE_DIR", os.fspath(source_dir))
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys

args = sys.argv[1:]
gemini_dir = pathlib.Path(args[args.index("--gemini_dir") + 1])
runtime_dir = gemini_dir / "antigravity-cli"
runtime_dir.mkdir(parents=True, exist_ok=True)
if not (runtime_dir / "antigravity-oauth-token").is_file():
    print("missing private authentication", file=sys.stderr)
    raise SystemExit(2)
state_path = runtime_dir / "fake-conversation.json"
session_id = (
    args[args.index("--conversation") + 1]
    if "--conversation" in args
    else "49f2a36e-43e4-4ba9-9f4f-817bee57f64c"
)
if "--conversation" in args:
    state = json.loads(state_path.read_text())
    if state["conversation_id"] != session_id:
        print("conversation is absent from private state", file=sys.stderr)
        raise SystemExit(3)
else:
    state_path.write_text(json.dumps({{"conversation_id": session_id}}))
model = (
    args[args.index("--model") + 1]
    if "--model" in args
    else "gemini-3.5-flash"
)
prompt = args[args.index("-p") + 1]
if prompt.startswith("AGY_DIFFERENT_SESSION"):
    session_id = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
if prompt.startswith("WRITE_ARTIFACT"):
    pathlib.Path(os.environ["AOP_OUTPUT_DIR"]).joinpath("paper.md").write_text(
        "# Extracted by agy\\n"
    )
print(json.dumps({{
    "event": "init",
    "init": {{
        "conversation_id": session_id,
        "model": model,
    }},
}}), flush=True)
print(json.dumps({{
    "event": "step_update",
    "step_update": {{
        "conversation_id": session_id,
        "step_type": "agent_response",
        "text_delta": "working",
        "duration_seconds": 0.25,
        "usage": {{
            "input_tokens": 40,
            "output_tokens": 5,
            "thinking_tokens": 2,
            "cache_read_tokens": 10,
            "total_tokens": 45,
        }},
    }},
}}), flush=True)
result = {{
    "conversation_id": session_id,
    "status": "SUCCESS",
    "response": f"answer:{{prompt}}",
    "duration_seconds": 1.25,
    "num_turns": 1,
    "usage": {{
        "input_tokens": 100,
        "output_tokens": 20,
        "thinking_tokens": 7,
        "cache_read_tokens": 30,
        "total_tokens": 120,
    }},
}}
if prompt.startswith("AGY_MISSING_SESSION"):
    result.pop("conversation_id")
if prompt.startswith("AGY_ERROR"):
    result.update({{
        "status": "ERROR",
        "response": None,
        "error": "synthetic agy failure",
    }})
print(json.dumps({{"event": "result", "result": result}}), flush=True)
"""
    )
    executable.chmod(0o755)
    return executable


@pytest.fixture
def fake_hermes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    executable = tmp_path / "hermes"
    state_path = tmp_path / "hermes-session.json"
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "auth.json").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", os.fspath(hermes_home))
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys

args = sys.argv[1:]
state_path = (
    pathlib.Path(os.environ["HERMES_HOME"]) / "fake-state.json"
    if os.environ.get("AOP_FAKE_HERMES_STATE_IN_HOME")
    else pathlib.Path({os.fspath(state_path)!r})
)

if args[:2] == ["sessions", "export"]:
    requested = args[args.index("--session-id") + 1]
    if not state_path.exists():
        print(f"Session '{{requested}}' not found.")
        raise SystemExit(0)
    state = json.loads(state_path.read_text())
    if state["id"] != requested:
        print(f"Session '{{requested}}' not found.")
        raise SystemExit(0)
    print(json.dumps(state))
    raise SystemExit(0)

if args[0] != "chat" or "-Q" not in args:
    raise RuntimeError(f"unexpected Hermes invocation: {{args}}")

prompt = args[args.index("-q") + 1]
if prompt.startswith("CHECK_PARTICIPANT"):
    required = {{"-Q", "--safe-mode", "--toolsets", "--max-turns"}}
    if not required.issubset(args):
        raise RuntimeError(f"missing participant flags: {{args}}")
    if args[args.index("--toolsets") + 1] != "__aop_no_tools__":
        raise RuntimeError(f"unexpected participant toolset: {{args}}")
    if args[args.index("--max-turns") + 1] != "1":
        raise RuntimeError(f"participant turn limit is not one: {{args}}")
    if "--yolo" in args or "--accept-hooks" in args:
        raise RuntimeError(f"participant inherited coding-agent flags: {{args}}")
    if os.environ.get("HERMES_KANBAN_TASK") or os.environ.get("HERMES_NO_TOOLS"):
        raise RuntimeError("participant inherited internal Hermes mode state")
session_id = (
    args[args.index("--resume") + 1]
    if "--resume" in args
    else "44ad6c7c-5213-4af8-a1d5-4a742c85cbcf"
)
state = (
    json.loads(state_path.read_text())
    if state_path.exists()
    else {{
        "id": session_id,
        "model": args[args.index("--model") + 1],
        "billing_provider": "nous",
        "input_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "estimated_cost_usd": 0.0,
        "actual_cost_usd": None,
        "cost_source": "provider_models_api",
        "pricing_version": "nous-models-2026-08-03",
        "messages": [],
    }}
)

if prompt.startswith("CHECK_HERMES_SANDBOX"):
    pathlib.Path("agent-write.txt").write_text("allowed")
    pathlib.Path(os.environ["AOP_OUTPUT_DIR"]).joinpath("report.md").write_text(
        "# Hermes artifact\\n"
    )
    for protected in [
        pathlib.Path(os.environ["AOP_ROOT"]) / "main-write.txt",
        pathlib.Path(".git"),
    ]:
        try:
            protected.write_text("forbidden")
        except OSError:
            pass
        else:
            raise RuntimeError(f"sandbox allowed write to {{protected}}")

state["input_tokens"] += 10
state["cache_read_tokens"] += 3
state["cache_write_tokens"] += 2
state["output_tokens"] += 4
state["reasoning_tokens"] += 1
state["estimated_cost_usd"] += 0.000001
state["messages"].extend([
    {{"id": f"user-{{len(state['messages'])}}", "role": "user", "content": prompt}},
    {{
        "id": f"assistant-{{len(state['messages']) + 1}}",
        "role": "assistant",
        "content": f"answer:{{prompt}}",
    }},
])
state_path.write_text(json.dumps(state))

print(f"narrated reasoning for {{prompt}}\\nanswer:{{prompt}}", flush=True)
print(f"session_id: {{session_id}}", file=sys.stderr, flush=True)
if prompt == "FAIL":
    raise SystemExit(3)
"""
    )
    executable.chmod(0o755)
    return executable
