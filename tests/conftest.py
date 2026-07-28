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
if prompt.startswith("WRITE_ARTIFACT"):
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
def fake_agy(tmp_path: Path) -> Path:
    executable = tmp_path / "agy"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys

args = sys.argv[1:]
session_id = (
    args[args.index("--conversation") + 1]
    if "--conversation" in args
    else "49f2a36e-43e4-4ba9-9f4f-817bee57f64c"
)
model = (
    args[args.index("--model") + 1]
    if "--model" in args
    else "gemini-3.5-flash"
)
prompt = args[args.index("-p") + 1]
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
