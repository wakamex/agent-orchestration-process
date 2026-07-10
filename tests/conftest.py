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
import pathlib
import subprocess
import sys
import time

args = sys.argv[1:]
prompt = sys.stdin.read()
output_path = pathlib.Path(args[args.index("--output-last-message") + 1])

if prompt == "SLEEP":
    time.sleep(10)

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
