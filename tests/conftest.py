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
import re
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

if prompt.startswith("AOP integration assignment"):
    def field(name):
        match = re.search(rf"^{{name}}: (.+)$", prompt, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"missing integration field: {{name}}")
        return match.group(1)

    root = field("Main worktree")
    main_head = field("Current main commit")
    base = field("Recorded task base")
    task_head = field("Original task head")
    rebase = subprocess.run(
        ["git", "rebase", "--onto", main_head, base, task_head], check=False
    )
    if rebase.returncode:
        conflicts = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=U"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()
        for name in conflicts:
            main_content = subprocess.run(
                ["git", "-C", root, "show", f"{{main_head}}:{{name}}"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.rstrip()
            task_content = subprocess.run(
                ["git", "show", f"{{task_head}}:{{name}}"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.rstrip()
            pathlib.Path(name).write_text(f"{{main_content}}\\n{{task_content}}\\n")
            subprocess.run(["git", "add", name], check=True)
        environment = dict(os.environ, GIT_EDITOR="true")
        subprocess.run(["git", "rebase", "--continue"], env=environment, check=True)
    rebased_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", root, "merge", "--ff-only", rebased_head], check=True
    )

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
