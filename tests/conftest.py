from __future__ import annotations

import os
import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import pytest


SESSION_ID = "019f4da1-342f-7670-8aac-25999973b294"

SEALED_PROVIDER_PROBE = r"""
if prompt.startswith("SEALED_PROVIDER_PROBE"):
    forbidden_environment = {"AOP_ROOT", "AOP_TASK", "AOP_WORKTREE", "AOP_RUN_ID", "OLDPWD"}
    leaked = forbidden_environment.intersection(os.environ)
    if leaked:
        raise RuntimeError(f"sealed environment leaked: {sorted(leaked)}")
    if "HERMES_HOME" in os.environ and os.environ.get(
        "HERMES_REAL_HOME"
    ) != os.environ.get("HERMES_HOME"):
        raise RuntimeError("sealed Hermes real home is not its task-private guest home")
    pid_one_environment = pathlib.Path("/proc/1/environ").read_bytes().split(b"\0")
    pid_one_names = {
        entry.partition(b"=")[0].decode(errors="replace")
        for entry in pid_one_environment
        if entry
    }
    pid_one_leaked = forbidden_environment.intersection(pid_one_names)
    if pid_one_leaked:
        raise RuntimeError(
            f"sealed PID 1 environment leaked: {sorted(pid_one_leaked)}"
        )
    if b"/code/" in b"\0".join(pid_one_environment):
        raise RuntimeError("sealed PID 1 environment exposed a controller path")
    pid_one_command = pathlib.Path("/proc/1/cmdline").read_bytes()
    if b"/code/" in pid_one_command:
        raise RuntimeError("sealed PID 1 command exposed a controller path")
    if b"--ro-bind" in pid_one_command or b"--bind" in pid_one_command:
        raise RuntimeError("sealed PID 1 command exposed sandbox mount arguments")
    if b"/.aop/" in pathlib.Path("/proc/self/mountinfo").read_bytes():
        raise RuntimeError("sealed mount metadata exposed controller state")
    workspace = pathlib.Path("/workspace")
    if pathlib.Path.cwd() != workspace or any(workspace.iterdir()):
        raise RuntimeError("sealed workspace is not empty and neutral")
    for hidden in (pathlib.Path("/repository"), pathlib.Path("/code"), pathlib.Path("/home")):
        if hidden.exists():
            raise RuntimeError(f"sealed run exposed host path: {hidden}")
    try:
        workspace.joinpath("forbidden-write").write_text("forbidden")
    except OSError:
        pass
    else:
        raise RuntimeError("sealed neutral workspace is writable")
    if pathlib.Path("/cache/shared-cache-canary").exists():
        raise RuntimeError("sealed run exposed the shared cache")
"""


@pytest.fixture(autouse=True)
def fresh_model_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    providers = {
        "openai": {
            "models": {
                "gpt-5.6-sol": {
                    "name": "GPT-5.6 Sol",
                    "cost": {
                        "input": 5,
                        "cache_read": 0.5,
                        "cache_write": 6.25,
                        "output": 30,
                        "tiers": [
                            {
                                "input": 10,
                                "cache_read": 1,
                                "cache_write": 12.5,
                                "output": 45,
                                "tier": {"type": "context", "size": 272000},
                            }
                        ],
                    },
                },
                "gpt-5.6-terra": {
                    "name": "GPT-5.6 Terra",
                    "cost": {"input": 2, "cache_read": 0.2, "output": 12},
                },
                "gpt-5.6-luna": {
                    "name": "GPT-5.6 Luna",
                    "cost": {"input": 0.2, "cache_read": 0.02, "output": 1.2},
                },
                "gpt-5.5": {
                    "name": "GPT-5.5",
                    "cost": {
                        "input": 5,
                        "cache_read": 0.5,
                        "output": 30,
                        "tiers": [
                            {
                                "input": 10,
                                "cache_read": 1,
                                "output": 45,
                                "tier": {"type": "context", "size": 272000},
                            }
                        ],
                    },
                },
                "gpt-5.4-mini": {
                    "name": "GPT-5.4 mini",
                    "cost": {"input": 0.75, "cache_read": 0.075, "output": 4.5},
                },
            }
        },
        "anthropic": {"models": {}},
        "deepseek": {
            "models": {
                "deepseek-v4-flash": {
                    "name": "DeepSeek-V4-Flash",
                    "cost": {"input": 0.3, "cache_read": 0.03, "output": 1.2},
                },
                "deepseek-v4-pro": {
                    "name": "DeepSeek-V4-Pro",
                    "cost": {"input": 0.6, "cache_read": 0.06, "output": 2.4},
                },
            }
        },
        "google": {
            "models": {
                "gemini-3.5-flash": {
                    "name": "Gemini 3.5 Flash",
                    "cost": {"input": 1.5, "cache_read": 0.15, "output": 9},
                }
            }
        },
        "opencode": {"models": {}},
        "xai": {
            "models": {
                "grok-4.5": {
                    "name": "Grok 4.5",
                    "cost": {"input": 2, "cache_read": 0.3, "output": 6},
                }
            }
        },
    }
    raw = json.dumps(providers, separators=(",", ":"), sort_keys=True).encode()
    path = tmp_path / "model-catalog.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fetched_at": time.time(),
                "source": "https://models.dev/api.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "providers": providers,
            }
        )
    )
    monkeypatch.setenv("AOP_MODEL_CATALOG_CACHE", os.fspath(path))
    return path


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
def fake_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source_home = tmp_path / "codex-home"
    source_home.mkdir()
    source_home.joinpath("auth.json").write_text('{"auth_mode": "chatgpt"}\n')
    source_home.joinpath("config.toml").write_text('model = "test"\n')
    source_home.joinpath("models_cache.json").write_text('{"models": []}\n')
    source_home.joinpath("history.jsonl").write_text("global history\n")
    source_home.joinpath("state_5.sqlite").write_text("global database\n")
    source_home.joinpath("cache").mkdir()
    source_home.joinpath("cache", "global-cache").write_text("global cache\n")
    source_home.joinpath("sessions").mkdir()
    source_home.joinpath("sessions", "global-session").write_text("global session\n")
    source_home.joinpath("rules").mkdir()
    source_home.joinpath("rules", "default.rules").write_text("allow test\n")
    source_home.joinpath("skills", ".system").mkdir(parents=True)
    source_home.joinpath("skills", ".system", "generated").write_text("generated\n")
    source_home.joinpath("skills", "user-skill").mkdir()
    source_home.joinpath("skills", "user-skill", "SKILL.md").write_text("user\n")
    monkeypatch.setenv("AOP_CODEX_SOURCE_HOME", os.fspath(source_home))
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
if args == ["login", "status"]:
    if os.environ.get("AOP_FAKE_CODEX_AUTH") == "api-key":
        print("Logged in using an API key")
    else:
        print("Logged in using ChatGPT")
    raise SystemExit(0)
prompt = sys.stdin.read()
{SEALED_PROVIDER_PROBE}
output_path = pathlib.Path(args[args.index("--output-last-message") + 1])
codex_home = pathlib.Path(os.environ["CODEX_HOME"])
if prompt.startswith("CHECK_READ_PATHS"):
    manifest = json.loads(pathlib.Path(os.environ["AOP_INPUT_MANIFEST"]).read_text())
    input_dir = pathlib.Path(os.environ["AOP_INPUT_DIR"])
    if not input_dir.is_dir() or len(manifest["inputs"]) != 2:
        raise RuntimeError("declared read paths were not exposed")
    for item in manifest["inputs"]:
        mounted = pathlib.Path(item["mounted_path"])
        if mounted.parent != input_dir or not mounted.exists():
            raise RuntimeError(f"invalid read path mapping: {{item}}")
        for protected in (mounted,):
            target = protected / "forbidden" if protected.is_dir() else protected
            try:
                if protected.is_dir():
                    target.write_text("forbidden")
                else:
                    with target.open("a") as handle:
                        handle.write("forbidden")
            except OSError:
                pass
            else:
                raise RuntimeError(f"declared read path was writable: {{protected}}")
if prompt.startswith("CHECK_SEALED"):
    forbidden_environment = {{
        "AOP_ROOT", "AOP_TASK", "AOP_WORKTREE", "AOP_RUN_ID", "OLDPWD"
    }}
    leaked = forbidden_environment.intersection(os.environ)
    if leaked:
        raise RuntimeError(f"sealed environment leaked: {{sorted(leaked)}}")
    if pathlib.Path.cwd() != pathlib.Path("/workspace") or any(pathlib.Path.cwd().iterdir()):
        raise RuntimeError("sealed workspace is not empty and neutral")
    for hidden in [
        pathlib.Path("/repository"),
        pathlib.Path("/code/aop"),
        pathlib.Path("/home/mihai"),
        pathlib.Path({os.fspath(source_home)!r}),
    ]:
        if hidden.exists():
            raise RuntimeError(f"sealed run exposed host path: {{hidden}}")
if prompt.startswith("CHECK_SEALED_WRITE_MARKERS"):
    for root in ("/cache", "/state", "/scratch", "/output"):
        pathlib.Path(root, "cross-task-canary").write_text(prompt)
if prompt.startswith("CHECK_SEALED_NO_MARKERS"):
    for root in ("/cache", "/state", "/scratch", "/output", "/workspace"):
        if pathlib.Path(root, "cross-task-canary").exists():
            raise RuntimeError(f"sealed run read another task's state: {{root}}")
if prompt.startswith("CHECK_SEALED_RESUME_MARKER"):
    if not pathlib.Path("/cache/cross-task-canary").is_file():
        raise RuntimeError("sealed exact resume did not reuse its session cache")
if prompt.startswith("CHECK_PROFILE_ACCESS"):
    _, expected, unrelated_path = prompt.split()
    if pathlib.Path(unrelated_path).exists():
        raise RuntimeError("profile exposed an unrelated host path")
    repository = pathlib.Path("/repository")
    if not repository.joinpath("README.md").is_file():
        raise RuntimeError("profile did not expose the primary repository read-only")
    if repository.joinpath(".aop").exists() and any(repository.joinpath(".aop").iterdir()):
        raise RuntimeError("profile exposed AOP controller state")
    try:
        repository.joinpath("README.md").open("a").write("forbidden")
    except OSError:
        pass
    else:
        raise RuntimeError("profile allowed primary repository mutation")
    workspace_probe = pathlib.Path("workspace-probe.txt")
    try:
        workspace_probe.write_text("probe")
    except OSError:
        if expected == "edit":
            raise RuntimeError("edit profile did not allow workspace writes")
    else:
        if expected == "review":
            raise RuntimeError("review profile allowed workspace writes")
        workspace_probe.unlink()
for required in ["auth.json"]:
    if not codex_home.joinpath(required).is_file():
        print(f"missing seeded Codex file: {{required}}", file=sys.stderr)
        raise SystemExit(1)
if codex_home.joinpath("sessions", "global-session").exists():
    print("global Codex sessions were copied", file=sys.stderr)
    raise SystemExit(1)
if codex_home.joinpath("skills", ".system", "generated").exists():
    print("generated Codex skills were copied", file=sys.stderr)
    raise SystemExit(1)

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
    if not codex_home.joinpath("sessions", session_id).is_file():
        print("private Codex session is missing", file=sys.stderr)
        raise SystemExit(1)
if prompt == "DIFFERENT_SESSION":
    session_id = "different-codex-thread"
codex_home.joinpath("sessions").mkdir(exist_ok=True)
codex_home.joinpath("sessions", session_id).write_text("private session\\n")
with codex_home.joinpath("history.jsonl").open("a") as history:
    history.write(f"{{prompt}}\\n")

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
if prompt == "INCOMPLETE":
    raise SystemExit(0)
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
if args == ["auth", "status", "--json"]:
    auth_method = (
        "api_key"
        if os.environ.get("AOP_FAKE_CLAUDE_AUTH") == "api-key"
        else "claude.ai"
    )
    print(json.dumps({{
        "loggedIn": True,
        "authMethod": auth_method,
        "apiProvider": "firstParty",
    }}))
    raise SystemExit(0)
prompt = sys.stdin.read()
{SEALED_PROVIDER_PROBE}
if prompt == "CHECK_SANDBOX":
    pathlib.Path("agent-write.txt").write_text("allowed")
    (pathlib.Path(os.environ["AOP_CACHE_DIR"]) / "provider-cache.txt").write_text(
        "shared"
    )
    for protected in [
        pathlib.Path("/repository") / "main-write.txt",
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
        pathlib.Path("/repository") / "main-write.txt",
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
if prompt == "DIFFERENT_SESSION":
    session_id = "different-claude-session"
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
if prompt == "INCOMPLETE":
    raise SystemExit(0)
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
{SEALED_PROVIDER_PROBE}
cursor_home = pathlib.Path(os.environ["HOME"])
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
        pathlib.Path("/repository") / "main-write.txt",
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
    "subtype": "error" if prompt == "CURSOR_ERROR_WITH_RESPONSE" else "success",
    "duration_ms": 1500,
    "duration_api_ms": 1250,
    "is_error": prompt == "CURSOR_ERROR_WITH_RESPONSE",
    "result": f"answer:{{prompt}}",
    "error": (
        "synthetic Cursor provider failure"
        if prompt == "CURSOR_ERROR_WITH_RESPONSE"
        else None
    ),
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
def fake_devin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    executable = tmp_path / "devin"
    source_data = tmp_path / "devin-data"
    source_data.mkdir()
    (source_data / "credentials.toml").write_text('token = "test"\n')
    (source_data / "cli").mkdir()
    (source_data / "cli" / "installed.bin").write_bytes(b"not copied")
    source_config = tmp_path / "devin-config"
    source_config.mkdir()
    (source_config / "config.json").write_text(
        '{"devin": {"org_id": "test"}, "mcp": {"secret": "instruction"}}\n'
    )
    monkeypatch.setenv("AOP_DEVIN_DATA_DIR", os.fspath(source_data))
    monkeypatch.setenv("AOP_DEVIN_CONFIG_DIR", os.fspath(source_config))
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys
import time

args = sys.argv[1:]
if args == ["models", "list", "--format", "json"]:
    print(json.dumps({{
        "families": [
            {{
                "family_label": "SWE-1.7",
                "family_uid": "swe-1.7",
                "variants": [
                    {{
                        "model_uid": "swe-1-7",
                        "label": "SWE-1.7 Max",
                        "cost_tier": "Free",
                    }},
                ],
            }},
        ],
    }}))
    raise SystemExit(0)

data_dir = pathlib.Path(os.environ["XDG_DATA_HOME"]) / "devin"
config_dir = pathlib.Path(os.environ["XDG_CONFIG_HOME"]) / "devin"
state_dir = pathlib.Path(os.environ["XDG_STATE_HOME"])
cache_dir = pathlib.Path(os.environ["XDG_CACHE_HOME"])
if not data_dir.joinpath("credentials.toml").is_file():
    raise RuntimeError("missing private Devin authentication")
if not config_dir.joinpath("config.json").is_file():
    raise RuntimeError("missing private Devin configuration")
if data_dir.joinpath("cli", "installed.bin").exists():
    raise RuntimeError("copied the installed Devin bundle")
for directory in [data_dir / "cli", state_dir, cache_dir]:
    directory.mkdir(parents=True, exist_ok=True)
state_path = data_dir / "cli" / "fake-session.json"
export_path = pathlib.Path(args[args.index("--export") + 1])
prompt = args[args.index("-p") + 1]
{SEALED_PROVIDER_PROBE}
concurrent_export = prompt.startswith("DEVIN_CONCURRENT_")
if concurrent_export:
    barrier = cache_dir / "concurrent-export-barrier"
    barrier.mkdir(parents=True, exist_ok=True)
    barrier.joinpath(prompt).touch()
    deadline = time.monotonic() + 5
    while len(list(barrier.iterdir())) < 2:
        if time.monotonic() >= deadline:
            raise RuntimeError("concurrent Devin test did not overlap")
        time.sleep(0.01)
if "--resume" in args:
    requested_session = args[args.index("--resume") + 1]
    state = json.loads(state_path.read_text())
    if state["session_id"] != requested_session:
        raise RuntimeError("session is absent from private state")
    session_id = requested_session
    model = state["model"]
else:
    session_id = (
        f"tested-{{prompt.rsplit('_', 1)[-1].lower()}}"
        if concurrent_export
        else "tested-basil"
    )
    model = args[args.index("--model") + 1]
    state_path.write_text(json.dumps({{"session_id": session_id, "model": model}}))
if prompt == "DEVIN_DIFFERENT_SESSION":
    session_id = "different-devin-session"
if prompt == "DEVIN_INCOMPLETE":
    print("partial", flush=True)
    raise SystemExit(0)
if prompt == "DEVIN_FAIL":
    print("synthetic Devin failure", file=sys.stderr)
    raise SystemExit(2)
if prompt.startswith("WRITE_ARTIFACT"):
    pathlib.Path(os.environ["AOP_OUTPUT_DIR"]).joinpath("report.md").write_text(
        "# Devin artifact\\n"
    )
if prompt == "CHECK_DEVIN_SANDBOX":
    pathlib.Path("agent-write.txt").write_text("allowed")
    for protected in [
        pathlib.Path("/repository") / "main-write.txt",
        pathlib.Path(".git"),
    ]:
        try:
            protected.write_text("forbidden")
        except OSError:
            pass
        else:
            raise RuntimeError(f"sandbox allowed write to {{protected}}")

message = f"answer:{{prompt}}"
export_path.write_text(json.dumps({{
    "schema_version": "ATIF-v1.7",
    "session_id": session_id,
    "agent": {{"name": "devin", "model_name": "SWE-1.7"}},
    "steps": [
        *(
            []
            if concurrent_export
            else [{{"step_id": 1, "source": "user", "message": prompt}}]
        ),
        {{
            "step_id": 2,
            "source": "agent",
            "message": message,
            "model_name": "SWE-1.7",
            "metrics": {{
                "prompt_tokens": 100,
                "cached_tokens": 30,
                "completion_tokens": 20,
            }},
            "extra": {{"generation_model": model}},
        }},
    ],
}}))
print(message, flush=True)
"""
    )
    executable.chmod(0o755)
    return executable


@pytest.fixture
def fake_opencode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    executable = tmp_path / "opencode"
    source_config = tmp_path / "opencode-config"
    source_config.mkdir()
    (source_config / "opencode.jsonc").write_text('{"instructions": ["AGENTS.md"]}\n')
    (source_config / "AGENTS.md").write_text("test instructions\n")
    dependencies = source_config / "node_modules" / "@opencode-ai" / "plugin"
    dependencies.mkdir(parents=True)
    (dependencies / "package.json").write_text('{"version": "test"}\n')
    source_data = tmp_path / "opencode-data"
    source_data.mkdir()
    (source_data / "auth.json").write_text('{"opencode": {"key": "test"}}\n')
    monkeypatch.setenv("AOP_OPENCODE_CONFIG_DIR", os.fspath(source_config))
    monkeypatch.setenv("AOP_OPENCODE_DATA_DIR", os.fspath(source_data))
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys

args = sys.argv[1:]
if args[0] != "run" or "--format" not in args or "--auto" not in args:
    raise RuntimeError(f"unexpected invocation: {{args}}")
prompt = args[args.index("--") + 1]
{SEALED_PROVIDER_PROBE}
config_dir = pathlib.Path(os.environ["XDG_CONFIG_HOME"]) / "opencode"
data_dir = pathlib.Path(os.environ["XDG_DATA_HOME"]) / "opencode"
state_dir = pathlib.Path(os.environ["XDG_STATE_HOME"]) / "opencode"
cache_dir = pathlib.Path(os.environ["XDG_CACHE_HOME"]) / "opencode"
auth_path = data_dir / "auth.json"
if not auth_path.is_file():
    raise RuntimeError("missing private OpenCode authentication")
if not config_dir.joinpath(
    "node_modules", "@opencode-ai", "plugin", "package.json"
).is_file():
    raise RuntimeError("missing shared OpenCode plugin dependencies")
auth_path.write_text('{{"opencode": {{"key": "refreshed"}}}}\\n')
config_dir.joinpath("package.json").write_text('{{"dependencies": {{}}}}\\n')
state_dir.mkdir(parents=True, exist_ok=True)
state_dir.joinpath("model.json").write_text('{{}}\\n')
cache_dir.mkdir(parents=True, exist_ok=True)
cache_dir.joinpath("version").write_text("test\\n")
session_id = (
    args[args.index("--session") + 1]
    if "--session" in args
    else "ses_opencode_test"
)
database = data_dir / "opencode.db"
if "--session" in args:
    state = json.loads(database.read_text())
    if state["session_id"] != session_id:
        raise RuntimeError("session is absent from private state")
else:
    database.write_text(json.dumps({{"session_id": session_id}}))
if prompt == "OPENCODE_DIFFERENT_SESSION":
    session_id = "ses_opencode_different"
if prompt.startswith("WRITE_ARTIFACT"):
    pathlib.Path(os.environ["AOP_OUTPUT_DIR"]).joinpath("report.md").write_text(
        "# OpenCode artifact\\n"
    )
if prompt == "CHECK_OPENCODE_SANDBOX":
    pathlib.Path("agent-write.txt").write_text("allowed")
    try:
        pathlib.Path("/repository").joinpath("main-write.txt").write_text(
            "forbidden"
        )
    except OSError:
        pass
    else:
        raise RuntimeError("sandbox allowed main-worktree mutation")

print(json.dumps({{
    "type": "step_start",
    "timestamp": 1000,
    "sessionID": session_id,
    "part": {{"type": "step-start"}},
}}), flush=True)
if prompt == "OPENCODE_TOOL_LOOP":
    print(json.dumps({{
        "type": "text",
        "timestamp": 1100,
        "sessionID": session_id,
        "part": {{"type": "text", "text": "intermediate"}},
    }}), flush=True)
    print(json.dumps({{
        "type": "step_finish",
        "timestamp": 1200,
        "sessionID": session_id,
        "part": {{
            "type": "step-finish",
            "reason": "tool-calls",
            "tokens": {{
                "total": 11,
                "input": 1,
                "output": 2,
                "reasoning": 1,
                "cache": {{"read": 3, "write": 4}},
            }},
            "cost": 0.0001,
        }},
    }}), flush=True)
    print(json.dumps({{
        "type": "step_start",
        "timestamp": 1300,
        "sessionID": session_id,
        "part": {{"type": "step-start"}},
    }}), flush=True)
if prompt in {{"OPENCODE_ERROR", "OPENCODE_ERROR_WITH_RESPONSE"}}:
    print(json.dumps({{
        "type": "error",
        "timestamp": 1200,
        "sessionID": session_id,
        "error": {{"name": "ProviderError", "data": {{"message": "synthetic failure"}}}},
    }}), flush=True)
if prompt != "OPENCODE_ERROR":
    print(json.dumps({{
        "type": "text",
        "timestamp": 1400,
        "sessionID": session_id,
        "part": {{"type": "text", "text": f"answer:{{prompt}}"}},
    }}), flush=True)
    print(json.dumps({{
        "type": "step_finish",
        "timestamp": 1500,
        "sessionID": session_id,
        "part": {{
            "type": "step-finish",
            "reason": "stop",
            "tokens": {{
                "total": 60,
                "input": 40,
                "output": 5,
                "reasoning": 2,
                "cache": {{"read": 10, "write": 3}},
            }},
            "cost": 0.00012345,
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
    (source_dir / "google_accounts.json").write_text('{"active": "test"}\n')
    (source_dir / "config").mkdir()
    (source_dir / "config" / "config.json").write_text("{}\n")
    (source_dir / "antigravity-cli").mkdir()
    (source_dir / "antigravity-cli" / "antigravity-oauth-token").write_text(
        "test-token\n"
    )
    (source_dir / "antigravity-cli" / "settings.json").write_text('{"study": true}\n')
    (source_dir / "antigravity").mkdir()
    (source_dir / "antigravity" / "browserAllowlist.txt").write_text("localhost\n")
    (source_dir / "antigravity" / "mcp_config.json").write_text(
        '{"mcpServers": {"study": {}}}\n'
    )
    (source_dir / "antigravity" / "user_settings.pb").write_text("study\n")
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
if args == ["--version"]:
    print("1.1.19")
    raise SystemExit(0)
gemini_dir = pathlib.Path(args[args.index("--gemini_dir") + 1])
runtime_dir = gemini_dir / "antigravity-cli"
runtime_dir.mkdir(parents=True, exist_ok=True)
prompt = args[args.index("-p") + 1]
settings_path = runtime_dir / "settings.json"
settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {{}}
direct_gemini = settings.get("modelProvider") == "gemini"
if direct_gemini and not os.environ.get("GEMINI_API_KEY"):
    print("missing Gemini API key", file=sys.stderr)
    raise SystemExit(2)
if not direct_gemini and not (runtime_dir / "antigravity-oauth-token").is_file():
    print("missing private authentication", file=sys.stderr)
    raise SystemExit(2)
if (
    prompt.startswith("CHECK_AGY_API_ROUTE")
    and os.environ.get("GOOGLE_GEMINI_BASE_URL") != "https://gemini.example.test/v1"
):
    print("missing Gemini endpoint override", file=sys.stderr)
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
{SEALED_PROVIDER_PROBE}
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
if prompt.startswith("AGY_ERROR_WITH_RESPONSE"):
    result["response"] = f"answer:{{prompt}}"
print(json.dumps({{"event": "result", "result": result}}), flush=True)
"""
    )
    executable.chmod(0o755)
    return executable


@pytest.fixture
def fake_hermes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    executable = tmp_path / "hermes"
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "auth.json").write_text("{}\n")
    (hermes_home / "skills" / "study-skill").mkdir(parents=True)
    (hermes_home / "skills" / "study-skill" / "SKILL.md").write_text(
        "concealed study instruction\n"
    )
    monkeypatch.setenv("HERMES_HOME", os.fspath(hermes_home))
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys
import time

args = sys.argv[1:]
state_path = pathlib.Path(os.environ["HERMES_HOME"]) / "fake-state.json"

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
{SEALED_PROVIDER_PROBE}
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
if prompt == "ROTATE_AUTH":
    auth_path = pathlib.Path(os.environ["HERMES_HOME"]) / "auth.json"
    auth = json.loads(auth_path.read_text())
    entry = auth["credential_pool"]["xai-oauth"][0]
    generation = int(entry["access_token"].removeprefix("generation-"))
    time.sleep(0.2)
    generation += 1
    entry["access_token"] = f"generation-{{generation}}"
    entry["refresh_token"] = f"refresh-{{generation}}"
    entry["last_refresh"] = f"2026-08-08T20:{{generation:02d}}:00Z"
    auth["updated_at"] = f"2026-08-08T20:{{generation:02d}}:01Z"
    auth_path.write_text(json.dumps(auth))
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
        "billing_provider": (
            args[args.index("--provider") + 1]
            if "--provider" in args
            else "nous"
        ),
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
        pathlib.Path("/repository") / "main-write.txt",
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


@pytest.fixture
def fake_dsh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source_home = tmp_path / "dsh-home"
    source_home.mkdir()
    source_home.joinpath(".credentials.yaml").write_text(
        'DEEPSEEK_API_KEY: "managed test: value"\n'
        "CUSTOM_ANTHROPIC_AUTH: managed-anthropic-value\n"
        "OPENAI_API_KEY: unrelated-secret\n"
    )
    source_home.joinpath("settings.yaml").write_text(
        "study: must-not-copy\n"
        "llm-pi-ai:\n"
        "  providers:\n"
        "    anthropic:\n"
        "      apiKeyEnv: CUSTOM_ANTHROPIC_AUTH\n"
        "    openai:\n"
        "      apiKeyEnv: OPENAI_API_KEY\n"
        "    amazon-bedrock: {}\n"
    )
    monkeypatch.setenv("AOP_DSH_SOURCE_HOME", os.fspath(source_home))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    executable = tmp_path / "dsh"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys

args = sys.argv[1:]
if args[:2] != ["--profile", "headless"] or "--patch" not in args:
    print("invalid dsh invocation", file=sys.stderr)
    raise SystemExit(2)
patch = pathlib.Path(args[args.index("--patch") + 1])
contents = patch.read_text()
if "aop-dsh-runner.mjs" not in contents:
    print("AOP dsh runner was not installed", file=sys.stderr)
    raise SystemExit(2)
if os.environ.get("DSH_TELEMETRY_DISABLED") != "1":
    print("dsh telemetry was not disabled", file=sys.stderr)
    raise SystemExit(2)
home = pathlib.Path(os.environ["DSH_HOME"])
if (
    "provider: \\\"amazon-bedrock\\\"" not in contents
    and not home.joinpath(".credentials.yaml").is_file()
    and not os.environ.get("DEEPSEEK_API_KEY")
):
    print("dsh credentials were not seeded", file=sys.stderr)
    raise SystemExit(2)
if home.joinpath("settings.yaml").exists() and "study:" in home.joinpath("settings.yaml").read_text():
    print("dsh user settings leaked into private state", file=sys.stderr)
    raise SystemExit(2)
session_id = os.environ["AOP_DSH_SESSION_ID"]
sessions = home / "fake-sessions"
sessions.mkdir(exist_ok=True)
session = sessions / session_id
if os.environ.get("AOP_DSH_RESUME") == "1" and not session.is_file():
    print("dsh resume state is missing", file=sys.stderr)
    raise SystemExit(1)
prompt = args[-1]
{SEALED_PROVIDER_PROBE}
if prompt == "CHECK_ENV_OVERRIDE" and os.environ.get("DEEPSEEK_API_KEY") != "environment-override":
    print("dsh environment credential did not take precedence", file=sys.stderr)
    raise SystemExit(2)
if prompt == "CHECK_PROVIDER":
    settings = home.joinpath("settings.yaml").read_text()
    credentials = home.joinpath(".credentials.yaml").read_text()
    if "provider: \\\"anthropic\\\"" not in contents or "model: \\\"claude-sonnet-4-5\\\"" not in contents:
        print("dsh provider selection was not patched", file=sys.stderr)
        raise SystemExit(2)
    if "CUSTOM_ANTHROPIC_AUTH" not in settings or "openai:" in settings:
        print("dsh provider settings were not projected", file=sys.stderr)
        raise SystemExit(2)
    if "CUSTOM_ANTHROPIC_AUTH" not in credentials or "DEEPSEEK_API_KEY" in credentials:
        print("dsh provider credential was not projected", file=sys.stderr)
        raise SystemExit(2)
    if os.environ.get("CUSTOM_ANTHROPIC_AUTH") != "environment-custom-value":
        print("dsh custom environment credential was not preserved", file=sys.stderr)
        raise SystemExit(2)
if prompt == "CHECK_NATIVE_PROVIDER" and home.joinpath(".credentials.yaml").exists():
    print("dsh provider-native auth received a managed credential", file=sys.stderr)
    raise SystemExit(2)
session.write_text(prompt)
if "anthropic.claude-sonnet-4-5-v1:0" in contents:
    model = "anthropic.claude-sonnet-4-5-v1:0"
elif "claude-sonnet-4-5" in contents:
    model = "claude-sonnet-4-5"
else:
    model = "deepseek-v4-pro" if "deepseek-v4-pro" in contents else "deepseek-v4-flash"
print(json.dumps({{"type": "aop.dsh.started", "session_id": session_id}}), flush=True)
if prompt == "FAIL":
    print(json.dumps({{
        "type": "aop.dsh.result",
        "session_id": session_id,
        "model": model,
        "final_message": None,
        "usage": {{}},
        "completed": False,
        "error": "synthetic dsh failure",
    }}), flush=True)
    raise SystemExit(1)
print(json.dumps({{
    "type": "aop.dsh.result",
    "session_id": session_id,
    "model": model,
    "final_message": f"answer:{{prompt}}",
    "usage": {{
        "input_tokens": 90,
        "cached_input_tokens": 30,
        "output_tokens": 20,
        "reasoning_output_tokens": 7,
    }},
    "completed": True,
    "error": (
        "synthetic dsh provider failure"
        if prompt == "DSH_ERROR_WITH_RESPONSE"
        else None
    ),
}}), flush=True)
"""
    )
    executable.chmod(0o755)
    return executable


@pytest.fixture
def fake_grok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source_home = tmp_path / "grok-home"
    source_home.mkdir()
    source_home.joinpath("auth.json").write_text(
        json.dumps(
            {
                "grok.com": {
                    "key": "test-token",
                    "auth_mode": "oidc",
                    "create_time": "2026-01-01T00:00:00Z",
                    "user_id": "test-user",
                }
            }
        )
    )
    source_home.joinpath("config.toml").write_text('[agent]\nname = "grok"\n')
    source_home.joinpath("rules").mkdir()
    source_home.joinpath("rules", "default.rules").write_text("test rule\n")
    source_home.joinpath("skills", "test-skill").mkdir(parents=True)
    source_home.joinpath("skills", "test-skill", "SKILL.md").write_text("test skill\n")
    source_home.joinpath("sessions").mkdir()
    source_home.joinpath("sessions", "global-session").write_text("global\n")
    source_home.joinpath("logs").mkdir()
    source_home.joinpath("logs", "unified.jsonl").write_text("global log\n")
    monkeypatch.setenv("AOP_GROK_SOURCE_HOME", os.fspath(source_home))

    executable = tmp_path / "grok"
    executable.write_text(
        f"""#!{sys.executable}
import json
import os
import pathlib
import sys

args = sys.argv[1:]
if args == ["models"]:
    print("Available models:")
    print("  * grok-build (default)")
    print("  * grok-4.5")
    raise SystemExit(0)

required = {{
    "--cwd", "--output-format", "--always-approve", "--no-plan",
    "--no-leader", "--no-auto-update", "--verbatim", "--sandbox", "--prompt-file",
    "--model",
}}
if not required.issubset(args):
    raise RuntimeError(f"unexpected Grok invocation: {{args}}")
if args[args.index("--output-format") + 1] != "streaming-json":
    raise RuntimeError("Grok output format is not streaming-json")
if args[args.index("--sandbox") + 1] != "off":
    raise RuntimeError("Grok nested sandbox was not disabled")

prompt = pathlib.Path(args[args.index("--prompt-file") + 1]).read_text()
{SEALED_PROVIDER_PROBE}
grok_home = pathlib.Path(os.environ["GROK_HOME"])
if not grok_home.joinpath("auth.json").is_file():
    raise RuntimeError("missing private Grok authentication")
if grok_home.joinpath("sessions", "global-session").exists():
    raise RuntimeError("global Grok sessions were copied")
if grok_home.joinpath("logs", "unified.jsonl").exists():
    raise RuntimeError("global Grok logs were copied")
if prompt == "CHECK_GROK_STORAGE" and os.environ.get("GROK_STORAGE_MODE") != "local":
    raise RuntimeError("GROK_STORAGE_MODE was not inherited")
if prompt == "CHECK_GROK_NO_STORAGE" and "GROK_STORAGE_MODE" in os.environ:
    raise RuntimeError("unset GROK_STORAGE_MODE was synthesized")

model = args[args.index("--model") + 1]
session_id = {SESSION_ID!r}
state_dir = grok_home / "sessions" / "fake-project"
state_dir.mkdir(parents=True, exist_ok=True)
state_path = state_dir / "session.json"
if "--resume" in args:
    session_id = args[args.index("--resume") + 1]
    state = json.loads(state_path.read_text())
    if state["session_id"] != session_id:
        raise RuntimeError("Grok session is absent from private state")
else:
    state_path.write_text(json.dumps({{"session_id": session_id}}))
if prompt == "GROK_DIFFERENT_SESSION":
    session_id = "11111111-2222-4333-8444-555555555555"

if prompt.startswith("WRITE_ARTIFACT"):
    pathlib.Path(os.environ["AOP_OUTPUT_DIR"]).joinpath("report.md").write_text(
        "# Grok artifact\\n"
    )

usage = {{
    "input_tokens": 40,
    "cache_read_input_tokens": 10,
    "cache_creation_input_tokens": 5,
    "output_tokens": 20,
    "reasoning_tokens": 7,
    "total_tokens": 75,
}}
terminal = {{
    "sessionId": session_id,
    "requestId": "grok-request",
    "usage": usage,
    "num_turns": 2,
    "modelUsage": {{
        model: {{
            "inputTokens": 40,
            "cacheReadInputTokens": 10,
            "cacheCreationInputTokens": 5,
            "outputTokens": 20,
            "modelCalls": 2,
            "costUSD": 0.00045678,
        }}
    }},
    "total_cost_usd": 0.00045678,
}}
if prompt == "GROK_ERROR":
    terminal.update({{"type": "error", "message": "synthetic Grok failure"}})
    print(json.dumps(terminal), flush=True)
    raise SystemExit(1)

print(json.dumps({{"type": "text", "data": "working"}}), flush=True)
print(json.dumps({{
    "type": "usage",
    "messageId": "intermediate",
    "stopReason": "tool_use",
}}), flush=True)
print(json.dumps({{"type": "text", "data": f"answer:{{prompt}}"}}), flush=True)
print(json.dumps({{
    "type": "usage",
    "messageId": "final",
    "stopReason": "end_turn",
}}), flush=True)
if prompt == "GROK_INCOMPLETE":
    raise SystemExit(0)
terminal.update({{
    "type": "end",
    "stopReason": (
        "max_turn_requests" if prompt == "GROK_MAX_TURNS" else "end_turn"
    ),
}})
print(json.dumps(terminal), flush=True)
"""
    )
    executable.chmod(0o755)
    return executable
