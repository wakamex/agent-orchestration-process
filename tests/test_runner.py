from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

from agent_orchestration_process import __version__
from agent_orchestration_process.cli import build_parser, main
from agent_orchestration_process.locks import exclusive_lock, task_lock_path
from agent_orchestration_process.models import RunRequest, RunResult
from agent_orchestration_process.runner import (
    AgentRunner,
    CodexAdapter,
    CursorAdapter,
    _codex_no_web_config,
    _filtered_environment,
    _launch_environment,
    _provider_command,
    _provider_runtime,
)
from agent_orchestration_process.worktrees import AOPError, WorktreeManager


SESSION_ID = "019f4da1-342f-7670-8aac-25999973b294"


def test_installed_codex_accepts_no_web_permission_profile_without_a_turn(
    tmp_path: Path,
) -> None:
    binary = shutil.which("codex")
    if binary is None:
        pytest.skip("Codex is not installed")

    codex_home = tmp_path / "codex-home"
    worktree = tmp_path / "worktree"
    codex_home.mkdir()
    worktree.mkdir()
    codex_home.joinpath("config.toml").write_text(
        'model_provider = "contract-check"\n'
        "[model_providers.contract-check]\n"
        'name = "Contract check"\n'
        'base_url = "http://127.0.0.1:9/v1"\n'
        'wire_api = "responses"\n'
        "requires_openai_auth = false\n"
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.endswith("_API_KEY")
    }
    environment["CODEX_HOME"] = os.fspath(codex_home)
    process = subprocess.Popen(
        [binary, "app-server", "--stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=environment,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    buffered = b""

    def send(value: dict[str, object]) -> None:
        process.stdin.write(json.dumps(value).encode() + b"\n")
        process.stdin.flush()

    def response(response_id: int) -> dict[str, object]:
        nonlocal buffered
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for _key, _events in selector.select(deadline - time.monotonic()):
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    pytest.fail("Codex app-server exited before responding")
                buffered += chunk
                while b"\n" in buffered:
                    line, buffered = buffered.split(b"\n", 1)
                    value = json.loads(line)
                    if value.get("id") == response_id:
                        return value
        pytest.fail(f"Codex app-server did not respond to request {response_id}")

    try:
        send(
            {
                "id": 1,
                "method": "initialize",
                "params": {
                    "clientInfo": {
                        "name": "aop-contract-check",
                        "title": "AOP contract check",
                        "version": "0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            }
        )
        assert "error" not in response(1)
        send({"method": "initialized", "params": {}})
        send(
            {
                "id": 2,
                "method": "thread/start",
                "params": {
                    "cwd": os.fspath(worktree),
                    "approvalPolicy": "never",
                    "permissions": "aop-no-web",
                    "ephemeral": True,
                    "config": _codex_no_web_config(
                        "edit", os.fspath(worktree), [os.fspath(worktree)]
                    ),
                },
            }
        )
        result = response(2)
        assert "error" not in result
        assert result["result"]["activePermissionProfile"]["id"] == "aop-no-web"
    finally:
        selector.close()
        process.stdin.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize(
    ("profile", "workspace_access", "root_access"),
    [
        ("edit", "write", "read"),
        ("review", "read", "read"),
        ("sealed", "read", "read"),
        ("host", None, "write"),
    ],
)
def test_codex_no_web_permission_profile_matches_aop_filesystem_contract(
    profile: str, workspace_access: str | None, root_access: str
) -> None:
    writable_roots = ["/output", "/scratch", "/state", "/cache", "/tmp"]
    config = _codex_no_web_config(profile, "/workspace", writable_roots)
    permission = config["permissions"]["aop-no-web"]
    filesystem = permission["filesystem"]

    assert config["web_search"] == "disabled"
    assert permission["network"] == {"enabled": False}
    assert filesystem[":root"] == root_access
    if workspace_access is None:
        assert "/workspace" not in filesystem
        assert set(filesystem) == {":root"}
    else:
        assert filesystem["/workspace"] == workspace_access
        for root in writable_roots:
            assert filesystem[root] == "write"
            for name in (".git", ".agents", ".codex"):
                assert filesystem[f"{root}/{name}"] == "write"
    if profile == "edit":
        assert filesystem["/workspace/.git"] == "read"
        assert filesystem["/workspace/.agents"] == "write"
        assert filesystem["/workspace/.codex"] == "write"


def test_codex_no_web_fails_closed_for_wrong_active_permission_profile(
    repository: Path, fake_codex: Path
) -> None:
    result = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    ).run(
        task="wrong-codex-permission-profile",
        prompt="test",
        model="wrong-active-profile",
        no_web=True,
    )

    assert not result.succeeded
    assert result.error == (
        "Codex did not activate the requested no-web permission profile aop-no-web"
    )


@pytest.mark.parametrize("profile", ["edit", "review", "sealed", "host"])
def test_installed_codex_executes_no_web_profile_inside_aop_filesystem_contract(
    repository: Path, tmp_path: Path, profile: str
) -> None:
    binary = shutil.which("codex")
    bwrap = shutil.which("bwrap")
    if binary is None:
        pytest.skip("Codex is not installed")
    if profile != "host" and bwrap is None:
        pytest.skip("bwrap is not installed")

    manager = WorktreeManager.discover(repository)
    worktree = manager.create(f"codex-{profile}")
    provider_state = tmp_path / "state"
    codex_home = provider_state / "codex" / "home"
    scratch = tmp_path / "scratch"
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    inputs = tmp_path / "inputs"
    for path in (codex_home, scratch, cache, output, inputs):
        path.mkdir(parents=True)

    protocol_cwd = os.fspath(worktree.path) if profile == "host" else "/workspace"
    writable_roots = (
        [] if profile == "host" else ["/output", "/scratch", "/state", "/cache", "/tmp"]
    )
    config = _codex_no_web_config(profile, protocol_cwd, writable_roots)
    permission = config["permissions"]["aop-no-web"]
    config_lines = [
        'default_permissions = "aop-no-web"',
        'web_search = "disabled"',
        "",
        "[permissions.aop-no-web.filesystem]",
    ]
    config_lines.extend(
        f"{json.dumps(path)} = {json.dumps(access)}"
        for path, access in permission["filesystem"].items()
    )
    config_lines.extend(["", "[permissions.aop-no-web.network]", "enabled = false", ""])
    codex_home.joinpath("config.toml").write_text("\n".join(config_lines))

    request = RunRequest(
        run_id="019f4da1-342f-7670-8aac-25999973b296",
        provider="codex",
        mode="agent",
        task=f"codex-{profile}",
        prompt="unused",
        base="HEAD",
        model=None,
        inference_provider=None,
        inference_route=None,
        effort=None,
        profile=profile,
        no_web=True,
        effective_policy={},
        timeout_seconds=10,
        session_id=None,
        parent_run_id=None,
        artifacts=(),
        inputs=(),
        created_at="2026-09-02T00:00:00+00:00",
    )
    environment = {
        **os.environ,
        "AOP_ROOT": os.fspath(repository),
        "AOP_CACHE_DIR": os.fspath(cache),
        "AOP_PROVIDER_STATE_DIR": os.fspath(provider_state),
        "AOP_SCRATCH_DIR": os.fspath(scratch),
        "AOP_OUTPUT_DIR": os.fspath(output),
        "AOP_INPUT_DIR": os.fspath(inputs),
        "AOP_PROFILE": profile,
        "CODEX_HOME": os.fspath(codex_home),
    }
    command = _provider_command(
        [binary, "app-server", "--stdio"], request, worktree, environment
    )
    process = subprocess.Popen(
        command,
        cwd=worktree.path,
        env=_launch_environment(command, environment),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)

    def send(value: dict[str, object]) -> None:
        process.stdin.write(json.dumps(value) + "\n")
        process.stdin.flush()

    def response(response_id: int) -> dict[str, object]:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            for _key, _events in selector.select(deadline - time.monotonic()):
                line = process.stdout.readline()
                if not line:
                    stderr = process.stderr.read() if process.stderr is not None else ""
                    pytest.fail(f"Codex app-server exited before responding: {stderr}")
                value = json.loads(line)
                if value.get("id") == response_id:
                    return value
        pytest.fail(f"Codex app-server did not respond to request {response_id}")

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(0.1)
    port = listener.getsockname()[1]
    output_target = (
        os.fspath(output / "probe") if profile == "host" else "/output/probe"
    )
    workspace_target = (
        os.fspath(worktree.path / "probe") if profile == "host" else "/workspace/probe"
    )
    script = (
        'printf aux > "$1"; '
        'if printf workspace > "$2"; then workspace=writable; else workspace=readonly; fi; '
        f"if exec 3<>/dev/tcp/127.0.0.1/{port}; then network=allowed; else network=blocked; fi; "
        'printf "%s|%s" "$workspace" "$network"'
    )

    try:
        send(
            {
                "id": 1,
                "method": "initialize",
                "params": {
                    "clientInfo": {
                        "name": "aop-contract-check",
                        "title": "AOP contract check",
                        "version": "0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            }
        )
        assert "error" not in response(1)
        send({"method": "initialized", "params": {}})
        send(
            {
                "id": 2,
                "method": "command/exec",
                "params": {
                    "command": [
                        "/bin/bash",
                        "-c",
                        script,
                        "aop-contract-check",
                        output_target,
                        workspace_target,
                    ],
                    "cwd": protocol_cwd,
                    "timeoutMs": 5000,
                    "permissionProfile": "aop-no-web",
                },
            }
        )
        result = response(2)
        assert "error" not in result, result
        expected_workspace = "writable" if profile in {"edit", "host"} else "readonly"
        assert result["result"]["exitCode"] == 0
        assert result["result"]["stdout"] == f"{expected_workspace}|blocked"
        assert output.joinpath("probe").read_text() == "aux"
        assert worktree.path.joinpath("probe").exists() is (profile in {"edit", "host"})
        with pytest.raises(TimeoutError):
            listener.accept()
    finally:
        listener.close()
        selector.close()
        process.stdin.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_filtered_environment_allows_optional_grok_storage_mode() -> None:
    assert _filtered_environment(
        {"GROK_STORAGE_MODE": "local", "AOP_INTERNAL_SECRET": "hidden"}
    ) == {"GROK_STORAGE_MODE": "local"}
    assert _filtered_environment({"AOP_INTERNAL_SECRET": "hidden"}) == {}


def test_dsh_runtime_preserves_a_user_npm_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = tmp_path / "npm"
    package = prefix / "lib" / "node_modules" / "@deepseek-ai" / "dsh"
    binary = package / "lib" / "bin.js"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/usr/bin/env node\n")
    binary.chmod(0o755)
    executable = prefix / "bin" / "dsh"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(binary)
    monkeypatch.setenv("PATH", f"{executable.parent}:{os.environ['PATH']}")

    command, mounts = _provider_runtime(["dsh", "--version"], provider="dsh")

    node_modules = prefix / "lib" / "node_modules"
    assert command == [os.fspath(binary), "--version"]
    assert mounts == [(os.fspath(node_modules), os.fspath(node_modules))]


def test_cursor_runtime_projects_wrapper_siblings_read_only(
    repository: Path,
    fake_cursor: Path,
    tmp_path: Path,
) -> None:
    version = tmp_path / "cursor-agent" / "versions" / "2026.08.11-test"
    version.mkdir(parents=True)
    wrapper = version / "cursor-agent"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'script_dir="$(dirname "$(realpath "$0")")"\n'
        'exec "$script_dir/node" "$script_dir/index.js" "$@"\n'
    )
    wrapper.chmod(0o755)
    node = version / "node"
    node.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'script_dir="$(dirname "$(realpath "$0")")"\n'
        'test -f "$script_dir/runtime-chunk.js"\n'
        'if touch "$script_dir/runtime-write" 2>/dev/null; then\n'
        "  exit 91\n"
        "fi\n"
        'exec /usr/bin/python3 "$@"\n'
    )
    node.chmod(0o755)
    version.joinpath("index.js").write_text(fake_cursor.read_text())
    version.joinpath("runtime-chunk.js").write_text("runtime chunk\n")

    result = AgentRunner(
        WorktreeManager.discover(repository),
        CursorAdapter(os.fspath(wrapper)),
    ).run(task="cursor-sibling-runtime", prompt="test", timeout_seconds=5)

    assert result.succeeded
    assert result.final_message == "answer:test"
    assert not version.joinpath("runtime-write").exists()
    assert ["--ro-bind", os.fspath(version), "/runtime/provider"] in [
        result.command[index : index + 3] for index in range(len(result.command) - 2)
    ]
    request = json.loads(
        repository.joinpath(".aop", "runs", result.run_id, "request.json").read_text()
    )
    assert request["effective_policy"]["provider_runtime"]["executable"] == (
        os.fspath(wrapper)
    )
    assert request["effective_policy"]["provider_runtime"]["sha256"]


def test_hermes_runtime_projects_only_editable_python_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_root = tmp_path / "hermes-agent"
    venv = agent_root / "venv"
    site_packages = venv / "lib" / "python3.11" / "site-packages"
    source_package = agent_root / "hermes_cli"
    source_module = agent_root / "batch_runner.py"
    interpreter_root = tmp_path / "python-runtime"
    interpreter = interpreter_root / "bin" / "python3.11"
    wrapper = tmp_path / "bin" / "hermes"
    for directory in (
        site_packages,
        source_package,
        interpreter.parent,
        wrapper.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("runtime\n")
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(interpreter)
    entrypoint = agent_root / "hermes"
    entrypoint.write_text("launcher\n")
    source_module.write_text("module\n")
    finder = site_packages / "__editable___hermes_agent_finder.py"
    finder.write_text(
        "from __future__ import annotations\n"
        "MAPPING: dict[str, str] = {"
        f"'hermes_cli': {os.fspath(source_package)!r}, "
        f"'batch_runner': {os.fspath(source_module.with_suffix(''))!r}"
        "}\n"
    )
    wrapper.write_text(f'#!/usr/bin/env bash\nexec "{python}" "{entrypoint}" "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{wrapper.parent}:{os.environ['PATH']}")

    command, mounts = _provider_runtime(
        ["hermes", "chat", "-q", "test"], provider="hermes"
    )

    assert command == [
        os.fspath(python),
        os.fspath(entrypoint),
        "chat",
        "-q",
        "test",
    ]
    assert mounts == [
        (os.fspath(source_module), os.fspath(source_module)),
        (os.fspath(entrypoint), os.fspath(entrypoint)),
        (os.fspath(source_package), os.fspath(source_package)),
        (os.fspath(venv), os.fspath(venv)),
        (os.fspath(interpreter_root), os.fspath(interpreter_root)),
    ]


def test_cli_reports_version_and_provider_neutral_resume_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"aop {__version__}\n"
    help_text = parser.format_help()
    assert "resume an agent session from a run" in help_text
    assert "resume the Codex session" not in help_text
    args = parser.parse_args(
        [
            "run",
            "player",
            "--agent",
            "hermes",
            "--mode",
            "participant",
            "--prompt",
            "play",
        ]
    )
    assert args.mode == "participant"
    provider_args = parser.parse_args(
        [
            "run",
            "routed",
            "--agent",
            "hermes",
            "--provider",
            "xai-oauth",
            "--model",
            "grok-build-0.1",
            "--prompt",
            "test",
        ]
    )
    assert provider_args.inference_provider == "xai-oauth"
    assert (
        parser.parse_args(
            ["run", "worker", "--agent", "opencode", "--prompt", "fix"]
        ).agent
        == "opencode"
    )
    read_args = parser.parse_args(
        [
            "run",
            "reader",
            "--input",
            "/sources/one",
            "--input",
            "/sources/two",
            "--prompt",
            "inspect",
        ]
    )
    assert read_args.input_paths == ["/sources/one", "/sources/two"]

    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(["resume", "--help"])
    assert help_exit.value.code == 0
    resume_help = capsys.readouterr().out
    assert "replace inherited input snapshots" in resume_help


def test_run_persists_structured_codex_artifacts(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    result = runner.run(
        task="implement",
        prompt="make the change",
        model="gpt-5.6-sol",
        effort="high",
        timeout_seconds=5,
    )

    assert result.succeeded
    assert result.session_id == SESSION_ID
    assert result.model == "gpt-5.6-sol"
    assert result.effort == "high"
    assert result.final_message == "answer:make the change"
    assert result.usage.input_tokens == 1
    assert result.usage.output_tokens == 1
    assert result.usage.reasoning_output_tokens == 0
    assert result.calculated_cost is not None
    assert result.calculated_cost.amount_usd == 0.000035
    assert result.provider_reported_cost is None
    assert result.billing.route == "subscription"
    assert result.billing.credential_source == "chatgpt-oauth"
    assert result.billing.detected_by == "codex login status"
    assert result.time_to_first_event_seconds is not None
    assert result.time_to_first_response_seconds is not None
    assert result.provider_duration_seconds is None
    assert result.provider_status == "completed"
    assert result.command[0] == "bwrap"
    assert result.command[-2:] == ["app-server", "--stdio"]

    run_dir = repository / ".aop" / "runs" / result.run_id
    assert (repository / ".aop" / "runs").stat().st_mode & 0o777 == 0o700
    assert run_dir.stat().st_mode & 0o777 == 0o700
    request = json.loads((run_dir / "request.json").read_text())
    persisted_result = json.loads((run_dir / "result.json").read_text())
    events = (run_dir / "events.jsonl").read_text()

    assert request["prompt"] == "make the change"
    assert request["model"] == "gpt-5.6-sol"
    assert request["inference_provider"] is None
    assert request["artifacts"] == []
    assert request["inputs"] == []
    assert persisted_result["succeeded"] is True
    assert persisted_result["usage_schema"] == "aop-token-usage-v2"
    assert persisted_result["accounting_status"] == "complete"
    assert persisted_result["artifacts"] == []
    assert persisted_result["inference_provider"] is None
    assert persisted_result["inputs"] == []
    assert not (run_dir / "input-manifest.json").exists()
    assert persisted_result["billing"] == {
        "route": "subscription",
        "credential_source": "chatgpt-oauth",
        "detected_by": "codex login status",
    }
    assert persisted_result["calculated_cost"]["pricing_version"].startswith(
        "models-dev-"
    )
    assert persisted_result["provider_reported_cost"] is None

    legacy_result = dict(persisted_result)
    legacy_result.pop("usage_schema")
    legacy_result["api_equivalent_cost"] = dict(legacy_result.pop("calculated_cost"))
    legacy_result["api_equivalent_cost"]["estimated"] = True
    legacy_result.pop("provider_reported_cost")
    legacy_result["billing"] = {
        **legacy_result["billing"],
        "actual_cost_known": False,
    }
    loaded_legacy = RunResult.from_dict(legacy_result)
    assert loaded_legacy.calculated_cost is not None
    assert loaded_legacy.calculated_cost.amount_usd == 0.000035
    assert loaded_legacy.provider_reported_cost is None
    legacy_result["read_paths"] = legacy_result.pop("inputs")
    loaded_read_paths = RunResult.from_dict(legacy_result)
    assert loaded_read_paths.inputs == loaded_legacy.inputs
    assert '"method": "thread/start"' in events
    assert '"approvalPolicy": "never"' in events
    assert '"sandbox": "danger-full-access"' in events
    assert (run_dir / "stderr.log").read_text() == ""


@pytest.mark.parametrize("profile", ["review", "edit"])
def test_declared_inputs_are_snapshotted_hashed_and_recorded(
    repository: Path,
    fake_codex: Path,
    tmp_path: Path,
    profile: str,
) -> None:
    sources = tmp_path / "sources"
    transcripts = sources / "transcripts"
    transcripts.mkdir(parents=True)
    day_four = transcripts / "day-4.md"
    day_four.write_text("Day four\n")
    nested = transcripts / "nested"
    nested.mkdir()
    day_five = nested / "day-5.md"
    day_five.write_text("Day five\n")
    ledger = sources / "ledger.json"
    ledger.write_text('{"status":"verified"}\n')
    manager = WorktreeManager.discover(repository)

    result = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).run(
        task="reader",
        prompt="CHECK_READ_PATHS",
        profile=profile,
        input_paths=[transcripts, ledger],
    )

    assert result.succeeded
    assert [Path(item.mounted_path).name for item in result.inputs] == [
        "transcripts",
        "ledger.json",
    ]
    transcript_input, ledger_input = result.inputs
    assert transcript_input.source_path == os.fspath(transcripts.resolve())
    assert transcript_input.kind == "directory"
    assert transcript_input.size_bytes == len(b"Day four\nDay five\n")
    assert [item.relative_path for item in transcript_input.files] == [
        "day-4.md",
        "nested/day-5.md",
    ]
    assert transcript_input.files[0].sha256 == hashlib.sha256(b"Day four\n").hexdigest()
    assert ledger_input.kind == "file"
    assert ledger_input.sha256 == hashlib.sha256(ledger.read_bytes()).hexdigest()
    assert ledger_input.files[0].relative_path == "ledger.json"
    assert not (transcripts / "forbidden").exists()
    assert ledger.read_text() == '{"status":"verified"}\n'

    assert [
        "--ro-bind",
        os.fspath(manager.state_dir / "snapshots" / result.run_id),
        "/inputs",
    ] in [result.command[index : index + 3] for index in range(len(result.command) - 2)]
    assert all(item.source_path not in result.command for item in result.inputs)

    run_dir = manager.state_dir / "runs" / result.run_id
    request = json.loads((run_dir / "request.json").read_text())
    persisted_result = json.loads((run_dir / "result.json").read_text())
    manifest = json.loads((run_dir / "input-manifest.json").read_text())
    assert request["inputs"] == persisted_result["inputs"]
    assert manifest == {"schema_version": 1, "inputs": request["inputs"]}
    assert transcript_input.mounted_path in request["prompt"]
    assert transcript_input.source_path not in request["prompt"]
    snapshot_root = manager.state_dir / "snapshots" / result.run_id
    assert (snapshot_root / "transcripts" / "day-4.md").read_text() == "Day four\n"
    assert (snapshot_root / "ledger.json").read_bytes() == ledger.read_bytes()
    assert snapshot_root.stat().st_mode & 0o777 == 0o700
    assert (snapshot_root / "transcripts").stat().st_mode & 0o777 == 0o700
    assert (snapshot_root / "ledger.json").stat().st_mode & 0o777 == 0o600


def test_resume_inherits_input_sources_and_refreshes_snapshots(
    repository: Path,
    fake_codex: Path,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.md"
    source.write_text("first\n")
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    first = runner.run(task="reader-resume", prompt="first", input_paths=[source])
    source.write_text("second\n")
    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert first.succeeded
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert resumed.inputs[0].source_path == first.inputs[0].source_path
    assert resumed.inputs[0].sha256 == hashlib.sha256(b"second\n").hexdigest()
    assert resumed.inputs[0].sha256 != first.inputs[0].sha256
    assert resumed.inputs[0].mounted_path == first.inputs[0].mounted_path


def test_declared_inputs_reject_unsafe_or_ambiguous_sources(
    repository: Path,
    fake_codex: Path,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "source.md"
    second = tmp_path / "second" / "source.md"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("one\n")
    second.write_text("two\n")
    linked = tmp_path / "linked.md"
    linked.symlink_to(first)
    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "linked.md").symlink_to(first)
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    with pytest.raises(AOPError, match="same basename"):
        runner.run(task="duplicate-read", prompt="unused", input_paths=[first, second])
    with pytest.raises(AOPError, match="may not be a symlink"):
        runner.run(task="linked-read", prompt="unused", input_paths=[linked])
    with pytest.raises(AOPError, match="contains a symlink"):
        runner.run(task="nested-linked-read", prompt="unused", input_paths=[directory])
    host_result = runner.run(
        task="host-input",
        prompt="unused",
        profile="host",
        input_paths=[first],
    )
    assert host_result.inputs


@pytest.mark.parametrize("profile", ["review", "edit", "host"])
def test_codex_uses_private_runtime_state_with_read_only_global_profile(
    repository: Path,
    fake_codex: Path,
    profile: str,
) -> None:
    source_home = Path(os.environ["AOP_CODEX_SOURCE_HOME"])
    for path in source_home.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source_home.chmod(0o555)
    task = f"managed-codex-{profile}"
    manager = WorktreeManager.discover(repository)

    try:
        runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))
        first = runner.run(task=task, prompt="first", profile=profile)
        resumed = runner.resume(run_id=first.run_id, prompt="second")
    finally:
        source_home.chmod(0o755)
        for path in source_home.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)

    private_home = manager.state_dir / "provider-state" / task / "codex" / "home"
    worktree = manager.get(task).path
    assert first.succeeded
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert private_home.joinpath("auth.json").is_file()
    assert private_home.joinpath("config.toml").is_file()
    assert private_home.joinpath("models_cache.json").is_file()
    assert private_home.joinpath("rules", "default.rules").is_file()
    assert private_home.joinpath("skills", "user-skill", "SKILL.md").is_file()
    assert not private_home.joinpath("skills", ".system", "generated").exists()
    assert private_home.joinpath("sessions", SESSION_ID).is_file()
    assert private_home.joinpath("history.jsonl").read_text() == "first\nsecond\n"
    assert source_home.joinpath("history.jsonl").read_text() == "global history\n"
    assert source_home.joinpath("state_5.sqlite").read_text() == "global database\n"
    assert source_home.joinpath("sessions", "global-session").is_file()
    assert not list(worktree.rglob("auth.json"))

    manager.remove(task)
    assert not private_home.exists()
    assert (manager.state_dir / "runs" / first.run_id / "result.json").is_file()
    assert (manager.state_dir / "runs" / resumed.run_id / "result.json").is_file()


def test_codex_records_metered_api_authentication_without_credentials(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOP_FAKE_CODEX_AUTH", "api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-persisted")
    result = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    ).run(task="codex-api-billing", prompt="test")

    assert result.succeeded
    assert result.billing.route == "metered-api"
    assert result.billing.credential_source == "openai-api-key"
    persisted = repository / ".aop" / "runs" / result.run_id / "result.json"
    assert "must-not-be-persisted" not in persisted.read_text()


def test_safe_profile_does_not_mount_unrelated_control_directories(
    repository: Path,
    fake_codex: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = tmp_path / "control"
    control.mkdir()
    manager = WorktreeManager.discover(repository)
    result = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).run(
        task="hidden-control", prompt="make the change", timeout_seconds=5
    )

    assert os.fspath(control) not in result.command
    assert ["--dev-bind", "/", "/"] not in [
        result.command[index : index + 3] for index in range(len(result.command) - 2)
    ]
    assert result.succeeded


def test_workspace_sandbox_mounts_shared_git_metadata_read_only(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    result = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).run(
        task="read-only-git", prompt="make the change", timeout_seconds=5
    )
    worktree = manager.get("read-only-git").path
    common = Path(
        subprocess.run(
            (
                "git",
                "-C",
                os.fspath(worktree),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    triplets = [
        result.command[index : index + 3] for index in range(len(result.command) - 2)
    ]
    assert ["--ro-bind", os.fspath(common), os.fspath(common)] in triplets
    assert ["--ro-bind", os.fspath(common), "/git"] in triplets
    assert result.succeeded


def test_declared_artifact_is_validated_and_archived(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    result = runner.run(
        task="extract",
        prompt="WRITE_ARTIFACT",
        profile="review",
        artifacts=["paper.md"],
    )

    assert result.succeeded
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    content = b"# Extracted\n"
    assert artifact.logical_path == "paper.md"
    assert artifact.archive_path == "artifacts/paper.md"
    assert artifact.size_bytes == len(content)
    assert artifact.sha256 == hashlib.sha256(content).hexdigest()

    run_dir = manager.state_dir / "runs" / result.run_id
    assert (run_dir / artifact.archive_path).read_bytes() == content
    assert "narrating before writing" in (run_dir / "events.jsonl").read_text()
    request = json.loads((run_dir / "request.json").read_text())
    assert "/output" in request["prompt"]
    assert request["artifacts"] == ["paper.md"]


def test_declared_artifact_directory_is_recursively_archived(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    result = runner.run(
        task="linked-extract",
        prompt="WRITE_ARTIFACT_TREE",
        profile="review",
        artifacts=["paper.md", "assets"],
    )

    assert result.succeeded
    assert [artifact.logical_path for artifact in result.artifacts] == [
        "paper.md",
        "assets/figure-1.png",
        "assets/nested/figure-2.svg",
    ]
    expected = {
        "paper.md": b"# Extracted\n\n![Figure](assets/figure-1.png)\n",
        "assets/figure-1.png": b"PNG figure 1",
        "assets/nested/figure-2.svg": b"<svg>figure 2</svg>\n",
    }
    run_dir = manager.state_dir / "runs" / result.run_id
    for artifact in result.artifacts:
        content = expected[artifact.logical_path]
        assert (run_dir / artifact.archive_path).read_bytes() == content
        assert artifact.size_bytes == len(content)
        assert artifact.sha256 == hashlib.sha256(content).hexdigest()


def test_declared_empty_artifact_directory_is_valid(
    repository: Path, fake_codex: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    result = runner.run(
        task="empty-assets",
        prompt="WRITE_EMPTY_ARTIFACT_DIRECTORY",
        profile="review",
        artifacts=["assets"],
    )

    assert result.succeeded
    assert result.artifacts == ()


@pytest.mark.parametrize(
    ("prompt", "diagnostic"),
    [
        ("WRITE_ARTIFACT_TREE_SYMLINK", "symlinks are not allowed"),
        ("WRITE_ARTIFACT_TREE_EMPTY_FILE", "empty"),
        ("WRITE_ARTIFACT_TREE_SPECIAL_FILE", "not a regular file"),
    ],
)
def test_unsafe_file_in_declared_artifact_directory_fails_the_run(
    repository: Path,
    fake_codex: Path,
    prompt: str,
    diagnostic: str,
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    result = runner.run(
        task=f"unsafe-tree-{diagnostic.split()[0]}",
        prompt=prompt,
        profile="review",
        artifacts=["assets"],
    )

    assert not result.succeeded
    assert result.error == f"artifact assets/figure-1.png: {diagnostic}"
    assert result.artifacts == ()


def test_overlapping_artifact_declarations_fail_before_launch(
    repository: Path, fake_codex: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    with pytest.raises(
        AOPError,
        match="overlapping artifact paths: assets and assets/figure-1.png",
    ):
        runner.run(
            task="overlapping-artifacts",
            prompt="WRITE_ARTIFACT_TREE",
            profile="review",
            artifacts=["assets", "assets/figure-1.png"],
        )


@pytest.mark.parametrize(
    ("prompt", "diagnostic"),
    [
        ("MISSING_ARTIFACT", "missing"),
        ("EMPTY_ARTIFACT", "empty"),
        ("SYMLINK_ARTIFACT", "symlinks are not allowed"),
    ],
)
def test_invalid_declared_artifact_fails(
    repository: Path,
    fake_codex: Path,
    prompt: str,
    diagnostic: str,
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    result = runner.run(
        task=f"invalid-{diagnostic.split()[0]}",
        prompt=prompt,
        profile="review",
        artifacts=["paper.md"],
    )

    assert not result.succeeded
    assert result.exit_code == 0
    assert result.error == f"artifact paper.md: {diagnostic}"
    assert result.artifacts == ()
    run_dir = repository / ".aop" / "runs" / result.run_id
    assert not (run_dir / "artifacts" / "paper.md").exists()


def test_previous_run_artifact_cannot_satisfy_a_new_run(
    repository: Path, fake_codex: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )
    first = runner.run(
        task="repeat-extract",
        prompt="WRITE_ARTIFACT",
        profile="review",
        artifacts=["paper.md"],
    )

    second = runner.run(
        task="repeat-extract",
        prompt="MISSING_ARTIFACT",
        profile="review",
        artifacts=["paper.md"],
    )

    assert first.succeeded
    assert not second.succeeded
    assert second.error == "artifact paper.md: missing"
    assert first.run_id != second.run_id


@pytest.mark.parametrize("path", ["../paper.md", "/tmp/paper.md", "."])
def test_unsafe_artifact_declarations_fail_before_launch(
    repository: Path, fake_codex: Path, path: str
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    )

    with pytest.raises(AOPError, match="relative and contained"):
        runner.run(task="unsafe-path", prompt="unused", artifacts=[path])

    assert not (repository / ".aop" / "worktrees" / "unsafe-path").exists()


def test_resume_uses_recorded_session_and_links_runs(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))
    first = runner.run(
        task="review",
        prompt="first",
        model="gpt-5.6-terra",
        effort="medium",
        timeout_seconds=5,
    )

    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert resumed.succeeded
    assert resumed.session_id == SESSION_ID
    assert resumed.model == "gpt-5.6-terra"
    assert resumed.effort == "medium"
    assert resumed.calculated_cost is not None
    assert resumed.final_message == "answer:second"
    assert resumed.command[-2:] == ["app-server", "--stdio"]
    first_events = [
        json.loads(line)
        for line in (manager.state_dir / "runs" / first.run_id / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    resumed_events = [
        json.loads(line)
        for line in (manager.state_dir / "runs" / resumed.run_id / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    first_start = next(
        event for event in first_events if event.get("method") == "thread/start"
    )
    first_turn = next(
        event for event in first_events if event.get("method") == "turn/start"
    )
    resume = next(
        event for event in resumed_events if event.get("method") == "thread/resume"
    )
    resumed_turn = next(
        event for event in resumed_events if event.get("method") == "turn/start"
    )
    assert first_start["params"]["model"] == "gpt-5.6-terra"
    assert first_turn["params"]["effort"] == "medium"
    assert resume["params"]["threadId"] == SESSION_ID
    assert "model" not in resume["params"]
    assert "effort" not in resumed_turn["params"]

    request_path = repository / ".aop" / "runs" / resumed.run_id / "request.json"
    request = json.loads(request_path.read_text())
    assert request["parent_run_id"] == first.run_id
    assert request["session_id"] == SESSION_ID
    assert request["timeout_seconds"] == 5


def test_resume_uses_a_fresh_explicit_artifact_contract(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))
    first = runner.run(
        task="artifact-resume",
        prompt="WRITE_ARTIFACT",
        profile="review",
        artifacts=["paper.md"],
    )

    resumed = runner.resume(run_id=first.run_id, prompt="MISSING_ARTIFACT")

    assert first.succeeded
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert resumed.artifacts == ()
    request = json.loads(
        (manager.state_dir / "runs" / resumed.run_id / "request.json").read_text()
    )
    assert request["artifacts"] == []
    assert request["parent_run_id"] == first.run_id


def test_resume_reuses_lock_held_by_aop_exec(
    repository: Path, fake_codex: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))
    first = runner.run(task="nested-resume", prompt="first", timeout_seconds=5)
    monkeypatch.setenv("AOP_TASK_LOCK_HELD", "nested-resume")

    with exclusive_lock(
        task_lock_path(manager.state_dir, "nested-resume"), "task nested-resume"
    ):
        resumed = runner.resume(run_id=first.run_id, prompt="inside exec")

    assert resumed.succeeded
    assert resumed.final_message == "answer:inside exec"


def test_timeout_terminates_process_and_records_result(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    result = runner.run(task="slow", prompt="SLEEP", timeout_seconds=1)

    assert not result.succeeded
    assert result.timed_out
    assert result.error == "timed out after 1 seconds"
    assert result.accounting_status == "partial"
    assert result.usage is not None
    assert result.usage.input_tokens == 10
    assert result.usage.cached_input_tokens == 4
    assert result.usage.output_tokens == 2
    assert result.usage.reasoning_output_tokens == 1
    assert result.calculated_cost is not None
    assert result.calculated_cost.amount_usd == 0.000092
    assert result.provider_status == "interrupted"
    assert result.provider_reported_cost is None
    result_path = repository / ".aop" / "runs" / result.run_id / "result.json"
    persisted = json.loads(result_path.read_text())
    assert persisted["timed_out"] is True
    assert persisted["usage"] == {
        "input_tokens": 10,
        "cached_input_tokens": 4,
        "output_tokens": 2,
        "reasoning_output_tokens": 1,
    }
    assert persisted["accounting_status"] == "partial"


def test_codex_timeout_falls_back_to_process_termination_when_interrupt_hangs(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    adapter = CodexAdapter(os.fspath(fake_codex))
    adapter._SHUTDOWN_GRACE_SECONDS = 0.05

    result = AgentRunner(manager, adapter).run(
        task="hung-interrupt",
        prompt="HANG_INTERRUPT",
        timeout_seconds=1,
    )

    assert result.timed_out
    assert result.accounting_status == "unavailable"
    assert result.usage is None
    events = (manager.state_dir / "runs" / result.run_id / "events.jsonl").read_text()
    assert '"method": "turn/interrupt"' in events


def test_turn_failure_is_not_reported_as_success(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    result = runner.run(task="failure", prompt="FAIL")

    assert not result.succeeded
    assert result.exit_code == 1
    assert result.error == "synthetic failure"
    assert result.final_message is None


def test_codex_requires_a_terminal_completion_event(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    result = runner.run(task="incomplete", prompt="INCOMPLETE")

    assert not result.succeeded
    assert result.exit_code == 0
    assert result.session_id == SESSION_ID
    assert result.final_message == "answer:INCOMPLETE"
    assert result.error == "Codex did not emit a terminal turn/completed notification"


def test_codex_rejects_a_changed_resume_thread(
    repository: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))
    first = runner.run(task="codex-resume-identity", prompt="first")
    private_home = (
        manager.state_dir
        / "provider-state"
        / "codex-resume-identity"
        / "codex"
        / "home"
    )
    private_home.joinpath("fake-resume-id").write_text("different-codex-thread")

    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert not resumed.succeeded
    assert resumed.exit_code == 0
    assert resumed.session_id is None
    assert resumed.error == (
        "Codex resumed as thread different-codex-thread instead of the requested "
        f"thread {first.session_id}"
    )


def test_cli_runs_codex_and_reports_resumable_ids(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))

    exit_code = main(["run", "cli-task", "--prompt", "from cli"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "answer:from cli\n"
    assert f"session_id={SESSION_ID}" in captured.err
    run_id = re.search(r"run_id=([0-9a-f-]+)", captured.err)
    assert run_id is not None
    assert (repository / ".aop" / "runs" / run_id.group(1) / "result.json").exists()


def test_cli_prints_machine_readable_run_and_resume_results(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))

    assert main(["run", "json-task", "--prompt", "first", "--json"]) == 0
    first_output = capsys.readouterr()
    first = json.loads(first_output.out)

    assert first_output.err == ""
    assert first["succeeded"] is True
    assert first["task"] == "json-task"
    assert first["final_message"] == "answer:first"

    assert main(["resume", first["run_id"], "--prompt", "second", "--json"]) == 0
    resumed_output = capsys.readouterr()
    resumed = json.loads(resumed_output.out)

    assert resumed_output.err == ""
    assert resumed["succeeded"] is True
    assert resumed["session_id"] == first["session_id"]
    assert resumed["final_message"] == "answer:second"


def test_cli_cleanup_discards_a_run_worktree_but_keeps_run_records(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))
    manager = WorktreeManager.discover(repository)
    result = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).run(
        task="disposable", prompt="first"
    )
    run_dir = manager.state_dir / "runs" / result.run_id

    assert main(["cleanup", result.run_id]) == 0
    assert capsys.readouterr().out == "disposable\n"
    assert not any(item.task == "disposable" for item in manager.list())
    assert (run_dir / "request.json").exists()
    assert (run_dir / "result.json").exists()

    assert main(["cleanup", result.run_id]) == 0
    assert capsys.readouterr().out == "disposable\n"


def test_cli_accepts_artifact_declarations(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))

    exit_code = main(
        [
            "run",
            "cli-artifact",
            "--profile",
            "review",
            "--artifact",
            "paper.md",
            "--prompt",
            "WRITE_ARTIFACT",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    run_id = re.search(r"run_id=([0-9a-f-]+)", captured.err)
    assert run_id is not None
    result = json.loads(
        (repository / ".aop" / "runs" / run_id.group(1) / "result.json").read_text()
    )
    assert result["artifacts"][0]["logical_path"] == "paper.md"
