from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import pytest

from agent_orchestration_process.pricing import TokenUsage
from agent_orchestration_process.runner import (
    AgyAdapter,
    AgentRunner,
    ClaudeAdapter,
    CodexAdapter,
    CursorAdapter,
    DevinAdapter,
    DeepSeekHarnessAdapter,
    HermesAdapter,
    _HermesSession,
    OpenCodeAdapter,
)
from agent_orchestration_process.worktrees import AOPError, WorktreeManager


def test_dsh_run_and_exact_resume_use_the_native_patch_interface(
    repository: Path, fake_dsh: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, DeepSeekHarnessAdapter(os.fspath(fake_dsh)))

    first = runner.run(
        task="dsh-task",
        prompt="first",
        model="deepseek-v4-pro",
        effort="max",
        timeout_seconds=5,
    )
    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert first.succeeded
    assert first.provider == "dsh"
    assert first.model == "deepseek-v4-pro"
    assert first.effort == "max"
    assert first.final_message == "answer:first"
    assert first.usage.input_tokens == 90
    assert first.usage.cached_input_tokens == 30
    assert first.usage.output_tokens == 20
    assert first.usage.reasoning_output_tokens == 7
    assert first.api_equivalent_cost is not None
    assert first.api_equivalent_cost.amount_usd == 0.0001038
    assert first.billing.route == "metered-api"
    assert first.billing.credential_source == "deepseek-api-key"
    assert first.command[0] == "bwrap"
    assert ["--profile", "headless"] == first.command[
        first.command.index("--profile") : first.command.index("--profile") + 2
    ]
    patch = Path(
        first.command[first.command.index("--patch") + 1].replace(
            "/scratch", os.fspath(manager.state_dir / "scratch" / "dsh-task"), 1
        )
    )
    patch_text = patch.read_text()
    assert 'model: "deepseek-v4-pro"' in patch_text
    assert 'reasoningEffort: "max"' in patch_text
    assert "- id: session-title-llm\n  disabled: true" in patch_text
    assert "- id: headless-runner\n  disabled: true" in patch_text
    assert "    - id: aop-headless-runner" in patch_text
    assert "file:///state/dsh/home/profiles/headless/aop-dsh-runner.mjs" in patch_text
    runner_source = private_home = (
        manager.state_dir / "provider-state" / "dsh-task" / "dsh" / "home"
    )
    assert (
        "installModelSelection"
        in runner_source.joinpath(
            "profiles", "headless", "aop-dsh-runner.mjs"
        ).read_text()
    )
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert resumed.final_message == "answer:second"

    private_home = manager.state_dir / "provider-state" / "dsh-task" / "dsh" / "home"
    assert private_home.joinpath(".credentials.yaml").is_file()
    assert not private_home.joinpath("settings.yaml").exists()
    assert (
        private_home.joinpath("fake-sessions", first.session_id).read_text() == "second"
    )


def test_dsh_maps_none_effort_and_normalizes_failures(
    repository: Path, fake_dsh: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository),
        DeepSeekHarnessAdapter(os.fspath(fake_dsh)),
    )
    failed = runner.run(task="dsh-failure", prompt="FAIL", effort="none")

    assert not failed.succeeded
    assert failed.error == "synthetic dsh failure"
    scratch = repository / ".aop" / "scratch" / "dsh-failure"
    patch = next(scratch.glob("aop-dsh-*.cordis.yml"))
    assert 'reasoningEffort: "off"' in patch.read_text()

    with pytest.raises(AOPError, match="dsh effort must be one of"):
        runner.run(task="dsh-bad-effort", prompt="test", effort="ultra")


def test_dsh_projects_the_selected_provider_and_its_exact_credential_ref(
    repository: Path,
    fake_dsh: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUSTOM_ANTHROPIC_AUTH", "environment-custom-value")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-anthropic-run")
    manager = WorktreeManager.discover(repository)
    result = AgentRunner(manager, DeepSeekHarnessAdapter(os.fspath(fake_dsh))).run(
        task="dsh-anthropic",
        prompt="CHECK_PROVIDER",
        inference_provider="anthropic",
        model="claude-sonnet-4-5",
        effort="low",
    )

    assert result.succeeded
    assert result.inference_provider == "anthropic"
    assert result.model == "claude-sonnet-4-5"
    assert result.billing.credential_source == "custom-anthropic-auth"
    private_home = (
        manager.state_dir / "provider-state" / "dsh-anthropic" / "dsh" / "home"
    )
    settings = private_home.joinpath("settings.yaml").read_text()
    credentials = private_home.joinpath(".credentials.yaml").read_text()
    assert "anthropic:" in settings
    assert "CUSTOM_ANTHROPIC_AUTH" in settings
    assert "openai:" not in settings
    assert "study:" not in settings
    assert "CUSTOM_ANTHROPIC_AUTH" in credentials
    assert "DEEPSEEK_API_KEY" not in credentials
    assert "OPENAI_API_KEY" not in credentials
    assert "environment-custom-value" not in json.dumps(result.to_dict())
    assert "OPENAI_API_KEY" not in result.command
    ref_index = result.command.index("CUSTOM_ANTHROPIC_AUTH")
    assert result.command[ref_index - 1 : ref_index + 2] == [
        "--setenv",
        "CUSTOM_ANTHROPIC_AUTH",
        "<redacted>",
    ]


def test_dsh_preserves_provider_native_auth_when_api_key_env_is_omitted(
    repository: Path,
    fake_dsh: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "native-access-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "native-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-bedrock-run")
    manager = WorktreeManager.discover(repository)
    result = AgentRunner(manager, DeepSeekHarnessAdapter(os.fspath(fake_dsh))).run(
        task="dsh-bedrock",
        prompt="CHECK_NATIVE_PROVIDER",
        inference_provider="amazon-bedrock",
        model="anthropic.claude-sonnet-4-5-v1:0",
    )

    assert result.succeeded
    assert result.billing.credential_source == "provider-native"
    assert "AWS_ACCESS_KEY_ID" in result.command
    assert "AWS_SECRET_ACCESS_KEY" in result.command
    assert "OPENAI_API_KEY" not in result.command
    assert "native-access-id" not in json.dumps(result.to_dict())
    assert "native-secret" not in json.dumps(result.to_dict())
    private_home = manager.state_dir / "provider-state" / "dsh-bedrock" / "dsh" / "home"
    assert private_home.joinpath("settings.yaml").is_file()
    assert not private_home.joinpath(".credentials.yaml").exists()


def test_sealed_dsh_projects_only_its_managed_credential(
    tmp_path: Path, fake_dsh: Path
) -> None:
    manager = WorktreeManager.standalone(tmp_path / "controller")
    result = AgentRunner(manager, DeepSeekHarnessAdapter(os.fspath(fake_dsh))).run(
        task="sealed-dsh", prompt="test", profile="sealed"
    )

    assert result.succeeded
    private_credentials = (
        manager.state_dir
        / "provider-state"
        / result.run_id
        / "dsh"
        / "home"
        / ".credentials.yaml"
    )
    content = private_credentials.read_text()
    assert "DEEPSEEK_API_KEY:" in content
    assert "managed test: value" in content
    assert "OPENAI_API_KEY" not in content
    assert "unrelated-secret" not in content
    assert private_credentials.stat().st_mode & 0o777 == 0o600
    assert "managed test: value" not in json.dumps(result.to_dict())


def test_sealed_dsh_preserves_and_redacts_native_environment_override(
    tmp_path: Path,
    fake_dsh: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-override")
    result = AgentRunner(
        WorktreeManager.standalone(tmp_path / "controller"),
        DeepSeekHarnessAdapter(os.fspath(fake_dsh)),
    ).run(task="sealed-dsh-env", prompt="CHECK_ENV_OVERRIDE", profile="sealed")

    assert result.succeeded
    assert "environment-override" not in json.dumps(result.to_dict())
    key_index = result.command.index("DEEPSEEK_API_KEY")
    assert result.command[key_index - 1 : key_index + 2] == [
        "--setenv",
        "DEEPSEEK_API_KEY",
        "<redacted>",
    ]


def test_claude_run_and_exact_resume(repository: Path, fake_claude: Path) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, ClaudeAdapter(os.fspath(fake_claude)))

    first = runner.run(
        task="claude-task",
        prompt="first",
        model="opus",
        effort="high",
        timeout_seconds=5,
    )
    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert first.succeeded
    assert first.provider == "claude"
    assert first.model == "claude-test-model"
    assert first.usage.input_tokens == 60
    assert first.usage.cached_input_tokens == 20
    assert first.usage.output_tokens == 40
    assert first.api_equivalent_cost is not None
    assert first.api_equivalent_cost.amount_usd == 0.0123
    assert first.billing.route == "subscription"
    assert first.billing.credential_source == "claude-oauth"
    assert first.billing.detected_by == "claude auth status"
    assert not first.billing.actual_cost_known
    assert first.command[0] == "bwrap"
    assert "--dangerously-skip-permissions" in first.command
    assert ["--model", "opus"] == first.command[
        first.command.index("--model") : first.command.index("--model") + 2
    ]
    assert ["--effort", "high"] == first.command[
        first.command.index("--effort") : first.command.index("--effort") + 2
    ]
    assert resumed.succeeded
    assert ["--resume", first.session_id] == resumed.command[-2:]
    assert "--model" not in resumed.command


def test_claude_records_metered_api_authentication_without_credentials(
    repository: Path,
    fake_claude: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOP_FAKE_CLAUDE_AUTH", "api-key")
    result = AgentRunner(
        WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
    ).run(task="claude-api-billing", prompt="test")

    assert result.succeeded
    assert result.billing.route == "metered-api"
    assert result.billing.credential_source == "anthropic-api-key"
    assert not result.billing.actual_cost_known


def test_claude_requires_a_terminal_result_event(
    repository: Path, fake_claude: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, ClaudeAdapter(os.fspath(fake_claude)))

    result = runner.run(task="claude-incomplete", prompt="INCOMPLETE")

    assert not result.succeeded
    assert result.exit_code == 0
    assert result.session_id is not None
    assert result.final_message is None
    assert result.error == "Claude did not emit a terminal result event"


def test_claude_rejects_a_changed_resume_session(
    repository: Path, fake_claude: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, ClaudeAdapter(os.fspath(fake_claude)))
    first = runner.run(task="claude-resume-identity", prompt="first")

    resumed = runner.resume(run_id=first.run_id, prompt="DIFFERENT_SESSION")

    assert not resumed.succeeded
    assert resumed.exit_code == 0
    assert resumed.session_id is None
    assert resumed.error == (
        "Claude resumed as session different-claude-session instead of the requested "
        f"session {first.session_id}"
    )


def test_cursor_defaults_to_composer_and_resumes_exact_chat(
    repository: Path, fake_cursor: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CursorAdapter(os.fspath(fake_cursor)))

    first = runner.run(task="cursor-task", prompt="first", timeout_seconds=5)
    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert first.succeeded
    assert first.provider == "cursor"
    assert first.model == "composer-2.5"
    assert first.final_message == "answer:first"
    assert first.usage.input_tokens == 100
    assert first.usage.cached_input_tokens == 30
    assert first.usage.output_tokens == 20
    assert first.provider_duration_seconds == 1.25
    assert first.api_equivalent_cost is None
    assert first.billing.route == "provider-credits"
    assert first.billing.credential_source == "cursor-account"
    assert not first.billing.actual_cost_known
    assert ["--model", "composer-2.5"] == first.command[
        first.command.index("--model") : first.command.index("--model") + 2
    ]
    assert ["--sandbox", "disabled"] == first.command[
        first.command.index("--sandbox") : first.command.index("--sandbox") + 2
    ]
    assert "--force" in first.command
    assert "--trust" in first.command
    assert first.time_to_first_response_seconds is not None
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert ["--resume", first.session_id] == resumed.command[
        resumed.command.index("--resume") : resumed.command.index("--resume") + 2
    ]
    assert "--model" not in resumed.command


def test_cursor_rejects_effort_and_changed_resume_chat(
    repository: Path, fake_cursor: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CursorAdapter(os.fspath(fake_cursor)))

    with pytest.raises(AOPError, match="does not accept a separate effort"):
        runner.run(task="cursor-effort", prompt="test", effort="high")
    assert manager.list() == []

    first = runner.run(task="cursor-resume", prompt="first")
    resumed = runner.resume(run_id=first.run_id, prompt="CURSOR_DIFFERENT_SESSION")

    assert not resumed.succeeded
    assert resumed.session_id is None
    assert "instead of the requested chat" in (resumed.error or "")


def test_cursor_workspace_sandbox_and_artifact(
    repository: Path, fake_cursor: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CursorAdapter(os.fspath(fake_cursor)))

    sandboxed = runner.run(task="cursor-sandbox", prompt="CHECK_CURSOR_SANDBOX")
    artifact = runner.run(
        task="cursor-artifact",
        prompt="WRITE_ARTIFACT",
        profile="review",
        artifacts=["report.md"],
    )

    assert sandboxed.succeeded
    assert (manager.get("cursor-sandbox").path / "agent-write.txt").read_text() == (
        "allowed"
    )
    assert not (repository / "main-write.txt").exists()
    assert artifact.succeeded
    assert (
        manager.state_dir
        / "runs"
        / artifact.run_id
        / artifact.artifacts[0].archive_path
    ).read_text() == "# Cursor artifact\n"


@pytest.mark.parametrize("profile", ["review", "edit"])
def test_cursor_uses_private_runtime_state_with_read_only_global_home(
    repository: Path,
    fake_cursor: Path,
    profile: str,
) -> None:
    source_home = Path(os.environ["AOP_CURSOR_HOME"])
    source_config = Path(os.environ["AOP_CURSOR_CONFIG_DIR"])
    source_auth = source_config / "auth.json"
    source_home.chmod(0o555)
    source_config.chmod(0o555)
    source_auth.chmod(0o444)
    task = f"managed-cursor-{profile}"
    manager = WorktreeManager.discover(repository)

    try:
        runner = AgentRunner(manager, CursorAdapter(os.fspath(fake_cursor)))
        first = runner.run(
            task=task,
            prompt="first",
            profile=profile,
            timeout_seconds=5,
        )
        resumed = runner.resume(run_id=first.run_id, prompt="second")
    finally:
        source_home.chmod(0o755)
        source_config.chmod(0o755)
        source_auth.chmod(0o644)

    private_state = manager.state_dir / "provider-state" / task / "cursor"
    assert first.succeeded
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert (private_state / "config" / "cursor" / "auth.json").read_text() == (
        '{"token": "refreshed"}\n'
    )
    assert (private_state / "home" / "skills-cursor" / "test.md").is_file()
    assert (
        private_state / "home" / "projects" / "test-project" / "repo.json"
    ).is_file()
    assert (
        private_state / "home" / "chats" / first.session_id / "state.json"
    ).read_text() == "second"
    assert not (source_home / "projects").exists()
    assert not (source_home / "chats").exists()
    assert source_auth.read_text() == '{"token": "test"}\n'
    assert "/state/cursor/home" in first.command
    assert os.fspath(source_home) not in first.command
    assert ["--setenv", "XDG_CONFIG_HOME", "/state/cursor/config"] == (
        first.command[
            first.command.index("XDG_CONFIG_HOME") - 1 : first.command.index(
                "XDG_CONFIG_HOME"
            )
            + 2
        ]
    )
    assert ["--setenv", "XDG_CACHE_HOME"] == first.command[
        first.command.index("XDG_CACHE_HOME") - 1 : first.command.index(
            "XDG_CACHE_HOME"
        )
        + 1
    ]

    manager.remove(task)
    assert not private_state.exists()
    assert (manager.state_dir / "runs" / first.run_id / "result.json").is_file()
    assert (manager.state_dir / "runs" / resumed.run_id / "result.json").is_file()


def test_devin_defaults_to_swe_and_resumes_exact_session(
    repository: Path, fake_devin: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, DevinAdapter(os.fspath(fake_devin)))

    first = runner.run(task="devin-task", prompt="first", timeout_seconds=5)
    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert first.succeeded
    assert first.provider == "devin"
    assert first.model == "swe-1-7"
    assert first.final_message == "answer:first"
    assert first.usage.input_tokens == 100
    assert first.usage.cached_input_tokens == 30
    assert first.usage.output_tokens == 20
    assert first.api_equivalent_cost is None
    assert first.billing.route == "provider-credits"
    assert first.billing.credential_source == "devin-account"
    assert not first.billing.actual_cost_known
    assert ["--model", "swe-1-7"] == first.command[
        first.command.index("--model") : first.command.index("--model") + 2
    ]
    assert ["--permission-mode", "dangerous"] == first.command[
        first.command.index("--permission-mode") : first.command.index(
            "--permission-mode"
        )
        + 2
    ]
    assert first.time_to_first_response_seconds is not None
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert resumed.model == "swe-1-7"
    assert ["--resume", first.session_id] == resumed.command[
        resumed.command.index("--resume") : resumed.command.index("--resume") + 2
    ]
    assert "--model" not in resumed.command
    assert (
        manager.state_dir / "runs" / first.run_id / "provider-result.json"
    ).is_file()


def test_devin_rejects_effort_incomplete_turn_and_changed_resume_session(
    repository: Path, fake_devin: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, DevinAdapter(os.fspath(fake_devin)))

    with pytest.raises(AOPError, match="does not accept a separate effort"):
        runner.run(task="devin-effort", prompt="test", effort="high")
    incomplete = runner.run(task="devin-incomplete", prompt="DEVIN_INCOMPLETE")
    first = runner.run(task="devin-resume", prompt="first")
    resumed = runner.resume(run_id=first.run_id, prompt="DEVIN_DIFFERENT_SESSION")

    assert not incomplete.succeeded
    assert incomplete.error == "Devin did not write a trajectory export"
    assert not resumed.succeeded
    assert resumed.session_id is None
    assert resumed.error == (
        "Devin resumed as session different-devin-session instead of the requested "
        f"session {first.session_id}"
    )


def test_devin_workspace_sandbox_and_artifact(
    repository: Path, fake_devin: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, DevinAdapter(os.fspath(fake_devin)))

    sandboxed = runner.run(task="devin-sandbox", prompt="CHECK_DEVIN_SANDBOX")
    artifact = runner.run(
        task="devin-artifact",
        prompt="WRITE_ARTIFACT",
        profile="review",
        artifacts=["report.md"],
    )

    assert sandboxed.succeeded
    assert (manager.get("devin-sandbox").path / "agent-write.txt").read_text() == (
        "allowed"
    )
    assert not (repository / "main-write.txt").exists()
    assert artifact.succeeded
    assert (
        manager.state_dir
        / "runs"
        / artifact.run_id
        / artifact.artifacts[0].archive_path
    ).read_text() == "# Devin artifact\n"


@pytest.mark.parametrize("profile", ["review", "edit", "host"])
def test_devin_uses_private_runtime_state_with_read_only_global_profile(
    repository: Path,
    fake_devin: Path,
    profile: str,
) -> None:
    source_data = Path(os.environ["AOP_DEVIN_DATA_DIR"])
    source_config = Path(os.environ["AOP_DEVIN_CONFIG_DIR"])
    source_credentials = source_data / "credentials.toml"
    source_data.chmod(0o555)
    source_config.chmod(0o555)
    source_credentials.chmod(0o444)
    task = f"managed-devin-{profile}"
    manager = WorktreeManager.discover(repository)

    try:
        runner = AgentRunner(manager, DevinAdapter(os.fspath(fake_devin)))
        first = runner.run(task=task, prompt="first", profile=profile)
        resumed = runner.resume(run_id=first.run_id, prompt="second")
    finally:
        source_data.chmod(0o755)
        source_config.chmod(0o755)
        source_credentials.chmod(0o644)

    private_state = manager.state_dir / "provider-state" / task / "devin"
    assert first.succeeded
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert (private_state / "data" / "devin" / "credentials.toml").is_file()
    assert (private_state / "config" / "devin" / "config.json").is_file()
    assert (private_state / "data" / "devin" / "cli" / "fake-session.json").is_file()
    assert not (private_state / "data" / "devin" / "cli" / "installed.bin").exists()
    assert not (source_data / "cli" / "fake-session.json").exists()
    assert source_credentials.read_text() == 'token = "test"\n'

    manager.remove(task)
    assert not private_state.exists()
    assert (manager.state_dir / "runs" / first.run_id / "result.json").is_file()
    assert (manager.state_dir / "runs" / resumed.run_id / "result.json").is_file()


def test_opencode_defaults_to_zen_model_and_resumes_exact_session(
    repository: Path, fake_opencode: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, OpenCodeAdapter(os.fspath(fake_opencode)))

    first = runner.run(
        task="opencode-task",
        prompt="first",
        effort="high",
        timeout_seconds=5,
    )
    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert first.succeeded
    assert first.provider == "opencode"
    assert first.model == "opencode/deepseek-v4-flash"
    assert first.effort == "high"
    assert first.final_message == "answer:first"
    assert first.usage.input_tokens == 53
    assert first.usage.cached_input_tokens == 10
    assert first.usage.output_tokens == 5
    assert first.usage.reasoning_output_tokens == 2
    assert first.provider_duration_seconds == 0.5
    assert first.api_equivalent_cost is not None
    assert first.api_equivalent_cost.amount_usd == 0.00012345
    assert not first.api_equivalent_cost.estimated
    assert first.billing.route == "provider-credits"
    assert first.billing.credential_source == "opencode-api-key"
    assert first.billing.actual_cost_known
    assert first.time_to_first_response_seconds is not None
    assert ["--model", "opencode/deepseek-v4-flash"] == first.command[
        first.command.index("--model") : first.command.index("--model") + 2
    ]
    assert ["--variant", "high"] == first.command[
        first.command.index("--variant") : first.command.index("--variant") + 2
    ]
    assert "--auto" in first.command
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert ["--session", first.session_id] == resumed.command[
        resumed.command.index("--session") : resumed.command.index("--session") + 2
    ]
    assert ["--model", first.model] == resumed.command[
        resumed.command.index("--model") : resumed.command.index("--model") + 2
    ]
    assert ["--variant", "high"] == resumed.command[
        resumed.command.index("--variant") : resumed.command.index("--variant") + 2
    ]


def test_opencode_accepts_short_zen_model_and_fails_closed_on_changed_resume(
    repository: Path, fake_opencode: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, OpenCodeAdapter(os.fspath(fake_opencode)))

    first = runner.run(
        task="opencode-resume",
        prompt="first",
        model="deepseek-v4-flash",
    )
    resumed = runner.resume(
        run_id=first.run_id,
        prompt="OPENCODE_DIFFERENT_SESSION",
    )

    assert first.model == "opencode/deepseek-v4-flash"
    assert not resumed.succeeded
    assert resumed.session_id is None
    assert "instead of the requested session" in (resumed.error or "")


def test_opencode_reports_structured_provider_errors(
    repository: Path, fake_opencode: Path
) -> None:
    result = AgentRunner(
        WorktreeManager.discover(repository),
        OpenCodeAdapter(os.fspath(fake_opencode)),
    ).run(task="opencode-error", prompt="OPENCODE_ERROR")

    assert not result.succeeded
    assert result.error == "synthetic failure"


def test_opencode_normalizes_multi_step_tool_loop(
    repository: Path, fake_opencode: Path
) -> None:
    result = AgentRunner(
        WorktreeManager.discover(repository),
        OpenCodeAdapter(os.fspath(fake_opencode)),
    ).run(task="opencode-tool-loop", prompt="OPENCODE_TOOL_LOOP")

    assert result.succeeded
    assert result.final_message == "answer:OPENCODE_TOOL_LOOP"
    assert result.usage.input_tokens == 61
    assert result.usage.cached_input_tokens == 13
    assert result.usage.output_tokens == 7
    assert result.usage.reasoning_output_tokens == 3
    assert result.api_equivalent_cost is not None
    assert result.api_equivalent_cost.amount_usd == 0.00022345


def test_opencode_workspace_sandbox_and_artifact(
    repository: Path, fake_opencode: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, OpenCodeAdapter(os.fspath(fake_opencode)))

    sandboxed = runner.run(task="opencode-sandbox", prompt="CHECK_OPENCODE_SANDBOX")
    artifact = runner.run(
        task="opencode-artifact",
        prompt="WRITE_ARTIFACT",
        profile="review",
        artifacts=["report.md"],
    )

    assert sandboxed.succeeded
    assert (manager.get("opencode-sandbox").path / "agent-write.txt").read_text() == (
        "allowed"
    )
    assert not (repository / "main-write.txt").exists()
    assert artifact.succeeded
    assert (
        manager.state_dir
        / "runs"
        / artifact.run_id
        / artifact.artifacts[0].archive_path
    ).read_text() == "# OpenCode artifact\n"


@pytest.mark.parametrize("profile", ["review", "edit"])
def test_opencode_uses_private_runtime_state_with_read_only_global_profile(
    repository: Path,
    fake_opencode: Path,
    profile: str,
) -> None:
    source_config = Path(os.environ["AOP_OPENCODE_CONFIG_DIR"])
    source_data = Path(os.environ["AOP_OPENCODE_DATA_DIR"])
    source_auth = source_data / "auth.json"
    source_dependencies = source_config / "node_modules"
    for path in (source_config, source_data, source_dependencies):
        path.chmod(0o555)
    source_auth.chmod(0o444)
    task = f"managed-opencode-{profile}"
    manager = WorktreeManager.discover(repository)

    try:
        runner = AgentRunner(manager, OpenCodeAdapter(os.fspath(fake_opencode)))
        first = runner.run(task=task, prompt="first", profile=profile)
        resumed = runner.resume(run_id=first.run_id, prompt="second")
    finally:
        for path in (source_config, source_data, source_dependencies):
            path.chmod(0o755)
        source_auth.chmod(0o644)

    private_state = manager.state_dir / "provider-state" / task / "opencode"
    assert first.succeeded
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert (private_state / "data" / "opencode" / "auth.json").read_text() == (
        '{"opencode": {"key": "refreshed"}}\n'
    )
    assert (private_state / "data" / "opencode" / "opencode.db").is_file()
    assert (private_state / "state" / "opencode" / "model.json").is_file()
    assert (private_state / "config" / "opencode" / "package.json").is_file()
    assert not (source_data / "opencode.db").exists()
    assert source_auth.read_text() == '{"opencode": {"key": "test"}}\n'
    assert ["--setenv", "XDG_DATA_HOME", "/state/opencode/data"] == (
        first.command[
            first.command.index("XDG_DATA_HOME") - 1 : first.command.index(
                "XDG_DATA_HOME"
            )
            + 2
        ]
    )
    assert ["--setenv", "XDG_CACHE_HOME", "/cache"] == (
        first.command[
            first.command.index("XDG_CACHE_HOME") - 1 : first.command.index(
                "XDG_CACHE_HOME"
            )
            + 2
        ]
    )
    assert (
        private_state
        / "config"
        / "opencode"
        / "node_modules"
        / "@opencode-ai"
        / "plugin"
        / "package.json"
    ).is_file()
    assert os.fspath(source_dependencies) not in first.command

    manager.remove(task)
    assert not private_state.exists()
    assert (manager.state_dir / "runs" / first.run_id / "result.json").is_file()
    assert (manager.state_dir / "runs" / resumed.run_id / "result.json").is_file()


def test_agy_passes_native_model_effort_and_resumes_exact_conversation(
    repository: Path, fake_agy: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, AgyAdapter(os.fspath(fake_agy)))

    first = runner.run(
        task="agy-task",
        prompt="first",
        model="gemini-3.5-flash",
        effort="low",
        timeout_seconds=5,
    )
    resumed = runner.resume(run_id=first.run_id, prompt="second")

    assert first.succeeded
    assert first.provider == "agy"
    assert first.model == "gemini-3.5-flash"
    assert first.effort == "low"
    assert ["--model", "gemini-3.5-flash"] == first.command[
        first.command.index("--model") : first.command.index("--model") + 2
    ]
    assert ["--effort", "low"] == first.command[
        first.command.index("--effort") : first.command.index("--effort") + 2
    ]
    assert ["--output-format", "stream-json"] == first.command[
        first.command.index("--output-format") : first.command.index("--output-format")
        + 2
    ]
    assert ["--print-timeout", "24h"] == first.command[
        first.command.index("--print-timeout") : first.command.index("--print-timeout")
        + 2
    ]
    assert "--log-file" not in first.command
    assert "--add-dir" not in first.command
    assert "--gemini_dir" in first.command
    assert first.final_message == "answer:first"
    assert first.usage.input_tokens == 100
    assert first.usage.cached_input_tokens == 30
    assert first.usage.output_tokens == 20
    assert first.usage.reasoning_output_tokens == 7
    assert first.provider_duration_seconds == 1.25
    assert first.time_to_first_response_seconds is not None
    assert first.api_equivalent_cost is not None
    assert first.api_equivalent_cost.amount_usd == 0.0003345
    assert first.api_equivalent_cost.model == "gemini-3.5-flash"
    assert first.api_equivalent_cost.priced_as == "gemini-3.5-flash"
    assert first.api_equivalent_cost.estimated
    assert first.billing.route == "subscription"
    assert first.billing.credential_source == "google-oauth"
    assert not first.billing.actual_cost_known
    events_path = manager.state_dir / "runs" / first.run_id / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "init",
        "step_update",
        "result",
    ]
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert ["--conversation", first.session_id] == resumed.command[
        resumed.command.index("--conversation") : resumed.command.index(
            "--conversation"
        )
        + 2
    ]
    assert "--model" not in resumed.command
    assert "--effort" not in resumed.command


@pytest.mark.parametrize("profile", ["review", "edit", "host"])
def test_agy_uses_private_persistent_runtime_state_in_every_sandbox(
    repository: Path,
    fake_agy: Path,
    profile: str,
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, AgyAdapter(os.fspath(fake_agy)))
    source_dir = Path(os.environ["AOP_AGY_SOURCE_DIR"])

    first = runner.run(
        task=f"private-agy-{profile}",
        prompt="first",
        profile=profile,
        timeout_seconds=5,
    )
    resumed = runner.resume(run_id=first.run_id, prompt="second")

    private_dir = (
        manager.state_dir
        / "provider-state"
        / f"private-agy-{profile}"
        / "agy"
        / "gemini"
    )
    assert first.succeeded
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert (private_dir / "oauth_creds.json").is_file()
    assert (private_dir / "config" / "config.json").is_file()
    assert (private_dir / "antigravity-cli" / "antigravity-oauth-token").is_file()
    assert (private_dir / "antigravity-cli" / "fake-conversation.json").is_file()
    assert not (
        private_dir / "antigravity-cli" / "conversations" / "stale.json"
    ).exists()
    assert not (source_dir / "antigravity-cli" / "fake-conversation.json").exists()
    first_dir_index = first.command.index("--gemini_dir")
    resumed_dir_index = resumed.command.index("--gemini_dir")
    expected_runtime_dir = (
        os.fspath(private_dir) if profile == "host" else "/state/agy/gemini"
    )
    assert first.command[first_dir_index + 1] == expected_runtime_dir
    assert resumed.command[resumed_dir_index + 1] == expected_runtime_dir
    if profile == "host":
        assert first.command[0] == os.fspath(fake_agy)
    else:
        assert first.command[0] == "bwrap"
        assert os.fspath(source_dir) not in first.command
        assert ["--bind", os.fspath(private_dir.parent.parent)] == first.command[
            first.command.index(os.fspath(private_dir.parent.parent))
            - 1 : first.command.index(os.fspath(private_dir.parent.parent)) + 1
        ]

    manager.remove(f"private-agy-{profile}")
    assert not private_dir.exists()
    assert (manager.state_dir / "runs" / first.run_id / "result.json").is_file()
    assert (manager.state_dir / "runs" / resumed.run_id / "result.json").is_file()


@pytest.mark.parametrize(
    ("prompt", "error"),
    [
        (
            "AGY_DIFFERENT_SESSION",
            "agy resumed as conversation aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee "
            "instead of the requested conversation",
        ),
        (
            "AGY_MISSING_SESSION",
            "agy resume result did not report the requested conversation ID",
        ),
    ],
)
def test_agy_resume_fails_closed_when_terminal_conversation_identity_is_not_exact(
    repository: Path,
    fake_agy: Path,
    prompt: str,
    error: str,
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, AgyAdapter(os.fspath(fake_agy)))
    first = runner.run(task="agy-identity", prompt="first", timeout_seconds=5)

    resumed = runner.resume(run_id=first.run_id, prompt=prompt)

    assert not resumed.succeeded
    assert resumed.session_id is None
    assert resumed.error is not None
    assert resumed.error.startswith(error)
    with pytest.raises(AOPError, match="run has no resumable agent session"):
        runner.resume(run_id=resumed.run_id, prompt="third")


def test_agy_requires_an_authenticated_source_profile(
    repository: Path,
    fake_agy: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-agy-profile"
    monkeypatch.setenv("AOP_AGY_SOURCE_DIR", os.fspath(missing))
    manager = WorktreeManager.discover(repository)

    with pytest.raises(AOPError, match="authenticate Agy first"):
        AgentRunner(manager, AgyAdapter(os.fspath(fake_agy))).run(
            task="missing-agy-profile",
            prompt="first",
            timeout_seconds=5,
        )

    assert not (manager.state_dir / "runs").exists()


def test_agy_terminal_error_status_fails_even_with_zero_exit(
    repository: Path, fake_agy: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), AgyAdapter(os.fspath(fake_agy))
    )

    result = runner.run(
        task="agy-error",
        prompt="AGY_ERROR",
        timeout_seconds=5,
    )

    assert result.exit_code == 0
    assert not result.succeeded
    assert result.error == "synthetic agy failure"


def test_agy_rejects_an_unsupported_effort(repository: Path, fake_agy: Path) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), AgyAdapter(os.fspath(fake_agy))
    )

    with pytest.raises(AOPError, match="agy effort must be one of: high, low, medium"):
        runner.run(
            task="bad-agy",
            prompt="test",
            effort="xhigh",
        )


def test_agy_defaults_to_gemini_35_flash_medium(
    repository: Path, fake_agy: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), AgyAdapter(os.fspath(fake_agy))
    )

    result = runner.run(task="default-agy", prompt="test", timeout_seconds=5)

    assert result.succeeded
    assert result.model == "gemini-3.5-flash"
    assert result.effort == "medium"
    assert ["--model", "gemini-3.5-flash"] == result.command[
        result.command.index("--model") : result.command.index("--model") + 2
    ]
    assert ["--effort", "medium"] == result.command[
        result.command.index("--effort") : result.command.index("--effort") + 2
    ]


def test_agy_passes_an_exact_model_without_adding_effort(
    repository: Path, fake_agy: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), AgyAdapter(os.fspath(fake_agy))
    )

    result = runner.run(
        task="agy-36",
        prompt="test",
        model="gemini-3.6-flash-high",
    )

    assert result.succeeded
    assert result.model == "gemini-3.6-flash-high"
    assert result.effort is None
    assert "--effort" not in result.command


def test_agy_can_produce_a_declared_artifact(repository: Path, fake_agy: Path) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, AgyAdapter(os.fspath(fake_agy)))

    result = runner.run(
        task="agy-artifact",
        prompt="WRITE_ARTIFACT",
        profile="review",
        artifacts=["paper.md"],
    )

    assert result.succeeded
    artifact = result.artifacts[0]
    assert (
        manager.state_dir / "runs" / result.run_id / artifact.archive_path
    ).read_text() == "# Extracted by agy\n"


def test_claude_workspace_uses_bwrap_to_protect_main_and_git_metadata(
    repository: Path, fake_claude: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, ClaudeAdapter(os.fspath(fake_claude)))

    result = runner.run(task="sandboxed", prompt="CHECK_SANDBOX", timeout_seconds=5)
    worktree = manager.get("sandboxed")

    assert result.succeeded
    assert (worktree.path / "agent-write.txt").read_text() == "allowed"
    assert (manager.cache_dir / "provider-cache.txt").read_text() == "shared"
    assert not (repository / "main-write.txt").exists()
    assert (worktree.path / ".git").read_text().startswith("gitdir:")


def test_review_profile_writes_only_to_controller_scratch(
    repository: Path, fake_claude: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, ClaudeAdapter(os.fspath(fake_claude)))

    result = runner.run(
        task="proposal",
        prompt="CHECK_SCRATCH",
        profile="review",
        timeout_seconds=5,
    )
    worktree = manager.get("proposal")

    assert result.succeeded
    assert (
        manager.state_dir / "scratch" / "proposal" / "analysis.txt"
    ).read_text() == "allowed"
    assert not (worktree.path / "agent-write.txt").exists()
    assert not (repository / "main-write.txt").exists()


def test_hermes_run_and_exact_resume_report_per_turn_usage(
    repository: Path,
    fake_hermes: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOP_HERMES_BIN", os.fspath(fake_hermes))
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, HermesAdapter(os.fspath(fake_hermes)))

    first = runner.run(
        task="hermes-task",
        prompt="first",
        model="deepseek/deepseek-v4-flash-0731",
        effort="high",
        timeout_seconds=5,
    )
    resumed = AgentRunner(manager).resume(run_id=first.run_id, prompt="second")

    assert first.succeeded
    assert first.provider == "hermes"
    assert first.model == "deepseek/deepseek-v4-flash-0731"
    assert first.effort == "high"
    assert first.final_message == "answer:first"
    assert first.command[0] == "bwrap"
    assert "--provider" not in first.command
    assert ["--model", "deepseek/deepseek-v4-flash-0731"] == first.command[
        first.command.index("--model") : first.command.index("--model") + 2
    ]
    assert ["--reasoning", "high"] == first.command[
        first.command.index("--reasoning") : first.command.index("--reasoning") + 2
    ]
    assert all(
        option in first.command
        for option in ["-Q", "--yolo", "--accept-hooks", "--source"]
    )
    assert first.usage.input_tokens == 15
    assert first.usage.cached_input_tokens == 3
    assert first.usage.output_tokens == 5
    assert first.usage.reasoning_output_tokens == 1
    assert first.api_equivalent_cost is not None
    assert first.api_equivalent_cost.amount_usd == 0.000001
    assert first.api_equivalent_cost.estimated
    assert first.billing.route == "subscription"
    assert first.billing.credential_source == "nous-oauth"
    assert not first.billing.actual_cost_known

    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert resumed.final_message == "answer:second"
    assert ["--resume", first.session_id] == resumed.command[
        resumed.command.index("--resume") : resumed.command.index("--resume") + 2
    ]
    assert "--no-restore-cwd" in resumed.command
    assert "--provider" not in resumed.command
    assert ["--model", "deepseek/deepseek-v4-flash-0731"] == resumed.command[
        resumed.command.index("--model") : resumed.command.index("--model") + 2
    ]
    assert ["--reasoning", "high"] == resumed.command[
        resumed.command.index("--reasoning") : resumed.command.index("--reasoning") + 2
    ]
    assert resumed.usage == first.usage
    assert resumed.api_equivalent_cost is not None
    assert resumed.api_equivalent_cost.amount_usd == 0.000001


def test_hermes_provider_override_is_recorded_and_reused_on_resume(
    repository: Path,
    fake_hermes: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOP_HERMES_BIN", os.fspath(fake_hermes))
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, HermesAdapter(os.fspath(fake_hermes)))

    first = runner.run(
        task="hermes-routed",
        prompt="first",
        model="grok-build-0.1",
        inference_provider="xai-oauth",
    )
    resumed = AgentRunner(manager).resume(run_id=first.run_id, prompt="second")

    assert first.succeeded
    assert resumed.succeeded
    assert first.inference_provider == "xai-oauth"
    assert resumed.inference_provider == "xai-oauth"
    for result in (first, resumed):
        assert ["--provider", "xai-oauth"] == result.command[
            result.command.index("--provider") : result.command.index("--provider") + 2
        ]
        assert ["--model", "grok-build-0.1"] == result.command[
            result.command.index("--model") : result.command.index("--model") + 2
        ]

    first_request = json.loads(
        (manager.state_dir / "runs" / first.run_id / "request.json").read_text()
    )
    resumed_result = json.loads(
        (manager.state_dir / "runs" / resumed.run_id / "result.json").read_text()
    )
    assert first_request["inference_provider"] == "xai-oauth"
    assert resumed_result["inference_provider"] == "xai-oauth"


def test_provider_override_requires_hermes_and_an_explicit_model(
    repository: Path,
    fake_codex: Path,
    fake_hermes: Path,
) -> None:
    manager = WorktreeManager.discover(repository)

    with pytest.raises(AOPError, match="codex does not support --provider"):
        AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).run(
            task="unsupported-provider",
            prompt="unused",
            model="gpt-5.6-sol",
            inference_provider="openai",
        )
    with pytest.raises(AOPError, match="requires an explicit --model"):
        AgentRunner(manager, HermesAdapter(os.fspath(fake_hermes))).run(
            task="missing-provider-model",
            prompt="unused",
            inference_provider="xai-oauth",
        )


def test_hermes_participant_mode_is_tool_free_bounded_and_stable_across_resume(
    repository: Path,
    fake_hermes: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOP_HERMES_BIN", os.fspath(fake_hermes))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "inherited-worker")
    monkeypatch.setenv("HERMES_NO_TOOLS", "inherited-internal-override")
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, HermesAdapter(os.fspath(fake_hermes)))

    first = runner.run(
        task="hermes-participant",
        prompt="CHECK_PARTICIPANT first",
        mode="participant",
        profile="review",
        timeout_seconds=5,
    )
    resumed = AgentRunner(manager).resume(
        run_id=first.run_id,
        prompt="CHECK_PARTICIPANT second",
    )

    assert first.succeeded
    assert resumed.succeeded
    assert first.mode == "participant"
    assert resumed.mode == "participant"
    assert resumed.session_id == first.session_id
    for result in (first, resumed):
        assert "--safe-mode" in result.command
        assert ["--toolsets", "__aop_no_tools__"] == result.command[
            result.command.index("--toolsets") : result.command.index("--toolsets") + 2
        ]
        assert ["--max-turns", "1"] == result.command[
            result.command.index("--max-turns") : result.command.index("--max-turns")
            + 2
        ]
        assert "--yolo" not in result.command
        assert "--accept-hooks" not in result.command
        assert "--no-tools" not in result.command

    first_request = json.loads(
        (manager.state_dir / "runs" / first.run_id / "request.json").read_text()
    )
    resumed_request = json.loads(
        (manager.state_dir / "runs" / resumed.run_id / "request.json").read_text()
    )
    assert first_request["mode"] == "participant"
    assert resumed_request["mode"] == "participant"


@pytest.mark.parametrize(
    ("adapter", "provider"),
    [
        (CodexAdapter, "codex"),
        (ClaudeAdapter, "claude"),
        (CursorAdapter, "cursor"),
        (OpenCodeAdapter, "opencode"),
        (AgyAdapter, "agy"),
        (DeepSeekHarnessAdapter, "dsh"),
    ],
)
def test_unsupported_adapters_reject_participant_mode_before_creating_a_worktree(
    repository: Path,
    fake_codex: Path,
    fake_claude: Path,
    fake_cursor: Path,
    fake_opencode: Path,
    fake_agy: Path,
    fake_dsh: Path,
    adapter: (
        type[CodexAdapter]
        | type[ClaudeAdapter]
        | type[CursorAdapter]
        | type[OpenCodeAdapter]
        | type[AgyAdapter]
        | type[DeepSeekHarnessAdapter]
    ),
    provider: str,
) -> None:
    binaries = {
        "codex": fake_codex,
        "claude": fake_claude,
        "cursor": fake_cursor,
        "opencode": fake_opencode,
        "agy": fake_agy,
        "dsh": fake_dsh,
    }
    binary = binaries[provider]
    manager = WorktreeManager.discover(repository)

    with pytest.raises(AOPError, match=f"{provider} does not support participant mode"):
        AgentRunner(manager, adapter(os.fspath(binary))).run(
            task=f"unsupported-{provider}",
            prompt="test",
            mode="participant",
        )

    assert manager.list() == []


def test_hermes_supports_workspace_sandbox_and_artifacts(
    repository: Path, fake_hermes: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, HermesAdapter(os.fspath(fake_hermes)))

    result = runner.run(
        task="hermes-artifact",
        prompt="CHECK_HERMES_SANDBOX",
        model="deepseek/deepseek-v4-flash-0731",
        profile="edit",
        artifacts=["report.md"],
    )

    assert result.succeeded
    assert (manager.get("hermes-artifact").path / "agent-write.txt").read_text() == (
        "allowed"
    )
    assert not (repository / "main-write.txt").exists()
    artifact = result.artifacts[0]
    assert (
        manager.state_dir / "runs" / result.run_id / artifact.archive_path
    ).read_text() == "# Hermes artifact\n"


def test_hermes_defaults_to_deepseek_v4_flash_0731(
    repository: Path, fake_hermes: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), HermesAdapter(os.fspath(fake_hermes))
    )

    result = runner.run(task="default-hermes", prompt="test")

    assert result.succeeded
    assert result.model == "deepseek/deepseek-v4-flash-0731"
    assert result.effort is None
    assert ["--model", "deepseek/deepseek-v4-flash-0731"] == result.command[
        result.command.index("--model") : result.command.index("--model") + 2
    ]
    assert "--reasoning" not in result.command


def test_hermes_rejects_an_unsupported_effort(
    repository: Path, fake_hermes: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), HermesAdapter(os.fspath(fake_hermes))
    )

    with pytest.raises(AOPError, match="Hermes effort must be one of"):
        runner.run(task="bad-hermes", prompt="test", effort="extreme")


def test_hermes_nonzero_exit_is_a_normalized_failure(
    repository: Path, fake_hermes: Path
) -> None:
    runner = AgentRunner(
        WorktreeManager.discover(repository), HermesAdapter(os.fspath(fake_hermes))
    )

    result = runner.run(
        task="failed-hermes",
        prompt="FAIL",
        model="deepseek/deepseek-v4-flash-0731",
    )

    assert not result.succeeded
    assert result.exit_code == 3
    assert result.error == "Hermes exited with status 3"


def test_hermes_failure_reports_provider_error_instead_of_resume_banner() -> None:
    error = HermesAdapter._exit_error(
        "reasoning\nAPI call failed after 3 retries: empty stream\n",
        "↻ Resumed session session-1 (1 user message, 2 total messages)\n"
        "session_id: session-1\n",
        1,
    )

    assert error == "API call failed after 3 retries: empty stream"


def test_hermes_does_not_return_a_stale_message_when_resume_adds_no_answer() -> None:
    previous = _HermesSession(
        model="grok-4.5",
        billing_provider="xai-oauth",
        final_message="PLAY: round one",
        last_assistant_id=None,
        usage=TokenUsage(),
        cost_usd=None,
        cost_estimated=True,
        cost_source="none",
        pricing_version="none",
    )
    unchanged = _HermesSession(
        model="grok-4.5",
        billing_provider="xai-oauth",
        final_message="PLAY: round one",
        last_assistant_id=None,
        usage=TokenUsage(),
        cost_usd=None,
        cost_estimated=True,
        cost_source="none",
        pricing_version="none",
    )

    assert HermesAdapter._final_message(previous, unchanged) is None


def test_hermes_unknown_session_cost_uses_catalog_api_equivalent() -> None:
    session = _HermesSession(
        model="grok-4.5",
        billing_provider="xai-oauth",
        final_message="answer",
        last_assistant_id="2",
        usage=TokenUsage(),
        cost_usd=0.0,
        cost_estimated=True,
        cost_source="none",
        pricing_version="hermes-cli-reported",
    )
    usage = TokenUsage(
        input_tokens=2_000,
        cached_input_tokens=500,
        output_tokens=1_000,
    )

    cost = HermesAdapter._cost_delta(None, session, "grok-4.5", usage)

    assert cost is not None
    assert cost.amount_usd == 0.00915
    assert cost.estimated
    assert cost.pricing_source == "https://models.dev/api.json"


def test_hermes_unknown_session_cost_with_no_usage_remains_unknown() -> None:
    session = _HermesSession(
        model="grok-4.5",
        billing_provider="xai-oauth",
        final_message=None,
        last_assistant_id=None,
        usage=TokenUsage(),
        cost_usd=0.0,
        cost_estimated=True,
        cost_source="none",
        pricing_version="hermes-cli-reported",
    )

    assert HermesAdapter._cost_delta(None, session, "grok-4.5", TokenUsage()) is None


@pytest.mark.parametrize("profile", ["review", "edit", "host"])
def test_hermes_run_and_resume_with_read_only_runtime_home(
    repository: Path,
    fake_hermes: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: str,
) -> None:
    hermes_home = tmp_path / f"managed-hermes-{profile}"
    hermes_home.mkdir()
    (hermes_home / "auth.json").write_text("{}\n")
    state_path = hermes_home / "fake-state.json"
    hermes_home.chmod(0o555)
    monkeypatch.setenv("HERMES_HOME", os.fspath(hermes_home))
    monkeypatch.setenv("AOP_FAKE_HERMES_STATE_IN_HOME", "1")
    monkeypatch.setenv("AOP_HERMES_BIN", os.fspath(fake_hermes))
    manager = WorktreeManager.discover(repository)

    try:
        first = AgentRunner(manager, HermesAdapter(os.fspath(fake_hermes))).run(
            task=f"managed-{profile}",
            prompt="first",
            profile=profile,
        )
        resumed = AgentRunner(manager).resume(
            run_id=first.run_id,
            prompt="second",
        )
    finally:
        hermes_home.chmod(0o755)

    assert first.succeeded
    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert resumed.final_message == "answer:second"
    assert not state_path.exists()
    isolated_home = (
        manager.state_dir / "provider-state" / f"managed-{profile}" / "hermes" / "home"
    )
    assert (isolated_home / "fake-state.json").is_file()
    assert (isolated_home / "auth.json").read_text() == "{}\n"
    assert (
        manager.state_dir / "shared-provider-state" / "hermes" / "auth.json"
    ).read_text() == "{}\n"
    assert not (
        manager.state_dir / "scratch" / f"managed-{profile}" / "provider-state"
    ).exists()
    assert "--overlay" not in first.command
    assert os.fspath(hermes_home) not in first.command
    assert os.fspath(hermes_home) not in resumed.command


def test_hermes_migrates_freshest_rotated_credentials_and_serializes_tasks(
    repository: Path,
    fake_hermes: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def auth(generation: int, *, updated_minute: int) -> dict[str, object]:
        refresh_day = 7 if generation == 0 else 8
        refresh_minute = 5 if generation == 0 else generation
        return {
            "version": 1,
            "updated_at": f"2026-08-08T20:{updated_minute:02d}:01Z",
            "providers": {},
            "credential_pool": {
                "xai-oauth": [
                    {
                        "id": "rotating-xai",
                        "label": "xai-oauth-oauth-1",
                        "source": "manual:device_code",
                        "access_token": f"generation-{generation}",
                        "refresh_token": f"refresh-{generation}",
                        "last_refresh": (
                            f"2026-08-{refresh_day:02d}T20:{refresh_minute:02d}:00Z"
                        ),
                    }
                ]
            },
        }

    source_home = tmp_path / "hermes-source"
    source_home.mkdir()
    source_auth = source_home / "auth.json"
    source_auth.write_text(json.dumps(auth(0, updated_minute=5)))
    monkeypatch.setenv("HERMES_HOME", os.fspath(source_home))
    monkeypatch.setenv("AOP_HERMES_BIN", os.fspath(fake_hermes))
    monkeypatch.setenv("AOP_FAKE_HERMES_STATE_IN_HOME", "1")
    monkeypatch.setenv("AOP_FAKE_HERMES_ROTATE_DELAY", "0.2")
    manager = WorktreeManager.discover(repository)

    newer_private = (
        manager.state_dir / "provider-state" / "previous-success" / "hermes" / "home"
    )
    newer_private.mkdir(parents=True)
    (newer_private / "auth.json").write_text(json.dumps(auth(1, updated_minute=14)))
    later_but_stale = (
        manager.state_dir / "provider-state" / "later-failure" / "hermes" / "home"
    )
    later_but_stale.mkdir(parents=True)
    (later_but_stale / "auth.json").write_text(json.dumps(auth(0, updated_minute=34)))
    manager.create("rotate-a")
    manager.create("rotate-b")

    def run(task: str):
        return AgentRunner(manager, HermesAdapter(os.fspath(fake_hermes))).run(
            task=task,
            prompt="ROTATE_AUTH",
            profile="review",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["rotate-a", "rotate-b"]))

    assert all(result.succeeded for result in results)
    shared_auth = manager.state_dir / "shared-provider-state" / "hermes" / "auth.json"
    shared = json.loads(shared_auth.read_text())
    entry = shared["credential_pool"]["xai-oauth"][0]
    assert entry["access_token"] == "generation-3"
    assert entry["refresh_token"] == "refresh-3"
    assert shared_auth.stat().st_mode & 0o777 == 0o600
    assert json.loads(source_auth.read_text()) == auth(0, updated_minute=5)

    manager.remove("rotate-a", force=True)
    assert shared_auth.is_file()
