from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from support import AOPTestCase, Capture, MonkeyPatch, parametrize, raises
import yaml
from shared_fixtures import api_cost

from agent_orchestration_process import runner as runner_module
from agent_orchestration_process.cli import main
from agent_orchestration_process.models import PromptPart, RunRequest
from agent_orchestration_process.pricing import TokenUsage
from agent_orchestration_process.runner import (
    AgyAdapter,
    AgentRunner,
    ClaudeAdapter,
    CodexAdapter,
    CursorAdapter,
    DevinAdapter,
    DeepSeekHarnessAdapter,
    GrokAdapter,
    HermesAdapter,
    _HermesSession,
    OpenCodeAdapter,
)
from agent_orchestration_process.worktrees import AOPError, Worktree, WorktreeManager


AT = "2026-09-01T12:00:00+00:00"


def _host_request() -> RunRequest:
    return RunRequest(
        run_id="run",
        provider="hermes",
        mode="agent",
        task="t",
        prompt="p",
        base="HEAD",
        model=None,
        inference_provider=None,
        inference_route=None,
        effort=None,
        profile="host",
        no_web=False,
        effective_policy={},
        timeout_seconds=None,
        session_id=None,
        parent_run_id=None,
        artifacts=(),
        inputs=(),
        created_at="2026-10-19T00:00:00+00:00",
    )


class ProvidersTest(AOPTestCase):
    def test_dsh_run_and_exact_resume_use_the_native_patch_interface(
        self,
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
        assert first.usage.input_tokens == 120
        assert first.usage.cached_input_tokens == 30
        assert first.usage.output_tokens == 20
        assert first.usage.reasoning_output_tokens == 7
        assert first.calculated_cost is not None
        assert first.calculated_cost.amount_usd == api_cost(
            "deepseek-v4-pro",
            first.started_at,
            input_tokens=120,
            cached_input_tokens=30,
            output_tokens=20,
            provider="deepseek",
        )
        assert first.provider_reported_cost is None
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
        assert not private_home.joinpath(".credentials.yaml").exists()
        assert not private_home.joinpath("settings.yaml").exists()
        assert (
            private_home.joinpath("fake-sessions", first.session_id).read_text() == "second"
        )

    def test_dsh_maps_none_effort_and_normalizes_failures(
        self,
        repository: Path, fake_dsh: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository),
            DeepSeekHarnessAdapter(os.fspath(fake_dsh)),
        )
        failed = runner.run(task="dsh-failure", prompt="FAIL", effort="none")
        response = runner.run(
            task="dsh-provider-error", prompt="DSH_ERROR_WITH_RESPONSE", effort="none"
        )

        assert not failed.succeeded
        assert failed.error == "synthetic dsh failure"
        assert failed.provider_error == "synthetic dsh failure"
        assert response.status == "response_available_with_provider_error"
        assert response.execution_completed
        assert response.response_available
        assert response.error is None
        assert response.provider_error == "synthetic dsh provider failure"
        assert response.final_message == "answer:DSH_ERROR_WITH_RESPONSE"
        scratch = repository / ".aop" / "scratch" / "dsh-failure"
        patch = next(scratch.glob("aop-dsh-*.cordis.yml"))
        assert 'reasoningEffort: "off"' in patch.read_text()

        with raises(AOPError, match="dsh effort must be one of"):
            runner.run(task="dsh-bad-effort", prompt="test", effort="ultra")

    def test_grok_run_and_exact_resume_use_native_headless_protocol(
        self,
        repository: Path,
        fake_grok: Path,
    ) -> None:
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, GrokAdapter(os.fspath(fake_grok)))

        first = runner.run(
            task="grok-task",
            prompt="first",
            model="grok-4.5",
            effort="high",
            timeout_seconds=5,
        )
        resumed = runner.resume(run_id=first.run_id, prompt="second")

        assert first.succeeded, first.error
        assert first.provider == "grok"
        assert first.model == "grok-4.5"
        assert first.effort == "high"
        assert first.final_message == "answer:first"
        assert first.usage.input_tokens == 55
        assert first.usage.cached_input_tokens == 10
        assert first.usage.output_tokens == 20
        assert first.usage.reasoning_output_tokens == 7
        assert first.calculated_cost is not None
        assert first.calculated_cost.amount_usd == 0.000213
        assert first.provider_reported_cost is None
        assert first.billing.route == "subscription"
        assert first.billing.credential_source == "xai-oauth"
        assert first.provider_duration_seconds is None
        assert first.command[0] == "bwrap"
        assert ["--output-format", "streaming-json"] == first.command[
            first.command.index("--output-format") : first.command.index("--output-format")
            + 2
        ]
        for flag in (
            "--always-approve",
            "--no-plan",
            "--no-leader",
            "--no-auto-update",
            "--verbatim",
        ):
            assert flag in first.command
        assert ["--sandbox", "off"] == first.command[
            first.command.index("--sandbox") : first.command.index("--sandbox") + 2
        ]

        private_home = manager.state_dir / "provider-state" / "grok-task" / "grok" / "home"
        assert private_home.joinpath("auth.json").is_file()
        assert private_home.joinpath("config.toml").is_file()
        assert private_home.joinpath("rules", "default.rules").is_file()
        assert private_home.joinpath("skills", "test-skill", "SKILL.md").is_file()
        assert not private_home.joinpath("sessions", "global-session").exists()
        assert not private_home.joinpath("logs").exists()

        assert resumed.succeeded, resumed.error
        assert resumed.session_id == first.session_id
        assert resumed.model == first.model
        assert resumed.effort == first.effort
        assert resumed.final_message == "answer:second"
        assert ["--resume", first.session_id] == resumed.command[
            resumed.command.index("--resume") : resumed.command.index("--resume") + 2
        ]

    def test_grok_normalizes_failures_resume_identity_and_effort(
        self,
        repository: Path,
        fake_grok: Path,
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository),
            GrokAdapter(os.fspath(fake_grok)),
        )

        failed = runner.run(task="grok-failure", prompt="GROK_ERROR")
        incomplete = runner.run(task="grok-incomplete", prompt="GROK_INCOMPLETE")
        max_turns = runner.run(task="grok-max-turns", prompt="GROK_MAX_TURNS")
        first = runner.run(task="grok-resume", prompt="first")
        mismatched = runner.resume(
            run_id=first.run_id,
            prompt="GROK_DIFFERENT_SESSION",
        )

        assert failed.error == "synthetic Grok failure"
        assert failed.exit_code == 1
        assert not incomplete.succeeded
        assert incomplete.error == "Grok did not report a session ID"
        assert not max_turns.succeeded
        assert max_turns.status == "response_available_with_provider_error"
        assert max_turns.error is None
        assert max_turns.provider_status == "max_turn_requests"
        assert max_turns.provider_error == "Grok stopped with reason max_turn_requests"
        assert max_turns.final_message == "answer:GROK_MAX_TURNS"
        assert not mismatched.succeeded
        assert mismatched.session_id is None
        assert "instead of the requested session" in mismatched.error
        with raises(AOPError, match="Grok effort must be one of"):
            runner.run(task="grok-bad-effort", prompt="test", effort="ultra")

    def test_grok_storage_mode_is_optional_and_inherited_when_set(
        self,
        repository: Path,
        fake_grok: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        manager = WorktreeManager.discover(repository)

        monkeypatch.setenv("GROK_STORAGE_MODE", "local")
        inherited = AgentRunner(manager, GrokAdapter(os.fspath(fake_grok))).run(
            task="grok-storage", prompt="CHECK_GROK_STORAGE"
        )
        assert inherited.succeeded, inherited.error
        index = inherited.command.index("GROK_STORAGE_MODE")
        assert inherited.command[index - 1 : index + 2] == [
            "--setenv",
            "GROK_STORAGE_MODE",
            "local",
        ]

        monkeypatch.delenv("GROK_STORAGE_MODE")
        absent = AgentRunner(manager, GrokAdapter(os.fspath(fake_grok))).run(
            task="grok-no-storage", prompt="CHECK_GROK_NO_STORAGE"
        )
        assert absent.succeeded, absent.error
        assert "GROK_STORAGE_MODE" not in absent.command

    def test_grok_keeps_reported_cost_for_api_key_billing(
        self,
        repository: Path,
        fake_grok: Path,
    ) -> None:
        source_home = Path(os.environ["AOP_GROK_SOURCE_HOME"])
        source_home.joinpath("auth.json").write_text(
            json.dumps({"xai-api": {"auth_mode": "api_key", "key": "secret"}})
        )

        result = AgentRunner(
            WorktreeManager.discover(repository),
            GrokAdapter(os.fspath(fake_grok)),
        ).run(task="grok-api-key", prompt="test", model="grok-4.5")

        assert result.succeeded, result.error
        assert result.billing.route == "metered-api"
        assert result.billing.credential_source == "xai-api-key"
        assert result.provider_reported_cost is not None
        assert result.provider_reported_cost.amount_usd == 0.00045678
        assert "secret" not in json.dumps(result.to_dict())

    def test_dsh_projects_the_selected_provider_and_its_exact_credential_ref(
        self,
        repository: Path,
        fake_dsh: Path,
        monkeypatch: MonkeyPatch,
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
        assert result.usage.input_tokens == 120
        assert result.usage.cached_input_tokens == 30
        assert result.usage.output_tokens == 20
        assert result.usage.reasoning_output_tokens == 7
        assert result.billing.credential_source == "custom-anthropic-auth"
        private_home = (
            manager.state_dir / "provider-state" / "dsh-anthropic" / "dsh" / "home"
        )
        settings = private_home.joinpath("settings.yaml").read_text()
        assert "anthropic:" in settings
        assert "CUSTOM_ANTHROPIC_AUTH" in settings
        assert "openai:" not in settings
        assert "study:" not in settings
        assert not private_home.joinpath(".credentials.yaml").exists()
        assert "environment-custom-value" not in json.dumps(result.to_dict())
        assert "OPENAI_API_KEY" not in result.command
        ref_index = result.command.index("CUSTOM_ANTHROPIC_AUTH")
        assert result.command[ref_index - 1 : ref_index + 2] == [
            "--setenv",
            "CUSTOM_ANTHROPIC_AUTH",
            "<redacted>",
        ]

    def test_dsh_preserves_provider_native_auth_when_api_key_env_is_omitted(
        self,
        repository: Path,
        fake_dsh: Path,
        monkeypatch: MonkeyPatch,
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

    def test_sealed_dsh_passes_only_its_managed_credential_through_the_environment(
        self,
        tmp_path: Path, fake_dsh: Path
    ) -> None:
        manager = WorktreeManager.standalone(tmp_path / "controller")
        result = AgentRunner(manager, DeepSeekHarnessAdapter(os.fspath(fake_dsh))).run(
            task="sealed-dsh", prompt="CHECK_MANAGED_KEY", profile="sealed"
        )

        assert result.succeeded, result.error
        private_home = (
            manager.sealed_runtime_dir / "provider-state" / result.run_id / "dsh" / "home"
        )
        # dsh's headless profile reads launch credentials, not its store.
        assert not private_home.joinpath(".credentials.yaml").exists()
        ref_index = result.command.index("DEEPSEEK_API_KEY")
        assert result.command[ref_index - 1 : ref_index + 2] == [
            "--setenv",
            "DEEPSEEK_API_KEY",
            "<redacted>",
        ]
        assert "OPENAI_API_KEY" not in result.command
        assert "managed test: value" not in json.dumps(result.to_dict())

    def test_dsh_rejects_a_credentials_file_older_than_version_1(
        self,
        tmp_path: Path, fake_dsh: Path, monkeypatch: MonkeyPatch
    ) -> None:
        source = Path(os.environ["AOP_DSH_SOURCE_HOME"])
        source.joinpath(".credentials.yaml").write_text("DEEPSEEK_API_KEY: flat\n")
        manager = WorktreeManager.standalone(tmp_path / "controller")

        with raises(AOPError, match="not a version 1 dsh credentials file"):
            AgentRunner(manager, DeepSeekHarnessAdapter(os.fspath(fake_dsh))).run(
                task="flat-dsh", prompt="test", profile="sealed"
            )

    def test_sealed_dsh_preserves_and_redacts_native_environment_override(
        self,
        tmp_path: Path,
        fake_dsh: Path,
        monkeypatch: MonkeyPatch,
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

    def test_claude_run_and_exact_resume(self, repository: Path, fake_claude: Path) -> None:
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, ClaudeAdapter(os.fspath(fake_claude)))

        first = runner.run(
            task="claude-task",
            prompt="first",
            model="opus",
            effort="high",
            timeout_seconds=5,
        )
        assert first.succeeded, first.error
        resumed = runner.resume(run_id=first.run_id, prompt="second")

        assert first.provider == "claude"
        assert first.model == "claude-test-model"
        assert first.usage.input_tokens == 60
        assert first.usage.cached_input_tokens == 20
        assert first.usage.output_tokens == 40
        assert first.calculated_cost is not None
        assert first.calculated_cost.amount_usd == 0.0123
        assert first.provider_reported_cost is None
        assert first.billing.route == "subscription"
        assert first.billing.credential_source == "claude-oauth"
        assert first.billing.detected_by == "claude auth status"
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
        self,
        repository: Path,
        fake_claude: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AOP_FAKE_CLAUDE_AUTH", "api-key")
        result = AgentRunner(
            WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
        ).run(task="claude-api-billing", prompt="test")

        assert result.succeeded
        assert result.billing.route == "metered-api"
        assert result.billing.credential_source == "anthropic-api-key"

    def test_claude_requires_a_terminal_result_event(
        self,
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
        self,
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
        self,
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
        assert first.usage.input_tokens == 130
        assert first.usage.cached_input_tokens == 30
        assert first.usage.output_tokens == 20
        assert first.provider_duration_seconds == 1.25
        assert first.calculated_cost is None
        assert first.provider_reported_cost is None
        assert first.billing.route == "provider-credits"
        assert first.billing.credential_source == "cursor-account"
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
        self,
        repository: Path, fake_cursor: Path
    ) -> None:
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, CursorAdapter(os.fspath(fake_cursor)))

        with raises(AOPError, match="does not accept a separate effort"):
            runner.run(task="cursor-effort", prompt="test", effort="high")
        assert manager.list() == []

        first = runner.run(task="cursor-resume", prompt="first")
        resumed = runner.resume(run_id=first.run_id, prompt="CURSOR_DIFFERENT_SESSION")

        assert not resumed.succeeded
        assert resumed.session_id is None
        assert "instead of the requested chat" in (resumed.error or "")

    def test_cursor_preserves_response_bearing_provider_errors(
        self,
        repository: Path, fake_cursor: Path
    ) -> None:
        result = AgentRunner(
            WorktreeManager.discover(repository),
            CursorAdapter(os.fspath(fake_cursor)),
        ).run(task="cursor-provider-error", prompt="CURSOR_ERROR_WITH_RESPONSE")

        assert result.status == "response_available_with_provider_error"
        assert result.execution_completed
        assert result.response_available
        assert result.error is None
        assert result.provider_status == "error"
        assert result.provider_error == "synthetic Cursor provider failure"
        assert result.final_message == "answer:CURSOR_ERROR_WITH_RESPONSE"

    def test_cursor_workspace_sandbox_and_artifact(
        self,
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

    @parametrize("profile", ["review", "edit"])
    def test_cursor_uses_private_runtime_state_with_read_only_global_home(
        self,
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

    @parametrize("profile", ["edit", "review", "sealed"])
    def test_sandboxed_runs_cannot_start_user_services(
        self, repository: Path, fake_cursor: Path, tmp_path: Path, profile: str
    ) -> None:
        manager = (
            WorktreeManager.standalone(tmp_path / "controller")
            if profile == "sealed"
            else WorktreeManager.discover(repository)
        )
        runner = AgentRunner(manager, CursorAdapter(os.fspath(fake_cursor)))

        result = runner.run(
            task=f"user-manager-{profile}", prompt="CHECK_NO_USER_MANAGER", profile=profile
        )

        assert result.succeeded, result.error

    def test_cursor_that_ends_its_turn_without_exiting_is_stopped(
        self, repository: Path, fake_cursor: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(CursorAdapter, "TURN_END_GRACE_SECONDS", 0.5)
        monkeypatch.setattr(runner_module, "_STOP_CHECK_SECONDS", 0.2)
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, CursorAdapter(os.fspath(fake_cursor)))

        hung = runner.run(
            task="cursor-hang", prompt="CURSOR_HANG_AFTER_TURN", timeout_seconds=30
        )
        # The earlier turn's end must not stop a resumed turn still working.
        working = runner.resume(run_id=hung.run_id, prompt="CURSOR_SILENT_WORK")

        assert not hung.succeeded
        assert not hung.timed_out
        assert hung.duration_seconds < 20
        assert "ended its turn" in hung.error
        assert hung.final_message == "Task done."
        assert hung.session_id == "b5b6dbdd-0d68-452f-9d5c-20238c970169"
        assert hung.accounting_status == "unavailable"
        assert working.succeeded, working.error
        assert working.session_id == hung.session_id

    def test_devin_defaults_to_swe_and_resumes_exact_session(
        self,
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
        assert first.calculated_cost is None
        assert first.provider_reported_cost is None
        assert first.billing.route == "provider-credits"
        assert first.billing.credential_source == "devin-account"
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
        self,
        repository: Path, fake_devin: Path
    ) -> None:
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, DevinAdapter(os.fspath(fake_devin)))

        with raises(AOPError, match="does not accept a separate effort"):
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

    def test_devin_matches_the_prompt_it_records_without_trailing_whitespace(
        self, repository: Path, fake_devin: Path
    ) -> None:
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, DevinAdapter(os.fspath(fake_devin)))

        result = runner.run(task="devin-newline", prompt="first\n")

        assert result.succeeded, result.error
        assert result.session_id == "tested-basil"

    def test_devin_waits_for_a_usage_limit_reset_and_continues_the_session(
        self, repository: Path, fake_devin: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DevinAdapter, "_LIMIT_MARGIN_SECONDS", 0.0)
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, DevinAdapter(os.fspath(fake_devin)))

        result = runner.run(
            task="devin-limit", prompt="DEVIN_RATE_LIMIT", timeout_seconds=120
        )

        assert result.succeeded, result.error
        assert result.session_id == "tested-basil"
        assert result.final_message.startswith("answer:Your previous turn stopped")
        assert result.usage.input_tokens == 150
        assert result.usage.output_tokens == 25
        stderr = (manager.state_dir / "runs" / result.run_id / "stderr.log").read_text()
        assert "Reached free model rate limit" in stderr

    def test_devin_limit_past_the_deadline_fails_with_a_resumable_session(
        self, repository: Path, fake_devin: Path
    ) -> None:
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, DevinAdapter(os.fspath(fake_devin)))

        limited = runner.run(
            task="devin-late", prompt="DEVIN_RATE_LIMIT_LATE", timeout_seconds=60
        )
        resumed = runner.resume(run_id=limited.run_id, prompt="second")

        assert not limited.succeeded
        assert "Reached free model rate limit" in limited.error
        assert limited.duration_seconds < 30
        assert limited.session_id == "tested-basil"
        assert resumed.succeeded, resumed.error
        assert resumed.session_id == "tested-basil"

    def test_devin_refuses_a_prompt_it_would_read_as_a_slash_command(
        self, repository: Path, fake_devin: Path
    ) -> None:
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, DevinAdapter(os.fspath(fake_devin)))
        first = runner.run(task="devin-slash", prompt="first")

        with raises(AOPError, match="slash command"):
            runner.run(task="devin-slash-new", prompt="/workspace is a worktree")
        with raises(AOPError, match="slash command"):
            runner.resume(run_id=first.run_id, prompt="  /compact")
        assert not any(item.task == "devin-slash-new" for item in manager.list())

    def test_devin_workspace_sandbox_and_artifact(
        self,
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

    @parametrize("profile", ["review", "edit", "host"])
    def test_devin_uses_private_runtime_state_with_read_only_global_profile(
        self,
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

    def test_opencode_long_prompt_uses_stdin(self, repository: Path, fake_opencode: Path) -> None:
        prompt = "x" * 140_000
        result = AgentRunner(WorktreeManager.discover(repository), OpenCodeAdapter(os.fspath(fake_opencode))).run(task="long-opencode", prompt=prompt, timeout_seconds=5)
        assert result.succeeded
        assert result.final_message == "answer:" + prompt
        assert prompt not in result.command

    def test_opencode_defaults_to_zen_model_and_resumes_exact_session(
        self,
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
        assert first.usage.output_tokens == 7
        assert first.usage.reasoning_output_tokens == 2
        assert first.provider_duration_seconds is not None
        assert first.calculated_cost is not None
        assert first.calculated_cost.priced_as == "deepseek/deepseek-v4-flash"
        assert first.calculated_cost.amount_usd == api_cost(
            "deepseek-v4-flash",
            first.started_at,
            input_tokens=53,
            cached_input_tokens=10,
            output_tokens=7,
            provider="deepseek",
        )
        assert first.provider_reported_cost is not None
        assert first.provider_reported_cost.amount_usd == 0.00012345
        assert first.provider_reported_cost.source == "OpenCode step_finish events"
        assert first.billing.route == "provider-credits"
        assert first.billing.credential_source == "opencode-api-key"
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
        self,
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
        self,
        repository: Path, fake_opencode: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository),
            OpenCodeAdapter(os.fspath(fake_opencode)),
        )
        result = runner.run(task="opencode-error", prompt="OPENCODE_ERROR")
        response = runner.run(
            task="opencode-provider-error", prompt="OPENCODE_ERROR_WITH_RESPONSE"
        )

        assert not result.succeeded
        assert result.error == "synthetic failure"
        assert result.provider_error == "synthetic failure"
        assert response.status == "response_available_with_provider_error"
        assert response.execution_completed
        assert response.response_available
        assert response.error is None
        assert response.provider_status == "error"
        assert response.provider_error == "synthetic failure"
        assert response.final_message == "answer:OPENCODE_ERROR_WITH_RESPONSE"

    def test_opencode_normalizes_multi_step_tool_loop(
        self,
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
        assert result.usage.output_tokens == 10
        assert result.usage.reasoning_output_tokens == 3
        assert result.provider_reported_cost is not None
        assert result.provider_reported_cost.amount_usd == 0.00022345

    def test_opencode_timeout_preserves_observed_partial_accounting(
        self,
        repository: Path, fake_opencode: Path
    ) -> None:
        result = AgentRunner(
            WorktreeManager.discover(repository),
            OpenCodeAdapter(os.fspath(fake_opencode)),
        ).run(
            task="opencode-partial-timeout",
            prompt="OPENCODE_PARTIAL_TIMEOUT",
            timeout_seconds=1,
        )

        assert result.timed_out
        assert result.accounting_status == "partial"
        assert result.usage == TokenUsage(
            input_tokens=8,
            cached_input_tokens=3,
            cache_write_input_tokens=4,
            output_tokens=3,
            reasoning_output_tokens=1,
        )
        assert result.provider_reported_cost is not None
        assert result.provider_reported_cost.amount_usd == 0.0001

    def test_opencode_workspace_sandbox_and_artifact(
        self,
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

    @parametrize("profile", ["review", "edit"])
    def test_opencode_uses_private_runtime_state_with_read_only_global_profile(
        self,
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
        self,
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
        assert first.usage.input_tokens == 130
        assert first.usage.cached_input_tokens == 30
        assert first.usage.output_tokens == 20
        assert first.usage.reasoning_output_tokens == 7
        assert first.provider_duration_seconds == 1.25
        assert first.time_to_first_response_seconds is not None
        assert first.calculated_cost is not None
        assert first.calculated_cost.amount_usd == 0.0003345
        assert first.calculated_cost.model == "gemini-3.5-flash"
        assert first.calculated_cost.priced_as == "google/gemini-3.5-flash"
        assert first.billing.route == "subscription"
        assert first.billing.credential_source == "google-oauth"
        events_path = manager.state_dir / "runs" / first.run_id / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        assert [event["event"] for event in events] == [
            "init",
            "step_update",
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
        # Agy's result reports the conversation's running total; each run records
        # only its own usage, so summing runs does not double count.
        assert resumed.usage == first.usage
        assert resumed.calculated_cost.amount_usd == first.calculated_cost.amount_usd

    @parametrize("profile", ["review", "edit", "host"])
    def test_agy_uses_private_persistent_runtime_state_in_every_sandbox(
        self,
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

    @parametrize("profile", ["review", "sealed"])
    def test_agy_preserves_direct_gemini_route_in_isolated_state(
        self,
        repository: Path,
        fake_agy: Path,
        monkeypatch: MonkeyPatch,
        profile: str,
    ) -> None:
        manager = WorktreeManager.discover(repository)
        source_dir = Path(os.environ["AOP_AGY_SOURCE_DIR"])
        source_settings = {
            "modelProvider": "gemini",
            "permissions": {"shell": "allow"},
            "trustedWorkspaces": ["/private/source"],
        }
        source_dir.joinpath("antigravity-cli", "settings.json").write_text(
            json.dumps(source_settings)
        )
        api_key = "test-gemini-api-key"
        endpoint = "https://gemini.example.test/v1"
        monkeypatch.setenv("GEMINI_API_KEY", api_key)
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", endpoint)

        result = AgentRunner(manager, AgyAdapter(os.fspath(fake_agy))).run(
            task=f"agy-direct-gemini-{profile}",
            prompt="CHECK_AGY_API_ROUTE",
            profile=profile,
            timeout_seconds=5,
        )

        assert result.succeeded
        assert result.billing.route == "metered-api"
        assert result.billing.credential_source == "gemini-api-key"
        request_path = manager.state_dir / "runs" / result.run_id / "request.json"
        request = json.loads(request_path.read_text())
        private_dir = Path(request["effective_policy"]["controller"]["provider_state"])
        private_agy = private_dir / "agy" / "gemini"
        assert json.loads(
            private_agy.joinpath("antigravity-cli", "settings.json").read_text()
        ) == ({"modelProvider": "gemini"} if profile == "sealed" else source_settings)
        assert not private_agy.joinpath("oauth_creds.json").exists()
        assert not private_agy.joinpath("google_accounts.json").exists()
        assert not private_agy.joinpath(
            "antigravity-cli", "antigravity-oauth-token"
        ).exists()
        inherited = request["effective_policy"]["environment"]["inherited_names"]
        assert "GEMINI_API_KEY" in inherited
        assert "GOOGLE_GEMINI_BASE_URL" in inherited
        recorded = request_path.read_text()
        assert api_key not in recorded
        assert endpoint not in recorded

    @parametrize(
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
        self,
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
        with raises(AOPError, match="run has no resumable agent session"):
            runner.resume(run_id=resumed.run_id, prompt="third")

    def test_agy_requires_an_authenticated_source_profile(
        self,
        repository: Path,
        fake_agy: Path,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        missing = tmp_path / "missing-agy-profile"
        monkeypatch.setenv("AOP_AGY_SOURCE_DIR", os.fspath(missing))
        manager = WorktreeManager.discover(repository)

        with raises(AOPError, match="authenticate Agy first"):
            AgentRunner(manager, AgyAdapter(os.fspath(fake_agy))).run(
                task="missing-agy-profile",
                prompt="first",
                timeout_seconds=5,
            )

        assert not (manager.state_dir / "runs").exists()

    def test_agy_terminal_provider_error_with_response_is_classified_separately(
        self,
        repository: Path, fake_agy: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), AgyAdapter(os.fspath(fake_agy))
        )

        result = runner.run(
            task="agy-error",
            prompt="AGY_ERROR_WITH_RESPONSE",
            timeout_seconds=5,
        )

        assert result.exit_code == 0
        assert result.execution_completed
        assert result.response_available
        assert not result.succeeded
        assert result.status == "response_available_with_provider_error"
        assert result.error is None
        assert result.provider_status == "ERROR"
        assert result.provider_error == "synthetic agy failure"
        assert result.final_message == "answer:AGY_ERROR_WITH_RESPONSE"

    def test_agy_terminal_provider_error_without_response_remains_failed(
        self,
        repository: Path, fake_agy: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), AgyAdapter(os.fspath(fake_agy))
        )

        result = runner.run(
            task="agy-error-without-response",
            prompt="AGY_ERROR",
            timeout_seconds=5,
        )

        assert result.execution_completed
        assert not result.response_available
        assert not result.succeeded
        assert result.status == "failed"
        assert result.error == "agy terminal result did not include a response"
        assert result.provider_status == "ERROR"
        assert result.provider_error == "synthetic agy failure"

    def test_agy_rejects_an_unsupported_effort(self, repository: Path, fake_agy: Path) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), AgyAdapter(os.fspath(fake_agy))
        )

        with raises(AOPError, match="agy effort must be one of: high, low, medium"):
            runner.run(
                task="bad-agy",
                prompt="test",
                effort="xhigh",
            )

    def test_agy_rejects_an_unsupported_version_before_creating_state(
        self,
        repository: Path,
        fake_agy: Path,
        monkeypatch: MonkeyPatch,
    ) -> None:
        manager = WorktreeManager.discover(repository)

        def reject(_binary: str) -> None:
            raise AOPError("Agy 1.1.15 is unsupported; AOP requires Agy 1.1.16 or newer")

        monkeypatch.setattr(runner_module, "require_supported_agy", reject)

        with raises(AOPError, match="Agy 1.1.15 is unsupported"):
            AgentRunner(manager, AgyAdapter(os.fspath(fake_agy))).run(
                task="old-agy",
                prompt="test",
            )

        assert manager.list() == []
        assert not manager.state_dir.joinpath("provider-state").exists()

    def test_agy_delegates_the_default_model_and_effort(
        self,
        repository: Path, fake_agy: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), AgyAdapter(os.fspath(fake_agy))
        )

        result = runner.run(task="default-agy", prompt="test", timeout_seconds=5)

        assert result.succeeded
        assert result.model == "gemini-3.5-flash"
        assert result.effort is None
        assert "--model" not in result.command
        assert "--effort" not in result.command

    def test_agy_passes_an_exact_model_without_adding_effort(
        self,
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

    def test_agy_can_produce_a_declared_artifact(self, repository: Path, fake_agy: Path) -> None:
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
        self,
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
        self,
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
        self,
        repository: Path,
        fake_hermes: Path,
        monkeypatch: MonkeyPatch,
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
        assert first.calculated_cost is not None
        assert first.calculated_cost.amount_usd == 0.000001
        assert first.billing.route == "subscription"
        assert first.billing.credential_source == "nous-oauth"

        assert resumed.succeeded
        assert resumed.session_id == first.session_id
        assert resumed.final_message == "answer:second"
        assert resumed.usage == first.usage
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
        assert resumed.calculated_cost is not None
        assert resumed.calculated_cost.amount_usd == 0.000001

    def test_hermes_provider_override_is_recorded_and_reused_on_resume(
        self,
        repository: Path,
        fake_hermes: Path,
        monkeypatch: MonkeyPatch,
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

    def test_provider_override_requires_a_supported_route_and_explicit_model(
        self,
        repository: Path,
        fake_codex: Path,
        fake_hermes: Path,
    ) -> None:
        manager = WorktreeManager.discover(repository)

        with raises(
            AOPError, match='codex inference provider "openai" is not supported'
        ):
            AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).run(
                task="unsupported-provider",
                prompt="unused",
                model="gpt-5.6-sol",
                inference_provider="openai",
            )
        with raises(AOPError, match="requires an explicit --model"):
            AgentRunner(manager, HermesAdapter(os.fspath(fake_hermes))).run(
                task="missing-provider-model",
                prompt="unused",
                inference_provider="xai-oauth",
            )

    def test_hermes_participant_mode_is_tool_free_bounded_and_stable_across_resume(
        self,
        repository: Path,
        fake_hermes: Path,
        monkeypatch: MonkeyPatch,
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

    def test_no_web_enforcement_is_recorded_for_supported_harnesses(
        self,
        repository: Path,
        fake_codex: Path,
        fake_claude: Path,
        fake_opencode: Path,
        fake_grok: Path,
        fake_hermes: Path,
    ) -> None:
        manager = WorktreeManager.discover(repository)
        cases = [
            (
                CodexAdapter(os.fspath(fake_codex)),
                "codex",
                {"app-server", "--disable", "plugins", "--stdio"},
            ),
            (
                ClaudeAdapter(os.fspath(fake_claude)),
                "claude",
                {"--safe-mode", "--no-chrome", "--strict-mcp-config", "--tools"},
            ),
            (
                OpenCodeAdapter(os.fspath(fake_opencode)),
                "opencode",
                {"--pure"},
            ),
            (
                GrokAdapter(os.fspath(fake_grok)),
                "grok",
                {"--disable-web-search", "--no-subagents", "--tools", "MCPTool"},
            ),
            (
                HermesAdapter(os.fspath(fake_hermes)),
                "hermes",
                {"--safe-mode", "--toolsets", "file,todo"},
            ),
        ]

        for adapter, provider, expected_arguments in cases:
            result = AgentRunner(manager, adapter).run(
                task=f"no-web-{provider}",
                prompt="test",
                no_web=True,
            )

            assert result.succeeded
            assert expected_arguments <= set(result.command)
            request = json.loads(
                (manager.state_dir / "runs" / result.run_id / "request.json").read_text()
            )
            capabilities = request["effective_policy"]["model_capabilities"]
            assert request["no_web"] is True
            assert capabilities["requested"]["external_retrieval"] == "denied"
            assert capabilities["effective"]["external_retrieval"] == "denied"
            assert capabilities["effective"]["tool_network_egress"] == "denied"
            assert capabilities["effective"]["inference_network"] == "allowed"
            assert capabilities["enforcement"]

            if provider == "codex":
                events = [
                    json.loads(line)
                    for line in (
                        manager.state_dir / "runs" / result.run_id / "events.jsonl"
                    )
                    .read_text()
                    .splitlines()
                ]
                start = next(
                    event for event in events if event.get("method") == "thread/start"
                )
                config = start["params"]["config"]
                assert config["web_search"] == "disabled"
                assert "tools" not in config
                permission = config["permissions"]["aop-no-web"]
                assert permission["network"]["enabled"] is False
                assert start["params"]["permissions"] == "aop-no-web"
                assert "sandbox" not in start["params"]
                recorded = capabilities["permission_profile"]
                assert recorded["id"] == "aop-no-web"
                assert recorded["active_profile_validation"] == "required"

            if provider in {"claude", "hermes"}:
                assert request["effective_policy"]["instructions"]["inherited_local"] == (
                    "none"
                )
                assert request["effective_policy"]["instruction_sources"] == []

            if provider == "grok":
                state = Path(request["effective_policy"]["controller"]["provider_state"])
                grok_home = state / "grok" / "home"
                assert grok_home.joinpath("auth.json").is_file()
                assert not grok_home.joinpath("config.toml").exists()

    @parametrize("provider", ["cursor", "agy"])
    def test_no_web_rejects_unsupported_harness_before_creating_a_worktree(
        self,
        repository: Path,
        fake_cursor: Path,
        fake_devin: Path,
        fake_agy: Path,
        fake_dsh: Path,
        provider: str,
    ) -> None:
        binaries = {
            "cursor": fake_cursor,
            "devin": fake_devin,
            "agy": fake_agy,
            "dsh": fake_dsh,
        }
        adapters = {
            "cursor": CursorAdapter,
            "devin": DevinAdapter,
            "agy": AgyAdapter,
            "dsh": DeepSeekHarnessAdapter,
        }
        manager = WorktreeManager.discover(repository)

        with raises(AOPError, match=f"{provider} cannot enforce --no-web"):
            AgentRunner(manager, adapters[provider](os.fspath(binaries[provider]))).run(
                task=f"unsupported-no-web-{provider}",
                prompt="test",
                no_web=True,
            )

        assert manager.list() == []

    def test_no_web_rejects_grok_host_profile_before_creating_a_worktree(
        self,
        repository: Path, fake_grok: Path
    ) -> None:
        manager = WorktreeManager.discover(repository)

        with raises(
            AOPError, match="grok cannot enforce --no-web with the host profile"
        ):
            AgentRunner(manager, GrokAdapter(os.fspath(fake_grok))).run(
                task="unsupported-no-web-grok-host",
                prompt="test",
                profile="host",
                no_web=True,
            )

        assert manager.list() == []

    @parametrize("profile", ["edit", "review", "sealed", "host"])
    def test_no_web_is_inherited_by_resume_with_the_same_private_state(
        self,
        repository: Path, fake_codex: Path, profile: str
    ) -> None:
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))
        first = runner.run(
            task="no-web-resume", prompt="first", profile=profile, no_web=True
        )
        resumed = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).resume(
            run_id=first.run_id, prompt="second"
        )

        assert resumed.succeeded
        events = [
            json.loads(line)
            for line in (manager.state_dir / "runs" / resumed.run_id / "events.jsonl")
            .read_text()
            .splitlines()
        ]
        resume = next(event for event in events if event.get("method") == "thread/resume")
        config = resume["params"]["config"]
        assert config["web_search"] == "disabled"
        assert "tools" not in config
        assert config["permissions"]["aop-no-web"]["network"]["enabled"] is False
        assert resume["params"]["permissions"] == "aop-no-web"
        assert "sandbox" not in resume["params"]
        filesystem = config["permissions"]["aop-no-web"]["filesystem"]
        assert filesystem[":root"] == ("write" if profile == "host" else "read")
        if profile != "host":
            assert filesystem["/workspace"] == ("write" if profile == "edit" else "read")
        first_request = json.loads(
            (manager.state_dir / "runs" / first.run_id / "request.json").read_text()
        )
        resumed_request = json.loads(
            (manager.state_dir / "runs" / resumed.run_id / "request.json").read_text()
        )
        assert resumed_request["no_web"] is True
        assert (
            resumed_request["effective_policy"]["controller"]["provider_state"]
            == first_request["effective_policy"]["controller"]["provider_state"]
        )
        provider_state = Path(
            resumed_request["effective_policy"]["controller"]["provider_state"]
        )
        private_home = provider_state / "codex" / "home"
        assert private_home.joinpath("auth.json").is_file()
        assert not private_home.joinpath("config.toml").exists()
        assert not private_home.joinpath("rules").exists()
        assert not private_home.joinpath("skills").exists()
        if profile != "sealed":
            manager.remove("no-web-resume")
            assert not provider_state.exists()

    @parametrize(
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
        self,
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

        with raises(AOPError, match=f"{provider} does not support participant mode"):
            AgentRunner(manager, adapter(os.fspath(binary))).run(
                task=f"unsupported-{provider}",
                prompt="test",
                mode="participant",
            )

        assert manager.list() == []

    def test_hermes_supports_workspace_sandbox_and_artifacts(
        self,
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

    def test_hermes_uses_and_records_its_configured_default_model(
        self,
        repository: Path, fake_hermes: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), HermesAdapter(os.fspath(fake_hermes))
        )

        result = runner.run(task="default-hermes", prompt="test")

        assert result.succeeded
        assert "--model" not in result.command
        assert result.model == "deepseek/deepseek-v4-flash-0731"
        assert result.effort is None
        assert "--reasoning" not in result.command

    def test_sealed_hermes_home_gets_only_the_model_configuration(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        (source / "auth.json").write_text("{}")
        (source / "config.yaml").write_text(
            "model:\n  provider: xai-oauth\n  default: grok-build-0.1\n"
            "toolsets: [web]\npersonality: concealed\n"
        )
        destination = tmp_path / "home"

        runner_module._prepare_hermes_home(source, destination, sealed=True)

        assert sorted(path.name for path in destination.iterdir()) == [
            "auth.json",
            "config.yaml",
        ]
        assert yaml.safe_load((destination / "config.yaml").read_text()) == {
            "model": {"provider": "xai-oauth", "default": "grok-build-0.1"}
        }

    def test_hermes_rejects_an_unsupported_effort(
        self,
        repository: Path, fake_hermes: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), HermesAdapter(os.fspath(fake_hermes))
        )

        with raises(AOPError, match="Hermes effort must be one of"):
            runner.run(task="bad-hermes", prompt="test", effort="extreme")

    def test_hermes_nonzero_exit_is_a_normalized_failure(
        self,
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

    def test_hermes_failure_reports_provider_error_instead_of_resume_banner(self) -> None:
        error = HermesAdapter._exit_error(
            "reasoning\nAPI call failed after 3 retries: empty stream\n",
            "↻ Resumed session session-1 (1 user message, 2 total messages)\n"
            "session_id: session-1\n",
            1,
        )

        assert error == "API call failed after 3 retries: empty stream"

    def test_hermes_resume_keeps_a_new_answer_that_repeats_the_previous_one(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # The shape of a Hermes 0.22 export: integer message IDs.
        def export(messages: list[dict]) -> str:
            return json.dumps({"id": "session-1", "model": "gpt-4.1", "messages": messages})

        first = [
            {"id": 1, "role": "user", "content": "Reply with silver."},
            {"id": 2, "role": "assistant", "content": "silver", "finish_reason": "stop"},
        ]
        second = [
            *first,
            {"id": 3, "role": "user", "content": "Repeat it."},
            {"id": 4, "role": "assistant", "content": "silver", "finish_reason": "stop"},
        ]
        sessions = []
        for messages in (first, second):
            monkeypatch.setattr(
                runner_module.subprocess,
                "run",
                lambda *args, output=export(messages), **kwargs: runner_module.subprocess.CompletedProcess(
                    args, 0, stdout=output, stderr=""
                ),
            )
            sessions.append(
                HermesAdapter("hermes")._session(
                    "session-1",
                    _host_request(),
                    Worktree(task="t", path=Path("/tmp"), head=""),
                    {},
                )
            )

        assert sessions[0].last_assistant_id == "2"
        assert sessions[1].last_assistant_id == "4"
        assert HermesAdapter._final_message(sessions[0], sessions[1]) == "silver"

    def test_hermes_does_not_return_a_stale_message_when_resume_adds_no_answer(self) -> None:
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

    def test_hermes_unknown_session_cost_uses_api_equivalent_rates(self) -> None:
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

        cost, reported_cost = HermesAdapter._cost_delta(
            None, session, "grok-4.5", usage, started_at=AT, finished_at=AT
        )

        assert cost is not None
        assert cost.amount_usd == 0.00915
        assert cost.priced_as == "xai/grok-4.5"
        assert reported_cost is None

    def test_hermes_unknown_session_cost_with_no_usage_remains_unknown(self) -> None:
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

        assert HermesAdapter._cost_delta(
            None, session, "grok-4.5", TokenUsage(), started_at=AT, finished_at=AT
        ) == (None, None)

    def test_hermes_separates_calculated_cost_from_provider_reported_cost(self) -> None:
        session = _HermesSession(
            model="grok-4.5",
            billing_provider="xai",
            final_message="answer",
            last_assistant_id="2",
            usage=TokenUsage(),
            cost_usd=0.02,
            cost_estimated=False,
            cost_source="provider",
            pricing_version="hermes-cli-reported",
        )
        usage = TokenUsage(
            input_tokens=2_000,
            cached_input_tokens=500,
            output_tokens=1_000,
        )

        cost, reported_cost = HermesAdapter._cost_delta(
            None, session, "grok-4.5", usage, started_at=AT, finished_at=AT
        )

        assert cost is not None
        assert cost.amount_usd == 0.00915
        assert reported_cost is not None
        assert reported_cost.amount_usd == 0.02
        assert reported_cost.source == "Hermes CLI session accounting (provider)"

    @parametrize("profile", ["review", "edit", "host"])
    def test_hermes_run_and_resume_with_read_only_runtime_home(
        self,
        repository: Path,
        fake_hermes: Path,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
        profile: str,
    ) -> None:
        hermes_home = tmp_path / f"managed-hermes-{profile}"
        hermes_home.mkdir()
        (hermes_home / "auth.json").write_text("{}\n")
        (hermes_home / "config.yaml").write_text(
            "model:\n  default: deepseek/deepseek-v4-flash-0731\n  provider: nous\n"
        )
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
            manager.state_dir / "provider-state" / f"managed-{profile}" / "hermes" / "profiles" / "aop"
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
        self,
        repository: Path,
        fake_hermes: Path,
        monkeypatch: MonkeyPatch,
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
        (source_home / "config.yaml").write_text(
            "model:\n  default: grok-4.6\n  provider: xai-oauth\n"
        )
        monkeypatch.setenv("HERMES_HOME", os.fspath(source_home))
        monkeypatch.setenv("AOP_HERMES_BIN", os.fspath(fake_hermes))
        monkeypatch.setenv("AOP_FAKE_HERMES_STATE_IN_HOME", "1")
        monkeypatch.setenv("AOP_FAKE_HERMES_ROTATE_DELAY", "0.2")
        manager = WorktreeManager.discover(repository)

        newer_private = (
            manager.state_dir / "provider-state" / "previous-success" / "hermes" / "profiles" / "aop"
        )
        newer_private.mkdir(parents=True)
        (newer_private / "auth.json").write_text(json.dumps(auth(1, updated_minute=14)))
        later_but_stale = (
            manager.state_dir / "provider-state" / "later-failure" / "hermes" / "profiles" / "aop"
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

    @parametrize("effort", ["minimal", "medium", "xhigh"])
    def test_dsh_rejects_unsupported_official_effort(self, repository, fake_dsh, effort):
        runner = AgentRunner(
            WorktreeManager.discover(repository), DeepSeekHarnessAdapter(str(fake_dsh))
        )
        with raises(AOPError, match="DeepSeek effort must be one of"):
            runner.run(task="bad-effort", prompt="test", effort=effort)

    def test_dsh_missing_usage_is_unavailable(self, repository, fake_dsh, monkeypatch):
        def unexpected_pricing(*args):
            raise AssertionError("missing usage must not trigger a pricing lookup")

        monkeypatch.setattr(DeepSeekHarnessAdapter, "_estimate_cost", unexpected_pricing)
        script = fake_dsh.read_text()
        # Simulate the driver's error path before an assistant usage event.
        fake_dsh.write_text(
            script
            + "\nprint(json.dumps({'type': 'aop.dsh.result', 'session_id': session_id, 'usage': None, 'completed': False, 'error': 'early failure'}))\n"
        )
        result = AgentRunner(
            WorktreeManager.discover(repository), DeepSeekHarnessAdapter(str(fake_dsh))
        ).run(task="no-usage", prompt="test")
        assert not result.succeeded
        assert result.usage is None
        assert result.calculated_cost is None
        assert result.accounting_status == "unavailable"

    def test_dsh_host_rejects_old_node(self, repository, tmp_path, monkeypatch):
        native = tmp_path / "dsh.js"
        native.write_text("#!/usr/bin/env node\n")
        native.chmod(0o755)
        node = tmp_path / "node"
        node.write_text("#!/bin/sh\necho v22.14.0\n")
        node.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
        runner = AgentRunner(
            WorktreeManager.discover(repository), DeepSeekHarnessAdapter(str(native))
        )
        with raises(AOPError, match="PATH selects.*v22.14.0"):
            runner.run(task="old-node", prompt="test", profile="host")

    @parametrize("profile", ["edit", "review", "sealed"])
    def test_opencode_reuses_private_dependencies_on_run_and_resume(
        self,
        repository: Path, fake_opencode: Path, profile: str
    ) -> None:
        source = Path(os.environ["AOP_OPENCODE_CONFIG_DIR"]) / "node_modules"
        (source / ".bin").mkdir()
        (source / ".bin" / "plugin").symlink_to("../@opencode-ai/plugin/package.json")
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, OpenCodeAdapter(os.fspath(fake_opencode)))
        first = runner.run(task="dependency-links", prompt="first", profile=profile)
        assert first.succeeded, first.error
        request = runner.store.load_request(first.run_id)
        private = (
            Path(request.effective_policy["controller"]["provider_state"])
            / "opencode/config/opencode/node_modules"
        )
        assert (private / ".bin/plugin").is_symlink()
        assert (private / ".bin/plugin").read_text() == '{"version": "test"}\n'
        # Dependencies become task-owned state, just like the rest of the native
        # config. Neither a resume nor another run should overwrite native updates.
        (private / "@opencode-ai/plugin/package.json").write_text('{"version": "task-local"}\n')
        for result in (
            runner.resume(run_id=first.run_id, prompt="resume"),
            runner.run(task="dependency-links", prompt="again", profile=profile),
        ):
            assert result.succeeded, result.error
        assert (private / ".bin/plugin").read_text() == '{"version": "task-local"}\n'
        assert (source / ".bin/plugin").read_text() == '{"version": "test"}\n'

    def test_claude_home_seed_skips_locks_and_files_removed_during_copy(
        self,
        tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        source = tmp_path / "source"
        config = source / ".claude"
        config.mkdir(parents=True)
        (config / "settings.json").write_text("{}")
        (config / "history.jsonl.lock").write_text("")
        (config / "vanishing.json").write_text("{}")
        copy = runner_module.shutil.copy2

        def copy_after_removal(source_path, target_path, *args, **kwargs):
            if Path(source_path).name == "vanishing.json":
                Path(source_path).unlink()
            return copy(source_path, target_path, *args, **kwargs)

        monkeypatch.setattr(runner_module.shutil, "copy2", copy_after_removal)
        environment = {
            "HOME": os.fspath(source),
            "AOP_PROVIDER_STATE_DIR": os.fspath(tmp_path / "state"),
        }

        runner_module._prepare_claude_environment(environment)

        seeded = Path(environment["HOME"]) / ".claude"
        assert sorted(path.name for path in seeded.iterdir()) == ["settings.json"]
        assert [path.name for path in (tmp_path / "state" / "claude").iterdir()] == [
            "home"
        ]

    def test_host_claude_keeps_home_and_uses_its_private_config_dir(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source"
        (source / ".claude").mkdir(parents=True)
        (source / ".claude" / "settings.json").write_text("{}")
        (source / ".claude.json").write_text('{"user": "main"}')
        environment = {
            "HOME": os.fspath(source),
            "AOP_PROFILE": "host",
            "AOP_PROVIDER_STATE_DIR": os.fspath(tmp_path / "state"),
        }

        runner_module._prepare_claude_environment(environment)

        config = tmp_path / "state" / "claude" / "home" / ".claude"
        assert environment["HOME"] == os.fspath(source)
        assert environment["CLAUDE_CONFIG_DIR"] == os.fspath(config)
        assert (config / "settings.json").is_file()
        assert (config / ".claude.json").read_text() == '{"user": "main"}'
        assert (source / ".claude.json").read_text() == '{"user": "main"}'

    @parametrize("profile", ["host", "sealed", "edit"])
    def test_reused_claude_home_takes_the_users_newer_sign_in(
        self, tmp_path: Path, profile: str
    ) -> None:
        source = tmp_path / "source"
        (source / ".claude").mkdir(parents=True)
        credentials = source / ".claude" / ".credentials.json"

        def sign_in(token: str, expires_at: int) -> None:
            credentials.write_text(json.dumps(
                {"claudeAiOauth": {"accessToken": token, "expiresAt": expires_at}}
            ))

        environment = {
            "HOME": os.fspath(source),
            "AOP_PROFILE": profile,
            "AOP_PROVIDER_STATE_DIR": os.fspath(tmp_path / "state"),
        }
        private = tmp_path / "state" / "claude" / "home" / ".claude" / ".credentials.json"

        sign_in("first", 1_000)
        runner_module._prepare_claude_environment(dict(environment))
        # The user signs in again after the task's home was seeded.
        sign_in("renewed", 2_000)
        runner_module._prepare_claude_environment(dict(environment))
        renewed = json.loads(private.read_text())["claudeAiOauth"]["accessToken"]
        # A login Claude refreshed inside the task is newer and is kept.
        private.write_text(json.dumps(
            {"claudeAiOauth": {"accessToken": "refreshed-in-task", "expiresAt": 3_000}}
        ))
        runner_module._prepare_claude_environment(dict(environment))

        assert renewed == "renewed"
        assert private.stat().st_mode & 0o777 == 0o600
        assert json.loads(private.read_text())["claudeAiOauth"]["accessToken"] == (
            "refreshed-in-task"
        )

    def test_failed_claude_home_seed_leaves_no_partial_home(
        self,
        tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        config = tmp_path / "source" / ".claude"
        config.mkdir(parents=True)
        (config / "settings.json").write_text("{}")

        def fail(*args, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(runner_module.shutil, "copy2", fail)
        environment = {
            "HOME": os.fspath(tmp_path / "source"),
            "AOP_PROVIDER_STATE_DIR": os.fspath(tmp_path / "state"),
        }

        with raises(AOPError, match="could not prepare isolated Claude state"):
            runner_module._prepare_claude_environment(environment)
        assert list((tmp_path / "state" / "claude").iterdir()) == []

    def test_claude_system_prompt_and_tools_are_recorded_and_kept_on_resume(
        self,
        repository: Path, fake_claude: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
        )

        first = runner.run(
            task="minimal",
            prompt="first",
            system_prompt="",
            tools=("Bash", "Read", "Write", "Edit"),
        )
        resumed = runner.resume(run_id=first.run_id, prompt="second")

        for result in (first, resumed):
            assert result.succeeded, result.error
            command = result.command
            assert command[command.index("--system-prompt") + 1] == ""
            assert command[command.index("--tools") + 1] == "Bash,Read,Write,Edit"
            request = runner.store.load_request(result.run_id)
            assert request.system_prompt == ""
            assert request.tools == ("Bash", "Read", "Write", "Edit")

    def test_claude_empty_tool_list_disables_every_tool(
        self,
        repository: Path, fake_claude: Path
    ) -> None:
        result = AgentRunner(
            WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
        ).run(task="no-tools", prompt="first", tools=())

        assert result.command[result.command.index("--tools") + 1] == ""
        assert "--system-prompt" not in result.command

    def test_no_web_claude_tools_stay_within_the_offline_set(
        self,
        repository: Path, fake_claude: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
        )

        with raises(AOPError, match="--no-web allows only"):
            runner.run(task="web", prompt="first", no_web=True, tools=("Read", "Bash"))
        result = runner.run(task="offline", prompt="first", no_web=True, tools=("Read",))
        assert result.command[result.command.index("--tools") + 1] == "Read"
        assert result.command.count("--tools") == 1

    def test_system_prompt_and_tools_are_claude_only(
        self,
        repository: Path, fake_codex: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
        )

        with raises(AOPError, match="does not support --system-prompt"):
            runner.run(task="codex", prompt="first", system_prompt="")

    def test_claude_provider_base_url_reaches_the_sandbox_and_is_kept_on_resume(
        self,
        repository: Path, fake_claude: Path, monkeypatch: MonkeyPatch
    ) -> None:
        script = fake_claude.read_text()
        fake_claude.write_text(
            script.replace(
                "prompt = sys.stdin.read()\n",
                "prompt = sys.stdin.read()\n"
                "if os.environ.get('ANTHROPIC_BASE_URL') != 'http://127.0.0.1:8765':\n"
                "    raise SystemExit('base URL missing')\n",
                1,
            )
        )
        runner = AgentRunner(
            WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
        )

        for profile in ("sealed", "edit"):
            first = runner.run(
                task=f"recorded-{profile}",
                prompt="first",
                profile=profile,
                provider_base_url="http://127.0.0.1:8765",
            )
            resumed = runner.resume(run_id=first.run_id, prompt="second")
            for result in (first, resumed):
                assert result.succeeded, result.error
                request = runner.store.load_request(result.run_id)
                assert request.provider_base_url == "http://127.0.0.1:8765"

    def test_ambient_claude_base_url_is_never_dropped_silently(
        self,
        repository: Path, fake_claude: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8765")
        runner = AgentRunner(
            WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
        )

        with raises(AOPError, match="pass --provider-base-url"):
            runner.run(task="ambient", prompt="first")
        with raises(AOPError, match="http"):
            runner.run(task="bad", prompt="first", provider_base_url="127.0.0.1:8765")

    def test_provider_base_url_is_claude_only(self, repository: Path, fake_codex: Path) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
        )

        with raises(AOPError, match="does not support --provider-base-url"):
            runner.run(task="codex", prompt="first", provider_base_url="http://127.0.0.1:1")

    def test_sealed_claude_run_reaches_a_loopback_recorder(
        self,
        repository: Path, fake_claude: Path
    ) -> None:
        import http.server
        import threading

        received: list[bytes] = []

        class Recorder(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                received.append(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(400)
                self.end_headers()

            def log_message(self, *args: object) -> None:
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Recorder)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}"
        script = fake_claude.read_text()
        fake_claude.write_text(
            script.replace(
                "prompt = sys.stdin.read()\n",
                "prompt = sys.stdin.read()\n"
                "import urllib.request, urllib.error\n"
                "try:\n"
                "    urllib.request.urlopen(urllib.request.Request(\n"
                "        os.environ['ANTHROPIC_BASE_URL'] + '/v1/messages',\n"
                "        data=prompt.encode(), method='POST'))\n"
                "except urllib.error.HTTPError:\n"
                "    pass\n",
                1,
            )
        )
        try:
            result = AgentRunner(
                WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
            ).run(task="recorder", prompt="hello", profile="sealed", provider_base_url=url)
        finally:
            server.shutdown()
            server.server_close()

        assert result.succeeded, result.error
        assert received == [b"hello"]

    @parametrize("provider", ["devin", "dsh"])
    def test_no_web_harnesses_relying_on_the_network_sandbox_reject_host(
        self,
        repository: Path, fake_devin: Path, fake_dsh: Path, provider: str
    ) -> None:
        adapter = (
            DevinAdapter(os.fspath(fake_devin))
            if provider == "devin"
            else DeepSeekHarnessAdapter(os.fspath(fake_dsh))
        )
        manager = WorktreeManager.discover(repository)

        with raises(AOPError, match="needs the network sandbox"):
            AgentRunner(manager, adapter).run(
                task=f"host-{provider}", prompt="test", no_web=True, profile="host"
            )
        assert manager.list() == []

    def test_no_web_dsh_disables_web_and_shell_plugins(
        self,
        repository: Path, fake_dsh: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), DeepSeekHarnessAdapter(os.fspath(fake_dsh))
        )

        result = runner.run(task="dsh-offline", prompt="test", no_web=True)

        assert result.succeeded, result.error
        patch = Path(result.command[result.command.index("--patch") + 1])
        request = runner.store.load_request(result.run_id)
        scratch = Path(request.effective_policy["controller"]["scratch"])
        rows = (scratch / patch.name).read_text()
        for plugin in (
            "tool-web",
            "web-fetch-http",
            "web-search-deepseek",
            "web",
            "tool-bash",
            "tool-pwsh",
            "tool-jobs",
            "tool-workflow",
        ):
            assert f"- id: {plugin}\n  disabled: true" in rows
        assert "api.deepseek.com:443" in request.effective_policy["network"][
            "allowed_endpoints"
        ]

    def test_no_web_devin_denies_shell_and_web_search(self, tmp_path: Path) -> None:
        config = tmp_path / "config.json"
        config.write_text('{"version": 1, "permissions": {"deny": ["Exec(rm)"]}}')

        runner_module._deny_devin_web_tools(config)

        assert json.loads(config.read_text())["permissions"]["deny"] == [
            "Exec(rm)",
            "exec",
            "web_search",
        ]

    def test_claude_runs_in_place_with_native_state_outside_git(
        self,
        tmp_path: Path,
        fake_claude: Path,
        monkeypatch: MonkeyPatch,
        capsys: Capture,
    ) -> None:
        workdir = tmp_path / "debate"
        workdir.mkdir()
        records = tmp_path / "profile"
        records.mkdir()
        monkeypatch.chdir(records)
        monkeypatch.setenv("AOP_CLAUDE_BIN", os.fspath(fake_claude))
        base = ["--agent", "claude", "--profile", "host", "--workdir", os.fspath(workdir)]

        assert main(["run", *base, "--native-state", "--prompt", "WHERE", "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["succeeded"], result["error"]
        where = json.loads(result["final_message"])
        assert where == {"cwd": os.fspath(workdir), "home": os.environ["HOME"], "config_dir": None}
        request = json.loads(
            (records / ".aop" / "runs" / result["run_id"] / "request.json").read_text()
        )
        assert request["workdir"] == os.fspath(workdir)
        assert request["native_state"] is True
        assert request["effective_policy"]["workspace"]["content"] == "caller-directory"
        assert request["effective_policy"]["state"] == "harness-native"
        assert not (workdir / ".aop").exists()

        assert main(["resume", result["run_id"], "--prompt", "WHERE", "--json"]) == 0
        resumed = json.loads(capsys.readouterr().out)
        assert resumed["session_id"] == result["session_id"]
        assert json.loads(resumed["final_message"])["cwd"] == os.fspath(workdir)

        # Without --native-state Claude uses AOP's private config, while the
        # host run keeps the user's HOME for other tools.
        assert main(["run", *base, "--prompt", "WHERE", "--json"]) == 0
        private = json.loads(json.loads(capsys.readouterr().out)["final_message"])
        assert private["cwd"] == os.fspath(workdir)
        assert private["home"] == os.environ["HOME"]
        assert private["config_dir"].endswith("/claude/home/.claude")

        assert main(["cleanup", result["run_id"]]) == 0
        capsys.readouterr()
        assert (workdir).is_dir()
        with raises(AOPError, match="applies only to the host profile"):
            AgentRunner(
                WorktreeManager.standalone(records), ClaudeAdapter(os.fspath(fake_claude))
            ).run(task="t", prompt="x", profile="sealed", workdir=workdir)
        assert main(["run", "--agent", "codex", "--profile", "host", "--native-state",
                     "--workdir", os.fspath(workdir), "--prompt", "x", "--dry-run"]) == 2
        assert "codex does not support --native-state" in capsys.readouterr().err

    @parametrize("profile", ["edit", "sealed"])
    def test_claude_runs_are_seeded_from_claude_config_dir(
        self,
        repository: Path,
        tmp_path: Path,
        fake_claude: Path,
        monkeypatch: MonkeyPatch,
        profile: str,
    ) -> None:
        second = tmp_path / "claude-b"
        second.mkdir()
        (second / ".credentials.json").write_text('{"account": "second"}\n')
        (second / "settings.json").write_text('{"model": "second"}\n')
        (second / ".claude.json").write_text('{"userID": "second"}\n')
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", os.fspath(second))
        manager = (
            WorktreeManager.standalone(tmp_path / "controller")
            if profile == "sealed"
            else WorktreeManager.discover(repository)
        )

        result = AgentRunner(manager, ClaudeAdapter(os.fspath(fake_claude))).run(
            task="second-account", prompt="WHERE", profile=profile
        )

        assert result.succeeded, result.error
        where = json.loads(result.final_message)
        # The run uses AOP's private copy in the standard place.
        assert where["config_dir"] is None
        private = (
            manager.sealed_runtime_dir / "provider-state" / result.run_id
            if profile == "sealed"
            else manager.state_dir / "provider-state" / "second-account"
        ) / "claude" / "home"
        assert (private / ".claude" / ".credentials.json").read_text() == '{"account": "second"}\n'
        if profile == "edit":
            assert (private / ".claude" / "settings.json").read_text() == '{"model": "second"}\n'
            assert (private / ".claude.json").read_text() == '{"userID": "second"}\n'

    def test_claude_thinking_display_is_recorded_and_kept_on_resume(
        self,
        repository: Path, fake_claude: Path, fake_codex: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
        )

        first = runner.run(task="thinking", prompt="first", thinking_display="summarized")
        resumed = runner.resume(run_id=first.run_id, prompt="second")

        for result in (first, resumed):
            assert result.succeeded, result.error
            flag = result.command.index("--thinking-display")
            assert result.command[flag + 1] == "summarized"
            assert runner.store.load_request(result.run_id).thinking_display == (
                "summarized"
            )
        with raises(AOPError, match="must be one of"):
            runner.run(task="bad", prompt="x", thinking_display="full")
        with raises(AOPError, match="does not support --thinking-display"):
            AgentRunner(
                WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
            ).run(task="codex", prompt="x", thinking_display="summarized")

    def test_claude_prompt_parts_become_content_blocks_with_one_cache_mark(
        self,
        repository: Path,
        fake_claude: Path,
        monkeypatch: MonkeyPatch,
        capsys: Capture,
    ) -> None:
        monkeypatch.chdir(repository)
        monkeypatch.setenv("AOP_CLAUDE_BIN", os.fspath(fake_claude))
        parts = repository.parent / "parts.json"
        parts.write_text(
            json.dumps(
                [
                    {"text": "view 1\n", "cache": False},
                    {"text": "view 2\n", "cache": True},
                    {"text": "question"},
                ]
            )
        )

        def check(arguments: list[str]) -> str:
            assert main([*arguments, "--prompt-parts", os.fspath(parts), "--json"]) == 0
            result = json.loads(capsys.readouterr().out)
            assert result["succeeded"], result["error"]
            flag = result["command"].index("--input-format")
            assert result["command"][flag + 1] == "stream-json"
            # The fake Claude answers with the stdin it received.
            sent = json.loads(result["final_message"].removeprefix("answer:"))
            assert sent["message"]["content"] == [
                {"type": "text", "text": "view 1\n"},
                {
                    "type": "text",
                    "text": "view 2\n",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
                {"type": "text", "text": "question"},
            ]
            request = json.loads(
                (repository / ".aop" / "runs" / result["run_id"] / "request.json").read_text()
            )
            assert request["prompt"] == "view 1\nview 2\nquestion"
            assert request["prompt_parts"] == [
                {"chars": 7, "cache": False},
                {"chars": 7, "cache": True},
                {"chars": 8, "cache": False},
            ]
            return result["run_id"]

        run_id = check(["run", "--agent", "claude", "parts"])
        check(["resume", run_id])

    def test_prompt_parts_are_claude_only_with_at_most_one_cache_mark(
        self,
        repository: Path, fake_claude: Path, fake_codex: Path
    ) -> None:
        claude = AgentRunner(
            WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
        )
        with raises(AOPError, match="one cached prompt part"):
            claude.run(
                task="marks",
                prompt="ab",
                prompt_parts=[PromptPart(1, cache=True), PromptPart(1, cache=True)],
            )
        with raises(AOPError, match="do not add up"):
            claude.run(task="sum", prompt="abc", prompt_parts=[PromptPart(2)])
        with raises(AOPError, match="none of them empty"):
            claude.run(task="empty", prompt="a", prompt_parts=[PromptPart(0), PromptPart(1)])
        with raises(AOPError, match="does not support --prompt-parts"):
            AgentRunner(
                WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
            ).run(task="codex", prompt="a", prompt_parts=[PromptPart(1)])
