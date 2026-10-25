from __future__ import annotations

import os
from pathlib import Path

from support import AOPTestCase, raises

from agent_orchestration_process.runner import AgentRunner, MuseAdapter
from agent_orchestration_process.worktrees import AOPError, WorktreeManager


class MuseAdapterTest(AOPTestCase):
    def test_muse_runs_with_its_login_and_resumes_the_same_session(
        self, repository: Path, tmp_path: Path, fake_muse: Path
    ) -> None:
        binary = fake_muse
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, MuseAdapter(os.fspath(binary)))

        first = runner.run(task="muse", prompt="first", effort="low")
        resumed = runner.resume(run_id=first.run_id, prompt="second")

        assert first.succeeded, first.error
        assert first.final_message == "answer:first"
        assert first.model == "muse-wonder-1.0"
        assert first.usage.input_tokens == 120
        assert first.usage.cached_input_tokens == 40
        assert first.usage.output_tokens == 30
        assert first.usage.reasoning_output_tokens == 10
        assert first.billing.route == "subscription"
        assert first.time_to_first_response_seconds is not None
        assert ["--reasoning-effort", "low"] == first.command[
            first.command.index("--reasoning-effort") :
        ][:2]
        assert resumed.succeeded, resumed.error
        assert resumed.session_id == first.session_id
        assert resumed.final_message == "answer:second"
        assert ["--session-id", first.session_id] == resumed.command[
            resumed.command.index("--session-id") :
        ][:2]

    def test_sealed_muse_keeps_only_login_and_privacy_settings(
        self, repository: Path, tmp_path: Path, fake_muse: Path
    ) -> None:
        binary = fake_muse
        runner = AgentRunner(
            WorktreeManager.standalone(tmp_path / "controller"),
            MuseAdapter(os.fspath(binary)),
        )
        edit = AgentRunner(
            WorktreeManager.discover(repository), MuseAdapter(os.fspath(binary))
        )

        sealed = runner.run(task="sealed", prompt="LIST_CONFIG", profile="sealed")
        full = edit.run(task="full", prompt="LIST_CONFIG")

        assert sealed.succeeded, sealed.error
        assert sealed.final_message == (
            "auth.json settings.json contributor_mode_enabled schema_version telemetry"
        )
        assert "--no-foreign-personal-context" in sealed.command
        assert full.succeeded, full.error
        assert full.final_message.startswith("AGENTS.md auth.json settings.json")
        assert "hooks" in full.final_message
        copied = (
            WorktreeManager.discover(repository).state_dir
            / "provider-state" / "full" / "muse" / "config" / "muse" / "AGENTS.md"
        )
        assert not copied.is_symlink() and copied.read_text() == "rules\n"

    def test_muse_failed_run_and_missing_login_and_no_web(
        self, repository: Path, tmp_path: Path, fake_muse: Path
    ) -> None:
        binary = fake_muse
        config = Path(os.environ["AOP_MUSE_CONFIG_DIR"])
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, MuseAdapter(os.fspath(binary)))

        failed = runner.run(task="muse-fail", prompt="MUSE_FAIL")
        with raises(AOPError, match="cannot enforce --no-web"):
            runner.run(task="muse-web", prompt="x", no_web=True)
        (config / "auth.json").unlink()
        with raises(AOPError, match="run `muse login` first"):
            runner.run(task="muse-login", prompt="x")

        assert not failed.succeeded
        assert failed.error == "Muse run ended failed: model call failed"
        assert failed.session_id
