from __future__ import annotations

import json
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
        # The main call's 120 input tokens plus a reminder call's 3,500.
        assert first.accounting_status == "complete"
        assert first.usage.input_tokens == 3620
        assert first.usage.cached_input_tokens == 40
        assert first.usage.output_tokens == 130
        assert first.usage.reasoning_output_tokens == 30
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

    def test_muse_usage_is_partial_when_its_session_log_does_not_reconcile(
        self, repository: Path, fake_muse: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), MuseAdapter(os.fspath(fake_muse))
        )

        mismatch = runner.run(task="mismatch", prompt="MUSE_LEDGER_MISMATCH")
        missing = runner.run(task="missing", prompt="MUSE_NO_LEDGER")

        for result in (mismatch, missing):
            assert result.succeeded, result.error
            assert result.accounting_status == "partial"
            assert result.usage.input_tokens == 120
            assert result.calculated_cost is None

    def test_muse_usage_covers_every_turn_of_one_invocation(
        self, repository: Path, fake_muse: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), MuseAdapter(os.fspath(fake_muse))
        )

        result = runner.run(task="turns", prompt="MUSE_TWO_TURNS")

        assert result.succeeded, result.error
        assert result.final_message == "answer:MUSE_TWO_TURNS"
        assert result.accounting_status == "complete"
        # 1,000 from the failed first turn, 120 and a 3,500 reminder from the last.
        assert result.usage.input_tokens == 4620
        assert result.usage.output_tokens == 135

    def test_muse_wonder_full_is_priced_at_its_estimate(
        self, repository: Path, fake_muse: Path
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), MuseAdapter(os.fspath(fake_muse))
        )

        result = runner.run(task="full", prompt="x", model="muse-wonder-full-1.0")

        cost = result.calculated_cost
        assert cost is not None
        assert cost.estimated_from == "meta/muse-wonder-1.0"
        assert "meta/muse-spark-1.3" in cost.priced_as

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

    def test_muse_failed_run_no_web_and_missing_login(
        self, repository: Path, tmp_path: Path, fake_muse: Path
    ) -> None:
        binary = fake_muse
        config = Path(os.environ["AOP_MUSE_CONFIG_DIR"])
        manager = WorktreeManager.discover(repository)
        runner = AgentRunner(manager, MuseAdapter(os.fspath(binary)))

        failed = runner.run(task="muse-fail", prompt="MUSE_FAIL")
        no_web = runner.run(task="muse-web", prompt="LIST_CONFIG", no_web=True)
        with raises(AOPError, match="needs the network sandbox"):
            runner.run(task="muse-host", prompt="x", profile="host", no_web=True)
        (config / "auth.json").unlink()
        with raises(AOPError, match="run `muse login` first"):
            runner.run(task="muse-login", prompt="x")

        assert not failed.succeeded
        assert failed.error == "Muse run ended failed: model call failed"
        assert failed.session_id
        assert no_web.succeeded, no_web.error
        assert {"--disable-web-tools", "--disable-shell"} <= set(no_web.command)
        assert no_web.final_message == (
            "auth.json settings.json contributor_mode_enabled schema_version telemetry"
        )


class MuseReminderTest(AOPTestCase):
    def test_reminders_select_per_run_and_by_profile_default(
        self, repository: Path, tmp_path: Path, fake_muse: Path, monkeypatch
    ) -> None:
        config = Path(os.environ["AOP_MUSE_CONFIG_DIR"])
        settings = json.loads((config / "settings.json").read_text())
        settings["runtime_capabilities"] = {
            "plugin:other:tool:x": {"enabled": True},
            "plugin:tbh-reminders:reminder:goal-reminder": {"enabled": False},
        }
        (config / "settings.json").write_text(json.dumps(settings))
        monkeypatch.setenv("AOP_MUSE_REMINDERS", "goal,verify sealed=none")
        edit = AgentRunner(
            WorktreeManager.discover(repository), MuseAdapter(os.fspath(fake_muse))
        )
        sealed = AgentRunner(
            WorktreeManager.standalone(tmp_path / "controller"),
            MuseAdapter(os.fspath(fake_muse)),
        )

        default = edit.run(task="default", prompt="x")
        resumed = edit.resume(run_id=default.run_id, prompt="y")
        off = sealed.run(task="sealed", prompt="LIST_CONFIG", profile="sealed")
        explicit = edit.run(task="explicit", prompt="x", reminders="all")

        def setting(result):
            value = result.command[result.command.index("-c") + 1]
            name, _, encoded = value.partition("=")
            assert name == "runtime_capabilities"
            return json.loads(encoded)

        assert setting(default) == {
            "plugin:other:tool:x": {"enabled": True},
            "plugin:goal-reminder:reminder:goal-reminder": {"enabled": True},
            "plugin:verify-reminder:reminder:verify-reminder": {"enabled": True},
            "plugin:skill-reminder:reminder:skill-reminder": {"enabled": False},
        }
        assert setting(resumed) == setting(default)
        assert "--disable-reminders" in off.command and "-c" not in off.command
        # A sealed run keeps the reminder switches, not other capabilities.
        assert off.final_message.endswith("runtime_capabilities schema_version telemetry")
        assert all(
            value["enabled"]
            for key, value in setting(explicit).items()
            if key.startswith("plugin:") and ":reminder:" in key and "tbh" not in key
        )

    def test_reminder_selection_is_validated(
        self, repository: Path, fake_muse: Path, monkeypatch
    ) -> None:
        runner = AgentRunner(
            WorktreeManager.discover(repository), MuseAdapter(os.fspath(fake_muse))
        )
        with raises(AOPError, match="--reminders must be"):
            runner.run(task="bad", prompt="x", reminders="goal,memory")
        monkeypatch.setenv("AOP_MUSE_REMINDERS", "edits=none")
        with raises(AOPError, match="unknown profile in AOP_MUSE_REMINDERS"):
            runner.run(task="bad-env", prompt="x")
        with raises(AOPError, match="does not support --reminders"):
            AgentRunner(WorktreeManager.discover(repository)).run(
                task="codex", prompt="x", reminders="none"
            )
