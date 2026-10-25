"""Opt-in contract against installed Muse Code, without a model request.

Muse's echo provider answers each prompt with "echo: PROMPT" and makes no
network request, so these runs exercise AOP's state seeding, sandbox, launch,
event parsing, and resume with the real binary and without a Meta login.
"""

from __future__ import annotations

import json
import os
import re

from support import AOPTestCase, parametrize, skip

from agent_orchestration_process.runner import AgentRunner, MuseAdapter
from agent_orchestration_process.worktrees import WorktreeManager


class MuseNativeTest(AOPTestCase):
    @parametrize(
        "profile,no_web",
        [("sealed", False), ("edit", False), ("host", False), ("edit", True)],
    )
    def test_installed_muse_answers_and_resumes_inside_each_profile(
        self, repository, tmp_path, monkeypatch, profile, no_web
    ):
        binary = os.environ.get("AOP_TEST_MUSE_BIN")
        if not binary:
            skip("set AOP_TEST_MUSE_BIN to test installed Muse Code")
        config = tmp_path / "muse-config"
        config.mkdir()
        (config / "auth.json").write_text('{"schema_version": 1, "providers": {}}\n')
        (config / "settings.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "provider": "echo",
                    "contributor_mode_enabled": False,
                    "telemetry": {"enabled": False},
                }
            )
        )
        monkeypatch.setenv("AOP_MUSE_CONFIG_DIR", os.fspath(config))
        monkeypatch.setenv("AOP_MUSE_DATA_DIR", os.fspath(tmp_path / "muse-data"))
        manager = (
            WorktreeManager.standalone(tmp_path / "controller")
            if profile == "sealed"
            else WorktreeManager.discover(repository)
        )
        runner = AgentRunner(manager, MuseAdapter(binary))

        first = runner.run(
            task="native-muse",
            prompt="say copper",
            profile=profile,
            no_web=no_web,
            timeout_seconds=60,
        )
        resumed = runner.resume(run_id=first.run_id, prompt="say silver")

        assert first.succeeded, first.error
        assert first.final_message == "echo: say copper"
        assert first.session_id
        assert first.accounting_status == "complete"
        assert resumed.succeeded, resumed.error
        assert resumed.session_id == first.session_id
        assert resumed.final_message == "echo: say silver"

    @parametrize("reminders,expected", [("goal", {"goal"}), ("none", set())])
    def test_installed_muse_runs_only_the_selected_reminders(
        self, repository, tmp_path, monkeypatch, reminders, expected
    ):
        binary = os.environ.get("AOP_TEST_MUSE_BIN")
        if not binary:
            skip("set AOP_TEST_MUSE_BIN to test installed Muse Code")
        config = tmp_path / "muse-config"
        config.mkdir()
        (config / "auth.json").write_text('{"schema_version": 1, "providers": {}}\n')
        (config / "settings.json").write_text(
            '{"schema_version": 1, "provider": "echo", '
            '"contributor_mode_enabled": false, "telemetry": {"enabled": false}}\n'
        )
        monkeypatch.setenv("AOP_MUSE_CONFIG_DIR", os.fspath(config))
        monkeypatch.setenv("AOP_MUSE_DATA_DIR", os.fspath(tmp_path / "muse-data"))
        manager = WorktreeManager.discover(repository)

        result = AgentRunner(manager, MuseAdapter(binary)).run(
            task="reminders", prompt="hi", reminders=reminders, timeout_seconds=60
        )

        events = (manager.state_dir / "runs" / result.run_id / "events.jsonl").read_text()
        started = set(re.findall(r'"task_kind":"reminder\.agent\.([a-z]+)-reminder"', events))
        assert result.succeeded, result.error
        assert started == expected
