"""Opt-in contract against installed Muse Code, without a model request.

Muse's echo provider answers each prompt with "echo: PROMPT" and makes no
network request, so these runs exercise AOP's state seeding, sandbox, launch,
event parsing, and resume with the real binary and without a Meta login.
"""

from __future__ import annotations

import json
import os

from support import AOPTestCase, parametrize, skip

from agent_orchestration_process.runner import AgentRunner, MuseAdapter
from agent_orchestration_process.worktrees import WorktreeManager


class MuseNativeTest(AOPTestCase):
    @parametrize("profile", ["sealed", "edit", "host"])
    def test_installed_muse_answers_and_resumes_inside_each_profile(
        self, repository, tmp_path, monkeypatch, profile
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
            task="native-muse", prompt="say copper", profile=profile, timeout_seconds=60
        )
        resumed = runner.resume(run_id=first.run_id, prompt="say silver")

        assert first.succeeded, first.error
        assert first.final_message == "echo: say copper"
        assert first.session_id
        assert first.accounting_status == "complete"
        assert resumed.succeeded, resumed.error
        assert resumed.session_id == first.session_id
        assert resumed.final_message == "echo: say silver"
