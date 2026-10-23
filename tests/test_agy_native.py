"""Opt-in contract against installed Antigravity, without a model request.

Antigravity announces its tools and working directory as soon as it starts and
exits when it receives no prompt, so these runs exercise AOP's whole setup,
state seeding, sandbox, and launch without spending quota.
"""

from __future__ import annotations

import json
import os

from support import AOPTestCase, MonkeyPatch, parametrize, skip

from agent_orchestration_process import runner as runner_module
from agent_orchestration_process.runner import AgentRunner, AgyAdapter
from agent_orchestration_process.worktrees import WorktreeManager


class AgyNativeTest(AOPTestCase):
    @parametrize("profile", ["sealed", "edit", "host"])
    def test_installed_agy_starts_inside_each_profile(
        self, repository, tmp_path, monkeypatch: MonkeyPatch, profile
    ):
        binary = os.environ.get("AOP_TEST_AGY_BIN")
        if not binary:
            skip("set AOP_TEST_AGY_BIN to test installed Antigravity")
        # The real home holds the Google login and settings AOP seeds from.
        monkeypatch.setenv("HOME", os.path.expanduser("~"))
        monkeypatch.setattr(
            runner_module.prompt_transport, "stdin_prompt", lambda provider, prompt: ""
        )
        manager = (
            WorktreeManager.standalone(tmp_path / "controller")
            if profile == "sealed"
            else WorktreeManager.discover(repository)
        )

        result = AgentRunner(manager, AgyAdapter(binary)).run(
            task="native-agy", prompt="unused", profile=profile, timeout_seconds=60
        )

        events = (
            manager.state_dir / "runs" / result.run_id / "events.jsonl"
        ).read_text()
        starts = [
            event["init"]
            for event in map(json.loads, events.splitlines())
            if event.get("event") == "init"
        ]
        assert starts, f"Antigravity did not start: {result.error}"
        assert starts[0]["cwd"] == (
            "/workspace" if profile != "host" else os.fspath(manager.get("native-agy").path)
        )
        assert {"view_file", "write_to_file", "run_command"} <= set(starts[0]["tools"])
        assert not result.timed_out
        # No model request was made, so the test costs no quota.
        assert '"usage"' not in events
