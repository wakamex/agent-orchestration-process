"""Opt-in contract against installed Grok: what a --no-web run offers the model."""

from __future__ import annotations

import json
import os

from support import AOPTestCase, skip

from agent_orchestration_process.runner import AgentRunner, GrokAdapter
from agent_orchestration_process.worktrees import WorktreeManager

NO_WEB_TOOLS = {"read_file", "search_replace", "grep", "list_dir", "todo_write"}


class GrokNativeTest(AOPTestCase):
    def test_no_web_grok_offers_only_local_file_tools(self, tmp_path):
        binary = os.environ.get("AOP_TEST_GROK_BIN")
        if not binary:
            skip("set AOP_TEST_GROK_BIN to test installed Grok")
        manager = WorktreeManager.standalone(tmp_path / "controller")

        # Grok announces its tools before its first model request, so this
        # holds whether or not the account has quota left.
        result = AgentRunner(manager, GrokAdapter(binary)).run(
            task="grok-no-web",
            prompt="Reply with only the word copper.",
            profile="sealed",
            no_web=True,
            timeout_seconds=120,
        )

        events = (manager.state_dir / "runs" / result.run_id / "events.jsonl").read_text()
        announced = [
            event["tools"]
            for event in map(json.loads, events.splitlines())
            if event.get("type") == "available_commands"
        ]
        assert announced, events[:500]
        assert set(announced[0]) <= NO_WEB_TOOLS, sorted(set(announced[0]) - NO_WEB_TOOLS)
