from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

import pytest

from agent_orchestration_process.model_listing import AGENTS
from agent_orchestration_process.runner import AgentRunner, adapter_for
from agent_orchestration_process.worktrees import WorktreeManager


SELECTED = {
    item.strip()
    for item in os.environ.get("AOP_LIVE_READ_PATH_AGENTS", "").split(",")
    if item.strip()
}


@pytest.mark.live
@pytest.mark.parametrize("provider", AGENTS)
def test_provider_reads_declared_path(
    provider: str,
    tmp_path: Path,
) -> None:
    if "all" not in SELECTED and provider not in SELECTED:
        pytest.skip("set AOP_LIVE_READ_PATH_AGENTS to select live providers")

    nonce = uuid.uuid4().hex
    source = tmp_path / "declared-source.txt"
    content = f"AOP declared read-path nonce: {nonce}\n"
    source.write_text(content)
    source_hash = hashlib.sha256(content.encode()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    manager = WorktreeManager.discover(root)
    task = f"live-read-{provider}-{nonce[:8]}"
    model = os.environ.get(f"AOP_LIVE_{provider.upper()}_MODEL")
    effort = os.environ.get(f"AOP_LIVE_{provider.upper()}_EFFORT")
    timeout = float(os.environ.get("AOP_LIVE_READ_PATH_TIMEOUT", "300"))

    try:
        result = AgentRunner(manager, adapter_for(provider)).run(
            task=task,
            prompt=(
                "Read declared-source.txt through its preferred task-local path. "
                f"Reply with exactly AOP_READ_OK:{nonce} and no other text."
            ),
            model=model,
            effort=effort,
            sandbox="scratch-write",
            timeout_seconds=timeout,
            read_paths=[source],
        )
    finally:
        if any(item.task == task for item in manager.list()):
            manager.remove(task, force=True)

    assert result.succeeded, result.error
    assert result.final_message is not None
    assert f"AOP_READ_OK:{nonce}" in result.final_message
    assert source.read_text() == content
    assert result.read_paths[0].sha256 == source_hash
