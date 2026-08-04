from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aop.runner import AgyAdapter, AgentRunner, ClaudeAdapter, HermesAdapter
from aop.worktrees import AOPError, WorktreeManager


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
    assert first.final_message == "answer:first"
    assert first.usage.input_tokens == 100
    assert first.usage.cached_input_tokens == 30
    assert first.usage.output_tokens == 20
    assert first.usage.reasoning_output_tokens == 7
    assert first.provider_duration_seconds == 1.25
    assert first.time_to_first_response_seconds is not None
    assert first.api_equivalent_cost is None
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
        sandbox="scratch-write",
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


def test_scratch_write_only_mounts_the_task_scratch_directory(
    repository: Path, fake_claude: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, ClaudeAdapter(os.fspath(fake_claude)))

    result = runner.run(
        task="proposal",
        prompt="CHECK_SCRATCH",
        sandbox="scratch-write",
        timeout_seconds=5,
    )
    worktree = manager.get("proposal")

    assert result.succeeded
    assert (worktree.path / "scratch" / "analysis.txt").read_text() == "allowed"
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
    assert ["--provider", "nous"] == first.command[
        first.command.index("--provider") : first.command.index("--provider") + 2
    ]
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

    assert resumed.succeeded
    assert resumed.session_id == first.session_id
    assert resumed.final_message == "answer:second"
    assert ["--resume", first.session_id] == resumed.command[
        resumed.command.index("--resume") : resumed.command.index("--resume") + 2
    ]
    assert "--no-restore-cwd" in resumed.command
    assert ["--provider", "nous"] == resumed.command[
        resumed.command.index("--provider") : resumed.command.index("--provider") + 2
    ]
    assert ["--model", "deepseek/deepseek-v4-flash-0731"] == resumed.command[
        resumed.command.index("--model") : resumed.command.index("--model") + 2
    ]
    assert ["--reasoning", "high"] == resumed.command[
        resumed.command.index("--reasoning") : resumed.command.index("--reasoning") + 2
    ]
    assert resumed.usage == first.usage
    assert resumed.api_equivalent_cost is not None
    assert resumed.api_equivalent_cost.amount_usd == 0.000001


def test_hermes_supports_workspace_sandbox_and_artifacts(
    repository: Path, fake_hermes: Path
) -> None:
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, HermesAdapter(os.fspath(fake_hermes)))

    result = runner.run(
        task="hermes-artifact",
        prompt="CHECK_HERMES_SANDBOX",
        model="deepseek/deepseek-v4-flash-0731",
        sandbox="workspace-write",
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


@pytest.mark.parametrize("sandbox", ["scratch-write", "workspace-write"])
def test_hermes_run_and_resume_with_read_only_runtime_home(
    repository: Path,
    fake_hermes: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sandbox: str,
) -> None:
    hermes_home = tmp_path / f"managed-hermes-{sandbox}"
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
            task=f"managed-{sandbox}",
            prompt="first",
            sandbox=sandbox,
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
        manager.state_dir / "provider-state" / f"managed-{sandbox}" / "hermes" / "home"
    )
    assert (isolated_home / "fake-state.json").is_file()
    assert (isolated_home / "auth.json").read_text() == "{}\n"
    assert not (
        manager.get(f"managed-{sandbox}").path / "scratch" / "provider-state"
    ).exists()
    assert "--setenv" in first.command
    assert "--overlay" not in first.command
    assert os.fspath(hermes_home) in first.command
    assert os.fspath(isolated_home) in first.command
    assert "--setenv" in resumed.command
