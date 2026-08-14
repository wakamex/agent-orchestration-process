from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent_orchestration_process.cli import main
from agent_orchestration_process.isolation import PROFILES, explain_profile
from agent_orchestration_process.runner import AgentRunner, CodexAdapter, adapter_for
from agent_orchestration_process.worktrees import WorktreeManager


def test_profiles_have_explicit_semantic_contracts() -> None:
    policies = {name: explain_profile(name) for name in PROFILES}

    assert policies["edit"]["workspace"]["access"] == "write"
    assert policies["review"]["workspace"]["access"] == "read"
    assert policies["sealed"]["workspace"]["access"] == "none"
    assert policies["sealed"]["host"]["access"] == "runtime-only"
    assert policies["sealed"]["instructions"]["inherited_local"] == "none"
    assert policies["sealed"]["instructions"]["provider_builtin"] == "present"
    assert policies["sealed"]["network"]["mode"] == "native"
    assert policies["sealed"]["network"]["isolation"] == "none"
    assert policies["sealed"]["writable_path_scopes"]["/cache"] == ("session-private")
    assert policies["sealed"]["identity"] == "opaque"
    assert policies["host"]["host"]["access"] == "native"
    assert policies["host"]["environment"]["mode"] == "native"


def test_legacy_sandbox_interface_is_removed() -> None:
    from agent_orchestration_process.cli import build_parser

    help_text = build_parser().format_help()
    run_help = build_parser()._subparsers._group_actions[0].choices["run"].format_help()
    assert "--sandbox" not in run_help
    assert "--read" not in run_help
    assert "scratch-write" not in help_text
    assert "workspace-write" not in help_text


def test_sealed_run_needs_no_git_and_exposes_only_snapshotted_input(
    tmp_path: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    neutral = tmp_path / "neutral-controller"
    neutral.mkdir()
    source = tmp_path / "declared.txt"
    source.write_text("original bytes\n")
    monkeypatch.setenv("STUDY_ARM", "study-condition-secret")
    monkeypatch.setenv("OLDPWD", "/code/secret-study")
    manager = WorktreeManager.standalone(neutral)
    assert manager.state_dir.stat().st_mode & 0o777 == 0o700

    result = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).run(
        task="human-study-label",
        prompt="CHECK_SEALED",
        profile="sealed",
        input_paths=[source],
    )

    assert result.succeeded
    assert "human-study-label" not in result.command
    assert os.fspath(source) not in result.command
    assert "study-condition-secret" not in json.dumps(result.to_dict())
    assert "OLDPWD" not in result.command
    assert ["--dev-bind", "/", "/"] not in [
        result.command[index : index + 3] for index in range(len(result.command) - 2)
    ]

    run_dir = manager.state_dir / "runs" / result.run_id
    request = json.loads((run_dir / "request.json").read_text())
    assert request["profile"] == "sealed"
    assert request["effective_policy"]["workspace"]["access"] == "none"
    assert request["effective_policy"]["host"]["access"] == "runtime-only"
    assert request["effective_policy"]["instructions"]["inherited_local"] == "none"
    assert request["effective_policy"]["instruction_sources"] == []
    assert os.fspath(source) not in request["prompt"]
    controller_paths = json.dumps(request["effective_policy"]["controller"])
    assert os.fspath(manager.state_dir) not in controller_paths
    assert os.fspath(manager.sealed_runtime_dir) in controller_paths

    snapshot = (
        Path(request["effective_policy"]["controller"]["input_snapshot"])
        / "declared.txt"
    )
    assert snapshot.read_text() == "original bytes\n"
    assert snapshot.stat().st_mode & 0o222 == 0
    source.write_text("changed later\n")
    assert snapshot.read_text() == "original bytes\n"
    source.unlink()

    codex_home = (
        Path(request["effective_policy"]["controller"]["provider_state"])
        / "codex"
        / "home"
    )
    assert (codex_home / "auth.json").is_file()
    assert not (codex_home / "config.toml").exists()
    assert not (codex_home / "rules").exists()
    assert not (codex_home / "skills").exists()

    resumed = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).resume(
        run_id=result.run_id,
        prompt="sealed resume",
    )
    assert resumed.succeeded
    assert resumed.session_id == result.session_id
    resumed_request = json.loads(
        (manager.state_dir / "runs" / resumed.run_id / "request.json").read_text()
    )
    assert (
        resumed_request["effective_policy"]["controller"]["provider_state"]
        == request["effective_policy"]["controller"]["provider_state"]
    )
    assert (
        resumed_request["effective_policy"]["controller"]["cache"]
        == request["effective_policy"]["controller"]["cache"]
    )
    assert resumed_request["inputs"][0]["sha256"] == request["inputs"][0]["sha256"]
    assert (
        resumed_request["inputs"][0]["source_path"]
        == request["inputs"][0]["source_path"]
    )
    assert "human-study-label" not in resumed.command


def test_cli_sealed_run_and_profile_explain_work_outside_git(
    tmp_path: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    neutral = tmp_path / "not-a-repository"
    neutral.mkdir()
    monkeypatch.chdir(neutral)
    monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))

    assert main(["profile", "explain", "sealed", "--json"]) == 0
    explanation = json.loads(capsys.readouterr().out)
    assert explanation["workspace"]["access"] == "none"

    assert main(["run", "--profile", "sealed", "--prompt", "hello", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["succeeded"] is True
    request = json.loads(
        (neutral / ".aop" / "runs" / result["run_id"] / "request.json").read_text()
    )
    assert request["profile"] == "sealed"

    assert main(["cleanup", result["run_id"]]) == 0
    capsys.readouterr()
    assert not Path(
        request["effective_policy"]["workspace"]["controller_path"]
    ).exists()
    assert not Path(
        request["effective_policy"]["controller"]["provider_state"]
    ).exists()
    assert not Path(request["effective_policy"]["controller"]["cache"]).exists()


def test_sealed_writable_state_is_private_between_tasks_and_reused_on_resume(
    tmp_path: Path, fake_codex: Path
) -> None:
    manager = WorktreeManager.standalone(tmp_path / "controller")
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    first = runner.run(
        task="first-concealed-arm",
        prompt="CHECK_SEALED_WRITE_MARKERS",
        profile="sealed",
    )
    second = runner.run(
        task="second-concealed-arm",
        prompt="CHECK_SEALED_NO_MARKERS",
        profile="sealed",
    )
    resumed = runner.resume(
        run_id=first.run_id,
        prompt="CHECK_SEALED_RESUME_MARKER",
    )

    assert first.succeeded, first.error
    assert second.succeeded, second.error
    assert resumed.succeeded, resumed.error
    first_request = json.loads(
        (manager.state_dir / "runs" / first.run_id / "request.json").read_text()
    )
    second_request = json.loads(
        (manager.state_dir / "runs" / second.run_id / "request.json").read_text()
    )
    resumed_request = json.loads(
        (manager.state_dir / "runs" / resumed.run_id / "request.json").read_text()
    )
    first_controller = first_request["effective_policy"]["controller"]
    second_controller = second_request["effective_policy"]["controller"]
    resumed_controller = resumed_request["effective_policy"]["controller"]
    for name in ("cache", "provider_state", "scratch"):
        assert first_controller[name] != second_controller[name]
        assert first_controller[name] == resumed_controller[name]


@pytest.mark.parametrize("profile", ["edit", "review"])
def test_repository_profiles_enforce_observed_access_and_hide_controller_state(
    profile: str,
    repository: Path,
    fake_codex: Path,
    tmp_path: Path,
) -> None:
    unrelated = tmp_path / "unrelated-control"
    unrelated.mkdir()
    unrelated.joinpath("secret.txt").write_text("not declared\n")
    result = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    ).run(
        task=f"access-{profile}",
        prompt=f"CHECK_PROFILE_ACCESS {profile} {unrelated}",
        profile=profile,
    )

    assert result.succeeded, result.error
    assert ["--dev-bind", "/", "/"] not in [
        result.command[index : index + 3] for index in range(len(result.command) - 2)
    ]
    persisted = json.loads(
        (repository / ".aop" / "runs" / result.run_id / "request.json").read_text()
    )
    assert persisted["effective_policy"]["instruction_sources"]
    assert all(
        len(source["sha256"]) == 64
        for source in persisted["effective_policy"]["instruction_sources"]
    )


def test_dry_run_does_not_create_a_task_worktree(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)

    assert (
        main(
            [
                "run",
                "preview",
                "--profile",
                "review",
                "--prompt",
                "inspect",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    policy = json.loads(capsys.readouterr().out)
    assert policy["profile"] == "review"
    assert WorktreeManager.discover(repository).list() == []


@pytest.mark.parametrize(
    ("provider", "fixture"),
    [
        ("codex", "fake_codex"),
        ("claude", "fake_claude"),
        ("cursor", "fake_cursor"),
        ("devin", "fake_devin"),
        ("opencode", "fake_opencode"),
        ("agy", "fake_agy"),
        ("grok", "fake_grok"),
        ("hermes", "fake_hermes"),
        ("dsh", "fake_dsh"),
    ],
)
def test_every_provider_runs_under_the_sealed_semantic_boundary(
    provider: str,
    fixture: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary = request.getfixturevalue(fixture)
    monkeypatch.setenv(f"AOP_{provider.upper()}_BIN", os.fspath(binary))
    manager = WorktreeManager.standalone(tmp_path / "controller")
    manager.cache_dir.mkdir(parents=True, exist_ok=True)
    manager.cache_dir.joinpath("shared-cache-canary").write_text(
        "must not enter sealed"
    )

    result = AgentRunner(manager, adapter_for(provider)).run(
        task="concealed-study-arm",
        prompt="SEALED_PROVIDER_PROBE",
        profile="sealed",
    )

    assert result.succeeded, result.error
    assert "concealed-study-arm" not in result.command
    persisted = json.loads(
        (manager.state_dir / "runs" / result.run_id / "request.json").read_text()
    )
    assert persisted["effective_policy"]["workspace"]["access"] == "none"
    assert persisted["effective_policy"]["instructions"]["inherited_local"] == ("none")
    state = Path(persisted["effective_policy"]["controller"]["provider_state"])
    controller = persisted["effective_policy"]["controller"]
    workspace = persisted["effective_policy"]["workspace"]["controller_path"]
    assert ["--ro-bind", workspace, "/workspace"] == result.command[
        result.command.index(workspace) - 1 : result.command.index(workspace) + 2
    ]
    assert ["--bind", controller["cache"], "/cache"] == result.command[
        result.command.index(controller["cache"]) - 1 : result.command.index(
            controller["cache"]
        )
        + 2
    ]
    assert os.fspath(manager.cache_dir) not in result.command
    forbidden_names = {"AGENTS.md", "SKILL.md", "default.rules"}
    assert not any(path.name in forbidden_names for path in state.rglob("*"))
    forbidden_directories = {"hooks", "memories", "plugins", "skills", "skills-cursor"}
    assert not any(
        path.is_dir() and path.name in forbidden_directories
        for path in state.rglob("*")
    )
    for config in state.rglob("config.json"):
        assert "mcp" not in config.read_text().lower()
    if provider == "agy":
        agy = state / "agy" / "gemini"
        files = {
            path.relative_to(agy).as_posix()
            for path in agy.rglob("*")
            if path.is_file()
        }
        assert files == {
            "antigravity-cli/antigravity-oauth-token",
            "antigravity-cli/fake-conversation.json",
            "google_accounts.json",
            "oauth_creds.json",
        }
