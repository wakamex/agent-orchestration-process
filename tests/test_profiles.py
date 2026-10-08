from __future__ import annotations

import json
import os
from pathlib import Path

from support import AOPTestCase, Capture, MonkeyPatch, Request, parametrize, raises

from agent_orchestration_process.cli import main
from agent_orchestration_process.isolation import PROFILES, explain_profile
from agent_orchestration_process.runner import (
    AgentRunner,
    ClaudeAdapter,
    CodexAdapter,
    adapter_for,
)
from agent_orchestration_process.worktrees import AOPError, WorktreeManager


class ProfilesTest(AOPTestCase):
    def test_profiles_have_explicit_semantic_contracts(self) -> None:
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

    def test_legacy_sandbox_interface_is_removed(self) -> None:
        from agent_orchestration_process.cli import build_parser

        help_text = build_parser().format_help()
        run_help = build_parser()._subparsers._group_actions[0].choices["run"].format_help()
        assert "--sandbox" not in run_help
        assert "--read" not in run_help
        assert "scratch-write" not in help_text
        assert "workspace-write" not in help_text

    def test_sealed_run_needs_no_git_and_exposes_only_snapshotted_input(
        self,
        tmp_path: Path,
        fake_codex: Path,
        monkeypatch: MonkeyPatch,
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
        assert request["effective_policy"]["controller"]["input_snapshot"] == os.fspath(
            manager.state_dir / "snapshots" / result.run_id
        )
        assert os.fspath(manager.sealed_runtime_dir) in controller_paths

        snapshot = (
            Path(request["effective_policy"]["controller"]["input_snapshot"])
            / "declared.txt"
        )
        assert snapshot.read_text() == "original bytes\n"
        assert snapshot.stat().st_mode & 0o777 == 0o600
        assert snapshot.parent.stat().st_mode & 0o777 == 0o700
        assert not Path(
            request["effective_policy"]["controller"]["input_projection"]
        ).exists()
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

        snapshot.write_text("tampered bytes\n")
        with raises(AOPError, match="sealed input snapshot changed"):
            AgentRunner(manager, CodexAdapter(os.fspath(fake_codex))).resume(
                run_id=result.run_id,
                prompt="reject tampered snapshot",
            )

    @parametrize("disk_runtime", [False, True])
    def test_cli_sealed_run_and_profile_explain_work_outside_git(
        self,
        disk_runtime: bool,
        tmp_path: Path,
        fake_codex: Path,
        monkeypatch: MonkeyPatch,
        capsys: Capture,
    ) -> None:
        if disk_runtime:
            monkeypatch.setenv("AOP_SEALED_RUNTIME_ROOT", str(tmp_path / "disk-runtime"))
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
        if disk_runtime:
            runtime = Path(request["effective_policy"]["workspace"]["controller_path"])
            assert runtime.is_relative_to(tmp_path / "disk-runtime")
            assert main(["resume", result["run_id"], "--prompt", "followup", "--json"]) == 0
            resumed = json.loads(capsys.readouterr().out)
            assert resumed["succeeded"]
            assert resumed["session_id"] == result["session_id"]
            # Cleanup finds the recorded runtime without the original setting.
            monkeypatch.delenv("AOP_SEALED_RUNTIME_ROOT")

        assert main(["cleanup", result["run_id"]]) == 0
        assert "aop: removed " in capsys.readouterr().err
        assert not Path(
            request["effective_policy"]["workspace"]["controller_path"]
        ).exists()
        assert not Path(
            request["effective_policy"]["controller"]["provider_state"]
        ).exists()
        assert not Path(request["effective_policy"]["controller"]["cache"]).exists()
        assert Path(request["effective_policy"]["controller"]["input_snapshot"]).is_dir()
        assert not Path(
            request["effective_policy"]["controller"]["input_projection_root"]
        ).exists()

    def test_sealed_writable_state_is_private_between_tasks_and_reused_on_resume(
        self,
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

    @parametrize("profile", ["edit", "review"])
    def test_repository_profiles_enforce_observed_access_and_hide_controller_state(
        self,
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
        self,
        repository: Path,
        monkeypatch: MonkeyPatch,
        capsys: Capture,
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

    def test_cli_no_web_dry_run_reports_effective_tool_policy_without_state(
        self,
        repository: Path,
        monkeypatch: MonkeyPatch,
        capsys: Capture,
    ) -> None:
        monkeypatch.chdir(repository)

        assert (
            main(
                [
                    "run",
                    "preview",
                    "--agent",
                    "codex",
                    "--no-web",
                    "--prompt",
                    "inspect",
                    "--dry-run",
                    "--json",
                ]
            )
            == 0
        )
        policy = json.loads(capsys.readouterr().out)
        capabilities = policy["model_capabilities"]
        assert capabilities["requested"]["external_retrieval"] == "denied"
        assert capabilities["effective"]["external_retrieval"] == "denied"
        assert capabilities["effective"]["model_tools"] == "allowlist"
        assert capabilities["effective"]["allowed_toolsets"] == [
            "local shell and file tools"
        ]
        assert WorktreeManager.discover(repository).list() == []

    @parametrize(
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
        self,
        provider: str,
        fixture: str,
        request: Request,
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
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
        assert ["--ro-bind", controller["input_projection"], "/inputs"] == (
            result.command[
                result.command.index(controller["input_projection"])
                - 1 : result.command.index(controller["input_projection"]) + 2
            ]
        )
        assert controller["input_snapshot"] not in result.command
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

    def test_cleanup_removes_only_this_projects_sealed_runtime(self, tmp_path: Path) -> None:
        from agent_orchestration_process.cli import _sealed_runtime_path

        manager = WorktreeManager.standalone(tmp_path / "project")
        owner = f"aop-sealed-{os.getuid()}"
        own = tmp_path / "runtime" / owner / manager.sealed_runtime_identity
        other = tmp_path / "runtime" / owner / ("0" * 24)
        for root in (own, other):
            (root / "provider-state" / "run").mkdir(parents=True)

        assert _sealed_runtime_path(
            manager, os.fspath(own / "provider-state" / "run"), "provider-state"
        ) == (own / "provider-state" / "run").resolve()
        assert _sealed_runtime_path(
            manager, os.fspath(other / "provider-state" / "run"), "provider-state"
        ) is None
        assert _sealed_runtime_path(
            manager, os.fspath(own / "provider-state" / "run"), "scratch"
        ) is None
        assert _sealed_runtime_path(manager, os.fspath(own / "provider-state"), "provider-state") is None

    def test_sealed_run_inside_a_repository_resumes_and_cleans_up_from_its_directory(
        self,
        repository: Path,
        fake_codex: Path,
        monkeypatch: MonkeyPatch,
        capsys: Capture,
    ) -> None:
        study = repository / "research" / "study"
        study.mkdir(parents=True)
        monkeypatch.chdir(study)
        monkeypatch.setenv("AOP_CODEX_BIN", os.fspath(fake_codex))

        assert main(["run", "--profile", "sealed", "--prompt", "first", "--json"]) == 0
        run_id = json.loads(capsys.readouterr().out)["run_id"]
        assert (study / ".aop" / "runs" / run_id).is_dir()

        assert main(["resume", run_id, "--prompt", "second", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["succeeded"]
        assert main(["cleanup", run_id]) == 0

    def test_writable_sealed_workspace_keeps_files_across_resume(
        self,
        repository: Path, fake_claude: Path
    ) -> None:
        script = fake_claude.read_text()
        fake_claude.write_text(
            script.replace(
                "prompt = sys.stdin.read()\n",
                "prompt = sys.stdin.read()\n"
                "workspace = pathlib.Path('/workspace')\n"
                "if prompt.startswith('WRITE'):\n"
                "    workspace.joinpath('dream.py').write_text('print(1)')\n"
                "if prompt.startswith('READ'):\n"
                "    pathlib.Path('/output/seen.txt').write_text(\n"
                "        workspace.joinpath('dream.py').read_text())\n"
                "if prompt.startswith('TRY'):\n"
                "    try:\n"
                "        workspace.joinpath('x').write_text('x')\n"
                "        outcome = 'written'\n"
                "    except OSError:\n"
                "        outcome = 'read-only'\n"
                "    pathlib.Path('/output/seen.txt').write_text(outcome)\n",
                1,
            )
        )
        runner = AgentRunner(
            WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
        )

        first = runner.run(
            task="dream", prompt="WRITE", profile="sealed", writable_workspace=True
        )
        resumed = runner.resume(run_id=first.run_id, prompt="READ", artifacts=["seen.txt"])
        default = runner.run(
            task="plain", prompt="TRY", profile="sealed", artifacts=["seen.txt"]
        )

        assert first.succeeded, first.error
        assert resumed.succeeded, resumed.error
        assert default.succeeded, default.error
        seen = runner.store.path(resumed.run_id) / "artifacts" / "seen.txt"
        assert seen.read_text() == "print(1)"
        request = runner.store.load_request(resumed.run_id)
        assert request.writable_workspace is True
        assert request.effective_policy["workspace"]["writable"] is True
        assert "/workspace" in request.effective_policy["writable_paths"]
        default_seen = runner.store.path(default.run_id) / "artifacts" / "seen.txt"
        assert default_seen.read_text() == "read-only"
        with raises(AOPError, match="only to the sealed profile"):
            runner.run(task="edit", prompt="x", writable_workspace=True)

    def test_sealed_claude_runs_exclude_account_connectors_and_skills(
        self,
        repository: Path, fake_claude: Path
    ) -> None:
        result = AgentRunner(
            WorktreeManager.discover(repository), ClaudeAdapter(os.fspath(fake_claude))
        ).run(task="sealed-claude", prompt="hello", profile="sealed")

        assert result.succeeded, result.error
        for flag in ("--safe-mode", "--strict-mcp-config"):
            assert flag in result.command
        assert result.command[result.command.index("--mcp-config") + 1] == (
            '{"mcpServers":{}}'
        )
