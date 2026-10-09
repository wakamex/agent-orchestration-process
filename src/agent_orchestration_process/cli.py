"""Command-line interface for AOP."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence, TextIO

import model_prices

from . import __version__, pricing
from .batch import BatchResult, BatchRunner
from .integration import CheckpointManager, IntegrationManager
from .isolation import PROFILES, explain_profile, resolve_policy
from .locks import exclusive_lock, task_lock_path
from .model_catalog import ModelCatalog, ensure_catalog_fresh
from .model_listing import AGENTS, AvailableModel, list_models
from .models import PromptPart, RunRequest, RunResult
from .runner import AgentRunner, RunStore, adapter_for, provider_runtime_record
from .migration import migrate_runs, migration_roots, print_counts
from .worktrees import (
    AOPError,
    WorktreeManager,
    register_state_root,
    registered_state_roots,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aop",
        description="Run concurrent agent tasks in isolated Git worktrees.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser(
        "models", help="list models and current per-million-token prices"
    )
    models.add_argument(
        "--agent",
        action="append",
        choices=AGENTS,
        help="limit results to an agent; may be repeated",
    )
    models.add_argument(
        "--refresh",
        action="store_true",
        help="refresh the shared catalog and supported native model caches",
    )
    models.add_argument(
        "--provider",
        help="list an inference provider inventory; currently Codex-only",
    )
    models.add_argument("--json", action="store_true", help="print JSON")

    commands.add_parser("init", help="prepare the current Git repository for AOP")

    migrate = commands.add_parser(
        "migrate-results",
        help="convert older run records to the current format, keeping each original",
    )
    migrate.add_argument(
        "runs_dirs",
        nargs="*",
        type=Path,
        help="run directories such as PROJECT/.aop/runs "
        "(default: every registered AOP state directory and the current project's)",
    )
    migrate.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing",
    )

    cleanup = commands.add_parser(
        "cleanup",
        help="discard the task worktree and private runtime state of a run's task",
    )
    cleanup.add_argument("run_id")

    checkpoint = commands.add_parser(
        "checkpoint", help="commit all current changes in a task worktree"
    )
    checkpoint.add_argument("task")
    checkpoint.add_argument("-m", "--message", required=True)

    integrate = commands.add_parser(
        "integrate", help="have the author rebase and fast-forward the task onto main"
    )
    integrate.add_argument("task")
    integrate.add_argument(
        "--timeout",
        type=_positive_timeout,
        help="author integration wall-clock seconds",
    )
    integrate.add_argument(
        "--remove-worktree",
        action="store_true",
        help="remove the task worktree after successful integration",
    )

    batch = commands.add_parser("batch", help="run a TOML task manifest in parallel")
    batch.add_argument("manifest", type=Path)
    batch.add_argument("--jobs", type=_positive_integer, default=4)

    profile_command = commands.add_parser(
        "profile", help="inspect execution profile guarantees"
    )
    profile_commands = profile_command.add_subparsers(
        dest="profile_command", required=True
    )
    explain = profile_commands.add_parser(
        "explain", help="print the declared boundary for a profile"
    )
    explain.add_argument("profile", choices=PROFILES)
    explain.add_argument("--agent", choices=AGENTS, default="codex")
    explain.add_argument("--json", action="store_true", help="print JSON")

    run = commands.add_parser("run", help="run an agent under an execution profile")
    run.add_argument("task", nargs="?", help="task label; optional for sealed runs")
    run.add_argument(
        "--agent",
        choices=AGENTS,
        default="codex",
    )
    run.add_argument("--base", default="HEAD", help="commit for a new task worktree")
    run.add_argument("--model", help="override the agent model")
    run.add_argument(
        "--provider",
        dest="inference_provider",
        help="override the inference provider for Codex, Hermes, or DeepSeek Harness",
    )
    run.add_argument(
        "--mode",
        choices=["agent", "participant"],
        default="agent",
        help="agent behavior mode; participant is currently Hermes-only",
    )
    run.add_argument(
        "--effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
        help="override agent reasoning effort",
    )
    run.add_argument(
        "--profile",
        choices=PROFILES,
        default="edit",
        help="edit workspace, review it read-only, run sealed inputs-only, or use the native host",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="preview the intended boundary without compiling or dispatching it",
    )
    run.add_argument(
        "--no-web",
        action="store_true",
        help="deny model-controlled external retrieval or fail before dispatch",
    )
    system_prompts = run.add_mutually_exclusive_group()
    system_prompts.add_argument(
        "--system-prompt",
        metavar="TEXT",
        help="replace the harness's base system prompt (claude only; '' for none)",
    )
    system_prompts.add_argument(
        "--system-prompt-file",
        type=Path,
        metavar="PATH",
        help="replace the harness's base system prompt with PATH's contents (claude only)",
    )
    run.add_argument(
        "--tools",
        type=_tool_list,
        metavar="LIST",
        help="exact comma-separated built-in tool set (claude only; '' for none)",
    )
    run.add_argument(
        "--thinking-display",
        choices=["summarized", "omitted", "highlights"],
        help="how Claude returns the model's thinking, such as official summaries "
        "(claude only)",
    )
    run.add_argument(
        "--writable-workspace",
        action="store_true",
        help="with --profile sealed, let the run write its private, initially empty "
        "/workspace; resumes keep its files",
    )
    run.add_argument(
        "--allow-url",
        action="append",
        default=[],
        dest="allowed_urls",
        metavar="URL",
        help="with --no-web, also allow the sandbox to reach URL's host and port; "
        "repeat for more",
    )
    run.add_argument(
        "--provider-base-url",
        metavar="URL",
        help="send the harness's inference requests to URL, such as a local "
        "recorder or proxy (claude only)",
    )
    run.add_argument("--timeout", type=_positive_timeout, help="wall-clock seconds")
    run.add_argument(
        "--json", action="store_true", help="print the normalized result as JSON"
    )
    _add_artifact_arguments(run)
    _add_input_arguments(run, default=[], resume=False)
    _add_prompt_arguments(run)

    resume = commands.add_parser("resume", help="resume an agent session from a run")
    resume.add_argument("run_id")
    resume.add_argument("--timeout", type=_positive_timeout, help="wall-clock seconds")
    resume.add_argument(
        "--json", action="store_true", help="print the normalized result as JSON"
    )
    _add_artifact_arguments(resume)
    _add_input_arguments(resume, default=None, resume=True)
    _add_prompt_arguments(resume)

    worktree = commands.add_parser("worktree", help="manage task worktrees")
    worktree_commands = worktree.add_subparsers(dest="worktree_command", required=True)

    create = worktree_commands.add_parser(
        "create", help="create a detached task worktree"
    )
    create.add_argument("task")
    create.add_argument(
        "--base", default="HEAD", help="commit to check out (default: HEAD)"
    )

    worktree_commands.add_parser("list", help="list AOP task worktrees")

    path = worktree_commands.add_parser("path", help="print a task worktree path")
    path.add_argument("task")

    remove = worktree_commands.add_parser("remove", help="remove a clean task worktree")
    remove.add_argument("task")
    remove.add_argument(
        "--force", action="store_true", help="discard uncommitted task changes"
    )

    execute = commands.add_parser("exec", help="run a command inside a task worktree")
    execute.add_argument("task")
    execute.add_argument(
        "--overlay",
        action="append",
        default=[],
        metavar="PATH",
        help="give PATH a private copy-on-write view of the main worktree directory",
    )
    execute.add_argument("exec_command", nargs=argparse.REMAINDER, metavar="-- COMMAND")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        raw_arguments = list(sys.argv[1:] if argv is None else argv)
        args = build_parser().parse_args(raw_arguments)

        if args.command in {
            "models",
            "run",
            "resume",
            "batch",
            "integrate",
            "migrate-results",
        }:
            price_warnings = pricing.refresh_prices(
                force=args.command == "models" and args.refresh
            )
            if args.command != "models":
                for warning in price_warnings:
                    print(f"aop: warning: {warning}", file=sys.stderr)

        if args.command == "migrate-results":
            return _migrate_results(args.runs_dirs, dry_run=args.dry_run)

        if args.command == "models":
            agents = args.agent or AGENTS
            if args.provider is not None and agents != ["codex"]:
                raise AOPError("models --provider requires exactly --agent codex")
            catalog_error = None
            try:
                catalog = ensure_catalog_fresh(force=args.refresh)
            except (AOPError, OSError) as error:
                # Native inventories remain useful without catalog metadata.
                catalog_error = str(error)
                catalog = ModelCatalog({}, 0, "", "")
            return _report_models(
                agents,
                catalog,
                inference_provider=args.provider,
                json_output=args.json,
                catalog_error=catalog_error,
                price_warnings=price_warnings,
                refresh=args.refresh,
            )

        if args.command == "profile":
            policy = explain_profile(args.profile)
            policy["provider"] = args.agent
            policy["provider_runtime"] = provider_runtime_record(
                adapter_for(args.agent)
            )
            _report_policy(policy, json_output=args.json)
            return 0

        if args.command == "run" and args.profile == "sealed":
            manager = WorktreeManager.standalone(Path.cwd())
        elif args.command in {"resume", "cleanup"}:
            manager = _manager_holding_run(args.run_id)
        else:
            manager = WorktreeManager.discover(Path.cwd())

        if args.command == "init":
            manager.initialize()
            print(f"initialized AOP state in {manager.state_dir}")
            return 0

        if args.command == "cleanup":
            request = RunStore(manager.state_dir / "runs").load_request(args.run_id)
            if request.profile == "sealed":
                _cleanup_sealed(manager, request)
            elif any(item.task == request.task for item in manager.list()):
                manager.remove(request.task, force=True)
            print(request.task)
            return 0

        if args.command == "checkpoint":
            result = CheckpointManager(manager).checkpoint(args.task, args.message)
            print(result.commit)
            print(f"aop: checkpoint={result.record_path}", file=sys.stderr)
            return 0

        if args.command == "integrate":
            result = IntegrationManager(manager).integrate(
                args.task,
                remove_worktree=args.remove_worktree,
                timeout_seconds=args.timeout,
            )
            if result.author_run.final_message:
                print(
                    result.author_run.final_message,
                    end=(
                        "" if result.author_run.final_message.endswith("\n") else "\n"
                    ),
                )
            print(result.integrated_head)
            print(
                f"aop: integrated={len(result.integrated_commits)} "
                f"author_run_id={result.author_run.run_id} record={result.record_path}",
                file=sys.stderr,
            )
            return 0

        if args.command == "exec":
            command = list(args.exec_command)
            while command[:1] == ["--overlay"]:
                if len(command) < 2:
                    raise AOPError("--overlay requires a path")
                args.overlay.append(command[1])
                del command[:2]
            if command[:1] == ["--"]:
                command = command[1:]
            return manager.run(args.task, command, overlays=args.overlay)

        if args.command == "batch":
            result = BatchRunner(manager, jobs=args.jobs).run(
                args.manifest,
                progress=lambda message: print(f"aop: {message}", file=sys.stderr),
            )
            return _report_batch(result, manager)

        if args.command == "run":
            task = args.task or str(uuid.uuid4())
            if args.task is None and args.profile != "sealed":
                raise AOPError("TASK is required unless --profile sealed is selected")
            preview = resolve_policy(
                args.profile,
                provider=args.agent,
                no_web=args.no_web,
                input_names=tuple(Path(path).name for path in args.input_paths),
            ).to_dict()
            preview["provider"] = args.agent
            preview["provider_runtime"] = provider_runtime_record(
                adapter_for(args.agent)
            )
            if args.dry_run:
                _report_policy(preview, json_output=args.json)
                return 0
            if not args.json:
                _report_policy(
                    preview,
                    json_output=False,
                    prefix="aop: ",
                    stream=sys.stderr,
                )
            prompt, prompt_parts = _read_prompt(args)
            result = AgentRunner(manager, adapter_for(args.agent)).run(
                task=task,
                prompt=prompt,
                prompt_parts=prompt_parts,
                base=args.base,
                model=args.model,
                inference_provider=args.inference_provider,
                effort=args.effort,
                mode=args.mode,
                profile=args.profile,
                no_web=args.no_web,
                timeout_seconds=args.timeout,
                artifacts=args.artifact,
                input_paths=args.input_paths,
                system_prompt=_read_system_prompt(args),
                tools=args.tools,
                provider_base_url=args.provider_base_url,
                allowed_urls=args.allowed_urls,
                writable_workspace=args.writable_workspace,
                thinking_display=args.thinking_display,
            )
            return _report_run(result, manager, json_output=args.json)

        if args.command == "resume":
            prompt, prompt_parts = _read_prompt(args)
            result = AgentRunner(manager).resume(
                run_id=args.run_id,
                prompt=prompt,
                prompt_parts=prompt_parts,
                timeout_seconds=args.timeout,
                artifacts=args.artifact,
                input_paths=args.input_paths,
            )
            return _report_run(result, manager, json_output=args.json)

        if args.worktree_command == "create":
            created = manager.create(args.task, args.base)
            print(created.path)
            return 0
        if args.worktree_command == "list":
            for item in manager.list():
                print(f"{item.task}\t{item.head[:12]}\t{item.path}")
            return 0
        if args.worktree_command == "path":
            print(manager.get(args.task).path)
            return 0
        if args.worktree_command == "remove":
            manager.remove(args.task, force=args.force)
            return 0

        raise AOPError("unsupported command")
    except AOPError as error:
        print(f"aop: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"aop: command not found: {error.filename}", file=sys.stderr)
        return 127
    except KeyboardInterrupt:
        print("aop: interrupted", file=sys.stderr)
        return 130


def _add_prompt_arguments(parser: argparse.ArgumentParser) -> None:
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    prompts.add_argument(
        "--prompt-parts",
        type=Path,
        metavar="PATH",
        help='JSON list of prompt blocks, each {"text": ..., "cache": true|false}; '
        "Claude caches the prompt through the one cached block (claude only)",
    )


def _add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="PATH",
        help="require and archive PATH relative to this run's output directory",
    )


def _add_input_arguments(
    parser: argparse.ArgumentParser,
    *,
    default: list[str] | None,
    resume: bool,
) -> None:
    help_text = (
        "replace inherited input snapshots with PATH; repeat for additional inputs"
        if resume
        else "snapshot PATH read-only at /inputs/BASENAME; repeat for additional inputs"
    )
    parser.add_argument(
        "--input",
        action="append",
        default=default,
        dest="input_paths",
        metavar="PATH",
        help=help_text,
    )


def _tool_list(value: str) -> tuple[str, ...]:
    return tuple(tool.strip() for tool in value.split(",")) if value.strip() else ()


def _read_system_prompt(args: argparse.Namespace) -> str | None:
    if args.system_prompt_file is None:
        return args.system_prompt
    try:
        return args.system_prompt_file.read_text()
    except OSError as error:
        raise AOPError(
            f"could not read system prompt file {args.system_prompt_file}: {error}"
        ) from error


def _read_prompt(args: argparse.Namespace) -> tuple[str, tuple[PromptPart, ...] | None]:
    if args.prompt is not None:
        return args.prompt, None
    path = args.prompt_file or args.prompt_parts
    try:
        text = path.read_text()
    except OSError as error:
        raise AOPError(f"could not read prompt file {path}: {error}") from error
    if args.prompt_file is not None:
        return text, None
    try:
        blocks = json.loads(text)
    except json.JSONDecodeError as error:
        raise AOPError(f"prompt parts file {path} is not JSON: {error}") from error
    if not isinstance(blocks, list) or not all(
        isinstance(block, dict)
        and set(block) <= {"text", "cache"}
        and isinstance(block.get("text"), str)
        and isinstance(block.get("cache", False), bool)
        for block in blocks
    ):
        raise AOPError(
            f"prompt parts file {path} must be a list of "
            '{"text": string, "cache": boolean} objects'
        )
    parts = tuple(
        PromptPart(chars=len(block["text"]), cache=block.get("cache", False))
        for block in blocks
    )
    return "".join(block["text"] for block in blocks), parts


def _report_policy(
    policy: dict[str, object],
    *,
    json_output: bool,
    prefix: str = "",
    stream: TextIO = sys.stdout,
) -> None:
    if json_output:
        print(json.dumps(policy, indent=2, sort_keys=True))
        return
    workspace = policy["workspace"]
    repository = policy["repository"]
    host = policy["host"]
    inputs = policy["inputs"]
    instructions = policy["instructions"]
    network = policy["network"]
    environment = policy["environment"]
    model_capabilities = policy["model_capabilities"]
    assert isinstance(workspace, dict)
    assert isinstance(repository, dict)
    assert isinstance(host, dict)
    assert isinstance(inputs, dict)
    assert isinstance(instructions, dict)
    assert isinstance(network, dict)
    assert isinstance(environment, dict)
    assert isinstance(model_capabilities, dict)
    print(f"{prefix}profile: {policy['profile']}", file=stream)
    print(
        f"{prefix}repository: {repository['access']} at "
        f"{repository.get('guest_path') or 'not mounted'}",
        file=stream,
    )
    print(
        f"{prefix}workspace: task access {workspace['access']}; "
        f"{workspace['content']} at {workspace.get('guest_path') or 'not mounted'}; "
        f"writable: {str(workspace['writable']).lower()}",
        file=stream,
    )
    print(f"{prefix}host: {host['access']}", file=stream)
    print(
        f"{prefix}inputs: {inputs['mode']} ({len(inputs.get('names', []))} declared)",
        file=stream,
    )
    print(
        f"{prefix}inherited local instructions: {instructions['inherited_local']}; "
        f"provider built-in prompt: {instructions['provider_builtin']}; "
        f"AOP generated instructions: {instructions['aop_generated']}",
        file=stream,
    )
    print(
        f"{prefix}environment: {environment['mode']}; "
        f"credentials: {environment['credential_exposure']} with recorded values "
        f"{environment['recorded_values']}; network: {network['mode']}; "
        f"network isolation: {network['isolation']}",
        file=stream,
    )
    print(
        f"{prefix}identity: {policy['identity']}; namespaces: "
        f"{', '.join(policy['namespaces']) or 'native'}; "
        f"capabilities: {policy['capabilities']}",
        file=stream,
    )
    requested_capabilities = model_capabilities["requested"]
    effective_capabilities = model_capabilities["effective"]
    assert isinstance(requested_capabilities, dict)
    assert isinstance(effective_capabilities, dict)
    print(
        f"{prefix}external retrieval: requested "
        f"{requested_capabilities['external_retrieval']}; effective "
        f"{effective_capabilities['external_retrieval']}; tool network egress: "
        f"{effective_capabilities['tool_network_egress']}; model tools: "
        f"{effective_capabilities['model_tools']}",
        file=stream,
    )
    print(
        f"{prefix}writable: "
        + ", ".join(
            f"{path} ({policy['writable_path_scopes'][path]})"
            for path in policy["writable_paths"]
        ),
        file=stream,
    )
    runtime = policy.get("provider_runtime")
    if isinstance(runtime, dict):
        print(
            f"{prefix}provider executable: {runtime.get('executable') or 'unresolved'} "
            f"sha256={runtime.get('sha256') or 'unavailable'}",
            file=stream,
        )


def _manager_holding_run(run_id: str) -> WorktreeManager:
    """The state of the run: the repository's, or the current directory's.

    Sealed runs record themselves under the current directory even inside a
    repository, so a resume or cleanup looks in both places.
    """
    try:
        repository = WorktreeManager.discover(Path.cwd())
    except AOPError:
        repository = None
    if repository is not None and (repository.state_dir / "runs" / run_id).is_dir():
        return repository
    if repository is None or (Path.cwd() / ".aop" / "runs" / run_id).is_dir():
        return WorktreeManager.standalone(Path.cwd())
    return repository


def _migrate_results(runs_dirs: Sequence[Path], *, dry_run: bool) -> int:
    if not runs_dirs:
        try:
            current = WorktreeManager.discover(Path.cwd()).state_dir
        except AOPError:
            current = Path.cwd() / ".aop"
        runs_dirs = migration_roots(registered_state_roots(), current)
    errors = 0
    for runs_dir in runs_dirs:
        if not runs_dir.is_dir():
            raise AOPError(f"not a run directory: {runs_dir}")
        counts = migrate_runs(runs_dir, dry_run=dry_run)
        if not dry_run:
            register_state_root(runs_dir.resolve().parent)
        print_counts(runs_dir, counts)
        errors += sum(
            count for outcome, count in counts.items() if outcome.endswith("error")
        )
    return 1 if errors else 0


def _cleanup_sealed(manager: WorktreeManager, request: RunRequest) -> None:
    task = request.task
    policy = request.effective_policy
    workspace = policy.get("workspace", {})
    controller = policy.get("controller", {})
    controller_path = workspace.get("controller_path")
    recorded = [
        (
            Path(controller_path).parent if isinstance(controller_path, str) else None,
            "sealed",
        ),
        (controller.get("provider_state"), "provider-state"),
        (controller.get("scratch"), "scratch"),
        (controller.get("cache"), "sealed-cache"),
        (controller.get("input_projection_root"), "input-projections"),
    ]
    with exclusive_lock(task_lock_path(manager.state_dir, task), f"task {task}"):
        for value, kind in recorded:
            path = _sealed_runtime_path(manager, value, kind)
            if path is not None:
                shutil.rmtree(path)
                print(f"aop: removed {path}", file=sys.stderr)


def _sealed_runtime_path(
    manager: WorktreeManager, value: object, kind: str
) -> Path | None:
    """Return a recorded directory only if it is this project's sealed runtime state.

    The check uses the recorded path's own structure, so cleanup works whatever
    AOP_SEALED_RUNTIME_ROOT is set to now.
    """
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value).resolve()
    identity = path.parent.parent
    if (
        path.parent.name != kind
        or identity.name != manager.sealed_runtime_identity
        or identity.parent.name != f"aop-sealed-{os.getuid()}"
        or not path.is_dir()
        or path.stat().st_uid != os.getuid()
    ):
        return None
    return path


def _positive_timeout(value: str) -> float:
    try:
        timeout = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a number") from error
    if timeout <= 0:
        raise argparse.ArgumentTypeError("timeout must be greater than zero")
    return timeout


def _positive_integer(value: str) -> int:
    try:
        integer = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if integer <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return integer


def _report_run(
    result: RunResult,
    manager: WorktreeManager,
    *,
    json_output: bool = False,
) -> int:
    if json_output:
        print(json.dumps(result.to_dict(), sort_keys=True))
        return _run_exit_code(result)
    if result.final_message:
        print(
            result.final_message,
            end="" if result.final_message.endswith("\n") else "\n",
        )
    token_count = result.usage.total_tokens if result.usage else "n/a"
    metrics = (
        f"time={result.duration_seconds:.2f}s tokens={token_count} "
        f"accounting={result.accounting_status}"
    )
    if result.calculated_cost:
        metrics += f" calculated_cost=${result.calculated_cost.amount_usd:.6f}"
    else:
        metrics += " calculated_cost=n/a"
    if result.provider_reported_cost:
        metrics += (
            f" provider_reported_cost=${result.provider_reported_cost.amount_usd:.6f}"
        )
    else:
        metrics += " provider_reported_cost=n/a"
    if result.inference_provider:
        metrics += f" provider={result.inference_provider}"
    metrics += f" billing={result.billing.route}"
    print(
        f"aop: run_id={result.run_id} session_id={result.session_id or '-'} {metrics} "
        f"artifacts={manager.state_dir / 'runs' / result.run_id}",
        file=sys.stderr,
    )
    if result.error:
        print(f"aop: {result.error}", file=sys.stderr)
    if result.provider_error:
        print(
            f"aop: provider status {result.provider_status or 'unknown'}: "
            f"{result.provider_error}",
            file=sys.stderr,
        )
    return _run_exit_code(result)


def _run_exit_code(result: RunResult) -> int:
    if result.succeeded:
        return 0
    if result.timed_out:
        return 124
    return result.exit_code if 0 < result.exit_code < 126 else 1


def _report_batch(result: BatchResult, manager: WorktreeManager) -> int:
    summary = manager.state_dir / "batches" / f"{result.batch_id}.json"
    print(
        f"aop: batch_id={result.batch_id} "
        f"succeeded={result.clean_successes}/{len(result.tasks)} "
        f"responses_with_provider_errors={result.responses_with_provider_errors} "
        f"without_response={result.runs_without_response} "
        f"summary={summary}",
        file=sys.stderr,
    )
    print(
        "aop: task\tagent\tprovider\tmodel\teffort\ttime\ttokens\tcalculated"
        "\tprovider-reported-cost\tbilling",
        file=sys.stderr,
    )
    for task in result.tasks:
        duration = (
            f"{task.duration_seconds:.2f}s"
            if task.duration_seconds is not None
            else "-"
        )
        tokens = (
            str((task.input_tokens or 0) + (task.output_tokens or 0))
            if task.input_tokens is not None
            else "-"
        )
        cost = (
            f"${task.calculated_cost_usd:.6f}"
            if task.calculated_cost_usd is not None
            else "n/a"
        )
        reported_cost = (
            f"${task.provider_reported_cost_usd:.6f}"
            if task.provider_reported_cost_usd is not None
            else "n/a"
        )
        print(
            f"aop: {task.task}\t{task.agent}\t"
            f"{task.inference_provider or '(configured)'}\t"
            f"{task.model or '(configured)'}\t"
            f"{task.effort or '(configured)'}\t{duration}\t{tokens}\t{cost}\t"
            f"{reported_cost}\t"
            f"{task.billing_route or 'unknown'}",
            file=sys.stderr,
        )
    if result.interrupted:
        return 130
    return 0 if result.succeeded else 1


def _report_models(
    agents: Sequence[str],
    catalog: ModelCatalog,
    *,
    inference_provider: str | None = None,
    json_output: bool,
    catalog_error: str | None = None,
    price_warnings: Sequence[str] = (),
    refresh: bool = False,
) -> int:
    models: list[AvailableModel] = []
    errors: dict[str, str] = {}
    # Native startup and network waits are independent across harnesses.
    # Submit once per harness so repeated --agent flags cannot race its state.
    with ThreadPoolExecutor(max_workers=min(len(AGENTS), len(agents) or 1)) as executor:
        discoveries = {
            agent: executor.submit(
                list_models, agent, catalog, inference_provider, refresh=refresh
            )
            for agent in dict.fromkeys(agents)
        }
        for agent, discovery in discoveries.items():
            try:
                models.extend(discovery.result())
            except AOPError as error:
                errors[agent] = str(error)
    models.sort(key=lambda item: (item.agent, item.model))
    warnings = {"catalog": catalog_error} if catalog_error else {}
    if price_warnings:
        warnings["prices"] = "; ".join(price_warnings)
    warnings.update(
        {
            model.agent: model.discovery_error
            for model in models
            if model.discovery_error
        }
    )
    fetched_at = (
        datetime.fromtimestamp(catalog.fetched_at, UTC).isoformat()
        if catalog.source
        else None
    )
    if json_output:
        print(
            json.dumps(
                {
                    "catalog": {
                        "fetched_at": fetched_at,
                        "sha256": catalog.sha256,
                        "source": catalog.source,
                    }
                    if catalog.source
                    else None,
                    "errors": errors,
                    "warnings": warnings,
                    "models": [model.to_dict() for model in models],
                    "pricing_basis": model_prices.pricing_basis(),
                },
                sort_keys=True,
            )
        )
    else:
        print(
            "agent\tprovider\tmodel\tavailability\tprice-scope\tinput\tcache-read\t"
            "cache-write\toutput"
        )
        for model in models:
            print(
                f"{model.agent}\t{model.inference_provider or '(native)'}\t"
                f"{model.model}\t{model.availability}\t"
                f"{model.price_scope}\t{_price(model.input_per_million_usd)}\t"
                f"{_price(model.cached_input_per_million_usd)}\t"
                f"{_price(model.cache_write_per_million_usd)}\t"
                f"{_price(model.output_per_million_usd)}"
            )
        print(
            f"aop: prices_per_million_usd basis={model_prices.pricing_basis()} "
            f"catalog_fetched_at={fetched_at} catalog_sha256={catalog.sha256}",
            file=sys.stderr,
        )
        for agent, error in errors.items():
            print(f"aop: {agent}: {error}", file=sys.stderr)
        for source, warning in warnings.items():
            print(f"aop: warning: {source}: {warning}", file=sys.stderr)
    return 0 if models and not errors else 1


def _price(value: float | None) -> str:
    return "n/a" if value is None else f"${value:g}"


if __name__ == "__main__":
    raise SystemExit(main())
