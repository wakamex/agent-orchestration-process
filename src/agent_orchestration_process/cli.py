"""Command-line interface for AOP."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence, TextIO

from . import __version__
from .batch import BatchResult, BatchRunner
from .integration import CheckpointManager, IntegrationManager
from .isolation import PROFILES, explain_profile, resolve_policy
from .locks import exclusive_lock, task_lock_path
from .model_catalog import ModelCatalog, ensure_catalog_fresh
from .model_listing import AGENTS, AvailableModel, list_models
from .models import RunRequest, RunResult
from .runner import AgentRunner, RunStore, adapter_for, provider_runtime_record
from .worktrees import AOPError, WorktreeManager


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
        "--refresh", action="store_true", help="refresh the shared model catalog now"
    )
    models.add_argument("--json", action="store_true", help="print JSON")

    commands.add_parser("init", help="prepare the current Git repository for AOP")

    cleanup = commands.add_parser(
        "cleanup", help="discard the task worktree associated with a run"
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
        help="override the inference provider for Hermes or DeepSeek Harness",
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

        if args.command == "models":
            catalog = ensure_catalog_fresh(force=args.refresh)
            return _report_models(args.agent or AGENTS, catalog, json_output=args.json)

        if args.command == "profile":
            policy = explain_profile(args.profile)
            policy["provider"] = args.agent
            policy["provider_runtime"] = provider_runtime_record(
                adapter_for(args.agent)
            )
            _report_policy(policy, json_output=args.json)
            return 0

        if args.command in {"run", "resume", "batch", "integrate"}:
            ensure_catalog_fresh()

        if args.command == "run" and args.profile == "sealed":
            manager = WorktreeManager.standalone(Path.cwd())
        else:
            try:
                manager = WorktreeManager.discover(Path.cwd())
            except AOPError:
                if args.command not in {"resume", "cleanup"}:
                    raise
                manager = WorktreeManager.standalone(Path.cwd())

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
            result = AgentRunner(manager, adapter_for(args.agent)).run(
                task=task,
                prompt=_read_prompt(args),
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
            )
            return _report_run(result, manager, json_output=args.json)

        if args.command == "resume":
            result = AgentRunner(manager).resume(
                run_id=args.run_id,
                prompt=_read_prompt(args),
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


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    try:
        return args.prompt_file.read_text()
    except OSError as error:
        raise AOPError(
            f"could not read prompt file {args.prompt_file}: {error}"
        ) from error


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


def _cleanup_sealed(manager: WorktreeManager, request: RunRequest) -> None:
    task = request.task
    policy = request.effective_policy
    with exclusive_lock(task_lock_path(manager.state_dir, task), f"task {task}"):
        workspace = policy.get("workspace", {})
        controller = policy.get("controller", {})
        controller_path = workspace.get("controller_path")
        if isinstance(controller_path, str):
            sealed_run_dir = Path(controller_path).parent
            sealed_root = (manager.sealed_runtime_dir / "sealed").resolve()
            resolved = sealed_run_dir.resolve()
            if resolved.is_relative_to(sealed_root) and resolved.exists():
                shutil.rmtree(resolved)
        provider_state = controller.get("provider_state")
        if isinstance(provider_state, str):
            state_path = Path(provider_state).resolve()
            state_root = (manager.sealed_runtime_dir / "provider-state").resolve()
            if state_path.is_relative_to(state_root) and state_path.exists():
                shutil.rmtree(state_path)
        scratch = controller.get("scratch")
        if isinstance(scratch, str):
            scratch_path = Path(scratch).resolve()
            scratch_root = (manager.sealed_runtime_dir / "scratch").resolve()
            if scratch_path.is_relative_to(scratch_root) and scratch_path.exists():
                shutil.rmtree(scratch_path)
        cache = controller.get("cache")
        if isinstance(cache, str):
            cache_path = Path(cache).resolve()
            cache_root = (manager.sealed_runtime_dir / "sealed-cache").resolve()
            if cache_path.is_relative_to(cache_root) and cache_path.exists():
                shutil.rmtree(cache_path)
        input_projection_root = controller.get("input_projection_root")
        if isinstance(input_projection_root, str):
            projection_path = Path(input_projection_root).resolve()
            projections_root = (
                manager.sealed_runtime_dir / "input-projections"
            ).resolve()
            if (
                projection_path.is_relative_to(projections_root)
                and projection_path.exists()
            ):
                shutil.rmtree(projection_path)


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
    metrics = f"time={result.duration_seconds:.2f}s tokens={result.usage.total_tokens}"
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
    agents: Sequence[str], catalog: ModelCatalog, *, json_output: bool
) -> int:
    models: list[AvailableModel] = []
    errors: dict[str, str] = {}
    for agent in agents:
        try:
            models.extend(list_models(agent, catalog))
        except AOPError as error:
            errors[agent] = str(error)
    models.sort(key=lambda item: (item.agent, item.model))
    fetched_at = datetime.fromtimestamp(catalog.fetched_at, UTC).isoformat()
    if json_output:
        print(
            json.dumps(
                {
                    "catalog": {
                        "fetched_at": fetched_at,
                        "sha256": catalog.sha256,
                        "source": catalog.source,
                    },
                    "errors": errors,
                    "models": [model.to_dict() for model in models],
                },
                sort_keys=True,
            )
        )
    else:
        print(
            "agent\tmodel\tavailability\tprice-scope\tinput\tcache-read\t"
            "cache-write\toutput"
        )
        for model in models:
            print(
                f"{model.agent}\t{model.model}\t{model.availability}\t"
                f"{model.price_scope}\t{_price(model.input_per_million_usd)}\t"
                f"{_price(model.cached_input_per_million_usd)}\t"
                f"{_price(model.cache_write_per_million_usd)}\t"
                f"{_price(model.output_per_million_usd)}"
            )
        sources = sorted(
            {model.pricing_source for model in models if model.pricing_source}
        )
        print(
            f"aop: prices_per_million_usd sources={','.join(sources)} "
            f"catalog_fetched_at={fetched_at} catalog_sha256={catalog.sha256}",
            file=sys.stderr,
        )
        for agent, error in errors.items():
            print(f"aop: {agent}: {error}", file=sys.stderr)
    return 0 if models and not errors else 1


def _price(value: float | None) -> str:
    return "n/a" if value is None else f"${value:g}"


if __name__ == "__main__":
    raise SystemExit(main())
