"""Command-line interface for AOP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .batch import BatchResult, BatchRunner
from .integration import CheckpointManager, IntegrationManager
from .models import RunResult
from .runner import AgentRunner, RunStore, adapter_for
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

    run = commands.add_parser("run", help="run an agent in an isolated task worktree")
    run.add_argument("task")
    run.add_argument(
        "--agent", choices=["codex", "claude", "agy", "hermes"], default="codex"
    )
    run.add_argument("--base", default="HEAD", help="commit for a new task worktree")
    run.add_argument("--model", help="override the agent model")
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
        "--sandbox",
        choices=["workspace-write", "scratch-write", "danger-full-access"],
        default="workspace-write",
    )
    run.add_argument("--timeout", type=_positive_timeout, help="wall-clock seconds")
    run.add_argument(
        "--json", action="store_true", help="print the normalized result as JSON"
    )
    _add_artifact_arguments(run)
    _add_prompt_arguments(run)

    resume = commands.add_parser("resume", help="resume an agent session from a run")
    resume.add_argument("run_id")
    resume.add_argument("--timeout", type=_positive_timeout, help="wall-clock seconds")
    resume.add_argument(
        "--json", action="store_true", help="print the normalized result as JSON"
    )
    _add_artifact_arguments(resume)
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
    args = build_parser().parse_args(argv)

    try:
        manager = WorktreeManager.discover(Path.cwd())

        if args.command == "init":
            manager.initialize()
            print(f"initialized AOP state in {manager.state_dir}")
            return 0

        if args.command == "cleanup":
            request = RunStore(manager.state_dir / "runs").load_request(args.run_id)
            if any(item.task == request.task for item in manager.list()):
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
            result = AgentRunner(manager, adapter_for(args.agent)).run(
                task=args.task,
                prompt=_read_prompt(args),
                base=args.base,
                model=args.model,
                effort=args.effort,
                mode=args.mode,
                sandbox=args.sandbox,
                timeout_seconds=args.timeout,
                artifacts=args.artifact,
            )
            return _report_run(result, manager, json_output=args.json)

        if args.command == "resume":
            result = AgentRunner(manager).resume(
                run_id=args.run_id,
                prompt=_read_prompt(args),
                timeout_seconds=args.timeout,
                artifacts=args.artifact,
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


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    try:
        return args.prompt_file.read_text()
    except OSError as error:
        raise AOPError(
            f"could not read prompt file {args.prompt_file}: {error}"
        ) from error


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
    if result.api_equivalent_cost:
        metrics += f" api_equiv=${result.api_equivalent_cost.amount_usd:.6f}"
    else:
        metrics += " api_equiv=n/a"
    print(
        f"aop: run_id={result.run_id} session_id={result.session_id or '-'} {metrics} "
        f"artifacts={manager.state_dir / 'runs' / result.run_id}",
        file=sys.stderr,
    )
    if result.error:
        print(f"aop: {result.error}", file=sys.stderr)
    return _run_exit_code(result)


def _run_exit_code(result: RunResult) -> int:
    if result.succeeded:
        return 0
    if result.timed_out:
        return 124
    return result.exit_code if 0 < result.exit_code < 126 else 1


def _report_batch(result: BatchResult, manager: WorktreeManager) -> int:
    summary = manager.state_dir / "batches" / f"{result.batch_id}.json"
    succeeded = sum(task.status == "succeeded" for task in result.tasks)
    print(
        f"aop: batch_id={result.batch_id} succeeded={succeeded}/{len(result.tasks)} "
        f"summary={summary}",
        file=sys.stderr,
    )
    print(
        "aop: task\tagent\tmodel\teffort\ttime\ttokens\tapi-equiv",
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
            f"${task.api_equivalent_cost_usd:.6f}"
            if task.api_equivalent_cost_usd is not None
            else "n/a"
        )
        print(
            f"aop: {task.task}\t{task.agent}\t{task.model or '(configured)'}\t"
            f"{task.effort or '(configured)'}\t{duration}\t{tokens}\t{cost}",
            file=sys.stderr,
        )
    if result.interrupted:
        return 130
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
