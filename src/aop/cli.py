"""Command-line interface for AOP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .batch import BatchResult, BatchRunner
from .models import RunResult
from .runner import AgentRunner
from .worktrees import AOPError, WorktreeManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aop",
        description="Run concurrent agent tasks in isolated Git worktrees.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="prepare the current Git repository for AOP")

    batch = commands.add_parser("batch", help="run a TOML task manifest in parallel")
    batch.add_argument("manifest", type=Path)
    batch.add_argument("--jobs", type=_positive_integer, default=4)

    run = commands.add_parser("run", help="run Codex in an isolated task worktree")
    run.add_argument("task")
    run.add_argument("--agent", choices=["codex"], default="codex")
    run.add_argument("--base", default="HEAD", help="commit for a new task worktree")
    run.add_argument("--model", help="override the configured Codex model")
    run.add_argument(
        "--effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
        help="override Codex model reasoning effort",
    )
    run.add_argument(
        "--sandbox",
        choices=["read-only", "workspace-write", "danger-full-access"],
        default="workspace-write",
    )
    run.add_argument("--timeout", type=_positive_timeout, help="wall-clock seconds")
    _add_prompt_arguments(run)

    resume = commands.add_parser("resume", help="resume the Codex session from a run")
    resume.add_argument("run_id")
    resume.add_argument("--timeout", type=_positive_timeout, help="wall-clock seconds")
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

        if args.command == "exec":
            command = list(args.exec_command)
            if command[:1] == ["--"]:
                command = command[1:]
            return manager.run(args.task, command)

        if args.command == "batch":
            result = BatchRunner(manager, jobs=args.jobs).run(
                args.manifest,
                progress=lambda message: print(f"aop: {message}", file=sys.stderr),
            )
            return _report_batch(result, manager)

        if args.command == "run":
            result = AgentRunner(manager).run(
                task=args.task,
                prompt=_read_prompt(args),
                base=args.base,
                model=args.model,
                effort=args.effort,
                sandbox=args.sandbox,
                timeout_seconds=args.timeout,
            )
            return _report_run(result, manager)

        if args.command == "resume":
            result = AgentRunner(manager).resume(
                run_id=args.run_id,
                prompt=_read_prompt(args),
                timeout_seconds=args.timeout,
            )
            return _report_run(result, manager)

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


def _report_run(result: RunResult, manager: WorktreeManager) -> int:
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
    if result.succeeded:
        return 0
    if result.error:
        print(f"aop: {result.error}", file=sys.stderr)
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
    print("aop: task\tmodel\teffort\ttime\ttokens\tapi-equiv", file=sys.stderr)
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
            f"aop: {task.task}\t{task.model or '(configured)'}\t"
            f"{task.effort or '(configured)'}\t{duration}\t{tokens}\t{cost}",
            file=sys.stderr,
        )
    if result.interrupted:
        return 130
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
