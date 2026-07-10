"""Command-line interface for AOP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .worktrees import AOPError, WorktreeManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aop",
        description="Run concurrent agent tasks in isolated Git worktrees.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="prepare the current Git repository for AOP")

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


if __name__ == "__main__":
    raise SystemExit(main())
