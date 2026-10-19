"""Start and stop AOP sandbox commands.

A sandbox command is `bwrap OPTIONS -- COMMAND`. Bubblewrap's own reaper runs
as the sandbox's process 1, whose command line every process inside can read,
so the options, which name host paths and mounts, travel through a private
file descriptor (`--args`) instead. Bubblewrap reports the sandbox's process
ID through `--info-fd`, which lets a stop signal reach the harness itself:
bubblewrap does not forward signals inward.
"""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import tempfile
from typing import Any

# Arguments each bubblewrap option takes, for the options AOP uses.
_OPTION_ARGUMENTS = {
    "--die-with-parent": 0,
    "--new-session": 0,
    "--clearenv": 0,
    "--unshare-pid": 0,
    "--unshare-ipc": 0,
    "--unshare-uts": 0,
    "--unshare-net": 0,
    "--cap-drop": 1,
    "--proc": 1,
    "--dev": 1,
    "--tmpfs": 1,
    "--dir": 1,
    "--chdir": 1,
    "--overlay-src": 1,
    "--tmp-overlay": 1,
    "--ro-bind": 2,
    "--bind": 2,
    "--setenv": 2,
    "--symlink": 2,
}


def is_sandbox(command: list[str]) -> bool:
    return "--clearenv" in command and "--unshare-pid" in command


def split_options(command: list[str]) -> tuple[list[str], list[str]]:
    """Bubblewrap's options and the command after them."""
    index = 1
    while index < len(command):
        option = command[index]
        if option == "--":
            return command[1:index], command[index + 1 :]
        if option not in _OPTION_ARGUMENTS:
            raise ValueError(f"unexpected bubblewrap option: {option}")
        index += 1 + _OPTION_ARGUMENTS[option]
    raise ValueError("bubblewrap command has no -- before its command")


def popen(command: list[str], **kwargs: Any) -> subprocess.Popen:
    """Start `command`, passing a sandbox's options through a private descriptor."""
    if not is_sandbox(command):
        return subprocess.Popen(command, **kwargs)
    options, tail = split_options(command)
    # An unlinked private file, since the options can exceed a pipe's buffer.
    stream = tempfile.TemporaryFile()
    stream.write(b"".join(option.encode() + b"\0" for option in options))
    stream.seek(0)
    arguments = stream.fileno()
    info_read, info_write = os.pipe()
    try:
        process = subprocess.Popen(
            [
                command[0],
                "--args",
                str(arguments),
                "--info-fd",
                str(info_write),
                "--",
                *tail,
            ],
            pass_fds=(arguments, info_write, *kwargs.pop("pass_fds", ())),
            **kwargs,
        )
    except BaseException:
        os.close(info_read)
        raise
    finally:
        stream.close()
        os.close(info_write)
    process.aop_info_fd = info_read  # type: ignore[attr-defined]
    return process


def stop(process: subprocess.Popen, grace_seconds: float) -> None:
    """Ask the command to exit, then kill whatever remains after the grace period."""
    harness = _harness_pid(process)
    try:
        if harness is not None:
            os.kill(harness, signal.SIGTERM)
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def close(process: subprocess.Popen) -> None:
    descriptor = getattr(process, "aop_info_fd", None)
    if descriptor is not None:
        os.close(descriptor)
        process.aop_info_fd = None  # type: ignore[attr-defined]


def _harness_pid(process: subprocess.Popen) -> int | None:
    """The host process ID of the command bubblewrap's reaper started."""
    descriptor = getattr(process, "aop_info_fd", None)
    if descriptor is None or process.poll() is not None:
        return None
    if not select.select([descriptor], [], [], 1)[0]:
        return None
    try:
        sandbox = json.loads(os.read(descriptor, 65536))["child-pid"]
        with open(f"/proc/{sandbox}/task/{sandbox}/children") as children:
            first = children.read().split()
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return int(first[0]) if first else None
