"""AOP's sandbox launcher against real bubblewrap."""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from support import AOPTestCase, raises, skip

from agent_orchestration_process import sandbox_launch
from agent_orchestration_process.runner import _capture_process, _isolated_root_command

PROBE = """\
import os, signal, sys, time
cmdline = open('/proc/1/cmdline', 'rb').read().replace(b'\\0', b' ').decode()
print('pid', os.getpid(), flush=True)
print('pid1', cmdline, flush=True)
signal.signal(signal.SIGTERM, lambda *a: (print('got SIGTERM', flush=True), sys.exit(0)))
os.system('(sleep 0.2 &); exit 0')  # leaves an orphan for process 1
time.sleep(1)
zombies = sum(
    1 for p in os.listdir('/proc')
    if p.isdigit() and open(f'/proc/{p}/stat').read().split()[2] == 'Z'
)
print('zombies', zombies, flush=True)
time.sleep(60)
"""


class SandboxLaunchTest(AOPTestCase):
    def test_options_split_by_their_arguments(self) -> None:
        options, tail = sandbox_launch.split_options(
            ["bwrap", "--setenv", "A", "--", "--clearenv", "--", "cmd", "--"]
        )
        assert options == ["--setenv", "A", "--", "--clearenv"]
        assert tail == ["cmd", "--"]
        with raises(ValueError, match="unexpected bubblewrap option"):
            sandbox_launch.split_options(["bwrap", "--share-net", "--", "cmd"])

    def test_timeout_reaches_the_harness_and_process_one_hides_the_sandbox(
        self, tmp_path: Path
    ) -> None:
        if shutil.which("bwrap") is None:
            skip("bwrap is not installed")
        (tmp_path / "probe.py").write_text(PROBE)
        command = [
            *_isolated_root_command("bwrap"),
            "--ro-bind",
            os.fspath(tmp_path),
            "/probe",
            "--",
            "/usr/bin/python3",
            "-u",
            "/probe/probe.py",
        ]

        started = time.monotonic()
        capture = _capture_process(
            command,
            cwd=tmp_path,
            environment=dict(os.environ),
            prompt=None,
            timeout_seconds=3,
            is_response=lambda line: True,
        )
        elapsed = time.monotonic() - started

        lines = dict(line.split(" ", 1) for line in capture.stdout.splitlines() if " " in line)
        assert capture.timed_out
        # The harness, not bubblewrap, received the stop signal and exited.
        assert "got SIGTERM" in capture.stdout, capture.stdout
        assert elapsed < 3 + 2, elapsed
        # Bubblewrap's reaper is process 1, the harness runs beside it, and
        # orphans are collected.
        assert lines["pid"] == "2"
        assert lines["zombies"] == "0"
        # Process 1's command line names no mount or host path.
        assert "--ro-bind" not in lines["pid1"]
        assert os.fspath(tmp_path) not in lines["pid1"]
        assert sys.executable not in lines["pid1"]
