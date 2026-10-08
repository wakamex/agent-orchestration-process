from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from support import AOPTestCase

from agent_orchestration_process import __version__


class EntryPointsTest(AOPTestCase):
    def test_console_script_and_module_report_the_version(self) -> None:
        for command in (
            [str(Path(sys.executable).with_name("aop")), "--version"],
            [sys.executable, "-m", "agent_orchestration_process", "--version"],
        ):
            with self.subTest(command=command[0]):
                completed = subprocess.run(
                    command, capture_output=True, text=True, check=True
                )
                self.assertIn(__version__, completed.stdout)
