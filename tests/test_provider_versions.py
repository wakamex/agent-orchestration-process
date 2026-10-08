from __future__ import annotations

import subprocess

from support import AOPTestCase, MonkeyPatch, parametrize, raises

from agent_orchestration_process import provider_versions
from agent_orchestration_process.worktrees import AOPError


class ProviderVersionsTest(AOPTestCase):
    @parametrize("output", ["1.1.16", "agy 1.1.19", "2.0.0-rc.1"])
    def test_supported_agy_versions_are_accepted(
        self,
        monkeypatch: MonkeyPatch, output: str
    ) -> None:
        monkeypatch.setattr(
            provider_versions.subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
        )

        assert provider_versions.require_supported_agy("agy") in output

    @parametrize("output", ["1.1.15", "1.1.16-rc.1"])
    def test_unsupported_agy_versions_report_the_required_floor(
        self,
        monkeypatch: MonkeyPatch, output: str
    ) -> None:
        monkeypatch.setattr(
            provider_versions.subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
        )

        with raises(AOPError, match=r"AOP requires Agy 1\.1\.16 or newer"):
            provider_versions.require_supported_agy("agy")

    def test_unrecognized_agy_version_fails_closed(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            provider_versions.subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args[0], 0, "development", ""
            ),
        )

        with raises(AOPError, match="Agy returned an invalid version"):
            provider_versions.require_supported_agy("agy")
