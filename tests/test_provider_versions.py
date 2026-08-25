from __future__ import annotations

import subprocess

import pytest

from agent_orchestration_process import provider_versions
from agent_orchestration_process.worktrees import AOPError


@pytest.mark.parametrize("output", ["1.1.16", "agy 1.1.19", "2.0.0-rc.1"])
def test_supported_agy_versions_are_accepted(
    monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    monkeypatch.setattr(
        provider_versions.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )

    assert provider_versions.require_supported_agy("agy") in output


@pytest.mark.parametrize("output", ["1.1.15", "1.1.16-rc.1"])
def test_unsupported_agy_versions_report_the_required_floor(
    monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    monkeypatch.setattr(
        provider_versions.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )

    with pytest.raises(AOPError, match=r"AOP requires Agy 1\.1\.16 or newer"):
        provider_versions.require_supported_agy("agy")


def test_unrecognized_agy_version_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        provider_versions.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "development", ""
        ),
    )

    with pytest.raises(AOPError, match="Agy returned an invalid version"):
        provider_versions.require_supported_agy("agy")
