"""Minimum supported versions for native agent harnesses."""

from __future__ import annotations

import re
import subprocess

from .worktrees import AOPError


AGY_MINIMUM_VERSION = (1, 1, 16)
AGY_MINIMUM_VERSION_TEXT = ".".join(str(part) for part in AGY_MINIMUM_VERSION)
_SEMVER = re.compile(
    r"(?<!\d)(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?"
)


def require_supported_agy(binary: str) -> str:
    try:
        result = subprocess.run(
            [binary, "--version"],
            text=True,
            capture_output=True,
            timeout=5,
        )
    except OSError as error:
        raise AOPError(f"could not query Agy version: {error}") from error
    except subprocess.TimeoutExpired as error:
        raise AOPError("Agy version check timed out") from error
    if result.returncode:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise AOPError(f"could not query Agy version: {detail}")

    output = result.stdout.strip() or result.stderr.strip()
    match = _SEMVER.search(output)
    if match is None:
        raise AOPError("Agy returned an invalid version")
    version = tuple(int(match.group(name)) for name in ("major", "minor", "patch"))
    prerelease = match.group("prerelease")
    if version < AGY_MINIMUM_VERSION or (
        version == AGY_MINIMUM_VERSION and prerelease is not None
    ):
        detected = match.group(0)
        raise AOPError(
            f"Agy {detected} is unsupported; AOP requires Agy "
            f"{AGY_MINIMUM_VERSION_TEXT} or newer"
        )
    return match.group(0)
