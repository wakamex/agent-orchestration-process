from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import model_prices
import pytest

from agent_orchestration_process import cli, model_listing
from agent_orchestration_process.model_catalog import ModelCatalog
from agent_orchestration_process.worktrees import AOPError


def catalog() -> ModelCatalog:
    return ModelCatalog(
        {
            "anthropic": {
                "models": {
                    "claude-sonnet-5": {},
                    "claude-catalog-only": {},
                }
            }
        },
        100,
        "https://models.dev/api.json",
        "catalog-hash",
    )


@pytest.fixture
def native_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    binary = tmp_path / "claude"
    binary.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
assert sys.argv[1:] == ["--print", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose", "--no-session-persistence", "--strict-mcp-config", "--settings", '{"disableAllHooks":true}']
request = json.loads(sys.stdin.readline())
assert request == {"type": "control_request", "request_id": "aop-models", "request": {"subtype": "initialize"}}
# Native configuration and provider selection must reach the native CLI intact.
assert os.environ["ANTHROPIC_BASE_URL"] == "https://route.invalid"
assert os.environ["CLAUDE_CONFIG_DIR"] == os.environ["AOP_TEST_CLAUDE_HOME"]
mode = os.environ.get("AOP_TEST_CLAUDE_MODE", "success")
if mode == "exit":
    sys.exit(1)
entries = [{"value": "opus", "resolvedModel": "claude-launch", "displayName": "Launch model"}, {"value": "sonnet", "resolvedModel": "claude-sonnet-5"}, {"value": "default", "resolvedModel": "claude-launch"}, {"value": "legacy-alias"}]
if mode == "invalid":
    entries = [{"value": "opus", "resolvedModel": 123}]
response = {"type": "control_response", "response": {"request_id": "aop-models", "subtype": "success", "response": {"models": entries}}}
# A notification and response in the same write exercise buffered framing.
print(json.dumps({"type": "system"}) + "\\n" + json.dumps(response), flush=True)
# No user turn should ever be submitted by inventory discovery.
for line in sys.stdin:
    raise AssertionError("unexpected request: " + line)
"""
    )
    binary.chmod(0o700)
    monkeypatch.setenv("AOP_CLAUDE_BIN", os.fspath(binary))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://route.invalid")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", os.fspath(tmp_path))
    monkeypatch.setenv("AOP_TEST_CLAUDE_HOME", os.fspath(tmp_path))
    return binary


def test_native_inventory_adds_launch_models_without_prices(native_cli: Path) -> None:
    rows = {row.model: row for row in model_listing.list_models("claude", catalog())}
    launch = rows["claude-launch"]
    assert launch.availability == "native-advertised"
    assert launch.inventory_source == "claude-sdk-initialize"
    assert launch.inventory_retrieved_at
    assert len(launch.inventory_sha256 or "") == 64
    assert launch.authenticated is False  # Advertisement is not entitlement proof.
    assert launch.price_scope == "unknown"
    assert launch.input_per_million_usd is None
    known = model_prices.rates("claude-sonnet-5", provider="anthropic")
    assert rows["claude-sonnet-5"].input_per_million_usd == known.input
    assert rows["claude-sonnet-5"].availability == "native-advertised"
    assert rows["legacy-alias"].price_scope == "unknown"
    fallback = rows["claude-catalog-only"]
    assert fallback.availability == "catalog"
    assert fallback.inventory_source == "https://models.dev/api.json"
    assert fallback.inventory_sha256 == "catalog-hash"
    assert len(rows) == 4


@pytest.mark.parametrize("mode", ["exit", "invalid"])
def test_native_discovery_failure_is_visible_in_catalog_fallback(
    native_cli: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setenv("AOP_TEST_CLAUDE_MODE", mode)
    rows = model_listing.list_models("claude", catalog())
    assert rows
    assert all(row.availability == "catalog" and row.discovery_error for row in rows)
    with pytest.raises(AOPError):
        model_listing.list_models("claude", ModelCatalog({}, 0, "", ""))


@pytest.mark.parametrize("refresh", [False, True])
def test_listing_survives_catalog_failure_and_queries_native_each_time(
    native_cli: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    refresh: bool,
) -> None:
    calls = []

    def unavailable(**options: object) -> ModelCatalog:
        calls.append(options)
        raise AOPError("pricing unavailable")

    monkeypatch.setattr(cli, "ensure_catalog_fresh", unavailable)
    args = ["models", "--agent", "claude", "--json"] + (
        ["--refresh"] if refresh else []
    )
    assert cli.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert calls == [{"force": refresh}]
    assert output["catalog"] is None
    assert output["warnings"] == {"catalog": "pricing unavailable"}
    assert output["errors"] == {}
    # Prices come from model-prices, so they survive a catalog failure.
    scopes = {row["model"]: row["price_scope"] for row in output["models"]}
    assert scopes["claude-launch"] == "unknown"
    assert scopes["claude-sonnet-5"] != "unknown"
    assert output["pricing_basis"] == model_prices.pricing_basis()
    monkeypatch.setenv("AOP_TEST_CLAUDE_MODE", "exit")
    assert cli.main(args) == 1
    output = json.loads(capsys.readouterr().out)
    assert "claude" in output["errors"]
    assert output["models"] == []


def test_catalog_fallback_warning_in_json_and_text(
    native_cli: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("AOP_TEST_CLAUDE_MODE", "exit")
    monkeypatch.setattr(cli, "ensure_catalog_fresh", lambda **kwargs: catalog())
    assert cli.main(["models", "--agent", "claude", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["warnings"]["claude"]
    assert cli.main(["models", "--agent", "claude"]) == 0
    assert "warning: claude:" in capsys.readouterr().err


@pytest.mark.parametrize("refresh", [False, True])
def test_models_refreshes_prices_and_reports_their_warnings(
    native_cli: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    refresh: bool,
) -> None:
    calls = []

    def refresh_prices(**options: object) -> list[str]:
        calls.append(options)
        return ["price data was last confirmed current 9 days ago"]

    monkeypatch.setattr(cli.pricing, "refresh_prices", refresh_prices)
    monkeypatch.setattr(cli, "ensure_catalog_fresh", lambda **kwargs: catalog())
    args = ["models", "--agent", "claude", "--json"] + (["--refresh"] if refresh else [])
    assert cli.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert calls == [{"force": refresh}]
    assert output["warnings"]["prices"] == (
        "price data was last confirmed current 9 days ago"
    )


def test_dispatch_refreshes_prices_and_warns_without_blocking(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)
    monkeypatch.setattr(
        cli.pricing, "refresh_prices", lambda **options: ["could not fetch price data"]
    )
    cli.main(["integrate", "no-such-task"])
    assert "aop: warning: could not fetch price data" in capsys.readouterr().err
