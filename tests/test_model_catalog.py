from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from aop import cli, model_catalog, model_listing
from aop.model_catalog import ModelCatalog, ensure_catalog_fresh
from aop.worktrees import AOPError


def test_fresh_catalog_does_not_contact_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_catalog,
        "_download_catalog",
        lambda: pytest.fail("fresh cache should not be refreshed"),
    )

    catalog = ensure_catalog_fresh()

    assert catalog.model("openai", "gpt-5.6-sol") is not None


def test_stale_catalog_is_refreshed_and_persisted(
    fresh_model_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = json.loads(fresh_model_catalog.read_text())
    cached["fetched_at"] = time.time() - model_catalog.CATALOG_TTL_SECONDS - 1
    fresh_model_catalog.write_text(json.dumps(cached))
    replacement = ModelCatalog(
        providers={"new": {"models": {}}},
        fetched_at=time.time(),
        source=model_catalog.CATALOG_URL,
        sha256=model_catalog._catalog_hash({"new": {"models": {}}}),
    )
    monkeypatch.setattr(model_catalog, "_download_catalog", lambda: replacement)

    actual = ensure_catalog_fresh()

    assert actual == replacement
    assert json.loads(fresh_model_catalog.read_text())["providers"] == {
        "new": {"models": {}}
    }


def test_stale_catalog_fails_closed_when_refresh_fails(
    fresh_model_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = json.loads(fresh_model_catalog.read_text())
    cached["fetched_at"] = time.time() - model_catalog.CATALOG_TTL_SECONDS - 1
    fresh_model_catalog.write_text(json.dumps(cached))

    def fail() -> ModelCatalog:
        raise OSError("offline")

    monkeypatch.setattr(model_catalog, "_download_catalog", fail)

    with pytest.raises(AOPError, match="cached data is 24.0 hours old: offline"):
        ensure_catalog_fresh()


def test_every_cli_invocation_runs_the_freshness_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        cli,
        "ensure_catalog_fresh",
        lambda **options: calls.append(options) or object(),
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])

    assert exit_info.value.code == 0
    assert calls == [{"force": False}]


def test_native_model_parsers_and_pricing(
    fresh_model_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = ensure_catalog_fresh()
    monkeypatch.setattr(model_listing, "_require_binary", lambda binary: None)

    monkeypatch.setattr(
        model_listing,
        "_codex_model_response",
        lambda binary: {
            "id": 2,
            "result": {
                "data": [
                    {"model": "gpt-5.6-sol", "displayName": "GPT-5.6 Sol"}
                ]
            },
        },
    )

    def run(command: list[str], *, input: str | None = None) -> str:
        if command[0] == "agent":
            return "Available models\n\ngpt-5.6-sol-high - Sol High\n"
        if command[0] == "agy":
            return "Fetching available models...\ngemini-3.5-flash-low\tGemini Flash\n"
        if command[0] == "opencode":
            return "opencode/deepseek-v4-flash\n"
        if command[0] == "devin":
            return json.dumps(
                {
                    "families": [
                        {
                            "family_label": "SWE-1.7",
                            "variants": [
                                {
                                    "model_uid": "swe-1-7",
                                    "label": "SWE-1.7 Max",
                                    "cost_tier": "Free",
                                },
                                {
                                    "model_uid": "swe-1-7-lightning",
                                    "label": "SWE-1.7 Lightning Max",
                                    "cost_tier": "Med cost",
                                    "cost_summary": (
                                        "$2.5 / MTok In · $12.5 / MTok Out"
                                    ),
                                },
                            ],
                        }
                    ]
                }
            )
        raise AssertionError(command)

    monkeypatch.setattr(model_listing, "_run", run)

    codex = model_listing.list_models("codex", catalog)[0]
    cursor = model_listing.list_models("cursor", catalog)[0]
    agy = model_listing.list_models("agy", catalog)[0]
    devin = model_listing.list_models("devin", catalog)
    opencode = model_listing.list_models("opencode", catalog)[0]

    assert codex.input_per_million_usd == 5
    assert codex.price_scope == "api-equivalent"
    assert cursor.input_per_million_usd == 5
    assert agy.availability == "account"
    assert devin[0].model == "swe-1-7"
    assert devin[0].input_per_million_usd == 0
    assert devin[0].output_per_million_usd == 0
    assert devin[1].input_per_million_usd == 2.5
    assert devin[1].output_per_million_usd == 12.5
    assert devin[1].pricing_source == "Devin CLI account model inventory"
    assert opencode.availability == "account"


def test_models_json_reports_catalog_provenance(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli,
        "list_models",
        lambda agent, catalog: [
            model_listing.AvailableModel(
                agent=agent,
                model="example",
                name="Example",
                availability="account",
                price_scope="unknown",
            )
        ],
    )

    assert cli.main(["models", "--agent", "codex", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["models"][0]["model"] == "example"
    assert output["catalog"]["source"] == model_catalog.CATALOG_URL
    assert len(output["catalog"]["sha256"]) == 64


def test_hermes_uses_live_provider_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_listing, "_require_binary", lambda binary: None)
    monkeypatch.setattr(model_listing, "_run", lambda command, **options: "nous\n")
    monkeypatch.setattr(
        model_listing,
        "_fetch_nous_models",
        lambda: [
            {
                "id": "deepseek/deepseek-v4-flash-0731",
                "name": "DeepSeek V4 Flash 0731",
                "pricing": {
                    "prompt": "0.000000009",
                    "completion": "0.000000018",
                    "input_cache_read": "0.0000000018",
                },
            }
        ],
    )

    model = model_listing.list_models("hermes", ensure_catalog_fresh())[0]

    assert model.availability == "account-endpoint"
    assert model.price_scope == "provider"
    assert model.input_per_million_usd == 0.009
    assert model.output_per_million_usd == 0.018
    assert model.pricing_source == model_listing.NOUS_MODELS_URL


def test_hermes_catalog_follows_the_configured_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_listing, "_require_binary", lambda binary: None)
    monkeypatch.setattr(
        model_listing, "_run", lambda command, **options: "xai-oauth\n"
    )

    model = model_listing.list_models("hermes", ensure_catalog_fresh())[0]

    assert model.model == "grok-4.5"
    assert model.availability == "catalog"
    assert model.price_scope == "api-equivalent"
    assert model.input_per_million_usd == 2
    assert model.output_per_million_usd == 6
