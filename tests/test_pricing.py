from __future__ import annotations

import pytest

from agent_orchestration_process.pricing import TokenUsage, estimate_api_cost


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.6-sol", 0.01055),
        ("gpt-5.6-terra", 0.00422),
        ("gpt-5.6-luna", 0.000422),
        ("gpt-5.5-2026-04-23", 0.01055),
        ("gpt-5.4-mini-2026-03-17", 0.0015825),
    ],
)
def test_estimated_standard_api_cost(model: str, expected: float) -> None:
    usage = TokenUsage(
        input_tokens=1_000,
        cached_input_tokens=100,
        output_tokens=200,
        reasoning_output_tokens=50,
    )

    estimate = estimate_api_cost(model, usage)

    assert estimate is not None
    assert estimate.amount_usd == expected
    assert estimate.pricing_version.startswith("models-dev-")
    assert estimate.pricing_source == "https://models.dev/api.json"
    assert estimate.pricing_retrieved_at is not None
    assert estimate.long_context_pricing is False


def test_reasoning_tokens_are_not_double_counted() -> None:
    without_reasoning = estimate_api_cost(
        "gpt-5.6-sol", TokenUsage(input_tokens=100, output_tokens=100)
    )
    with_reasoning = estimate_api_cost(
        "gpt-5.6-sol",
        TokenUsage(input_tokens=100, output_tokens=100, reasoning_output_tokens=80),
    )

    assert without_reasoning is not None
    assert with_reasoning is not None
    assert with_reasoning.amount_usd == without_reasoning.amount_usd


def test_total_tokens_contains_only_input_and_output_totals() -> None:
    usage = TokenUsage(
        input_tokens=100,
        cached_input_tokens=80,
        output_tokens=50,
        reasoning_output_tokens=40,
    )

    assert usage.total_tokens == 150


def test_normalized_cached_input_preserves_legacy_additive_price() -> None:
    estimate = estimate_api_cost(
        "gemini-3.5-flash-low",
        TokenUsage(
            input_tokens=400,
            cached_input_tokens=300,
            output_tokens=20,
            reasoning_output_tokens=7,
        ),
        providers=("google",),
        catalog_model="gemini-3.5-flash",
    )

    assert estimate is not None
    assert estimate.amount_usd == 0.000375
    assert estimate.model == "gemini-3.5-flash-low"
    assert estimate.priced_as == "gemini-3.5-flash"


def test_token_usage_rejects_overlapping_subsets_larger_than_totals() -> None:
    with pytest.raises(ValueError, match="cached_input_tokens"):
        TokenUsage(input_tokens=10, cached_input_tokens=11)
    with pytest.raises(ValueError, match="reasoning_output_tokens"):
        TokenUsage(output_tokens=10, reasoning_output_tokens=11)


def test_long_context_multiplier_is_applied_when_documented() -> None:
    estimate = estimate_api_cost(
        "gpt-5.5",
        TokenUsage(
            input_tokens=300_000, cached_input_tokens=100_000, output_tokens=1_000
        ),
    )

    assert estimate is not None
    assert estimate.long_context_pricing is True
    assert estimate.amount_usd == 2.145


def test_repeated_short_requests_do_not_trigger_long_context_multiplier() -> None:
    requests = (
        TokenUsage(
            input_tokens=150_000, cached_input_tokens=100_000, output_tokens=1_000
        ),
        TokenUsage(
            input_tokens=150_000, cached_input_tokens=100_000, output_tokens=1_000
        ),
    )
    aggregate = TokenUsage(
        input_tokens=300_000, cached_input_tokens=200_000, output_tokens=2_000
    )

    estimate = estimate_api_cost("gpt-5.5", aggregate, request_usages=requests)

    assert estimate is not None
    assert estimate.long_context_pricing is False
    assert estimate.amount_usd == 0.66


def test_mixed_context_tiers_are_priced_per_request() -> None:
    requests = (
        TokenUsage(
            input_tokens=100_000, cached_input_tokens=20_000, output_tokens=1_000
        ),
        TokenUsage(
            input_tokens=300_000, cached_input_tokens=100_000, output_tokens=1_000
        ),
    )
    aggregate = TokenUsage(
        input_tokens=400_000, cached_input_tokens=120_000, output_tokens=2_000
    )

    estimate = estimate_api_cost("gpt-5.5", aggregate, request_usages=requests)

    assert estimate is not None
    assert estimate.long_context_pricing is True
    assert estimate.amount_usd == 2.585


def test_request_usage_must_match_aggregate_usage() -> None:
    with pytest.raises(ValueError, match="request usage does not match"):
        estimate_api_cost(
            "gpt-5.5",
            TokenUsage(input_tokens=10),
            request_usages=(TokenUsage(input_tokens=9),),
        )


def test_unknown_or_implicit_model_has_no_cost_estimate() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=10)

    assert estimate_api_cost(None, usage) is None
    assert estimate_api_cost("future-model", usage) is None


@pytest.mark.parametrize(
    "metadata", [None, {"name": "Flash"}, {"cost": {"input": 0.3}}]
)
def test_deepseek_official_fallback_prices_and_provenance(monkeypatch, metadata):
    from dataclasses import replace
    from types import SimpleNamespace
    from agent_orchestration_process import pricing, model_listing
    from agent_orchestration_process.model_catalog import ensure_catalog_fresh

    monkeypatch.setattr(
        pricing,
        "time",
        SimpleNamespace(time=lambda: pricing.DEEPSEEK_PRICE_VERIFIED_AT),
    )
    catalog = ensure_catalog_fresh()
    models = {} if metadata is None else {"deepseek-flash": metadata}
    catalog = replace(catalog, providers={"deepseek": {"models": models}})
    usage = TokenUsage(1000, 100, 200)
    cost = estimate_api_cost("deepseek-flash", usage, catalog, providers=("deepseek",))
    assert cost.amount_usd == 0.0005106
    assert cost.pricing_basis == "peak"
    assert cost.pricing_source == pricing.DEEPSEEK_PRICE_SOURCE
    assert cost.pricing_retrieved_at == pricing.DEEPSEEK_PRICE_VERIFIED_AT
    assert cost.pricing_version == "deepseek-official-2026-09-10-peak"
    assert pricing.CalculatedCost.from_dict(cost.to_dict()) == cost
    row = model_listing._record(
        "dsh",
        "deepseek-flash",
        "Flash",
        "installed-default",
        "api-equivalent",
        catalog,
        "deepseek",
        "deepseek-flash",
    )
    assert (
        row.input_per_million_usd,
        row.cached_input_per_million_usd,
        row.output_per_million_usd,
    ) == (0.3, 0.006, 1.2)
    assert row.price_scope == "api-equivalent-peak"
    assert row.pricing_source == cost.pricing_source
    assert row.pricing_retrieved_at == cost.pricing_retrieved_at
    assert catalog.providers["deepseek"]["models"] == models
    assert (
        estimate_api_cost("deepseek-flash", usage, catalog, providers=("openai",))
        is None
    )
    assert (
        estimate_api_cost("deepseek-v4-flash", usage, catalog, providers=("deepseek",))
        is None
    )


def test_deepseek_catalog_prices_take_precedence(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from agent_orchestration_process import pricing
    from agent_orchestration_process.model_catalog import ensure_catalog_fresh

    monkeypatch.setattr(
        pricing,
        "time",
        SimpleNamespace(time=lambda: pricing.DEEPSEEK_PRICE_VERIFIED_AT),
    )
    catalog = replace(
        ensure_catalog_fresh(),
        providers={
            "deepseek": {
                "models": {
                    "deepseek-flash": {
                        "cost": {"input": 2, "output": 3, "cache_read": 1}
                    }
                }
            }
        },
    )
    cost = estimate_api_cost(
        "deepseek-flash", TokenUsage(1000, 100, 200), catalog, providers=("deepseek",)
    )
    assert cost.amount_usd == 0.0025
    assert cost.pricing_source == catalog.source
    assert cost.pricing_basis is None


@pytest.mark.parametrize("offset", [-1, 7 * 24 * 3600])
def test_deepseek_fallback_does_not_claim_unverified_or_expired_prices(
    monkeypatch, offset
):
    from dataclasses import replace
    from types import SimpleNamespace
    from agent_orchestration_process import pricing
    from agent_orchestration_process.model_catalog import ensure_catalog_fresh

    monkeypatch.setattr(
        pricing,
        "time",
        SimpleNamespace(time=lambda: pricing.DEEPSEEK_PRICE_VERIFIED_AT + offset),
    )
    catalog = replace(ensure_catalog_fresh(), providers={"deepseek": {"models": {}}})
    assert (
        estimate_api_cost(
            "deepseek-flash", TokenUsage(1000), catalog, providers=("deepseek",)
        )
        is None
    )


def test_dsh_run_persists_official_fallback(repository, fake_dsh, monkeypatch):
    from types import SimpleNamespace
    from agent_orchestration_process import pricing
    from agent_orchestration_process.runner import AgentRunner, DeepSeekHarnessAdapter
    from agent_orchestration_process.worktrees import WorktreeManager

    monkeypatch.setattr(
        pricing,
        "time",
        SimpleNamespace(time=lambda: pricing.DEEPSEEK_PRICE_VERIFIED_AT),
    )
    fake_dsh.write_text(
        fake_dsh.read_text().replace(
            '"deepseek-v4-flash"', '"deepseek-flash"'
        )
    )
    runner = AgentRunner(
        WorktreeManager.discover(repository), DeepSeekHarnessAdapter(str(fake_dsh))
    )
    result = runner.run(task="flash-pricing", prompt="test")
    assert result.succeeded, result.error
    assert result.model == "deepseek-flash"
    for cost in (
        result.calculated_cost,
        runner.store.load_result(result.run_id).calculated_cost,
    ):
        assert cost.amount_usd == 0.00005118
        assert cost.pricing_basis == "peak"
        assert cost.pricing_source == pricing.DEEPSEEK_PRICE_SOURCE
        assert cost.pricing_retrieved_at == pricing.DEEPSEEK_PRICE_VERIFIED_AT
