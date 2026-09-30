from __future__ import annotations

import llm_prices
import pytest

from agent_orchestration_process.pricing import (
    CalculatedCost,
    TokenUsage,
    estimate_api_cost,
)

# GPT-5.6 Sol cost $5/$30 until 2026-08-22 and $4/$20 after.
BEFORE_SOL_CUT = "2026-08-20T12:00:00+00:00"
AFTER_SOL_CUT = "2026-09-01T12:00:00+00:00"
# DeepSeek peak hours include 01:00-04:00 UTC on weekdays; 2026-09-30 is a Wednesday.
DEEPSEEK_PEAK = "2026-09-30T02:00:00+00:00"
DEEPSEEK_OFF_PEAK = "2026-09-30T05:00:00+00:00"


def cost_at(model, usage, at, *, provider="openai", finished_at=None, requests=None):
    return estimate_api_cost(
        model,
        usage,
        provider=provider,
        started_at=at,
        finished_at=finished_at or at,
        requests=requests,
    )


@pytest.mark.parametrize(
    ("at", "expected"),
    [(BEFORE_SOL_CUT, 0.01055), (AFTER_SOL_CUT, 0.00764)],
)
def test_cost_uses_rates_in_effect_when_the_run_happened(at, expected) -> None:
    usage = TokenUsage(
        input_tokens=1_000,
        cached_input_tokens=100,
        output_tokens=200,
        reasoning_output_tokens=50,
    )

    estimate = cost_at("gpt-5.6-sol", usage, at)

    assert estimate is not None
    assert estimate.amount_usd == expected
    assert estimate.priced_as == "openai/gpt-5.6-sol"
    assert estimate.pricing_basis == llm_prices.pricing_basis()
    assert estimate.pricing_version.startswith("llm-prices-")
    assert estimate.pricing_period is None
    assert estimate.long_context_pricing is False
    assert CalculatedCost.from_dict(estimate.to_dict()) == estimate


def test_run_spanning_a_price_change_without_request_times_is_unpriced() -> None:
    usage = TokenUsage(input_tokens=1_000, output_tokens=10)

    assert (
        cost_at("gpt-5.6-sol", usage, BEFORE_SOL_CUT, finished_at=AFTER_SOL_CUT) is None
    )


def test_same_prices_from_a_new_source_are_not_a_price_change() -> None:
    # A correction ends where models.dev recorded the same price.
    usage = TokenUsage(input_tokens=1_000, output_tokens=10)

    estimate = cost_at(
        "gpt-5.6-sol",
        usage,
        "2026-08-25T04:00:00+00:00",
        finished_at="2026-08-25T05:00:00+00:00",
    )

    assert estimate is not None
    assert estimate.amount_usd == 0.0042


@pytest.mark.parametrize(
    ("at", "period", "expected"),
    [(DEEPSEEK_PEAK, "peak", 0.0005106), (DEEPSEEK_OFF_PEAK, "off_peak", 0.0002553)],
)
def test_deepseek_time_of_day_pricing(at, period, expected) -> None:
    estimate = cost_at(
        "deepseek-flash", TokenUsage(1000, 100, 200), at, provider="deepseek"
    )

    assert estimate is not None
    assert estimate.amount_usd == expected
    assert estimate.pricing_period == period


def test_run_crossing_a_peak_window_is_priced_only_per_request() -> None:
    usage = TokenUsage(1000, 100, 200)
    start, finish = "2026-09-30T00:50:00+00:00", "2026-09-30T01:10:00+00:00"

    assert (
        cost_at("deepseek-flash", usage, start, provider="deepseek", finished_at=finish)
        is None
    )
    # A run covering a whole peak window has off-peak endpoints but is still unpriced.
    assert (
        cost_at(
            "deepseek-flash",
            usage,
            "2026-09-30T00:30:00+00:00",
            provider="deepseek",
            finished_at="2026-09-30T04:30:00+00:00",
        )
        is None
    )
    first, second = TokenUsage(400, 0, 100), TokenUsage(600, 100, 100)
    estimate = cost_at(
        "deepseek-flash",
        usage,
        start,
        provider="deepseek",
        finished_at=finish,
        requests=[(first, "2026-09-30T00:55:00Z"), (second, "2026-09-30T01:05:00Z")],
    )
    assert estimate is not None
    assert estimate.pricing_period == "mixed"
    assert estimate.amount_usd == round(
        llm_prices.cost(
            "deepseek-flash",
            input_tokens=400,
            output_tokens=100,
            at="2026-09-30T00:55:00Z",
        )
        + llm_prices.cost(
            "deepseek-flash",
            input_tokens=500,
            cache_read_tokens=100,
            output_tokens=100,
            at="2026-09-30T01:05:00Z",
        ),
        8,
    )


def test_reasoning_tokens_are_not_double_counted() -> None:
    without_reasoning = cost_at(
        "gpt-5.6-sol", TokenUsage(input_tokens=100, output_tokens=100), AFTER_SOL_CUT
    )
    with_reasoning = cost_at(
        "gpt-5.6-sol",
        TokenUsage(input_tokens=100, output_tokens=100, reasoning_output_tokens=80),
        AFTER_SOL_CUT,
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


def test_harness_model_names_resolve_to_the_lab_price() -> None:
    estimate = cost_at(
        "gemini-3.5-flash-low",
        TokenUsage(
            input_tokens=400,
            cached_input_tokens=300,
            output_tokens=20,
            reasoning_output_tokens=7,
        ),
        AFTER_SOL_CUT,
        provider="google",
    )

    assert estimate is not None
    assert estimate.amount_usd == 0.000375
    assert estimate.model == "gemini-3.5-flash-low"
    assert estimate.priced_as == "google/gemini-3.5-flash"
    assert estimate.removed_suffix == "-low"


def test_zai_flash_uses_the_sourced_correction() -> None:
    estimate = cost_at(
        "glm-5.3-flash",
        TokenUsage(1000, 100, 200),
        "2026-09-12T00:00:00+00:00",
        provider="zai",
    )

    assert estimate is not None
    assert estimate.amount_usd == 0.000238
    assert estimate.pricing_source == "https://docs.z.ai/guides/overview/pricing"


def test_token_usage_rejects_overlapping_subsets_larger_than_totals() -> None:
    with pytest.raises(ValueError, match="cached_input_tokens"):
        TokenUsage(input_tokens=10, cached_input_tokens=11)
    with pytest.raises(ValueError, match="reasoning_output_tokens"):
        TokenUsage(output_tokens=10, reasoning_output_tokens=11)


def test_long_context_tier_is_applied_per_request() -> None:
    long_request = TokenUsage(
        input_tokens=300_000, cached_input_tokens=100_000, output_tokens=1_000
    )

    estimate = cost_at("gpt-5.5", long_request, AFTER_SOL_CUT)

    assert estimate is not None
    assert estimate.long_context_pricing is True
    assert estimate.amount_usd == 2.145


def test_repeated_short_requests_do_not_trigger_long_context_tier() -> None:
    request = TokenUsage(
        input_tokens=150_000, cached_input_tokens=100_000, output_tokens=1_000
    )
    aggregate = TokenUsage(
        input_tokens=300_000, cached_input_tokens=200_000, output_tokens=2_000
    )

    estimate = cost_at(
        "gpt-5.5",
        aggregate,
        AFTER_SOL_CUT,
        requests=[(request, None), (request, None)],
    )

    assert estimate is not None
    assert estimate.long_context_pricing is False
    assert estimate.amount_usd == 0.66


def test_mixed_context_tiers_are_priced_per_request() -> None:
    requests = [
        (
            TokenUsage(
                input_tokens=100_000, cached_input_tokens=20_000, output_tokens=1_000
            ),
            None,
        ),
        (
            TokenUsage(
                input_tokens=300_000, cached_input_tokens=100_000, output_tokens=1_000
            ),
            None,
        ),
    ]
    aggregate = TokenUsage(
        input_tokens=400_000, cached_input_tokens=120_000, output_tokens=2_000
    )

    estimate = cost_at("gpt-5.5", aggregate, AFTER_SOL_CUT, requests=requests)

    assert estimate is not None
    assert estimate.long_context_pricing is True
    assert estimate.amount_usd == 2.585


def test_request_usage_must_match_aggregate_usage() -> None:
    with pytest.raises(ValueError, match="request usage does not match"):
        cost_at(
            "gpt-5.5",
            TokenUsage(input_tokens=10),
            AFTER_SOL_CUT,
            requests=[(TokenUsage(input_tokens=9), None)],
        )


def test_zero_measured_requests_still_require_a_known_model() -> None:
    estimate = cost_at("gpt-5.5", TokenUsage(), AFTER_SOL_CUT, requests=[])

    assert estimate is not None
    assert estimate.amount_usd == 0
    assert cost_at("future-model", TokenUsage(), AFTER_SOL_CUT, requests=[]) is None


def test_unknown_or_implicit_model_has_no_cost_estimate() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=10)

    assert cost_at(None, usage, AFTER_SOL_CUT) is None
    assert cost_at("future-model", usage, AFTER_SOL_CUT) is None


def test_listing_shows_current_rates_and_time_of_day_period() -> None:
    from agent_orchestration_process import model_listing
    from agent_orchestration_process.model_catalog import ModelCatalog

    row = model_listing._record(
        "dsh",
        "deepseek-flash",
        "Flash",
        "installed-default",
        "api-equivalent",
        ModelCatalog({}, 0, "", ""),
        "deepseek",
        "deepseek-flash",
    )
    rates = llm_prices.rates("deepseek-flash", provider="deepseek")

    assert (
        row.input_per_million_usd,
        row.cached_input_per_million_usd,
        row.output_per_million_usd,
    ) == (rates.input, rates.cache_read, rates.output)
    assert row.price_scope == f"api-equivalent-{rates.period.replace('_', '-')}"
    assert row.pricing_source == rates.source


def test_legacy_calculated_cost_records_still_load() -> None:
    legacy = {
        "amount_usd": 0.1,
        "currency": "USD",
        "model": "deepseek-flash",
        "priced_as": "deepseek-flash",
        "pricing_version": "deepseek-official-2026-09-10-peak",
        "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing/",
        "long_context_pricing": False,
        "pricing_retrieved_at": 1789000000.0,
        "pricing_basis": "peak",
    }

    loaded = CalculatedCost.from_dict(legacy)

    assert loaded is not None
    assert loaded.to_dict() == {
        **legacy,
        "pricing_period": None,
        "removed_suffix": None,
    }
