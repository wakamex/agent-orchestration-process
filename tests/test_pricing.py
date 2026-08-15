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


def test_unknown_or_implicit_model_has_no_cost_estimate() -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=10)

    assert estimate_api_cost(None, usage) is None
    assert estimate_api_cost("future-model", usage) is None
