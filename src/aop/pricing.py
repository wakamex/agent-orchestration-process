"""Versioned estimates of standard API-equivalent token cost."""

from __future__ import annotations

from dataclasses import asdict, dataclass


PRICING_VERSION = "2026-07-10"
GPT_56_PRICING_SOURCE = "https://openai.com/index/gpt-5-6/"
MODEL_PRICING_SOURCE = "https://developers.openai.com/api/docs/models"


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object] | None) -> TokenUsage:
        if value is None:
            return cls()
        return cls(
            input_tokens=_nonnegative_integer(value.get("input_tokens")),
            cached_input_tokens=_nonnegative_integer(value.get("cached_input_tokens")),
            output_tokens=_nonnegative_integer(value.get("output_tokens")),
            reasoning_output_tokens=_nonnegative_integer(
                value.get("reasoning_output_tokens")
            ),
        )


@dataclass(frozen=True)
class ModelPrice:
    input_per_million_usd: float
    cached_input_per_million_usd: float
    output_per_million_usd: float
    source: str
    long_context_threshold: int | None = None
    long_context_input_multiplier: float = 1.0
    long_context_output_multiplier: float = 1.0


@dataclass(frozen=True)
class EstimatedCost:
    amount_usd: float
    currency: str
    estimated: bool
    model: str
    priced_as: str
    pricing_version: str
    pricing_source: str
    long_context_pricing: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object] | None) -> EstimatedCost | None:
        if value is None:
            return None
        return cls(**value)


PRICES = {
    "gpt-5.6-sol": ModelPrice(5.00, 0.50, 30.00, GPT_56_PRICING_SOURCE),
    "gpt-5.6-terra": ModelPrice(2.50, 0.25, 15.00, GPT_56_PRICING_SOURCE),
    "gpt-5.6-luna": ModelPrice(1.00, 0.10, 6.00, GPT_56_PRICING_SOURCE),
    "gpt-5.5": ModelPrice(5.00, 0.50, 30.00, MODEL_PRICING_SOURCE, 272_000, 2.0, 1.5),
    "gpt-5.4": ModelPrice(2.50, 0.25, 15.00, MODEL_PRICING_SOURCE, 272_000, 2.0, 1.5),
    "gpt-5.4-mini": ModelPrice(0.75, 0.075, 4.50, MODEL_PRICING_SOURCE),
    "gpt-5.4-nano": ModelPrice(0.20, 0.02, 1.25, MODEL_PRICING_SOURCE),
    "gpt-5.3-codex": ModelPrice(1.75, 0.175, 14.00, MODEL_PRICING_SOURCE),
}


def estimate_api_cost(model: str | None, usage: TokenUsage) -> EstimatedCost | None:
    if model is None:
        return None
    priced_as = _canonical_model(model)
    if priced_as is None:
        return None
    price = PRICES[priced_as]
    long_context = (
        price.long_context_threshold is not None
        and usage.input_tokens > price.long_context_threshold
    )
    input_multiplier = price.long_context_input_multiplier if long_context else 1.0
    output_multiplier = price.long_context_output_multiplier if long_context else 1.0
    cached = min(usage.cached_input_tokens, usage.input_tokens)
    uncached = usage.input_tokens - cached
    amount = (
        uncached * price.input_per_million_usd * input_multiplier
        + cached * price.cached_input_per_million_usd * input_multiplier
        + usage.output_tokens * price.output_per_million_usd * output_multiplier
    ) / 1_000_000
    return EstimatedCost(
        amount_usd=round(amount, 8),
        currency="USD",
        estimated=True,
        model=model,
        priced_as=priced_as,
        pricing_version=PRICING_VERSION,
        pricing_source=price.source,
        long_context_pricing=long_context,
    )


def _canonical_model(model: str) -> str | None:
    for candidate in sorted(PRICES, key=len, reverse=True):
        if model == candidate or model.startswith(f"{candidate}-20"):
            return candidate
    return None


def _nonnegative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)
