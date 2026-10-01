"""API-equivalent token cost at the rates in effect when each request ran."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version

import llm_prices


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0

    def __post_init__(self) -> None:
        if (
            min(
                self.input_tokens,
                self.cached_input_tokens,
                self.output_tokens,
                self.reasoning_output_tokens,
            )
            < 0
        ):
            raise ValueError("token counts cannot be negative")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        if self.reasoning_output_tokens > self.output_tokens:
            raise ValueError("reasoning_output_tokens cannot exceed output_tokens")

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
class CalculatedCost:
    amount_usd: float
    currency: str
    model: str
    priced_as: str
    pricing_version: str
    pricing_source: str
    long_context_pricing: bool
    pricing_retrieved_at: float | None = None
    pricing_basis: str | None = None
    # "peak", "off_peak", or "mixed" under time-of-day pricing, else None.
    pricing_period: str | None = None
    # Text llm-prices removed from the model name to find its price, such as "-max".
    removed_suffix: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object] | None) -> CalculatedCost | None:
        if value is None:
            return None
        fields = dict(value)
        fields.pop("estimated", None)
        return cls(**fields)


def estimate_api_cost(
    model: str | None,
    usage: TokenUsage,
    *,
    provider: str | None,
    started_at: str,
    finished_at: str,
    requests: Sequence[tuple[TokenUsage, str | None]] | None = None,
) -> CalculatedCost | None:
    """Price usage at llm-prices API rates in effect when each request ran.

    `requests` lists each measured request's usage and, when the provider
    reported it, the time it was served, which must fall within the run. A
    request without a time is priced only when one price covered the whole run.
    """
    if model is None:
        return None
    priced = tuple(requests) if requests is not None else ((usage, None),)
    if _sum_usage([request_usage for request_usage, _ in priced]) != usage:
        raise ValueError("request usage does not match aggregate usage")
    # With no measured requests, pricing zero usage still requires known rates.
    priced = priced or ((usage, None),)
    amount = 0.0
    long_context = False
    found = []
    for request_usage, at in priced:
        rates = _request_rates(model, provider, at, started_at, finished_at)
        if rates is None:
            return None
        breakdown = rates.breakdown(
            input_tokens=request_usage.input_tokens - request_usage.cached_input_tokens,
            cache_read_tokens=request_usage.cached_input_tokens,
            output_tokens=request_usage.output_tokens,
        )
        amount += breakdown.total
        long_context = long_context or breakdown.tier_above is not None
        found.append(rates)
    return CalculatedCost(
        amount_usd=round(amount, 8),
        currency="USD",
        model=model,
        priced_as=" + ".join(
            sorted({f"{rates.provider}/{rates.model}" for rates in found})
        ),
        pricing_version=f"llm-prices-{version('llm-prices')}",
        pricing_source=" ".join(sorted({rates.source for rates in found})),
        long_context_pricing=long_context,
        pricing_basis=llm_prices.pricing_basis(),
        pricing_period=merge_periods(rates.period for rates in found),
        removed_suffix=merge_suffixes(rates.removed_suffix for rates in found),
    )


def merge_periods(periods: Iterable[str | None]) -> str | None:
    distinct = set(periods)
    return distinct.pop() if len(distinct) == 1 else "mixed"


def merge_suffixes(suffixes: Iterable[str | None]) -> str | None:
    """Join the name suffixes llm-prices removed to find a price, if any."""
    return " ".join(sorted(set(suffixes) - {"", None})) or None


def _request_rates(
    model: str, provider: str | None, at: str | None, started_at: str, finished_at: str
) -> llm_prices.Rates | None:
    if at is not None:
        # A reported time outside the run is a clock or parsing error.
        if not _utc(started_at) <= _utc(at) <= _utc(finished_at):
            return None
        return llm_prices.rates(model, at=at, provider=provider)
    spans = llm_prices.rates_between(model, started_at, finished_at, provider=provider)
    # Without a request time, only one price covering the whole run applies.
    return spans[0][1] if len(spans) == 1 else None


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _sum_usage(usages: Sequence[TokenUsage]) -> TokenUsage:
    return TokenUsage(
        input_tokens=sum(usage.input_tokens for usage in usages),
        cached_input_tokens=sum(usage.cached_input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        reasoning_output_tokens=sum(usage.reasoning_output_tokens for usage in usages),
    )


def _nonnegative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)
