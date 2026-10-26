"""API-equivalent token cost at the rates in effect when each request ran."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
import warnings

import model_prices

# Fetch newer price data at most daily, and warn when the data in use has not
# been confirmed current for longer than PRICE_DATA_STALE_AFTER.
PRICE_DATA_MAX_AGE = timedelta(days=1)
PRICE_DATA_STALE_AFTER = timedelta(days=3)


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    cache_write_input_tokens: int = 0
    cache_write_1h_input_tokens: int = 0

    def __post_init__(self) -> None:
        if (
            min(
                self.input_tokens,
                self.cached_input_tokens,
                self.output_tokens,
                self.reasoning_output_tokens,
                self.cache_write_input_tokens,
                self.cache_write_1h_input_tokens,
            )
            < 0
        ):
            raise ValueError("token counts cannot be negative")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        if self.cache_write_input_tokens > self.input_tokens:
            raise ValueError("cache_write_input_tokens cannot exceed input_tokens")
        if self.cached_input_tokens + self.cache_write_input_tokens > self.input_tokens:
            raise ValueError(
                "cached_input_tokens and cache_write_input_tokens cannot exceed input_tokens"
            )
        if self.cache_write_1h_input_tokens > self.cache_write_input_tokens:
            raise ValueError(
                "cache_write_1h_input_tokens cannot exceed cache_write_input_tokens"
            )
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
            input_tokens=nonnegative_int(value.get("input_tokens")),
            cached_input_tokens=nonnegative_int(value.get("cached_input_tokens")),
            output_tokens=nonnegative_int(value.get("output_tokens")),
            reasoning_output_tokens=nonnegative_int(
                value.get("reasoning_output_tokens")
            ),
            cache_write_input_tokens=nonnegative_int(
                value.get("cache_write_input_tokens")
            ),
            cache_write_1h_input_tokens=nonnegative_int(
                value.get("cache_write_1h_input_tokens")
            ),
        )

    @classmethod
    def from_disjoint(
        cls,
        *,
        uncached_input_tokens: object = 0,
        cached_input_tokens: object = 0,
        cache_write_input_tokens: object = 0,
        cache_write_1h_input_tokens: object = 0,
        output_tokens: object = 0,
        reasoning_output_tokens: object = 0,
        reasoning_in_output: bool = True,
    ) -> TokenUsage:
        uncached = nonnegative_int(uncached_input_tokens)
        cached = nonnegative_int(cached_input_tokens)
        write = nonnegative_int(cache_write_input_tokens)
        write_1h = min(nonnegative_int(cache_write_1h_input_tokens), write)
        output = nonnegative_int(output_tokens)
        reasoning = nonnegative_int(reasoning_output_tokens)
        return cls(
            input_tokens=uncached + cached + write,
            cached_input_tokens=cached,
            output_tokens=output if reasoning_in_output else output + reasoning,
            reasoning_output_tokens=reasoning,
            cache_write_input_tokens=write,
            cache_write_1h_input_tokens=write_1h,
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
    # Text model-prices removed from the model name to find its price, such as "-max".
    removed_suffix: str | None = None
    # For a model sold with no per-token price, the "provider/model" whose
    # rates model-prices uses as its estimate; None when every rate is a price.
    estimated_from: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object] | None) -> CalculatedCost | None:
        if value is None:
            return None
        return cls(**value)


def estimate_api_cost(
    model: str | None,
    usage: TokenUsage,
    *,
    provider: str | None,
    started_at: str,
    finished_at: str,
    requests: Sequence[tuple[TokenUsage, str | None]] | None = None,
    estimate: bool = False,
) -> CalculatedCost | None:
    """Price usage at model-prices API rates in effect when each request ran.

    `requests` lists each measured request's usage and, when the provider
    reported it, the time it was served, which must fall within the run. A
    request without a time is priced only when one price covered the whole run.
    With `estimate`, a model sold with no per-token price is priced at the
    rates model-prices names as its stand-in, and the cost records which model.
    """
    if model is None:
        return None
    priced = tuple(requests) if requests is not None else ((usage, None),)
    if sum_usage(request_usage for request_usage, _ in priced) != usage:
        raise ValueError("request usage does not match aggregate usage")
    # With no measured requests, pricing zero usage still requires known rates.
    priced = priced or ((usage, None),)
    amount = 0.0
    long_context = False
    found = []
    for request_usage, at in priced:
        rates = _request_rates(model, provider, at, started_at, finished_at, estimate)
        if rates is None:
            return None
        breakdown = rates.breakdown(
            input_tokens=(
                request_usage.input_tokens
                - request_usage.cached_input_tokens
                - request_usage.cache_write_input_tokens
            ),
            cache_read_tokens=request_usage.cached_input_tokens,
            cache_write_tokens=(
                request_usage.cache_write_input_tokens
                - request_usage.cache_write_1h_input_tokens
            ),
            cache_write_1h_tokens=request_usage.cache_write_1h_input_tokens,
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
        pricing_version=f"model-prices-{version('model-prices')}",
        pricing_source=" ".join(sorted({rates.source for rates in found})),
        long_context_pricing=long_context,
        pricing_basis=model_prices.pricing_basis(),
        pricing_period=merge_periods(rates.period for rates in found),
        removed_suffix=merge_suffixes(rates.removed_suffix for rates in found),
        estimated_from=merge_estimates(rates.estimated_from for rates in found),
    )


def refresh_prices(*, force: bool = False) -> list[str]:
    """Load the newest published price data and return warnings about it.

    A failed download keeps the newest cached or bundled data, so pricing never
    blocks a run; each calculated cost records the basis it used.
    """
    with warnings.catch_warnings():
        # The returned status carries the same failure, reported once below.
        warnings.simplefilter("ignore")
        status = model_prices.refresh(
            max_age=timedelta(0) if force else PRICE_DATA_MAX_AGE
        )
    found = [status.error] if status.error else []
    age = datetime.now(UTC) - status.published_at
    if age > PRICE_DATA_STALE_AFTER:
        found.append(
            f"price data was last confirmed current {age.days} days ago "
            f"({status.basis})"
        )
    return found


def merge_periods(periods: Iterable[str | None]) -> str | None:
    distinct = set(periods)
    return distinct.pop() if len(distinct) == 1 else "mixed"


def merge_estimates(sources: Iterable[str | None]) -> str | None:
    """Join the stand-in models of estimated rates, if any rate was estimated."""
    return " + ".join(sorted(set(sources) - {None})) or None


def merge_suffixes(suffixes: Iterable[str | None]) -> str | None:
    """Join the name suffixes model-prices removed to find a price, if any."""
    return " ".join(sorted(set(suffixes) - {"", None})) or None


def _request_rates(
    model: str,
    provider: str | None,
    at: str | None,
    started_at: str,
    finished_at: str,
    estimate: bool = False,
) -> model_prices.Rates | None:
    if at is not None:
        # A reported time outside the run is a clock or parsing error.
        if not _utc(started_at) <= _utc(at) <= _utc(finished_at):
            return None
        return model_prices.rates(model, at=at, provider=provider, estimate=estimate)
    spans = model_prices.rates_between(model, started_at, finished_at, provider=provider)
    if not spans and estimate:
        # Stand-in rates have no schedule of their own to span the run.
        return model_prices.rates(
            model, at=started_at, provider=provider, estimate=True
        )
    # Without a request time, only one price covering the whole run applies.
    return spans[0][1] if len(spans) == 1 else None


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def sum_usage(usages: Iterable[TokenUsage]) -> TokenUsage:
    items = tuple(usages)
    return TokenUsage(
        input_tokens=sum(usage.input_tokens for usage in items),
        cached_input_tokens=sum(usage.cached_input_tokens for usage in items),
        output_tokens=sum(usage.output_tokens for usage in items),
        reasoning_output_tokens=sum(usage.reasoning_output_tokens for usage in items),
        cache_write_input_tokens=sum(
            usage.cache_write_input_tokens for usage in items
        ),
        cache_write_1h_input_tokens=sum(
            usage.cache_write_1h_input_tokens for usage in items
        ),
    )


def nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def nonnegative_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(float(value), 0.0)
