"""Fresh estimates of standard API-equivalent token cost."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .model_catalog import ModelCatalog, ensure_catalog_fresh


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
    long_context_threshold: int | None = None
    long_context_input_per_million_usd: float | None = None
    long_context_cached_input_per_million_usd: float | None = None
    long_context_output_per_million_usd: float | None = None


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
    pricing_retrieved_at: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object] | None) -> EstimatedCost | None:
        if value is None:
            return None
        return cls(**value)


def estimate_api_cost(
    model: str | None,
    usage: TokenUsage,
    catalog: ModelCatalog | None = None,
    *,
    providers: tuple[str, ...] = ("openai",),
    additive_cached_input: bool = False,
    catalog_model: str | None = None,
) -> EstimatedCost | None:
    if model is None:
        return None
    catalog = catalog or ensure_catalog_fresh()
    resolved = _catalog_price(catalog, catalog_model or model, providers)
    if resolved is None:
        return None
    priced_as, price = resolved
    long_context = (
        price.long_context_threshold is not None
        and usage.input_tokens > price.long_context_threshold
    )
    input_rate = (
        price.long_context_input_per_million_usd
        if long_context
        else price.input_per_million_usd
    )
    cached_rate = (
        price.long_context_cached_input_per_million_usd
        if long_context
        else price.cached_input_per_million_usd
    )
    output_rate = (
        price.long_context_output_per_million_usd
        if long_context
        else price.output_per_million_usd
    )
    assert (
        input_rate is not None and cached_rate is not None and output_rate is not None
    )
    cached = usage.cached_input_tokens
    uncached = usage.input_tokens
    if not additive_cached_input:
        cached = min(cached, uncached)
        uncached -= cached
    amount = (
        uncached * input_rate + cached * cached_rate + usage.output_tokens * output_rate
    ) / 1_000_000
    return EstimatedCost(
        amount_usd=round(amount, 8),
        currency="USD",
        estimated=True,
        model=model,
        priced_as=priced_as,
        pricing_version=catalog.version,
        pricing_source=catalog.source,
        long_context_pricing=long_context,
        pricing_retrieved_at=catalog.fetched_at,
    )


def _catalog_price(
    catalog: ModelCatalog, model: str, providers: tuple[str, ...]
) -> tuple[str, ModelPrice] | None:
    unqualified = model.partition("/")[2] or model
    aliases = {"gpt-5.6": "gpt-5.6-sol"}
    requested = aliases.get(unqualified, unqualified)
    for provider in providers:
        models = catalog.providers.get(provider, {}).get("models", {})
        if not isinstance(models, dict):
            continue
        candidates = [requested]
        candidates.extend(
            candidate
            for candidate in models
            if isinstance(candidate, str)
            and requested.startswith(f"{candidate}-20")
        )
        for candidate in sorted(set(candidates), key=len, reverse=True):
            metadata = models.get(candidate)
            if not isinstance(metadata, dict):
                continue
            price = _model_price(metadata.get("cost"))
            if price is not None:
                return candidate, price
    return None


def _model_price(value: object) -> ModelPrice | None:
    if not isinstance(value, dict):
        return None
    input_rate = _rate(value.get("input"))
    output_rate = _rate(value.get("output"))
    cached_rate = _rate(value.get("cache_read"))
    if input_rate is None or output_rate is None:
        return None
    if cached_rate is None:
        cached_rate = input_rate
    tier = _context_tier(value.get("tiers"))
    return ModelPrice(
        input_per_million_usd=input_rate,
        cached_input_per_million_usd=cached_rate,
        output_per_million_usd=output_rate,
        long_context_threshold=tier[0] if tier else None,
        long_context_input_per_million_usd=tier[1] if tier else None,
        long_context_cached_input_per_million_usd=tier[2] if tier else None,
        long_context_output_per_million_usd=tier[3] if tier else None,
    )


def _context_tier(value: object) -> tuple[int, float, float, float] | None:
    if not isinstance(value, list):
        return None
    for tier in value:
        if not isinstance(tier, dict) or not isinstance(tier.get("tier"), dict):
            continue
        condition = tier["tier"]
        size = condition.get("size")
        input_rate = _rate(tier.get("input"))
        output_rate = _rate(tier.get("output"))
        cached_rate = _rate(tier.get("cache_read"))
        if (
            condition.get("type") == "context"
            and isinstance(size, int)
            and input_rate is not None
            and output_rate is not None
        ):
            return size, input_rate, cached_rate or input_rate, output_rate
    return None


def _rate(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _nonnegative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)
