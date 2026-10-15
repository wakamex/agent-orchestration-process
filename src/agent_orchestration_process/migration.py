"""One-time migration of derived run records to their current formats.

Raw provider artifacts such as events.jsonl are never changed. Each rewritten
record keeps its original beside it and records the conversions applied.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import TOKEN_USAGE_SCHEMA, RunRequest, RunResult
from .pricing import TokenUsage, estimate_api_cost
from .runner import (
    AgyAdapter,
    ClaudeAdapter,
    CursorAdapter,
    DeepSeekHarnessAdapter,
    GrokAdapter,
    OpenCodeAdapter,
    RunStore,
)

ORIGINAL_SUFFIX = ".pre-v3.json"
LEGACY_TOKEN_USAGE_SCHEMA = "aop-token-usage-v1"
# v2 is v3's convention, except that Agy runs recorded Agy's result usage, the
# conversation's running total across every invocation that resumed it.
PREVIOUS_TOKEN_USAGE_SCHEMA = "aop-token-usage-v2"
_CACHE_WRITE_EVENT_PROVIDERS = frozenset(
    {"claude", "cursor", "dsh", "grok", "opencode"}
)
_DERIVED_RESULT_FIELDS = (
    "execution_completed",
    "response_available",
    "status",
    "succeeded",
)


def migrate_runs(runs_dir: Path, *, dry_run: bool = False) -> Counter[str]:
    """Bring every run record under runs_dir to the current formats."""
    counts: Counter[str] = Counter()
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        for name, upgrade in (
            ("request.json", _upgraded_request),
            ("result.json", _upgraded_result),
        ):
            path = run_dir / name
            if not path.is_file():
                continue
            try:
                raw = json.loads(path.read_text())
                upgraded = upgrade(raw, run_dir)
            except (OSError, ValueError, TypeError, KeyError) as error:
                counts[f"{name}: error"] += 1
                print(f"aop: could not migrate {path}: {error}", flush=True)
                continue
            if upgraded is None:
                counts[f"{name}: current"] += 1
                continue
            value, outcome = upgraded
            counts[f"{name}: {outcome}"] += 1
            if not dry_run:
                original = path.with_name(
                    path.name.removesuffix(".json") + ORIGINAL_SUFFIX
                )
                if not original.exists():
                    shutil.copy2(path, original)
                RunStore.write_json(path, value)
    return counts


def _upgraded_request(
    raw: dict[str, Any], run_dir: Path
) -> tuple[dict[str, Any], str] | None:
    try:
        RunRequest.from_dict(raw)
        return None
    except (TypeError, ValueError, KeyError):
        pass
    fields = dict(raw)
    conversions = []
    if "sandbox" in fields:
        # A Codex sandbox mode that no execution profile resolves, so a resume
        # of such a run fails instead of guessing a profile.
        fields.setdefault("profile", fields.pop("sandbox"))
        conversions.append("sandbox-to-profile")
    if "read_paths" in fields:
        fields.setdefault("inputs", fields.pop("read_paths"))
        conversions.append("read-paths-to-inputs")
    for key, default in (
        ("mode", "agent"),
        ("inference_provider", None),
        ("inference_route", None),
        ("no_web", False),
        ("effective_policy", {}),
        ("artifacts", []),
        ("inputs", []),
    ):
        if key not in fields:
            fields[key] = default
            conversions.append(f"default-{key.replace('_', '-')}")
    fields["migration"] = _record(None, conversions, "request.json")
    request = RunRequest.from_dict(fields)
    return request.to_dict(), "+".join(conversions) or "normalized"


def _upgraded_result(
    raw: dict[str, Any], run_dir: Path
) -> tuple[dict[str, Any], str] | None:
    try:
        current = RunResult.from_dict(raw)
        return _upgraded_v3_result(raw, current, run_dir)
    except (TypeError, ValueError, KeyError):
        pass
    schema = raw.get("usage_schema")
    previous_usage = _raw_usage_counts(raw.get("usage"))
    fields = {
        key: value for key, value in raw.items() if key not in _DERIVED_RESULT_FIELDS
    }
    conversions = []
    for key in (
        "mode",
        "model",
        "effort",
        "time_to_first_event_seconds",
        "time_to_first_response_seconds",
        "provider_duration_seconds",
        "provider_status",
        "provider_error",
        "inference_provider",
        "provider_reported_cost",
    ):
        fields.setdefault(key, "agent" if key == "mode" else None)
    if "read_paths" in fields:
        fields.setdefault("inputs", fields.pop("read_paths"))
        conversions.append("read-paths-to-inputs")
    _fold_legacy_costs(fields, conversions)

    if schema is None:
        conversions.append("unversioned-usage")
        _legacy_accounting(fields, _legacy_usage(fields))
    elif schema == LEGACY_TOKEN_USAGE_SCHEMA:
        conversions.append("v1-accounting")
        _legacy_accounting(fields, TokenUsage.from_dict(fields.get("usage")))
    elif schema == PREVIOUS_TOKEN_USAGE_SCHEMA:
        conversions.append("v2-schema")
    elif schema != TOKEN_USAGE_SCHEMA:
        raise ValueError(f"unsupported token usage schema: {schema}")
    if fields.get("provider") == "agy" and schema != TOKEN_USAGE_SCHEMA:
        conversions.append("agy-step-usage")
        _agy_step_usage(fields, run_dir)
    if isinstance(fields.get("usage"), TokenUsage):
        fields["usage"] = fields["usage"].to_dict()

    fields["usage_schema"] = TOKEN_USAGE_SCHEMA
    fields["migration"] = _record(schema, conversions, "result.json")
    result = RunResult.from_dict(fields)
    recovered, event_conversions = _recover_event_accounting(result, run_dir)
    if event_conversions:
        conversions.extend(event_conversions)
        result = replace(
            recovered,
            migration=_record(schema, conversions, "result.json"),
        )
    outcome = "+".join(conversions) or "normalized"
    if result.usage != previous_usage:
        if "claude-resume-cost" not in event_conversions:
            result = _repriced(result)
        outcome += " (usage changed)"
    return result.to_dict(), outcome


def _upgraded_v3_result(
    raw: dict[str, Any], current: RunResult, run_dir: Path
) -> tuple[dict[str, Any], str] | None:
    raw_usage = raw.get("usage")
    if (
        current.usage is None
        or not isinstance(raw_usage, dict)
        or "cache_write_input_tokens" in raw_usage
        or current.provider not in _CACHE_WRITE_EVENT_PROVIDERS
    ):
        return None
    recovered, conversions = _recover_event_accounting(current, run_dir)
    if not conversions:
        return None
    usage_changed = recovered.usage != current.usage
    if usage_changed and "claude-resume-cost" not in conversions:
        recovered = _repriced(recovered)
    recovered = replace(
        recovered,
        migration=_record(TOKEN_USAGE_SCHEMA, conversions, "result.json"),
    )
    outcome = "+".join(conversions)
    if usage_changed:
        outcome += " (usage changed)"
    return recovered.to_dict(), outcome


def _recover_event_accounting(
    result: RunResult, run_dir: Path
) -> tuple[RunResult, list[str]]:
    if result.usage is None or result.provider not in _CACHE_WRITE_EVENT_PROVIDERS:
        return result, []
    try:
        events_text = (run_dir / "events.jsonl").read_text()
    except OSError:
        return result, []
    conversions: list[str] = []
    new_usage = result.usage
    new_cost = result.calculated_cost
    if result.provider == "claude":
        parent_run_id = _parent_run_id(run_dir)
        previous_total_cost_usd = (
            ClaudeAdapter._reported_total_cost_usd(
                run_dir.parent / parent_run_id / "events.jsonl"
            )
            if parent_run_id
            else None
        )
        parsed = ClaudeAdapter._parse(
            events_text,
            previous_total_cost_usd=previous_total_cost_usd,
        )
        if not parsed["usage_observed"]:
            return result, []
        event_usage = parsed["usage"]
        if (
            event_usage.cache_write_input_tokens > 0
            and result.usage.cached_input_tokens + event_usage.cache_write_input_tokens
            <= result.usage.input_tokens
        ):
            new_usage = replace(
                result.usage,
                cache_write_input_tokens=event_usage.cache_write_input_tokens,
                cache_write_1h_input_tokens=event_usage.cache_write_1h_input_tokens,
            )
        if parsed["cost_is_cumulative"]:
            new_cost = parsed["cost"]
            if new_cost is None and result.model:
                new_cost = estimate_api_cost(
                    result.model,
                    new_usage,
                    provider="anthropic",
                    started_at=result.started_at,
                    finished_at=result.finished_at,
                )
    else:
        if result.provider == "grok":
            parsed = GrokAdapter._parse(events_text, result.model)
        elif result.provider == "cursor":
            parsed = CursorAdapter._parse(events_text)
        elif result.provider == "opencode":
            parsed = OpenCodeAdapter._parse(events_text, result.model)
        else:
            parsed = DeepSeekHarnessAdapter._parse(events_text)
        if not parsed["usage_observed"]:
            return result, []
        event_usage = parsed["usage"]
        if (
            event_usage.cache_write_input_tokens > 0
            and result.usage.cached_input_tokens + event_usage.cache_write_input_tokens
            <= result.usage.input_tokens
        ):
            new_usage = replace(
                result.usage,
                cache_write_input_tokens=event_usage.cache_write_input_tokens,
                cache_write_1h_input_tokens=event_usage.cache_write_1h_input_tokens,
            )
    if new_usage != result.usage:
        conversions.append("cache-write-usage")
    if new_cost != result.calculated_cost:
        conversions.append("claude-resume-cost")
    if not conversions:
        return result, []
    return replace(result, usage=new_usage, calculated_cost=new_cost), conversions


def _parent_run_id(run_dir: Path) -> str | None:
    try:
        raw_request = json.loads((run_dir / "request.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw_request, dict):
        return None
    parent_run_id = raw_request.get("parent_run_id")
    return parent_run_id if isinstance(parent_run_id, str) and parent_run_id else None


def _record(schema: str | None, conversions: list[str], name: str) -> dict[str, Any]:
    return {
        "from_usage_schema": schema,
        "conversions": conversions,
        "original": name.removesuffix(".json") + ORIGINAL_SUFFIX,
        "migrated_at": datetime.now(UTC).isoformat(),
    }


def _fold_legacy_costs(fields: dict[str, Any], conversions: list[str]) -> None:
    legacy_cost = fields.pop("api_equivalent_cost", None)
    billing = fields.get("billing")
    reported = isinstance(billing, dict) and billing.get("actual_cost_known") is True
    if isinstance(billing, dict) and "actual_cost_known" in billing:
        fields["billing"] = {
            key: value for key, value in billing.items() if key != "actual_cost_known"
        }
    if legacy_cost is not None:
        conversions.append("api-equivalent-cost")
        if reported and fields.get("provider_reported_cost") is None:
            fields["provider_reported_cost"] = {
                "amount_usd": legacy_cost["amount_usd"],
                "currency": legacy_cost.get("currency", "USD"),
                "source": legacy_cost.get("pricing_source", "legacy run result"),
            }
        elif not reported:
            fields.setdefault("calculated_cost", legacy_cost)
    fields.setdefault("calculated_cost", None)
    cost = fields.get("calculated_cost")
    if isinstance(cost, dict) and "estimated" in cost:
        fields["calculated_cost"] = {
            key: value for key, value in cost.items() if key != "estimated"
        }


def _raw_usage_counts(raw: object) -> TokenUsage | None:
    if not isinstance(raw, dict):
        return None
    counts = {
        key: max(value, 0)
        if isinstance(value := raw.get(key), int) and not isinstance(value, bool)
        else 0
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "cache_write_1h_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
    }
    try:
        return TokenUsage(**counts)
    except ValueError:
        return None


def _legacy_usage(fields: dict[str, Any]) -> TokenUsage:
    raw = fields.get("usage") if isinstance(fields.get("usage"), dict) else {}
    counts = {
        key: max(value, 0)
        if isinstance(value := raw.get(key), int) and not isinstance(value, bool)
        else 0
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_input_tokens",
            "cache_write_1h_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        )
    }
    # These adapters persisted their provider's disjoint uncached input
    # bucket; OpenCode persisted visible output without its reasoning.
    if fields.get("provider") in {"agy", "cursor", "dsh"}:
        counts["input_tokens"] += (
            counts["cached_input_tokens"] + counts["cache_write_input_tokens"]
        )
    if fields.get("provider") == "opencode":
        counts["output_tokens"] += counts["reasoning_output_tokens"]
    return TokenUsage(**counts)


def _legacy_accounting(fields: dict[str, Any], usage: TokenUsage) -> None:
    if not fields.get("timed_out"):
        fields["usage"] = usage
        fields["accounting_status"] = "complete"
        return
    observed = usage.total_tokens > 0
    if not observed:
        fields["calculated_cost"] = None
    fields["usage"] = usage if observed else None
    fields["accounting_status"] = (
        "partial"
        if observed or fields.get("provider_reported_cost") is not None
        else "unavailable"
    )


def _agy_step_usage(fields: dict[str, Any], run_dir: Path) -> None:
    try:
        parsed = AgyAdapter._parse((run_dir / "events.jsonl").read_text())
    except OSError:
        parsed = {"usage_observed": False}
    if parsed["usage_observed"]:
        fields["usage"] = parsed["usage"]
        if fields.get("accounting_status") == "unavailable":
            fields["accounting_status"] = "partial"
        return
    fields.update(
        usage=None,
        calculated_cost=None,
        provider_reported_cost=None,
        accounting_status="unavailable",
    )


def _repriced(result: RunResult) -> RunResult:
    """Price changed usage at the rates in effect when the run happened."""
    if (
        result.provider == "claude"
        and result.calculated_cost is not None
        and result.calculated_cost.pricing_version == "claude-cli-reported"
    ):
        return result
    provider = {
        "agy": "google",
        "claude": "anthropic",
        "dsh": DeepSeekHarnessAdapter.pricing_provider(result.inference_provider),
        "grok": "xai",
    }.get(result.provider)
    priced = result.provider in {"agy", "claude", "dsh", "grok", "opencode"}
    cost = (
        estimate_api_cost(
            result.model,
            result.usage,
            provider=provider,
            started_at=result.started_at,
            finished_at=result.finished_at,
        )
        if priced and result.usage is not None and result.model
        else None
    )
    return replace(result, calculated_cost=cost)


def migration_roots(registered: Iterable[Path], current: Path | None) -> list[Path]:
    roots = [root / "runs" for root in registered]
    if current is not None:
        roots.append(current / "runs")
    return [root for root in dict.fromkeys(roots) if root.is_dir()]


def print_counts(root: Path, counts: Counter[str]) -> None:
    print(root)
    for outcome, count in sorted(counts.items()):
        print(f"  {count:>7}  {outcome}")
