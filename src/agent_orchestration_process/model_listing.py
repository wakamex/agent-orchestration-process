"""Model discovery across AOP's supported agent CLIs."""

from __future__ import annotations

import json
import hashlib
import os
import re
import selectors
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .codex_routes import (
    ZAI_CODING_PLAN,
    ZAI_CODING_PLAN_ENDPOINT,
    fetch_zai_inventory,
    resolve_codex_route,
)
from .model_catalog import ModelCatalog
from .models import InferenceRoute
from .pricing import pricing_metadata
from .provider_versions import require_supported_agy
from .worktrees import AOPError


AGENTS = (
    "codex",
    "claude",
    "cursor",
    "devin",
    "opencode",
    "agy",
    "grok",
    "hermes",
    "dsh",
    "zcode",
)
NOUS_MODELS_URL = "https://inference-api.nousresearch.com/v1/models"


@dataclass(frozen=True)
class AvailableModel:
    agent: str
    model: str
    name: str
    availability: str
    price_scope: str
    inference_provider: str | None = None
    inventory_retrieved_at: str | None = None
    inventory_sha256: str | None = None
    authenticated: bool = False
    inventory_source: str | None = None
    discovery_error: str | None = None
    input_per_million_usd: float | None = None
    cached_input_per_million_usd: float | None = None
    cache_write_per_million_usd: float | None = None
    output_per_million_usd: float | None = None
    pricing_source: str | None = None
    pricing_retrieved_at: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def list_models(
    agent: str,
    catalog: ModelCatalog,
    inference_provider: str | None = None,
) -> list[AvailableModel]:
    if agent == "codex":
        return _codex_models(catalog, inference_provider)
    if inference_provider is not None:
        raise AOPError(
            "--provider model inventory is currently supported only for Codex"
        )
    if agent == "cursor":
        return _simple_cli_models(
            agent, _binary("cursor", "AOP_CURSOR_BIN", "agent"), ["models"], catalog
        )
    if agent == "agy":
        return _agy_models(catalog)
    if agent == "devin":
        return _devin_models()
    if agent == "opencode":
        return _opencode_models(catalog)
    if agent == "claude":
        return _claude_models(catalog)
    if agent == "hermes":
        binary = _binary("hermes", "AOP_HERMES_BIN", "hermes")
        try:
            config = json.loads(_run([binary, "config", "get", "model", "--json"]))
        except json.JSONDecodeError as error:
            raise AOPError("Hermes returned invalid model configuration") from error
        if not isinstance(config, dict) or not isinstance(config.get("provider"), str):
            raise AOPError("Hermes model configuration has no provider")
        model = config.get("default") or config.get("name")
        if model is not None and not isinstance(model, str):
            raise AOPError("Hermes model configuration has an invalid model ID")
        return _hermes_models(catalog, config["provider"], model)
    if agent == "grok":
        return _grok_models(catalog)
    if agent == "zcode":
        return _zcode_models(catalog)
    if agent == "dsh":
        _binary("dsh", "AOP_DSH_BIN", "dsh")
        return [
            _record(
                "dsh",
                model,
                _name(catalog.model("deepseek", model), name),
                "installed-default",
                "api-equivalent",
                catalog,
                "deepseek",
                model,
            )
            for model, name in (
                ("deepseek-flash", "DeepSeek-V4.1-Flash"),
                ("deepseek-v4-flash", "DeepSeek-V4-Flash"),
                ("deepseek-v4-pro", "DeepSeek-V4-Pro"),
            )
        ]
    raise AOPError(f"unsupported agent: {agent}")


def _claude_models(catalog: ModelCatalog) -> list[AvailableModel]:
    binary = _binary("claude", "AOP_CLAUDE_BIN", "claude")
    records = {
        row.model: replace(
            row,
            inventory_source=catalog.source,
            inventory_retrieved_at=datetime.fromtimestamp(
                catalog.fetched_at, UTC
            ).isoformat(),
            inventory_sha256=catalog.sha256,
        )
        for row in _catalog_models("claude", "anthropic", catalog)
    }
    try:
        entries = _claude_model_response(binary)
    except AOPError as error:
        if not records:
            raise
        return [replace(row, discovery_error=str(error)) for row in records.values()]
    retrieved_at = datetime.now(UTC).isoformat()
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for entry in entries:
        # New CLIs resolve aliases using native account/provider configuration.
        # Older CLIs may advertise only aliases; leave their prices unknown.
        model = entry.get("resolvedModel") or entry["value"]
        records[model] = replace(
            _record(
                "claude",
                model,
                entry.get("displayName") or model,
                "native-advertised",
                "api-equivalent",
                catalog,
                "anthropic",
                model,
                inventory_retrieved_at=retrieved_at,
                inventory_sha256=digest,
            ),
            inventory_source="claude-sdk-initialize",
        )
    return sorted(records.values(), key=lambda row: row.model)


def _claude_model_response(binary: str) -> list[dict[str, Any]]:
    """Read the native SDK inventory without submitting a user turn or extracting auth."""
    try:
        process = subprocess.Popen(
            [
                binary,
                "--print",
                "--input-format",
                "stream-json",
                "--output-format",
                "stream-json",
                "--verbose",
                "--no-session-persistence",
                "--strict-mcp-config",
                "--settings",
                '{"disableAllHooks":true}',
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        raise AOPError("could not start Claude model discovery") from error
    assert process.stdin is not None and process.stdout is not None
    selector = selectors.DefaultSelector()
    try:
        process.stdin.write(
            json.dumps(
                {
                    "type": "control_request",
                    "request_id": "aop-models",
                    "request": {"subtype": "initialize"},
                }
            ).encode()
            + b"\n"
        )
        process.stdin.flush()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + 20
        buffered = b""
        size = 0
        while time.monotonic() < deadline:
            if not selector.select(max(0, deadline - time.monotonic())):
                break
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                break
            size += len(chunk)
            if size > 4_000_000:
                raise AOPError("Claude model discovery response exceeded 4 MB")
            buffered += chunk
            while b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                value = json.loads(line)
                if (
                    not isinstance(value, dict)
                    or value.get("type") != "control_response"
                ):
                    continue
                response = value.get("response")
                if (
                    not isinstance(response, dict)
                    or response.get("request_id") != "aop-models"
                ):
                    continue
                body = response.get("response")
                entries = body.get("models") if isinstance(body, dict) else None
                if (
                    response.get("subtype") != "success"
                    or not isinstance(entries, list)
                    or not entries
                ):
                    raise AOPError("Claude did not return a native model inventory")
                for entry in entries:
                    if (
                        not isinstance(entry, dict)
                        or not isinstance(entry.get("value"), str)
                        or not entry["value"]
                        or (
                            "resolvedModel" in entry
                            and (
                                not isinstance(entry["resolvedModel"], str)
                                or not entry["resolvedModel"]
                            )
                        )
                        or (
                            "displayName" in entry
                            and not isinstance(entry["displayName"], str)
                        )
                    ):
                        raise AOPError("Claude returned an invalid model inventory")
                return entries
        raise AOPError("Claude model discovery ended or timed out without an inventory")
    except (OSError, ValueError) as error:
        raise AOPError("could not read Claude model inventory") from error
    finally:
        selector.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdin.close()
        process.stdout.close()


def _zcode_models(catalog: ModelCatalog) -> list[AvailableModel]:
    from .zcode import configured_models, credential_env_names, read_config, source_home

    _require_binary(_binary("zcode", "AOP_ZCODE_BIN", "zcode"))
    config = read_config(source_home(dict(os.environ)) / "cli" / "config.json")
    models = configured_models(config)
    native_model = os.environ.get("ZCODE_MODEL")
    if native_model:
        if "/" not in native_model:
            native_model = "anthropic/" + native_model
        if native_model not in models:
            models.append(native_model)
    records = {
        model: AvailableModel(
            agent="zcode",
            model=model,
            name=model,
            availability="configured",
            price_scope="unavailable",
        )
        for model in models
    }
    # The official Coding Plan inventory is shared across harnesses. Query it
    # only for a configured canonical Z.ai route, never for arbitrary endpoints.
    for provider, definition in config.get("provider", {}).items():
        options = definition.get("options", {})
        if (
            definition.get("kind") != "anthropic"
            or options.get("baseURL") != "https://api.z.ai/api/anthropic"
        ):
            continue
        for model in models:
            native_provider, _, model_id = model.partition("/")
            if native_provider == provider:
                records[model] = _record(
                    "zcode",
                    model,
                    model,
                    "configured",
                    "api-equivalent",
                    catalog,
                    "zai",
                    model_id,
                    inference_provider=ZAI_CODING_PLAN,
                )
        credential = options.get("apiKey") or next(
            (
                os.environ[name]
                for name in credential_env_names(provider, definition)
                if os.environ.get(name)
            ),
            None,
        )
        if credential is None:
            continue
        if not isinstance(credential, str) or not credential.strip():
            raise AOPError("Zcode Z.ai API key must be a nonempty string")
        # Each configured route may select a different account's credentials.
        inventory = fetch_zai_inventory(ZAI_CODING_PLAN_ENDPOINT, credential)
        entries, retrieved_at, digest = inventory
        for entry in entries:
            model = f"{provider}/{entry['slug']}"
            records[model] = _record(
                "zcode",
                model,
                entry.get("display_name") or entry["slug"],
                "authenticated-endpoint",
                "api-equivalent",
                catalog,
                "zai",
                entry["slug"],
                inference_provider=ZAI_CODING_PLAN,
                authenticated=True,
                inventory_retrieved_at=retrieved_at,
                inventory_sha256=digest,
            )
    return sorted(records.values(), key=lambda record: record.model)


def _codex_models(
    catalog: ModelCatalog, inference_provider: str | None
) -> list[AvailableModel]:
    binary = _binary("codex", "AOP_CODEX_BIN", "codex")
    configured = os.environ.get("AOP_CODEX_SOURCE_HOME") or os.environ.get("CODEX_HOME")
    source_home = Path(configured).expanduser().resolve() if configured else None
    provider, _, route = resolve_codex_route(
        binary,
        source_home,
        Path.cwd(),
        inference_provider,
        None,
        os.environ,
        require_explicit_model=False,
    )
    if route is not None:
        return _codex_route_models(catalog, route)
    if provider is not None:
        raise AOPError(f"Codex did not resolve provider {inference_provider}")
    response = _codex_model_response(binary)
    data = response.get("result", {}).get("data") if response else None
    if not isinstance(data, list):
        raise AOPError("Codex did not return a model catalog")
    records = []
    for value in data:
        if not isinstance(value, dict) or not isinstance(value.get("model"), str):
            continue
        model = value["model"]
        records.append(
            _record(
                "codex",
                model,
                value.get("displayName") or model,
                "account",
                "api-equivalent",
                catalog,
                "openai",
                model,
            )
        )
    return records


def _codex_route_models(
    catalog: ModelCatalog, route: InferenceRoute
) -> list[AvailableModel]:
    if route.provider != ZAI_CODING_PLAN:
        raise AOPError(f"Codex did not resolve provider {route.provider}")
    records = []
    for value in route.inventory_models:
        model = value["slug"]
        records.append(
            _record(
                "codex",
                model,
                str(value.get("display_name") or model),
                "authenticated-endpoint",
                "api-equivalent",
                catalog,
                "zai",
                model,
                inference_provider=route.provider,
                inventory_retrieved_at=route.inventory_retrieved_at,
                inventory_sha256=route.inventory_sha256,
                authenticated=route.authenticated,
            )
        )
    return records


def _codex_model_response(binary: str) -> dict[str, Any]:
    try:
        process = subprocess.Popen(
            [binary, "app-server", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        raise AOPError(f"could not start Codex model discovery: {error}") from error
    assert process.stdin is not None and process.stdout is not None
    try:
        _write_json_line(
            process.stdin,
            {
                "method": "initialize",
                "id": 1,
                "params": {
                    "clientInfo": {
                        "name": "aop",
                        "title": "AOP",
                        "version": "0.1.0",
                    }
                },
            },
        )
        _read_json_response(process, 1)
        _write_json_line(process.stdin, {"method": "initialized", "params": {}})
        _write_json_line(
            process.stdin,
            {"method": "model/list", "id": 2, "params": {"limit": 1000}},
        )
        return _read_json_response(process, 2)
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _write_json_line(stream: Any, value: object) -> None:
    stream.write(f"{json.dumps(value)}\n")
    stream.flush()


def _read_json_response(
    process: subprocess.Popen[str], response_id: int
) -> dict[str, Any]:
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline:
            ready = selector.select(deadline - time.monotonic())
            if not ready:
                break
            line = process.stdout.readline()
            if not line:
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("id") == response_id:
                return value
    finally:
        selector.close()
    raise AOPError(f"Codex did not return response {response_id}")


def _simple_cli_models(
    agent: str,
    binary: str,
    arguments: list[str],
    catalog: ModelCatalog,
) -> list[AvailableModel]:
    output = _run([binary, *arguments])
    records = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line in {"Available models", "Fetching available models..."}:
            continue
        if "\t" in line:
            model, name = line.split("\t", 1)
        elif " - " in line:
            model, name = line.split(" - ", 1)
        else:
            continue
        provider, priced_model = _registry_identity(agent, model)
        records.append(
            _record(
                agent,
                model,
                name.removesuffix(" (default)"),
                "account",
                "api-equivalent" if provider else "unknown",
                catalog,
                provider,
                priced_model,
            )
        )
    if not records:
        raise AOPError(f"{agent} did not return any models")
    return records


def _agy_models(catalog: ModelCatalog) -> list[AvailableModel]:
    binary = _binary("agy", "AOP_AGY_BIN", "agy")
    require_supported_agy(binary)
    output = _run([binary, "--output-format", "json", "models"])
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise AOPError("agy returned an invalid model catalog") from error
    command = value.get("command") if isinstance(value, dict) else None
    data = command.get("data") if isinstance(command, dict) else None
    models = data.get("models") if isinstance(data, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("status") != "SUCCESS"
        or not isinstance(command, dict)
        or command.get("name") != "models"
        or not isinstance(models, list)
    ):
        raise AOPError("agy returned an invalid model catalog")

    records = []
    for item in models:
        if not isinstance(item, dict):
            continue
        model = item.get("id")
        name = item.get("label")
        if not isinstance(model, str) or not model:
            continue
        if not isinstance(name, str) or not name:
            continue
        provider, priced_model = _registry_identity("agy", model)
        records.append(
            _record(
                "agy",
                model,
                name,
                "account",
                "api-equivalent",
                catalog,
                provider,
                priced_model,
            )
        )
    if not records:
        raise AOPError("agy did not return any models")
    return records


def _opencode_models(catalog: ModelCatalog) -> list[AvailableModel]:
    binary = _binary("opencode", "AOP_OPENCODE_BIN", "opencode")
    output = _run([binary, "models"])
    records = []
    for line in output.splitlines():
        model = line.strip()
        if not model or "/" not in model:
            continue
        provider, priced_model = model.split("/", 1)
        metadata = catalog.model(provider, priced_model)
        records.append(
            _record(
                "opencode",
                model,
                _name(metadata, model),
                "account",
                "provider",
                catalog,
                provider,
                priced_model,
            )
        )
    if not records:
        raise AOPError("OpenCode did not return any models")
    return records


def _grok_models(catalog: ModelCatalog) -> list[AvailableModel]:
    binary = _binary("grok", "AOP_GROK_BIN", "grok")
    output = _run([binary, "models"])
    records = []
    for line in output.splitlines():
        match = re.fullmatch(r"\s*\*\s+(.+?)(?:\s+\(default\))?\s*", line)
        if match is None:
            continue
        model = match.group(1)
        records.append(
            _record(
                "grok",
                model,
                _name(catalog.model("xai", model), model),
                "account",
                "api-equivalent",
                catalog,
                "xai",
                model,
            )
        )
    if not records:
        raise AOPError("Grok did not return any models")
    return records


def _devin_models() -> list[AvailableModel]:
    binary = _binary("devin", "AOP_DEVIN_BIN", "devin")
    try:
        value = json.loads(_run([binary, "models", "list", "--format", "json"]))
    except json.JSONDecodeError as error:
        raise AOPError("Devin returned an invalid model inventory") from error
    families = value.get("families") if isinstance(value, dict) else None
    if not isinstance(families, list):
        raise AOPError("Devin returned an invalid model inventory")
    records = []
    for family in families:
        variants = family.get("variants") if isinstance(family, dict) else None
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            model = variant.get("model_uid")
            if not isinstance(model, str) or not model:
                continue
            input_price, output_price = _devin_prices(variant)
            priced = input_price is not None or output_price is not None
            records.append(
                AvailableModel(
                    agent="devin",
                    model=model,
                    name=str(variant.get("label") or model),
                    availability="account",
                    price_scope="provider" if priced else "unknown",
                    input_per_million_usd=input_price,
                    output_per_million_usd=output_price,
                    pricing_source=(
                        "Devin CLI account model inventory" if priced else None
                    ),
                )
            )
    if not records:
        raise AOPError("Devin did not return any models")
    return records


def _devin_prices(variant: dict[str, Any]) -> tuple[float | None, float | None]:
    if str(variant.get("cost_tier", "")).lower() == "free":
        return 0.0, 0.0
    summary = variant.get("cost_summary")
    if not isinstance(summary, str):
        return None, None
    prices = re.search(
        r"\$([0-9]+(?:\.[0-9]+)?)\s*/\s*MTok\s*In.*?"
        r"\$([0-9]+(?:\.[0-9]+)?)\s*/\s*MTok\s*Out",
        summary,
        re.IGNORECASE,
    )
    if prices is None:
        return None, None
    return float(prices.group(1)), float(prices.group(2))


def _catalog_models(
    agent: str, provider: str, catalog: ModelCatalog
) -> list[AvailableModel]:
    models = catalog.providers.get(provider, {}).get("models", {})
    if not isinstance(models, dict):
        return []
    return [
        _record(
            agent,
            model,
            _name(metadata, model),
            "catalog",
            "api-equivalent",
            catalog,
            provider,
            model,
        )
        for model, metadata in sorted(models.items())
        if isinstance(model, str) and isinstance(metadata, dict)
    ]


def _hermes_models(
    catalog: ModelCatalog, provider: str, configured_model: str | None = None
) -> list[AvailableModel]:
    catalog_provider = {
        "gemini": "google",
        "openai-codex": "openai",
        "xai-oauth": "xai",
    }.get(provider, provider)
    if catalog_provider != "nous":
        records = _catalog_models("hermes", catalog_provider, catalog)
        if configured_model:
            records = [row for row in records if row.model != configured_model]
            records.append(
                replace(
                    _record(
                        "hermes",
                        configured_model,
                        configured_model,
                        "configured",
                        "api-equivalent",
                        catalog,
                        catalog_provider,
                        configured_model,
                        inference_provider=provider,
                        inventory_retrieved_at=datetime.now(UTC).isoformat(),
                    ),
                    inventory_source="hermes config get model --json",
                )
            )
        if not records:
            raise AOPError(
                f"the model catalog has no entries for Hermes provider {provider}"
            )
        return sorted(records, key=lambda row: row.model)
    values = _fetch_nous_models()
    records = []
    for value in values:
        model = value.get("id")
        if not isinstance(model, str):
            continue
        price = value.get("pricing")
        price = price if isinstance(price, dict) else {}
        records.append(
            AvailableModel(
                agent="hermes",
                model=model,
                name=str(value.get("name") or model),
                availability="account-endpoint",
                price_scope="provider" if price else "unknown",
                input_per_million_usd=_per_million(price.get("prompt")),
                cached_input_per_million_usd=_per_million(
                    price.get("input_cache_read")
                ),
                cache_write_per_million_usd=_per_million(
                    price.get("input_cache_write")
                ),
                output_per_million_usd=_per_million(price.get("completion")),
                pricing_source=NOUS_MODELS_URL if price else None,
            )
        )
    if not records:
        raise AOPError("Nous Portal did not return any models")
    return records


def _fetch_nous_models() -> list[dict[str, Any]]:
    request = urllib.request.Request(
        NOUS_MODELS_URL,
        headers={"Accept": "application/json", "User-Agent": "aop-model-list/0.1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            value = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise AOPError(f"could not fetch Hermes models: {error}") from error
    data = value.get("data") if isinstance(value, dict) else None
    if not isinstance(data, list):
        raise AOPError("Hermes model endpoint returned an invalid catalog")
    return [item for item in data if isinstance(item, dict)]


def _record(
    agent: str,
    model: str,
    name: str,
    availability: str,
    price_scope: str,
    catalog: ModelCatalog,
    provider: str | None,
    priced_model: str | None,
    *,
    inference_provider: str | None = None,
    inventory_retrieved_at: str | None = None,
    inventory_sha256: str | None = None,
    authenticated: bool = False,
) -> AvailableModel:
    metadata = (
        pricing_metadata(catalog, provider, priced_model)
        if provider and priced_model
        else None
    )
    cost = metadata.get("cost") if isinstance(metadata, dict) else None
    provenance = metadata.get("aop_pricing", {}) if isinstance(metadata, dict) else {}
    cost = cost if isinstance(cost, dict) else {}
    return AvailableModel(
        agent=agent,
        model=model,
        name=name,
        availability=availability,
        price_scope=(
            "api-equivalent-peak" if provenance.get("basis") == "peak" else price_scope
        )
        if cost
        else "unknown",
        inference_provider=inference_provider,
        inventory_retrieved_at=inventory_retrieved_at,
        inventory_sha256=inventory_sha256,
        authenticated=authenticated,
        input_per_million_usd=_number(cost.get("input")),
        cached_input_per_million_usd=_number(cost.get("cache_read")),
        cache_write_per_million_usd=_number(cost.get("cache_write")),
        output_per_million_usd=_number(cost.get("output")),
        pricing_source=provenance.get("source", catalog.source) if cost else None,
        pricing_retrieved_at=provenance.get("verified_at", catalog.fetched_at)
        if cost
        else None,
    )


def _registry_identity(agent: str, model: str) -> tuple[str | None, str | None]:
    if agent == "cursor":
        candidate = re.sub(r"-(?:low|medium|high|xhigh)(?:-fast)?$", "", model)
        candidate = candidate.removesuffix("-fast")
        return ("openai", candidate) if candidate.startswith("gpt-") else (None, None)
    if agent == "agy":
        candidate = re.sub(r"-(?:low|medium|high)$", "", model)
        return "google", candidate
    return None, None


def _name(metadata: object, fallback: str) -> str:
    if isinstance(metadata, dict) and isinstance(metadata.get("name"), str):
        return metadata["name"]
    return fallback


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _per_million(value: object) -> float | None:
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


def _binary(agent: str, variable: str, default: str) -> str:
    binary = os.environ.get(variable, default)
    _require_binary(binary)
    return binary


def _require_binary(binary: str) -> None:
    if os.path.sep not in binary and shutil.which(binary) is None:
        raise AOPError(f"{binary} is not installed")
    if os.path.sep in binary and not Path(binary).is_file():
        raise AOPError(f"{binary} is not installed")


def _run(command: list[str], *, input: str | None = None) -> str:
    try:
        result = subprocess.run(
            command,
            input=input,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired as error:
        raise AOPError(f"{command[0]} model discovery timed out") from error
    if result.returncode:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise AOPError(f"{command[0]} model discovery failed: {detail}")
    return result.stdout
