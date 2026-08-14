"""Model discovery across AOP's supported agent CLIs."""

from __future__ import annotations

import json
import os
import re
import selectors
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .model_catalog import ModelCatalog
from .worktrees import AOPError


AGENTS = ("codex", "claude", "cursor", "devin", "opencode", "agy", "hermes", "dsh")
NOUS_MODELS_URL = "https://inference-api.nousresearch.com/v1/models"


@dataclass(frozen=True)
class AvailableModel:
    agent: str
    model: str
    name: str
    availability: str
    price_scope: str
    input_per_million_usd: float | None = None
    cached_input_per_million_usd: float | None = None
    cache_write_per_million_usd: float | None = None
    output_per_million_usd: float | None = None
    pricing_source: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def list_models(agent: str, catalog: ModelCatalog) -> list[AvailableModel]:
    if agent == "codex":
        return _codex_models(catalog)
    if agent == "cursor":
        return _simple_cli_models(
            agent, _binary("cursor", "AOP_CURSOR_BIN", "agent"), ["models"], catalog
        )
    if agent == "agy":
        return _simple_cli_models(
            agent, _binary("agy", "AOP_AGY_BIN", "agy"), ["models"], catalog
        )
    if agent == "devin":
        return _devin_models()
    if agent == "opencode":
        return _opencode_models(catalog)
    if agent == "claude":
        _require_binary(_binary("claude", "AOP_CLAUDE_BIN", "claude"))
        return _catalog_models(agent, "anthropic", catalog)
    if agent == "hermes":
        binary = _binary("hermes", "AOP_HERMES_BIN", "hermes")
        provider = _run([binary, "config", "get", "model.provider"]).strip()
        return _hermes_models(catalog, provider)
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
                ("deepseek-v4-flash", "DeepSeek-V4-Flash"),
                ("deepseek-v4-pro", "DeepSeek-V4-Pro"),
            )
        ]
    raise AOPError(f"unsupported agent: {agent}")


def _codex_models(catalog: ModelCatalog) -> list[AvailableModel]:
    binary = _binary("codex", "AOP_CODEX_BIN", "codex")
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


def _hermes_models(catalog: ModelCatalog, provider: str) -> list[AvailableModel]:
    catalog_provider = {
        "gemini": "google",
        "openai-codex": "openai",
        "xai-oauth": "xai",
    }.get(provider, provider)
    if catalog_provider != "nous":
        records = _catalog_models("hermes", catalog_provider, catalog)
        if not records:
            raise AOPError(
                f"the model catalog has no entries for Hermes provider {provider}"
            )
        return records
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
) -> AvailableModel:
    metadata = (
        catalog.model(provider, priced_model) if provider and priced_model else None
    )
    cost = metadata.get("cost") if isinstance(metadata, dict) else None
    cost = cost if isinstance(cost, dict) else {}
    return AvailableModel(
        agent=agent,
        model=model,
        name=name,
        availability=availability,
        price_scope=price_scope if cost else "unknown",
        input_per_million_usd=_number(cost.get("input")),
        cached_input_per_million_usd=_number(cost.get("cache_read")),
        cache_write_per_million_usd=_number(cost.get("cache_write")),
        output_per_million_usd=_number(cost.get("output")),
        pricing_source=catalog.source if cost else None,
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
