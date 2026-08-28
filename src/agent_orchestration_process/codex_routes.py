"""Validated Codex inference routes and authenticated model inventories."""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import InferenceRoute
from .worktrees import AOPError


ZAI_CODING_PLAN = "zai-coding-plan"
ZAI_CODING_PLAN_ENDPOINT = "https://api.z.ai/api/v1"
_MAX_INVENTORY_BYTES = 5_000_000
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_PROVIDER_NAME = re.compile(r"[A-Za-z0-9._-]+\Z")
_MODEL_FIELDS = {
    "apply_patch_tool_type",
    "base_instructions",
    "context_window",
    "default_reasoning_level",
    "default_reasoning_summary",
    "description",
    "display_name",
    "effective_context_window_percent",
    "experimental_supported_tools",
    "input_modalities",
    "max_context_window",
    "priority",
    "shell_type",
    "slug",
    "support_verbosity",
    "supported_in_api",
    "supported_reasoning_levels",
    "supports_parallel_tool_calls",
    "supports_reasoning_summaries",
    "truncation_policy",
    "visibility",
}


def resolve_codex_route(
    binary: str,
    source_home: Path | None,
    cwd: Path,
    inference_provider: str | None,
    model: str | None,
    environment: dict[str, str],
    *,
    require_explicit_model: bool = True,
) -> tuple[str | None, str | None, InferenceRoute | None]:
    """Resolve a requested or native Codex route through effective config."""
    if inference_provider not in {None, ZAI_CODING_PLAN}:
        raise AOPError(
            f'codex inference provider "{inference_provider}" is not supported'
        )
    response = _codex_config_read(binary, source_home, cwd, environment)
    config = response.get("result", {}).get("config")
    if not isinstance(config, dict):
        raise AOPError("Codex config/read did not return effective configuration")
    providers = config.get("model_providers", {})
    if not isinstance(providers, dict):
        raise AOPError("Codex effective model_providers must be a mapping")

    matches = [
        (name, value)
        for name, value in providers.items()
        if isinstance(name, str)
        and isinstance(value, dict)
        and _is_zai_coding_plan(value)
        and _trusted_route_origin(response, name)
    ]
    selected = config.get("model_provider")
    selected_match = next((item for item in matches if item[0] == selected), None)
    if inference_provider is None and selected_match is None:
        return None, model, None
    if inference_provider == ZAI_CODING_PLAN:
        if not matches:
            raise AOPError(
                "Codex has no valid Z.AI Coding Plan Responses provider configured"
            )
        if len(matches) != 1:
            names = ", ".join(sorted(name for name, _ in matches))
            raise AOPError(
                f"Codex has multiple Z.AI Coding Plan providers configured: {names}"
            )
        selected_match = matches[0]
    assert selected_match is not None

    native_provider, provider_config = selected_match
    credential_env = _credential_env(provider_config)
    credential = environment.get(credential_env)
    if not credential:
        raise AOPError(
            f"Codex provider {native_provider} requires {credential_env} in the environment"
        )
    effective_model = model or config.get("model")
    if require_explicit_model and inference_provider is not None and model is None:
        raise AOPError("codex --provider requires an explicit --model")
    if effective_model is not None and not isinstance(effective_model, str):
        raise AOPError("Codex effective model must be a string")
    if require_explicit_model and not effective_model:
        raise AOPError("Z.AI Coding Plan configuration has no effective model")

    inventory, retrieved_at, digest = fetch_zai_inventory(
        ZAI_CODING_PLAN_ENDPOINT, credential
    )
    if effective_model and effective_model not in _inventory_slugs(inventory):
        raise AOPError(
            f'Z.AI Coding Plan inventory does not contain model "{effective_model}"'
        )
    route = InferenceRoute(
        provider=ZAI_CODING_PLAN,
        native_provider=native_provider,
        endpoint=ZAI_CODING_PLAN_ENDPOINT,
        wire_api="responses",
        credential_env=credential_env,
        authenticated=True,
        inventory_retrieved_at=retrieved_at,
        inventory_sha256=digest,
        inventory_models=inventory,
    )
    return ZAI_CODING_PLAN, effective_model, route


def verify_codex_route(
    route: InferenceRoute, model: str | None, environment: dict[str, str]
) -> None:
    """Verify pinned route availability without replacing original provenance."""
    _validate_route_snapshot(route, model)
    credential = environment.get(route.credential_env)
    if not credential:
        raise AOPError(
            f"Codex provider {route.native_provider} requires "
            f"{route.credential_env} in the environment"
        )
    inventory, _, _ = fetch_zai_inventory(route.endpoint, credential)
    if model and model not in _inventory_slugs(inventory):
        raise AOPError(
            f'Z.AI Coding Plan inventory does not contain pinned model "{model}"'
        )


def projected_codex_config(
    route: InferenceRoute, model: str, catalog_path: str = "models.json"
) -> str:
    """Return the minimum Codex TOML needed for a selected custom route."""
    _validate_route_snapshot(route, model)
    quote = json.dumps
    return "\n".join(
        [
            f"model_provider = {quote(route.native_provider)}",
            f"model = {quote(model)}",
            f"model_catalog_json = {quote(catalog_path)}",
            "",
            f"[model_providers.{quote(route.native_provider)}]",
            'name = "Z.AI Coding Plan"',
            f"base_url = {quote(route.endpoint)}",
            f"env_key = {quote(route.credential_env)}",
            f"wire_api = {quote(route.wire_api)}",
            "requires_openai_auth = false",
            "",
        ]
    )


def inventory_document(route: InferenceRoute) -> str:
    _validate_route_snapshot(route, None)
    return f"{json.dumps({'models': route.inventory_models}, sort_keys=True)}\n"


def fetch_zai_inventory(
    endpoint: str, credential: str
) -> tuple[tuple[dict[str, Any], ...], str, str]:
    if endpoint != ZAI_CODING_PLAN_ENDPOINT:
        raise AOPError("recorded Z.AI Coding Plan endpoint is not canonical")
    request = urllib.request.Request(
        f"{endpoint}/models",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {credential}",
            "User-Agent": "aop-zai-inventory/0.1",
        },
    )
    opener = urllib.request.build_opener(_NoRedirects())
    try:
        with opener.open(request, timeout=20) as response:
            content = response.read(_MAX_INVENTORY_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise AOPError(
            f"Z.AI Coding Plan model inventory returned HTTP {error.code}"
        ) from None
    except (urllib.error.URLError, OSError):
        raise AOPError("Z.AI Coding Plan model inventory is unavailable") from None
    if len(content) > _MAX_INVENTORY_BYTES:
        raise AOPError("Z.AI Coding Plan model inventory exceeded 5 MB")
    try:
        value = json.loads(content)
    except (UnicodeError, json.JSONDecodeError):
        raise AOPError("Z.AI Coding Plan model inventory is invalid JSON") from None
    models = value.get("models") if isinstance(value, dict) else None
    if not isinstance(models, list) or not models:
        raise AOPError("Z.AI Coding Plan model inventory has no models")
    normalized: list[dict[str, Any]] = []
    slugs: set[str] = set()
    for item in models:
        if not isinstance(item, dict) or not isinstance(item.get("slug"), str):
            raise AOPError("Z.AI Coding Plan model inventory has invalid entries")
        slug = item["slug"]
        if not slug or slug in slugs:
            raise AOPError("Z.AI Coding Plan model inventory has duplicate model IDs")
        slugs.add(slug)
        projected = {key: item[key] for key in sorted(_MODEL_FIELDS & item.keys())}
        try:
            normalized.append(json.loads(json.dumps(projected, sort_keys=True)))
        except (TypeError, ValueError):
            raise AOPError(
                "Z.AI Coding Plan model inventory has invalid entries"
            ) from None
    normalized.sort(key=lambda item: item["slug"])
    canonical = json.dumps(normalized, separators=(",", ":"), sort_keys=True).encode()
    retrieved_at = datetime.now(UTC).isoformat()
    return tuple(normalized), retrieved_at, hashlib.sha256(canonical).hexdigest()


def _codex_config_read(
    binary: str,
    source_home: Path | None,
    cwd: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    child_environment = {
        name: environment[name]
        for name in (
            "HOME",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "PATH",
            "TERM",
            "TZ",
            "AOP_FAKE_CODEX_CONFIG_READ",
        )
        if name in environment
    }
    if source_home is not None:
        child_environment["CODEX_HOME"] = os.fspath(source_home)
    try:
        process = subprocess.Popen(
            [binary, "app-server", "--stdio"],
            cwd=cwd,
            env=child_environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError as error:
        raise AOPError(f"could not start Codex config discovery: {error}") from error
    assert process.stdin is not None
    try:
        _write_line(
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
        _read_response(process, 1)
        _write_line(process.stdin, {"method": "initialized", "params": {}})
        _write_line(
            process.stdin,
            {
                "method": "config/read",
                "id": 2,
                "params": {"cwd": os.fspath(cwd), "includeLayers": True},
            },
        )
        return _read_response(process, 2)
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _write_line(stream: Any, value: object) -> None:
    stream.write(f"{json.dumps(value)}\n")
    stream.flush()


def _read_response(process: subprocess.Popen[str], response_id: int) -> dict[str, Any]:
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
                if "error" in value:
                    raise AOPError(
                        f"Codex config discovery failed for response {response_id}"
                    )
                return value
    finally:
        selector.close()
    raise AOPError(f"Codex did not return config response {response_id}")


def _is_zai_coding_plan(value: dict[str, Any]) -> bool:
    if value.get("base_url") != ZAI_CODING_PLAN_ENDPOINT:
        return False
    if value.get("wire_api") != "responses":
        return False
    if value.get("requires_openai_auth", False) is not False:
        return False
    if _credential_env(value, strict=False) is None:
        return False
    forbidden = (
        "experimental_bearer_token",
        "auth",
        "aws",
        "query_params",
        "http_headers",
        "env_http_headers",
    )
    return all(value.get(name) in (None, {}) for name in forbidden)


def _credential_env(value: dict[str, Any], *, strict: bool = True) -> str | None:
    credential_env = value.get("env_key")
    if isinstance(credential_env, str) and _ENV_NAME.fullmatch(credential_env):
        return credential_env
    if strict:
        raise AOPError("Z.AI Coding Plan provider has an invalid env_key")
    return None


def _trusted_route_origin(response: dict[str, Any], native_provider: str) -> bool:
    result = response.get("result")
    origins = result.get("origins") if isinstance(result, dict) else None
    if not isinstance(origins, dict):
        return False
    for field in ("base_url", "env_key", "wire_api"):
        metadata = origins.get(f"model_providers.{native_provider}.{field}")
        name = metadata.get("name") if isinstance(metadata, dict) else None
        origin_type = name.get("type") if isinstance(name, dict) else None
        if not isinstance(origin_type, str) or origin_type == "project":
            return False
    return True


def _inventory_slugs(models: tuple[dict[str, Any], ...]) -> set[str]:
    return {item["slug"] for item in models}


def _validate_route_snapshot(route: InferenceRoute, model: str | None) -> None:
    if route.provider != ZAI_CODING_PLAN:
        raise AOPError(f"unsupported recorded Codex route: {route.provider}")
    if route.endpoint != ZAI_CODING_PLAN_ENDPOINT or route.wire_api != "responses":
        raise AOPError("recorded Z.AI Coding Plan route is not canonical")
    if not route.authenticated:
        raise AOPError("recorded Z.AI Coding Plan inventory is not authenticated")
    if not _PROVIDER_NAME.fullmatch(route.native_provider):
        raise AOPError("recorded Codex provider key is invalid")
    if not _ENV_NAME.fullmatch(route.credential_env):
        raise AOPError("recorded Codex credential reference is invalid")
    if not route.inventory_models:
        raise AOPError("recorded Z.AI Coding Plan inventory is empty")
    slugs: list[str] = []
    for item in route.inventory_models:
        slug = item.get("slug") if isinstance(item, dict) else None
        if not isinstance(slug, str) or not slug:
            raise AOPError("recorded Z.AI Coding Plan inventory is invalid")
        slugs.append(slug)
    if len(slugs) != len(set(slugs)):
        raise AOPError("recorded Z.AI Coding Plan inventory has duplicate model IDs")
    canonical = json.dumps(
        list(route.inventory_models), separators=(",", ":"), sort_keys=True
    ).encode()
    if hashlib.sha256(canonical).hexdigest() != route.inventory_sha256:
        raise AOPError("recorded Z.AI Coding Plan inventory hash does not match")
    try:
        retrieved_at = datetime.fromisoformat(route.inventory_retrieved_at)
    except ValueError:
        raise AOPError(
            "recorded Z.AI Coding Plan inventory retrieval time is invalid"
        ) from None
    if retrieved_at.tzinfo is None:
        raise AOPError("recorded Z.AI Coding Plan inventory retrieval time is invalid")
    if model and model not in slugs:
        raise AOPError(
            f'recorded Z.AI Coding Plan inventory does not contain model "{model}"'
        )


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Any,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        return None
