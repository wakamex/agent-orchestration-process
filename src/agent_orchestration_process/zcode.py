"""Native Zcode configuration projection and stream interpretation."""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from .pricing import TokenUsage, CalculatedCost, estimate_api_cost
from .model_catalog import ensure_catalog_fresh
from .worktrees import AOPError


def source_home(environment: dict[str, str]) -> Path:
    return (
        Path(
            environment.get(
                "AOP_ZCODE_SOURCE_HOME", str(Path(environment["HOME"]) / ".zcode")
            )
        )
        .expanduser()
        .resolve()
    )


def read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise AOPError(f"could not read Zcode configuration: {path}") from error
    if not isinstance(value, dict):
        raise AOPError(f"Zcode configuration must be an object: {path}")
    for key in ("provider", "model"):
        if key in value and not isinstance(value[key], dict):
            raise AOPError(f"Zcode {key} configuration must be an object: {path}")
    for definition in value.get("provider", {}).values():
        if not isinstance(definition, dict) or any(
            key in definition and not isinstance(definition[key], dict)
            for key in ("options", "models")
        ):
            raise AOPError(f"invalid Zcode provider definition: {path}")
    return value


def configured_models(config: dict[str, Any]) -> list[str]:
    models = []
    for provider, definition in config.get("provider", {}).items():
        for model in definition.get("models", {}):
            models.append(f"{provider}/{model}")
    for entry in config.get("model", {}).values():
        if isinstance(entry, str) and "/" in entry and entry not in models:
            models.append(entry)
    return models


def project_config(
    config: dict[str, Any], model: str | None, *, sealed: bool
) -> tuple[dict[str, Any], str, list[str]]:
    selection = model or config.get("model", {}).get("main")
    if not isinstance(selection, str) or "/" not in selection:
        raise AOPError(
            "Zcode requires a configured provider/model or an explicit --model provider/model"
        )
    provider, model_id = selection.split("/", 1)
    if not provider or not model_id:
        raise AOPError("Zcode model must be provider/model")
    definition = config.get("provider", {}).get(provider)
    if not isinstance(definition, dict):
        raise AOPError(
            f"Zcode provider {provider!r} is not configured in the native provider map"
        )
    # Zcode owns provider validation and model execution. Project one provider,
    # including its lite model, without exporting other provider credentials.
    selected_models = {"main": selection}
    lite = config.get("model", {}).get("lite")
    if isinstance(lite, str) and lite.startswith(provider + "/"):
        selected_models["lite"] = lite
    result = {"provider": {provider: definition}, "model": selected_models}
    for key in ("network", "modelStream", "toolConcurrency", "modelAnomalyGuard"):
        if key in config:
            result[key] = config[key]
    if not sealed:
        for key in (
            "plugins",
            "mcp",
            "skills",
            "skillOverrides",
            "commandOverrides",
            "hooks",
            "memory",
            "ui",
            "features",
        ):
            if key in config:
                result[key] = config[key]
    else:
        result.update(
            {
                "plugins": {"enabled": False},
                "skills": {"enabled": False, "includeInstructions": False},
                "memory": {"use": False},
                "features": {"memory": False, "skill": False, "mcp": False},
            }
        )
    return result, selection, credential_env_names(provider, definition)


def credential_env_names(provider: str, definition: dict[str, Any]) -> list[str]:
    candidates = []
    kind = definition.get("kind")
    if kind == "anthropic":
        candidates.append("ANTHROPIC_API_KEY")
    elif kind == "openai":
        candidates.append("OPENAI_API_KEY")
    # Match Zcode's native FRi credential precedence.
    import re

    for name in (
        definition.get("name"),
        provider,
        re.sub(r"^default[-_]", "", provider),
    ):
        if isinstance(name, str):
            normalized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
            if normalized:
                candidates.append(normalized + "_API_KEY")
    candidates.append("ZCODE_API_KEY")
    return list(dict.fromkeys(candidates))


def selection_fingerprint(config: dict[str, Any]) -> str:
    selection = {key: config.get(key) for key in ("model", "provider")}
    return hashlib.sha256(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def token_usage(raw: Any) -> TokenUsage | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or not any(
        name in raw for name in ("inputTokens", "outputTokens")
    ):
        raise ValueError("Zcode reported invalid token usage")
    counts = [
        raw.get(name, 0)
        for name in (
            "inputTokens",
            "cacheReadTokens",
            "outputTokens",
            "reasoningTokens",
        )
    ]
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in counts
    ):
        raise ValueError("Zcode reported invalid token usage")
    try:
        return TokenUsage(*counts)
    except ValueError as error:
        raise ValueError("Zcode reported inconsistent token usage") from error


def sum_usage(usages: list[TokenUsage]) -> TokenUsage:
    return TokenUsage(
        *(
            sum(getattr(usage, name) for usage in usages)
            for name in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
            )
        )
    )


def calculate_cost(
    model: str, usage: TokenUsage | None, requests: list[tuple[str, TokenUsage]]
) -> CalculatedCost | None:
    if (
        usage is None
        or not requests
        or sum_usage([value for _, value in requests]) != usage
    ):
        return None
    catalog = ensure_catalog_fresh()
    costs = [
        estimate_api_cost(name, value, catalog, providers=("zai",))
        for name, value in requests
    ]
    if any(cost is None for cost in costs):
        return None
    return replace(
        costs[0],
        model=model,
        amount_usd=round(sum(cost.amount_usd for cost in costs), 8),
        priced_as=" + ".join(sorted({cost.priced_as for cost in costs})),
        long_context_pricing=any(cost.long_context_pricing for cost in costs),
    )


def parse_stream(stdout: str) -> dict[str, Any]:
    result = None
    session = None
    session_conflict = False
    model = None
    error = None
    malformed = False
    completed = False
    active_turn = None
    requests = {}
    provider_error = None
    duration = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            malformed = True
            continue
        if not isinstance(event, dict):
            malformed = True
            continue
        if event.get("type") == "turn.started":
            active_turn = event.get("turnId")
        if active_turn and event.get("turnId") not in (None, active_turn):
            continue
        reported_session = event.get("sessionId")
        if isinstance(reported_session, str):
            if session is not None and session != reported_session:
                error = "Zcode stream contains conflicting session identities"
                session_conflict = True
            session = reported_session
        payload = event.get("payload")
        if isinstance(payload, dict):
            ref = payload.get("modelRef") or payload.get("model")
            if isinstance(ref, dict) and ref.get("role", "main") == "main":
                if isinstance(ref.get("providerId"), str) and isinstance(
                    ref.get("modelId"), str
                ):
                    observed_model = f"{ref['providerId']}/{ref['modelId']}"
                    if model is not None and model != observed_model:
                        error = "Zcode changed its main model during the turn"
                    model = observed_model
        if (
            isinstance(payload, dict)
            and payload.get("type") == "model_request_completed"
        ):
            ref = payload.get("model", {})
            request_id = payload.get("requestId")
            try:
                measured = token_usage(payload.get("usage"))
                if measured is not None:
                    if not isinstance(request_id, str) or not request_id:
                        raise ValueError("Zcode usage is missing its request identity")
                    if (
                        not isinstance(ref, dict)
                        or not ref.get("providerId")
                        or not ref.get("modelId")
                    ):
                        raise ValueError("Zcode usage is missing its model identity")
                    key = (event.get("turnId"), request_id, payload.get("attempt", 1))
                    measurement = (f"{ref['providerId']}/{ref['modelId']}", measured)
                    if key in requests and requests[key] != measurement:
                        raise ValueError("Zcode reported conflicting request usage")
                    requests[key] = measurement
            except (ValueError, TypeError) as failure:
                error = str(failure)
        if event.get("type") == "turn.failed":
            detail = payload.get("error", {}) if isinstance(payload, dict) else {}
            provider_error = detail.get("message") if isinstance(detail, dict) else None
            error = provider_error or "Zcode reported a failed turn"
        if event.get("type") == "turn.completed":
            completed = True
            if isinstance(payload, dict):
                milliseconds = payload.get("duration")
                if (
                    isinstance(milliseconds, (int, float))
                    and not isinstance(milliseconds, bool)
                    and milliseconds >= 0
                ):
                    duration = milliseconds / 1000
            if (
                isinstance(payload, dict)
                and payload.get("resultType", "success") != "success"
            ):
                error = "Zcode reported an unsuccessful completed turn"
        if event.get("type") == "result":
            if result is not None:
                error = "Zcode emitted multiple terminal results"
            result = event
    result = result or {}
    if result.get("sessionId") != session:
        error = error or "Zcode terminal result did not confirm the session ID"
    projection = result.get("projection", {})
    status = projection.get("status") if isinstance(projection, dict) else None
    if status not in {"idle", "completed"} or not completed:
        error = error or "Zcode did not emit a completed terminal turn"
    if malformed:
        error = error or "Zcode emitted invalid JSON stream data"
    response = result.get("response")
    if not isinstance(response, str) or not response.strip():
        response = None
        error = error or "Zcode did not emit a final response"
    if (
        session_conflict
        or not isinstance(session, str)
        or not session.startswith("sess_")
    ):
        session = None
        error = error or "Zcode did not report a valid session ID"
    raw_usage = result.get("usage")
    usage = None
    complete_usage = False
    if isinstance(raw_usage, dict) and raw_usage.get("source") == "provider":
        try:
            usage = token_usage(raw_usage)
            complete_usage = usage is not None
        except ValueError as failure:
            error = error or str(failure)
    if usage is None and requests:
        usage = sum_usage([value for _, value in requests.values()])
    accounting_status = (
        "unavailable"
        if usage is None
        else "complete"
        if complete_usage and completed and status in {"idle", "completed"}
        else "partial"
    )
    return {
        "session_id": session,
        "model": model,
        "error": error,
        "final_message": response,
        "usage": usage,
        "requests": list(requests.values()),
        "accounting_status": accounting_status,
        "provider_error": provider_error,
        "duration_seconds": duration,
        "status": status,
    }


def protocol_config(
    config: dict[str, Any], environment: dict[str, str]
) -> dict[str, Any]:
    """Project the selected legacy native provider into the app-server schema."""
    model = config["model"]["main"]
    provider, model_id = model.split("/", 1)
    definition = config["provider"][provider]
    options = definition.get("options", {})
    unsupported = set(options) - {"apiKey", "baseURL", "headers"}
    # Native legacy import maps a required key to the api-key access below.
    # No-authentication providers remain unsupported by this projection.
    if options.get("apiKeyRequired") is True:
        unsupported.discard("apiKeyRequired")
    if unsupported:
        raise AOPError(
            "Zcode app-server cannot project provider options: "
            + ", ".join(sorted(unsupported))
        )
    key = options.get("apiKey") or next(
        (
            environment[name]
            for name in credential_env_names(provider, definition)
            if environment.get(name)
        ),
        None,
    )
    if not key:
        raise AOPError(
            "Zcode app-server requires the selected provider's native API key; its client protocol does not load standalone OAuth credentials"
        )
    kinds = {
        "anthropic": "anthropic-messages",
        "openai": "openai-responses",
        "openai-compatible": "openai-chat-completions",
    }
    kind = definition.get("kind", "openai-compatible")
    if kind not in kinds:
        raise AOPError(f"Unsupported Zcode provider kind: {kind}")
    api = {"type": kinds[kind]}
    if options.get("baseURL"):
        api["baseUrl"] = options["baseURL"]
    headers = {**definition.get("headers", {}), **options.get("headers", {})}
    if headers:
        api["headers"] = headers
    model_rules = []
    for member_id, member in definition.get("models", {}).items():
        context_window = member.get(
            "contextWindow", member.get("limit", {}).get("context")
        )
        if context_window is not None:
            model_rules.append(
                {
                    "providerId": provider,
                    "modelId": member_id,
                    "config": {"properties": {"contextWindow": context_window}},
                }
            )
    return {
        "schemaVersion": 1,
        "config": {
            "providerConfigRules": {
                "providerRules": [
                    {
                        "providerId": provider,
                        "config": {
                            "group": "standard-personal",
                            "access": {"type": "api-key", "apiKey": key},
                            "api": api,
                            "personalModelIds": list(
                                dict.fromkeys([model_id, *definition.get("models", {})])
                            ),
                        },
                    }
                ]
            },
            "modelConfigRules": {
                "providerModelRules": model_rules,
                "manualProviderModelRules": [],
            },
            "defaultModelSelection": {"providerId": provider, "modelId": model_id},
        },
    }


def parse_protocol(stdout: str) -> dict[str, Any]:
    """Read native snapshots and subscribed events without counting replayed history."""
    events = []
    terminal = None
    snapshot = None
    for line in stdout.splitlines():
        try:
            message = json.loads(line)
        except ValueError:
            events.append(line)
            continue
        if not isinstance(message, dict):
            events.append(line)
            continue
        if message.get("method") == "session/event":
            event = message.get("params")
            if not isinstance(event, dict) or not isinstance(
                event.get("payload", {}), dict
            ):
                events.append("null")
                continue
            payload = event.get("payload", {})
            if (
                payload.get("type") == "model_request_completed"
                and "model" not in payload
            ):
                event = {
                    **event,
                    "payload": {
                        **payload,
                        "model": {
                            "providerId": payload.get("providerId"),
                            "modelId": payload.get("modelId"),
                            "role": "main"
                            if payload.get("modelRequestSessionType", "main") == "main"
                            else "auxiliary",
                        },
                    },
                }
            events.append(json.dumps(event))
            if event.get("type") == "turn.completed":
                terminal = event
        result = message.get("result")
        if isinstance(result, dict) and isinstance(result.get("session"), dict):
            snapshot = result
            events.append(
                json.dumps(
                    {
                        "type": "session.updated",
                        "sessionId": result["session"].get("sessionId"),
                        "payload": {"modelRef": result["session"].get("model")},
                    }
                )
            )
    if terminal and snapshot:
        payload = terminal.get("payload", {})
        events.append(
            json.dumps(
                {
                    "type": "result",
                    "sessionId": snapshot["session"].get("sessionId"),
                    "response": payload.get("response"),
                    "usage": payload.get("usage"),
                    "projection": snapshot.get("projection"),
                }
            )
        )
    return parse_stream("\n".join(events))
