from __future__ import annotations

import hashlib
import json
import os
import tomllib
import urllib.error
from pathlib import Path

import pytest

from agent_orchestration_process import codex_routes
from agent_orchestration_process.codex_routes import (
    ZAI_CODING_PLAN,
    fetch_zai_inventory,
    projected_codex_config,
    resolve_codex_route,
)
from agent_orchestration_process.runner import AgentRunner, CodexAdapter
from agent_orchestration_process.worktrees import AOPError, WorktreeManager


MODEL = {
    "slug": "glm-5.3-flash",
    "display_name": "GLM-5.3 Flash",
    "context_window": 200_000,
}
MODEL_DIGEST = hashlib.sha256(
    json.dumps([MODEL], separators=(",", ":"), sort_keys=True).encode()
).hexdigest()


def _effective_config(provider: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "model": "glm-5.3-flash",
        "model_provider": "local-zai",
        "model_providers": {
            "local-zai": provider
            or {
                "name": "Z.AI Coding Plan",
                "base_url": "https://api.z.ai/api/v1",
                "env_key": "ZAI_API_KEY",
                "wire_api": "responses",
                "requires_openai_auth": False,
            },
            "unrelated": {
                "base_url": "https://example.invalid/v1",
                "env_key": "OPENAI_API_KEY",
                "wire_api": "responses",
            },
        },
    }


def _stub_discovery(monkeypatch: pytest.MonkeyPatch, config: dict[str, object]) -> None:
    origins = {
        f"model_providers.{provider}.{field}": {"name": {"type": "user"}}
        for provider in ("local-zai", "second-zai")
        for field in ("base_url", "env_key", "wire_api")
    }
    monkeypatch.setattr(
        codex_routes,
        "_codex_config_read",
        lambda *args, **kwargs: {"result": {"config": config, "origins": origins}},
    )
    monkeypatch.setattr(
        codex_routes,
        "fetch_zai_inventory",
        lambda endpoint, credential: (
            (MODEL,),
            "2026-08-28T00:00:00+00:00",
            MODEL_DIGEST,
        ),
    )


def test_native_codex_route_is_normalized_with_original_provider_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_discovery(monkeypatch, _effective_config())

    provider, model, route = resolve_codex_route(
        "codex",
        tmp_path,
        tmp_path,
        None,
        None,
        {"ZAI_API_KEY": "secret"},
    )

    assert provider == ZAI_CODING_PLAN
    assert model == "glm-5.3-flash"
    assert route is not None
    assert route.native_provider == "local-zai"
    assert route.credential_env == "ZAI_API_KEY"
    assert route.inventory_models == (MODEL,)


def test_explicit_codex_route_requires_model_and_unambiguous_valid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_discovery(monkeypatch, _effective_config())
    with pytest.raises(AOPError, match="requires an explicit --model"):
        resolve_codex_route(
            "codex",
            tmp_path,
            tmp_path,
            ZAI_CODING_PLAN,
            None,
            {"ZAI_API_KEY": "secret"},
        )

    config = _effective_config()
    providers = config["model_providers"]
    assert isinstance(providers, dict)
    providers["second-zai"] = dict(providers["local-zai"])
    _stub_discovery(monkeypatch, config)
    with pytest.raises(AOPError, match="multiple Z.AI Coding Plan providers"):
        resolve_codex_route(
            "codex",
            tmp_path,
            tmp_path,
            ZAI_CODING_PLAN,
            "glm-5.3-flash",
            {"ZAI_API_KEY": "secret"},
        )


def test_project_config_cannot_choose_the_credential_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _effective_config()
    origins = {
        f"model_providers.local-zai.{field}": {
            "name": {"type": "project" if field == "env_key" else "user"}
        }
        for field in ("base_url", "env_key", "wire_api")
    }
    monkeypatch.setattr(
        codex_routes,
        "_codex_config_read",
        lambda *args, **kwargs: {"result": {"config": config, "origins": origins}},
    )

    with pytest.raises(AOPError, match="no valid Z.AI Coding Plan"):
        resolve_codex_route(
            "codex",
            tmp_path,
            tmp_path,
            ZAI_CODING_PLAN,
            "glm-5.3-flash",
            {"ZAI_API_KEY": "secret"},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_url", "https://api.z.ai/api/v1/"),
        ("base_url", "http://api.z.ai/api/v1"),
        ("base_url", "https://user@api.z.ai/api/v1"),
        ("base_url", "https://api.z.ai:443/api/v1"),
        ("base_url", "https://api.z.ai/api/v1?route=other"),
        ("wire_api", "chat"),
        ("requires_openai_auth", True),
        ("http_headers", {"Authorization": "secret"}),
        ("env_http_headers", {"Authorization": "OTHER_KEY"}),
        ("query_params", {"key": "value"}),
        ("experimental_bearer_token", "secret"),
        ("env_key", "INVALID-NAME"),
    ],
)
def test_route_recognition_rejects_noncanonical_or_mixed_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    config = _effective_config()
    providers = config["model_providers"]
    assert isinstance(providers, dict)
    provider = providers["local-zai"]
    assert isinstance(provider, dict)
    provider[field] = value
    _stub_discovery(monkeypatch, config)

    with pytest.raises(AOPError, match="no valid Z.AI Coding Plan"):
        resolve_codex_route(
            "codex",
            tmp_path,
            tmp_path,
            ZAI_CODING_PLAN,
            "glm-5.3-flash",
            {"ZAI_API_KEY": "secret"},
        )


def test_projected_config_contains_only_pinned_route() -> None:
    route = codex_routes.InferenceRoute(
        provider=ZAI_CODING_PLAN,
        native_provider="local-zai",
        endpoint="https://api.z.ai/api/v1",
        wire_api="responses",
        credential_env="ZAI_API_KEY",
        authenticated=True,
        inventory_retrieved_at="2026-08-28T00:00:00+00:00",
        inventory_sha256=MODEL_DIGEST,
        inventory_models=(MODEL,),
    )

    config = projected_codex_config(route, "glm-5.3-flash")

    assert '[model_providers."local-zai"]' in config
    assert "https://api.z.ai/api/v1" in config
    assert "ZAI_API_KEY" in config
    assert "secret" not in config
    parsed = tomllib.loads(config)
    assert parsed["model_providers"]["local-zai"]["wire_api"] == "responses"


def test_recorded_route_snapshot_fails_closed_on_corrupt_provenance() -> None:
    route = codex_routes.InferenceRoute(
        provider=ZAI_CODING_PLAN,
        native_provider="local-zai",
        endpoint="https://api.z.ai/api/v1",
        wire_api="responses",
        credential_env="ZAI_API_KEY",
        authenticated=True,
        inventory_retrieved_at="2026-08-28T00:00:00+00:00",
        inventory_sha256="0" * 64,
        inventory_models=(MODEL,),
    )

    with pytest.raises(AOPError, match="inventory hash does not match"):
        codex_routes.verify_codex_route(
            route, "glm-5.3-flash", {"ZAI_API_KEY": "secret"}
        )


class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self.content[:limit]


class _Opener:
    def __init__(self, response: object):
        self.response = response

    def open(self, request: object, timeout: int) -> object:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_inventory_is_normalized_hashed_and_secret_free_on_http_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = json.dumps({"models": [MODEL]}).encode()
    monkeypatch.setattr(
        codex_routes.urllib.request,
        "build_opener",
        lambda handler: _Opener(_Response(content)),
    )

    models, _, digest = fetch_zai_inventory("https://api.z.ai/api/v1", "do-not-record")

    canonical = json.dumps(list(models), separators=(",", ":"), sort_keys=True).encode()
    assert models == (MODEL,)
    assert digest == hashlib.sha256(canonical).hexdigest()

    error = urllib.error.HTTPError("redacted", 401, "unauthorized", {}, None)
    monkeypatch.setattr(
        codex_routes.urllib.request,
        "build_opener",
        lambda handler: _Opener(error),
    )
    with pytest.raises(AOPError, match="HTTP 401") as caught:
        fetch_zai_inventory("https://api.z.ai/api/v1", "do-not-record")
    assert "do-not-record" not in str(caught.value)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"not-json", "invalid JSON"),
        (json.dumps({"models": []}).encode(), "has no models"),
        (
            json.dumps({"models": [MODEL, MODEL]}).encode(),
            "duplicate model IDs",
        ),
        (b"x" * (codex_routes._MAX_INVENTORY_BYTES + 1), "exceeded 5 MB"),
    ],
)
def test_inventory_fails_closed_on_invalid_or_oversized_responses(
    monkeypatch: pytest.MonkeyPatch, content: bytes, message: str
) -> None:
    monkeypatch.setattr(
        codex_routes.urllib.request,
        "build_opener",
        lambda handler: _Opener(_Response(content)),
    )

    with pytest.raises(AOPError, match=message):
        fetch_zai_inventory("https://api.z.ai/api/v1", "secret")


def test_codex_route_run_projects_one_credential_and_resume_keeps_snapshot(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _effective_config()
    monkeypatch.setenv("AOP_FAKE_CODEX_CONFIG_READ", json.dumps(config))
    monkeypatch.setenv("ZAI_API_KEY", "first-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-secret")
    monkeypatch.setattr(
        codex_routes,
        "fetch_zai_inventory",
        lambda endpoint, credential: (
            (MODEL,),
            "2026-08-28T00:00:00+00:00",
            MODEL_DIGEST,
        ),
    )
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, CodexAdapter(os.fspath(fake_codex)))

    first = runner.run(
        task="zai-route",
        prompt="CHECK_ZAI_ROUTE first",
        model="glm-5.3-flash",
        inference_provider=ZAI_CODING_PLAN,
    )
    monkeypatch.setenv(
        "AOP_FAKE_CODEX_CONFIG_READ",
        json.dumps(
            {"model": "changed", "model_provider": "openai", "model_providers": {}}
        ),
    )
    monkeypatch.setenv("ZAI_API_KEY", "rotated-secret")
    resumed = runner.resume(run_id=first.run_id, prompt="CHECK_ZAI_ROUTE second")

    assert first.succeeded
    assert resumed.succeeded
    assert first.inference_provider == ZAI_CODING_PLAN
    assert resumed.inference_route == first.inference_route
    assert first.billing.route == "subscription"
    assert first.billing.credential_source == "ZAI_API_KEY"
    assert first.calculated_cost is not None
    assert first.calculated_cost.priced_as == "glm-5.3-flash"
    persisted = "\n".join(
        path.read_text() for path in (manager.state_dir / "runs").glob("*/request.json")
    )
    assert "first-secret" not in persisted
    assert "rotated-secret" not in persisted
    assert "unrelated-secret" not in persisted


def test_codex_route_no_web_keeps_only_generated_provider_config(
    repository: Path,
    fake_codex: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AOP_FAKE_CODEX_CONFIG_READ", json.dumps(_effective_config()))
    monkeypatch.setenv("ZAI_API_KEY", "secret")
    monkeypatch.setattr(
        codex_routes,
        "fetch_zai_inventory",
        lambda endpoint, credential: (
            (MODEL,),
            "2026-08-28T00:00:00+00:00",
            MODEL_DIGEST,
        ),
    )

    result = AgentRunner(
        WorktreeManager.discover(repository), CodexAdapter(os.fspath(fake_codex))
    ).run(
        task="zai-no-web",
        prompt="CHECK_ZAI_ROUTE no-web",
        model="glm-5.3-flash",
        inference_provider=ZAI_CODING_PLAN,
        no_web=True,
    )

    assert result.succeeded
    assert result.command[-2:] == ["app-server", "--stdio"]
    events = [
        json.loads(line)
        for line in (repository / ".aop" / "runs" / result.run_id / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    thread_start = next(
        event for event in events if event.get("method") == "thread/start"
    )
    config = thread_start["params"]["config"]
    assert config["web_search"] == "disabled"
    assert "tools" not in config
    assert config["permissions"]["aop-no-web"]["network"]["enabled"] is False
    assert thread_start["params"]["permissions"] == "aop-no-web"
    assert "sandbox" not in thread_start["params"]
