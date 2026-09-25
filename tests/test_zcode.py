from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_orchestration_process import zcode
from agent_orchestration_process.model_listing import AGENTS, list_models
from agent_orchestration_process.model_catalog import ensure_catalog_fresh
from agent_orchestration_process.runner import AgentRunner, ZcodeAdapter, adapter_for
from agent_orchestration_process.worktrees import AOPError, WorktreeManager


@pytest.fixture
def zcode_config(tmp_path, monkeypatch):
    source = tmp_path / "source-zcode"
    (source / "cli").mkdir(parents=True)
    config = {
        "model": {"main": "test/test-model", "lite": "other/other-model"},
        "provider": {
            "test": {
                "kind": "anthropic",
                "options": {"apiKey": "selected-key", "baseURL": "http://127.0.0.1:1"},
                "models": {"test-model": {}, "second-model": {}},
            },
            "other": {
                "options": {"apiKey": "unrelated-secret"},
                "models": {"other-model": {}},
            },
        },
        "plugins": {"enabled": False},
        "features": {"mcp": False},
    }
    (source / "cli" / "config.json").write_text(json.dumps(config))
    (source / "AGENTS.md").write_text("User instructions")
    (source / "cli" / "history.json").write_text("private history")
    monkeypatch.setenv("AOP_ZCODE_SOURCE_HOME", str(source))
    monkeypatch.delenv("ZCODE_MODEL", raising=False)
    return source, config


@pytest.fixture
def fake_zcode(tmp_path):
    binary = tmp_path / "zcode"
    binary.write_text((Path(__file__).parent / "fixtures/fake_zcode.py").read_text())
    binary.chmod(0o755)
    return binary


@pytest.mark.parametrize("profile", ["edit", "review", "sealed", "host"])
def test_run_and_exact_resume(repository, zcode_config, fake_zcode, profile):
    manager = WorktreeManager.discover(repository)
    runner = AgentRunner(manager, ZcodeAdapter(str(fake_zcode)))
    first = runner.run(
        task="zcode-task", prompt="first", profile=profile, timeout_seconds=10
    )
    assert first.succeeded, first.error
    assert first.model == "test/test-model"
    assert first.usage.input_tokens == 120
    assert first.usage.cached_input_tokens == 30
    assert first.usage.output_tokens == 20
    assert first.time_to_first_response_seconds is not None
    assert first.calculated_cost is None
    assert first.accounting_status == "complete"
    source, config = zcode_config
    config["model"]["main"] = "test/second-model"
    (source / "cli/config.json").write_text(json.dumps(config))
    resumed = runner.resume(run_id=first.run_id, prompt="second")
    assert resumed.succeeded, resumed.error
    assert resumed.model == first.model
    assert resumed.session_id == first.session_id
    request = json.loads(
        (manager.state_dir / "runs" / first.run_id / "request.json").read_text()
    )
    assert request["effective_policy"]["zcode"]["model"] == first.model
    assert request["effective_policy"]["zcode"]["credential_environment_names"] == []
    assert "selected-key" not in json.dumps(first.to_dict())
    wrong = runner.resume(run_id=resumed.run_id, prompt="wrong-session")
    assert not wrong.succeeded
    assert wrong.session_id is None


@pytest.mark.parametrize("prompt", ["timeout", "truncated"])
def test_incomplete_run_fails(repository, zcode_config, fake_zcode, prompt):
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    )
    result = runner.run(task="incomplete", prompt=prompt, timeout_seconds=0.2)
    assert not result.succeeded
    assert result.accounting_status == "unavailable"
    assert result.usage is None


def test_capabilities_and_inventory(
    repository, zcode_config, fake_zcode, monkeypatch, fresh_model_catalog
):
    monkeypatch.setenv("AOP_ZCODE_BIN", str(fake_zcode))
    assert "zcode" in AGENTS
    assert isinstance(adapter_for("zcode"), ZcodeAdapter)
    models = list_models("zcode", ensure_catalog_fresh())
    assert {entry.model for entry in models} == {
        "test/test-model",
        "test/second-model",
        "other/other-model",
    }
    assert all(entry.availability == "configured" for entry in models)
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    )
    for kwargs in (
        {"no_web": True},
        {"mode": "participant"},
        {"inference_provider": "test", "model": "test/test-model"},
    ):
        with pytest.raises(AOPError):
            runner.run(task="unsupported", prompt="test", **kwargs)


def test_usage_validation():
    events = [
        {"type": "turn.completed", "sessionId": "sess_test"},
        {
            "type": "result",
            "sessionId": "sess_test",
            "response": "ok",
            "projection": {"status": "idle"},
            "usage": {"source": "provider", "inputTokens": 1, "cacheReadTokens": 2},
        },
    ]
    parsed = zcode.parse_stream("\n".join(map(json.dumps, events)))
    assert parsed["usage"] is None
    assert "inconsistent token usage" in parsed["error"]


def test_selected_environment_credential(
    repository, zcode_config, fake_zcode, monkeypatch
):
    source, config = zcode_config
    del config["provider"]["test"]["options"]["apiKey"]
    (source / "cli/config.json").write_text(json.dumps(config))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "selected-environment-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-environment-key")
    script = fake_zcode.read_text().replace(
        "config = json.loads(",
        "assert os.environ['ANTHROPIC_API_KEY'] == 'selected-environment-key'\nassert 'OPENAI_API_KEY' not in os.environ\nconfig = json.loads(",
    )
    fake_zcode.write_text(script)
    manager = WorktreeManager.discover(repository)
    result = AgentRunner(manager, ZcodeAdapter(str(fake_zcode))).run(
        task="credential", prompt="test", profile="sealed", timeout_seconds=10
    )
    assert result.succeeded, result.error
    record = json.loads(
        (manager.state_dir / "runs" / result.run_id / "request.json").read_text()
    )
    assert record["effective_policy"]["environment"]["credential_names"] == [
        "ANTHROPIC_API_KEY"
    ]
    assert (
        "OPENAI_API_KEY"
        not in record["effective_policy"]["environment"]["allowed_names"]
    )
    assert "selected-environment-key" not in json.dumps(result.to_dict())
    assert "unrelated-environment-key" not in json.dumps(record)


@pytest.mark.parametrize(
    "config",
    [{"model": []}, {"provider": {"bad": []}}, {"provider": {"bad": {"models": []}}}],
)
def test_invalid_native_config_fails_cleanly(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(AOPError):
        zcode.read_config(path)


def test_authenticated_zai_inventory(zcode_config, fake_zcode, monkeypatch):
    from agent_orchestration_process import model_listing
    from agent_orchestration_process.codex_routes import ZAI_CODING_PLAN_ENDPOINT

    source, config = zcode_config
    config["provider"]["test"]["options"]["baseURL"] = "https://api.z.ai/api/anthropic"
    (source / "cli/config.json").write_text(json.dumps(config))
    monkeypatch.setenv("AOP_ZCODE_BIN", str(fake_zcode))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key-must-not-override-config")

    def fetch(endpoint, credential):
        assert endpoint == ZAI_CODING_PLAN_ENDPOINT
        assert credential == "selected-key"
        return (
            ({"slug": "glm-5.3-flash", "display_name": "GLM-5.3-Flash"},),
            "2026-09-06T00:00:00+00:00",
            "inventory-sha256",
        )

    monkeypatch.setattr(model_listing, "fetch_zai_inventory", fetch)
    models = {
        model.model: model for model in list_models("zcode", ensure_catalog_fresh())
    }
    flash = models["test/glm-5.3-flash"]
    assert flash.authenticated
    assert flash.availability == "authenticated-endpoint"
    assert flash.inventory_sha256 == "inventory-sha256"
    assert flash.inference_provider == "zai-coding-plan"
    assert flash.price_scope == "api-equivalent"
    assert flash.input_per_million_usd == 0.5
    assert flash.cached_input_per_million_usd == 0.1
    assert flash.output_per_million_usd == 3
    assert flash.cache_write_per_million_usd is None
    assert flash.pricing_source == ensure_catalog_fresh().source
    assert models["test/test-model"].price_scope == "unknown"
    assert models["test/test-model"].input_per_million_usd is None
    assert models["other/other-model"].availability == "configured"
    assert not models["test/test-model"].authenticated


def test_zai_inventory_requires_credentials_and_propagates_failure(
    zcode_config, fake_zcode, monkeypatch
):
    from agent_orchestration_process import model_listing

    source, config = zcode_config
    definition = config["provider"]["test"]
    definition["options"] = {"baseURL": "https://api.z.ai/api/anthropic"}
    (source / "cli/config.json").write_text(json.dumps(config))
    monkeypatch.setenv("AOP_ZCODE_BIN", str(fake_zcode))
    for name in zcode.credential_env_names("test", definition):
        monkeypatch.delenv(name, raising=False)

    def fetch(endpoint, credential):
        assert credential == "native-environment-key"
        raise AOPError("Z.AI Coding Plan model inventory returned HTTP 401")

    monkeypatch.setattr(model_listing, "fetch_zai_inventory", fetch)
    assert all(
        not entry.authenticated
        for entry in list_models("zcode", ensure_catalog_fresh())
    )
    monkeypatch.setenv("ZCODE_API_KEY", "native-environment-key")
    with pytest.raises(AOPError, match="HTTP 401"):
        list_models("zcode", ensure_catalog_fresh())


def test_custom_zcode_endpoint_does_not_receive_inventory_credentials(
    zcode_config, fake_zcode, monkeypatch
):
    from agent_orchestration_process import model_listing

    monkeypatch.setenv("AOP_ZCODE_BIN", str(fake_zcode))

    def unexpected_fetch(*args):
        pytest.fail("custom native endpoints must not trigger a Z.ai inventory request")

    monkeypatch.setattr(model_listing, "fetch_zai_inventory", unexpected_fetch)
    assert all(
        not entry.authenticated
        for entry in list_models("zcode", ensure_catalog_fresh())
    )


def test_configured_zai_models_use_catalog_without_credentials(
    zcode_config, fake_zcode, monkeypatch
):
    source, config = zcode_config
    definition = config["provider"]["test"]
    definition["options"] = {"baseURL": "https://api.z.ai/api/anthropic"}
    definition["models"] = {"glm-5.3-flash": {}}
    (source / "cli/config.json").write_text(json.dumps(config))
    monkeypatch.setenv("AOP_ZCODE_BIN", str(fake_zcode))
    for name in zcode.credential_env_names("test", definition):
        monkeypatch.delenv(name, raising=False)
    models = {
        entry.model: entry for entry in list_models("zcode", ensure_catalog_fresh())
    }
    flash = models["test/glm-5.3-flash"]
    assert flash.availability == "configured"
    assert not flash.authenticated
    assert flash.price_scope == "api-equivalent"
    assert flash.input_per_million_usd == 0.5
    assert flash.output_per_million_usd == 3
    assert models["other/other-model"].price_scope == "unavailable"


def test_custom_endpoint_cannot_borrow_zai_prices(
    zcode_config, fake_zcode, monkeypatch
):
    source, config = zcode_config
    config["provider"]["zai"] = config["provider"].pop("test")
    config["provider"]["zai"]["models"] = {"glm-5.3-flash": {}}
    config["model"] = {"main": "zai/glm-5.3-flash"}
    (source / "cli/config.json").write_text(json.dumps(config))
    monkeypatch.setenv("AOP_ZCODE_BIN", str(fake_zcode))
    models = {
        entry.model: entry for entry in list_models("zcode", ensure_catalog_fresh())
    }
    flash = models["zai/glm-5.3-flash"]
    assert flash.price_scope == "unavailable"
    assert flash.input_per_million_usd is None
    assert flash.pricing_source is None


@pytest.mark.parametrize("profile", ["edit", "review", "sealed", "host"])
@pytest.mark.parametrize("mutation", ["model", "provider"])
def test_resume_rejects_changed_private_selection(
    repository, zcode_config, fake_zcode, profile, mutation
):
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    )
    first = runner.run(
        task="pinned", prompt="first", profile=profile, timeout_seconds=5
    )
    request = runner.store.load_request(first.run_id)
    from pathlib import Path

    private_config = (
        Path(request.effective_policy["controller"]["provider_state"])
        / "zcode/home/.zcode/cli/config.json"
    )
    config = json.loads(private_config.read_text())
    if mutation == "model":
        config["model"]["main"] = "test/second-model"
    else:
        config["provider"]["test"]["options"]["baseURL"] = "https://different.example"
    private_config.write_text(json.dumps(config))
    with pytest.raises(AOPError, match="differs from the pinned selection"):
        runner.resume(run_id=first.run_id, prompt="second")
    assert request.model == first.model


def request_event(
    model="zai/glm-5.3-flash", request_id="request-one", turn="turn_current", **counts
):
    provider, model_id = model.split("/", 1)
    return {
        "type": "session.updated",
        "sessionId": "sess_test",
        "turnId": turn,
        "payload": {
            "type": "model_request_completed",
            "requestId": request_id,
            "attempt": 1,
            "model": {"providerId": provider, "modelId": model_id, "role": "main"},
            "usage": {
                "inputTokens": 120,
                "cacheReadTokens": 30,
                "outputTokens": 20,
                "reasoningTokens": 7,
                **counts,
            },
        },
    }


def test_partial_usage_deduplicates_and_excludes_previous_turn():
    measured = request_event()
    events = [
        {"type": "turn.started", "sessionId": "sess_test", "turnId": "turn_current"},
        request_event(turn="turn_previous"),
        measured,
        measured,
    ]
    parsed = zcode.parse_stream("\n".join(map(json.dumps, events)))
    assert parsed["accounting_status"] == "partial"
    assert parsed["usage"].input_tokens == 120
    assert len(parsed["requests"]) == 1
    assert parsed["error"]


@pytest.mark.parametrize("ending", ["truncated", "timeout"])
def test_run_retains_partial_usage_and_cost(
    repository, zcode_config, fake_zcode, ending
):
    source, config = zcode_config
    config["model"]["main"] = "test/glm-5.3-flash"
    config["provider"]["test"]["options"]["baseURL"] = "https://api.z.ai/api/anthropic"
    config["provider"]["test"]["models"]["glm-5.3-flash"] = {}
    (source / "cli/config.json").write_text(json.dumps(config))
    script = fake_zcode.read_text()
    marker = "            event('turn.completed'"
    position = script.index(marker)
    script = (
        script[:position]
        + (
            "            time.sleep(10)\n"
            if ending == "timeout"
            else "            raise SystemExit(0)\n"
        )
        + script[position:]
    )
    fake_zcode.write_text(script)
    result = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    ).run(task="partial", prompt="test", timeout_seconds=0.5)
    assert not result.succeeded
    assert result.accounting_status == "partial"
    assert result.usage.input_tokens == 120
    assert result.calculated_cost.amount_usd == 0.000108
    assert result.billing.route == "subscription"
    assert result.billing.credential_source == "native-config"
    assert result.inference_provider == "zai-coding-plan"
    assert result.provider_reported_cost is None


def test_successful_run_cost_and_billing(repository, zcode_config, fake_zcode):
    source, config = zcode_config
    config["model"]["main"] = "test/glm-5.3-flash"
    config["provider"]["test"]["options"]["baseURL"] = "https://api.z.ai/api/anthropic"
    config["provider"]["test"]["models"]["glm-5.3-flash"] = {}
    (source / "cli/config.json").write_text(json.dumps(config))
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    )
    first = runner.run(task="priced", prompt="first", timeout_seconds=5)
    resumed = runner.resume(run_id=first.run_id, prompt="second")
    for result in (first, resumed):
        assert result.succeeded, result.error
        assert result.calculated_cost.amount_usd == 0.000108
        assert result.calculated_cost.pricing_source == ensure_catalog_fresh().source
        assert result.inference_provider == "zai-coding-plan"
        assert result.billing.route == "subscription"
        assert result.accounting_status == "complete"


def test_mixed_model_cost_and_missing_prices():
    from dataclasses import replace
    from agent_orchestration_process.pricing import TokenUsage
    from agent_orchestration_process.model_catalog import ModelCatalog
    from unittest.mock import patch

    catalog = ensure_catalog_fresh()
    providers = {
        **catalog.providers,
        "zai": {
            "models": {
                "fast": {"cost": {"input": 1, "cache_read": 0.1, "output": 2}},
                "lite": {"cost": {"input": 0.1, "cache_read": 0.01, "output": 0.2}},
            }
        },
    }
    assert isinstance(catalog, ModelCatalog)
    requests = [
        ("zai/fast", TokenUsage(1000, 100, 100)),
        ("zai/lite", TokenUsage(1000, 100, 100)),
    ]
    total = zcode.sum_usage([usage for _, usage in requests])
    with patch.object(
        zcode,
        "ensure_catalog_fresh",
        return_value=replace(catalog, providers=providers),
    ):
        cost = zcode.calculate_cost("zai/fast", total, requests)
        assert cost.amount_usd == 0.001221
        assert zcode.calculate_cost("zai/fast", total, requests[:1]) is None
        assert zcode.calculate_cost("zai/fast", total, [("zai/missing", total)]) is None


@pytest.mark.parametrize("filename", ["zcode.json", ".zcode/config.json"])
def test_resume_rejects_changed_project_lite_model(
    repository, zcode_config, fake_zcode, filename
):
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    )
    first = runner.run(task="lite-pinned", prompt="first", timeout_seconds=5)
    assert first.succeeded, first.error
    path = runner.manager.get("lite-pinned").path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": {"lite": "test/second-model"}}))
    with pytest.raises(AOPError, match="project lite model conflicts"):
        runner.resume(run_id=first.run_id, prompt="second")


@pytest.mark.parametrize("effort", [None, "low", "high", "max"])
def test_effort_is_verified_recorded_and_resumed(
    repository, zcode_config, fake_zcode, effort
):
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    )
    first = runner.run(task="effort", prompt="first", effort=effort, timeout_seconds=5)
    assert first.succeeded, first.error
    assert first.effort == (effort or "max")
    request = runner.store.load_request(first.run_id)
    assert request.effort == first.effort
    assert request.effective_policy["zcode"]["effective_effort"] == first.effort
    resumed = runner.resume(run_id=first.run_id, prompt="second")
    assert resumed.succeeded, resumed.error
    assert resumed.effort == first.effort


def test_unsupported_effort_fails_before_prompt(repository, zcode_config, fake_zcode):
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    )
    result = runner.run(
        task="unsupported-effort",
        prompt="must not run",
        effort="medium",
        timeout_seconds=5,
    )
    assert not result.succeeded
    assert "supported levels: low, high, max" in result.error
    events = (runner.store.root / result.run_id / "events.jsonl").read_text()
    assert '"method": "session/event"' not in events
    assert result.usage is None


def test_resume_rejects_changed_native_config(repository, zcode_config, fake_zcode):
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    )
    first = runner.run(task="native-pinned", prompt="first", timeout_seconds=5)
    assert first.succeeded, first.error
    request = runner.store.load_request(first.run_id)
    path = (
        Path(request.effective_policy["controller"]["provider_state"])
        / "zcode/home/.zcode/v2/provider_config.json"
    )
    value = json.loads(path.read_text())
    value["config"]["providerConfigRules"]["providerRules"][0]["config"]["access"][
        "apiKey"
    ] = "different-account"
    path.write_text(json.dumps(value))
    with pytest.raises(AOPError, match="differs from the pinned selection"):
        runner.resume(run_id=first.run_id, prompt="second")


def test_protocol_prompt_write_obeys_deadline(repository, zcode_config, fake_zcode):
    script = fake_zcode.read_text()
    marker = "    elif method == 'session/subscribe':"
    start = script.index(marker)
    # The server acknowledges subscription but never reads the large command.
    stop = script.index("    else:\n", start)
    fake_zcode.write_text(script[:stop] + "        time.sleep(10)\n" + script[stop:])
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    )
    result = runner.run(task="blocked-input", prompt="x" * 700000, timeout_seconds=0.3)
    assert result.timed_out
    assert result.duration_seconds < 5
    assert result.usage is None


@pytest.mark.parametrize("required", [None, True])
def test_protocol_requires_selected_key_without_reading_other_credentials(
    zcode_config, required
):
    _, config = zcode_config
    projected, _, _ = zcode.project_config(config, None, sealed=True)
    del projected["provider"]["test"]["options"]["apiKey"]
    if required is not None:
        projected["provider"]["test"]["options"]["apiKeyRequired"] = required
    with pytest.raises(AOPError, match="selected provider's native API key"):
        zcode.protocol_config(projected, {"UNRELATED_API_KEY": "unrelated-secret"})


def test_protocol_projects_required_zai_environment_key(zcode_config):
    _, config = zcode_config
    provider = config["provider"].pop("test")
    provider["options"].pop("apiKey")
    provider["options"]["apiKeyRequired"] = True
    config["provider"]["zai"] = provider
    config["model"]["main"] = "zai/test-model"
    projected = zcode.protocol_config(config, {"ZAI_API_KEY": "synthetic-zai-key"})
    rule = projected["config"]["providerConfigRules"]["providerRules"][0]
    assert rule["providerId"] == "zai"
    assert rule["config"]["access"] == {
        "type": "api-key",
        "apiKey": "synthetic-zai-key",
    }
    assert "apiKeyRequired" not in json.dumps(projected)


def test_required_environment_key_run_and_resume(
    repository, zcode_config, fake_zcode, monkeypatch
):
    source, config = zcode_config
    options = config["provider"]["test"]["options"]
    options.pop("apiKey")
    options["apiKeyRequired"] = True
    (source / "cli/config.json").write_text(json.dumps(config))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-test-key")
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    )
    first = runner.run(task="required-key", prompt="first", profile="sealed")
    assert first.succeeded, first.error
    resumed = runner.resume(run_id=first.run_id, prompt="second")
    assert resumed.succeeded, resumed.error
    assert resumed.session_id == first.session_id


@pytest.mark.parametrize("required", [False, "true", 1, None])
def test_protocol_rejects_unsupported_key_requirement(zcode_config, required):
    _, config = zcode_config
    config["provider"]["test"]["options"]["apiKeyRequired"] = required
    with pytest.raises(AOPError, match="apiKeyRequired"):
        zcode.protocol_config(config, {})


@pytest.mark.parametrize(
    "variable",
    ["ZCODE_PERSONAL_PROVIDER_CONFIG_FILE", "ZCODE_BUILTIN_PROVIDER_CONFIG_FILE"],
)
def test_provider_file_override_cannot_bypass_projection(
    repository, zcode_config, fake_zcode, monkeypatch, variable
):
    monkeypatch.setenv(variable, "/outside-provider-config.json")
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    )
    with pytest.raises(AOPError, match="provider-file overrides"):
        runner.run(task="external-config", prompt="test")


@pytest.mark.parametrize("mutation", ["missing", "different-account"])
def test_resume_pins_environment_credential(
    repository, zcode_config, fake_zcode, monkeypatch, mutation
):
    source, config = zcode_config
    del config["provider"]["test"]["options"]["apiKey"]
    (source / "cli/config.json").write_text(json.dumps(config))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "first-account")
    runner = AgentRunner(
        WorktreeManager.discover(repository), ZcodeAdapter(str(fake_zcode))
    )
    first = runner.run(task="account-pinned", prompt="first", timeout_seconds=5)
    assert first.succeeded, first.error
    request = runner.store.load_request(first.run_id)
    path = (
        Path(request.effective_policy["controller"]["provider_state"])
        / "zcode/home/.zcode/v2/provider_config.json"
    )
    if mutation == "missing":
        path.unlink()
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "second-account")
        value = json.loads(path.read_text())
        value["config"]["providerConfigRules"]["providerRules"][0]["config"]["access"][
            "apiKey"
        ] = "second-account"
        path.write_text(json.dumps(value))
    with pytest.raises(AOPError, match="differs from the pinned selection"):
        runner.resume(run_id=first.run_id, prompt="second")


def test_conflicting_protocol_session_cannot_be_restored_by_snapshot():
    snapshot = {
        "session": {
            "sessionId": "sess_expected",
            "model": {"providerId": "test", "modelId": "model"},
        },
        "projection": {"status": "idle"},
    }
    # A completed event can arrive in the same pipe read as the first event
    # from a conflicting session. The terminal snapshot must not restore trust.
    messages = [
        {"id": 1, "result": snapshot},
        {
            "method": "session/event",
            "params": {
                "type": "turn.completed",
                "sessionId": "sess_wrong",
                "payload": {"resultType": "success", "response": "wrong session"},
            },
        },
        {"id": 2, "result": snapshot},
    ]
    parsed = zcode.parse_protocol("\n".join(map(json.dumps, messages)))
    assert parsed["session_id"] is None
    assert "conflicting session identities" in parsed["error"]
