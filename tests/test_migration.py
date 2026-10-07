from __future__ import annotations

import json
from pathlib import Path

import model_prices
import pytest

from agent_orchestration_process.cli import main
from agent_orchestration_process.migration import migrate_runs
from agent_orchestration_process.models import TOKEN_USAGE_SCHEMA
from agent_orchestration_process.runner import RunStore
from agent_orchestration_process.worktrees import AOPError, registered_state_roots

RUN_ID = "f1e2d3c4-0000-4000-8000-000000000001"


def write_run(
    runs: Path,
    result: dict | None = None,
    *,
    request: dict | None = None,
    events: list[dict] | None = None,
) -> Path:
    run_dir = runs / RUN_ID
    run_dir.mkdir(parents=True)
    if result is not None:
        (run_dir / "result.json").write_text(json.dumps(result))
    if request is not None:
        (run_dir / "request.json").write_text(json.dumps(request))
    if events is not None:
        (run_dir / "events.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events)
        )
    return run_dir


def result(provider: str, usage: dict | None, **fields) -> dict:
    return {
        "run_id": RUN_ID,
        "provider": provider,
        "task": "task",
        "model": "gemini-3.7-flash-high" if provider == "agy" else "gpt-5.6-sol",
        "session_id": "session",
        "command": [provider],
        "started_at": "2026-08-30T07:09:00+00:00",
        "finished_at": "2026-08-30T07:09:09+00:00",
        "duration_seconds": 9.0,
        "exit_code": 0,
        "timed_out": False,
        "error": None,
        "final_message": "answer",
        "usage": usage,
        "calculated_cost": None,
        **fields,
    }


def agy_step(input_tokens: int, cache_read: int, output: int, thinking: int) -> dict:
    return {
        "event": "step_update",
        "step_update": {
            "usage": {
                "input_tokens": input_tokens,
                "cache_read_tokens": cache_read,
                "output_tokens": output,
                "thinking_tokens": thinking,
            }
        },
    }


def usage(input_tokens: int, cached: int, output: int, reasoning: int) -> dict:
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output,
        "reasoning_output_tokens": reasoning,
    }


def test_v2_agy_result_takes_its_own_step_usage_and_is_repriced(tmp_path):
    runs = tmp_path / "runs"
    # A resumed Agy run recorded the conversation's running total.
    original = result(
        "agy",
        usage(147_694_920, 142_567_888, 1_603_877, 0),
        accounting_status="complete",
        usage_schema="aop-token-usage-v2",
        calculated_cost={
            "amount_usd": 99.0,
            "currency": "USD",
            "model": "gemini-3.7-flash-high",
            "priced_as": "gemini-3.7-flash",
            "pricing_version": "models-dev-2026-08-29",
            "pricing_source": "https://models.dev/api.json",
            "long_context_pricing": False,
        },
    )
    run_dir = write_run(runs, original, events=[agy_step(6_476, 220_102, 2_504, 426)])

    counts = migrate_runs(runs)

    assert counts == {"result.json: v2-schema+agy-step-usage (usage changed)": 1}
    migrated = json.loads((run_dir / "result.json").read_text())
    assert migrated["usage_schema"] == TOKEN_USAGE_SCHEMA
    assert migrated["usage"] == usage(226_578, 220_102, 2_504, 426)
    cost = migrated["calculated_cost"]
    assert cost["pricing_basis"] == model_prices.pricing_basis()
    assert cost["amount_usd"] == round(
        model_prices.cost(
            "gemini-3.7-flash-high",
            provider="google",
            input_tokens=6_476,
            cache_read_tokens=220_102,
            output_tokens=2_504,
            at="2026-08-30T07:09:00Z",
        ),
        8,
    )
    assert migrated["migration"]["from_usage_schema"] == "aop-token-usage-v2"
    assert migrated["migration"]["conversions"] == ["v2-schema", "agy-step-usage"]
    assert json.loads((run_dir / "result.pre-v3.json").read_text()) == original
    assert RunStore(runs).load_result(RUN_ID).usage.input_tokens == 226_578


def test_agy_result_without_events_becomes_unavailable(tmp_path):
    runs = tmp_path / "runs"
    run_dir = write_run(
        runs,
        result(
            "agy",
            usage(10, 4, 2, 1),
            accounting_status="complete",
            usage_schema="aop-token-usage-v2",
        ),
    )

    migrate_runs(runs)

    migrated = json.loads((run_dir / "result.json").read_text())
    assert migrated["usage"] is None
    assert migrated["calculated_cost"] is None
    assert migrated["accounting_status"] == "unavailable"


def test_v2_results_of_other_providers_only_change_their_marker(tmp_path):
    runs = tmp_path / "runs"
    original = result(
        "codex",
        usage(10, 4, 2, 1),
        accounting_status="complete",
        usage_schema="aop-token-usage-v2",
    )
    run_dir = write_run(runs, original)

    assert migrate_runs(runs) == {"result.json: v2-schema": 1}

    migrated = json.loads((run_dir / "result.json").read_text())
    assert migrated["usage"] == original["usage"]
    assert migrated["calculated_cost"] is None
    assert migrated["migration"]["conversions"] == ["v2-schema"]


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("claude", usage(10, 8, 20, 7)),
        ("codex", usage(10, 8, 20, 7)),
        ("cursor", usage(40, 30, 20, 7)),
        ("dsh", usage(40, 30, 20, 7)),
        ("opencode", usage(10, 8, 27, 7)),
    ],
)
def test_unversioned_results_take_the_legacy_provider_conversion(
    tmp_path, provider, expected
):
    runs = tmp_path / "runs"
    cached = 30 if provider in {"cursor", "dsh"} else 8
    run_dir = write_run(runs, result(provider, usage(10, cached, 20, 7)))

    assert not any("error" in outcome for outcome in migrate_runs(runs))

    migrated = json.loads((run_dir / "result.json").read_text())
    assert migrated["usage"] == expected
    assert migrated["accounting_status"] == "complete"


def test_legacy_cost_fields_fold_into_current_ones(tmp_path):
    runs = tmp_path / "runs"
    cost = {
        "amount_usd": 0.5,
        "currency": "USD",
        "model": "gpt-5.6-sol",
        "priced_as": "gpt-5.6-sol",
        "pricing_version": "models-dev-2026-08-10",
        "pricing_source": "https://models.dev/api.json",
        "long_context_pricing": False,
        "estimated": True,
    }
    legacy = result("codex", usage(10, 4, 2, 0), usage_schema="aop-token-usage-v1")
    del legacy["calculated_cost"]
    legacy.update(
        api_equivalent_cost=cost,
        billing={"route": "subscription", "actual_cost_known": False},
        read_paths=[],
    )
    run_dir = write_run(runs, legacy)

    migrate_runs(runs)

    migrated = json.loads((run_dir / "result.json").read_text())
    assert "api_equivalent_cost" not in migrated
    assert "read_paths" not in migrated
    assert migrated["calculated_cost"]["amount_usd"] == 0.5
    assert "estimated" not in migrated["calculated_cost"]
    assert "actual_cost_known" not in migrated["billing"]


def test_legacy_timed_out_zero_usage_becomes_unavailable(tmp_path):
    runs = tmp_path / "runs"
    run_dir = write_run(
        runs,
        result(
            "codex",
            usage(0, 0, 0, 0),
            usage_schema="aop-token-usage-v1",
            timed_out=True,
            exit_code=-15,
        ),
    )

    migrate_runs(runs)

    migrated = json.loads((run_dir / "result.json").read_text())
    assert migrated["accounting_status"] == "unavailable"
    assert migrated["usage"] is None


def test_requests_from_before_profiles_and_inputs_are_migrated(tmp_path):
    runs = tmp_path / "runs"
    original = {
        "run_id": RUN_ID,
        "provider": "codex",
        "task": "task",
        "prompt": "hello",
        "base": "HEAD",
        "model": None,
        "effort": None,
        "sandbox": "scratch-write",
        "timeout_seconds": None,
        "session_id": None,
        "parent_run_id": None,
        "created_at": "2026-08-10T00:00:00+00:00",
    }
    run_dir = write_run(runs, request=original)

    migrate_runs(runs)

    request = RunStore(runs).load_request(RUN_ID)
    assert request.profile == "scratch-write"
    assert request.artifacts == ()
    assert request.migration["conversions"][0] == "sandbox-to-profile"
    assert json.loads((run_dir / "request.pre-v3.json").read_text()) == original


def test_migration_is_idempotent_and_dry_run_writes_nothing(tmp_path):
    runs = tmp_path / "runs"
    run_dir = write_run(
        runs,
        result("codex", usage(10, 4, 2, 1), usage_schema="aop-token-usage-v2"),
    )
    before = (run_dir / "result.json").read_text()

    assert migrate_runs(runs, dry_run=True) == {"result.json: v2-schema": 1}
    assert (run_dir / "result.json").read_text() == before
    assert not (run_dir / "result.pre-v3.json").exists()

    migrate_runs(runs)
    migrated = (run_dir / "result.json").read_text()
    assert migrate_runs(runs) == {"result.json: current": 1}
    assert (run_dir / "result.json").read_text() == migrated
    assert (run_dir / "result.pre-v3.json").read_text() == before


def test_unmigrated_records_fail_with_the_migration_command(tmp_path):
    runs = tmp_path / "runs"
    write_run(
        runs,
        result("codex", usage(10, 4, 2, 1), usage_schema="aop-token-usage-v2"),
    )

    with pytest.raises(AOPError, match="aop migrate-results"):
        RunStore(runs).load_result(RUN_ID)


def test_cli_migrates_given_roots_and_registers_them(tmp_path, capsys):
    runs = tmp_path / "project" / ".aop" / "runs"
    write_run(
        runs,
        result("codex", usage(10, 4, 2, 1), usage_schema="aop-token-usage-v2"),
    )

    assert main(["migrate-results", str(runs)]) == 0

    assert "result.json: v2-schema" in capsys.readouterr().out
    assert (runs.parent).resolve() in registered_state_roots()
    assert RunStore(runs).load_result(RUN_ID).usage_schema == TOKEN_USAGE_SCHEMA
