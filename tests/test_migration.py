from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import model_prices
from support import AOPTestCase, parametrize, raises

from agent_orchestration_process.cli import main
from agent_orchestration_process.migration import migrate_archive, migrate_runs
from agent_orchestration_process.models import TOKEN_USAGE_SCHEMA
from agent_orchestration_process.runner import RunStore
from agent_orchestration_process.worktrees import (
    AOPError,
    compact_state_roots,
    register_state_root,
    registered_state_roots,
    results_archive_dir,
    state_roots_file,
)

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


def usage(
    input_tokens: int,
    cached: int,
    output: int,
    reasoning: int,
    *,
    cache_write: int = 0,
    cache_write_1h: int = 0,
) -> dict:
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "cache_write_input_tokens": cache_write,
        "cache_write_1h_input_tokens": cache_write_1h,
        "output_tokens": output,
        "reasoning_output_tokens": reasoning,
    }


class MigrationTest(AOPTestCase):
    def test_v2_agy_result_takes_its_own_step_usage_and_is_repriced(self, tmp_path):
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

        assert counts == {
            "result.json: v2-schema+agy-step-usage+state-root (usage changed)": 1,
            "archive: copied": 1,
        }
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
        assert migrated["migration"]["conversions"] == [
            "v2-schema",
            "agy-step-usage",
            "state-root",
        ]
        assert json.loads((run_dir / "result.pre-v3.json").read_text()) == original
        assert RunStore(runs).load_result(RUN_ID).usage.input_tokens == 226_578

    def test_agy_result_without_events_becomes_unavailable(self, tmp_path):
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

    def test_v2_results_of_other_providers_only_change_their_marker(self, tmp_path):
        runs = tmp_path / "runs"
        original = result(
            "codex",
            usage(10, 4, 2, 1),
            accounting_status="complete",
            usage_schema="aop-token-usage-v2",
        )
        run_dir = write_run(runs, original)

        assert migrate_runs(runs) == {
            "result.json: v2-schema+state-root": 1,
            "archive: copied": 1,
        }

        migrated = json.loads((run_dir / "result.json").read_text())
        assert migrated["usage"] == original["usage"]
        assert migrated["calculated_cost"] is None
        assert migrated["migration"]["conversions"] == ["v2-schema", "state-root"]

    @parametrize(
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
        self,
        tmp_path, provider, expected
    ):
        runs = tmp_path / "runs"
        cached = 30 if provider in {"cursor", "dsh"} else 8
        run_dir = write_run(runs, result(provider, usage(10, cached, 20, 7)))

        assert not any("error" in outcome for outcome in migrate_runs(runs))

        migrated = json.loads((run_dir / "result.json").read_text())
        assert migrated["usage"] == expected
        assert migrated["accounting_status"] == "complete"

    def test_legacy_cost_fields_fold_into_current_ones(self, tmp_path):
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

    def test_legacy_timed_out_zero_usage_becomes_unavailable(self, tmp_path):
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

    def test_requests_from_before_profiles_and_inputs_are_migrated(self, tmp_path):
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

    def test_migration_is_idempotent_and_dry_run_writes_nothing(self, tmp_path):
        runs = tmp_path / "runs"
        run_dir = write_run(
            runs,
            result("codex", usage(10, 4, 2, 1), usage_schema="aop-token-usage-v2"),
        )
        before = (run_dir / "result.json").read_text()

        assert migrate_runs(runs, dry_run=True) == {
            "result.json: v2-schema+state-root": 1,
            "archive: copied": 1,
        }
        assert (run_dir / "result.json").read_text() == before
        assert not (run_dir / "result.pre-v3.json").exists()

        migrate_runs(runs)
        migrated = (run_dir / "result.json").read_text()
        assert migrate_runs(runs) == {"result.json: current": 1, "archive: current": 1}
        assert (run_dir / "result.json").read_text() == migrated
        assert (run_dir / "result.pre-v3.json").read_text() == before

    def test_unmigrated_records_fail_with_the_migration_command(self, tmp_path):
        runs = tmp_path / "runs"
        write_run(
            runs,
            result("codex", usage(10, 4, 2, 1), usage_schema="aop-token-usage-v2"),
        )

        with raises(AOPError, match="aop migrate-results"):
            RunStore(runs).load_result(RUN_ID)

    def test_cli_migrates_given_roots_and_registers_them(self, tmp_path, capsys):
        runs = tmp_path / "project" / ".aop" / "runs"
        write_run(
            runs,
            result("codex", usage(10, 4, 2, 1), usage_schema="aop-token-usage-v2"),
        )

        assert main(["migrate-results", str(runs)]) == 0

        assert "result.json: v2-schema" in capsys.readouterr().out
        assert (runs.parent).resolve() in registered_state_roots()
        assert RunStore(runs).load_result(RUN_ID).usage_schema == TOKEN_USAGE_SCHEMA

    def test_v3_claude_results_recover_cache_write_and_resumed_turn_cost(
        self, tmp_path
    ):
        runs = tmp_path / "runs"
        parent_id = "f1e2d3c4-0000-4000-8000-000000000000"
        parent_dir = runs / parent_id
        parent_dir.mkdir(parents=True)
        (parent_dir / "events.jsonl").write_text(
            json.dumps({"type": "result", "total_cost_usd": 0.30}) + "\n"
        )
        v3_result = result(
            "claude",
            {
                "input_tokens": 710,
                "cached_input_tokens": 500,
                "output_tokens": 40,
                "reasoning_output_tokens": 0,
            },
            mode="agent",
            model="claude-haiku-5-5",
            effort=None,
            time_to_first_event_seconds=0.1,
            time_to_first_response_seconds=0.2,
            provider_duration_seconds=None,
            provider_status=None,
            provider_error=None,
            inference_provider=None,
            provider_reported_cost=None,
            accounting_status="complete",
            usage_schema=TOKEN_USAGE_SCHEMA,
            billing={
                "route": "subscription",
                "credential_source": "claude-oauth",
                "detected_by": "claude auth status",
            },
            inputs=[],
            artifacts=[],
            migration=None,
            calculated_cost={
                "amount_usd": 0.45,
                "currency": "USD",
                "model": "claude-haiku-5-5",
                "priced_as": "claude-haiku-5-5",
                "pricing_version": "claude-cli-reported",
                "pricing_source": "Claude Code result event",
                "long_context_pricing": False,
                "pricing_retrieved_at": None,
                "pricing_basis": None,
                "pricing_period": None,
                "removed_suffix": None,
            },
        )
        events = [
            {"type": "system", "subtype": "init", "model": "claude-haiku-5-5"},
            {
                "type": "result",
                "subtype": "success",
                "session_id": "session",
                "result": "answer",
                "total_cost_usd": 0.45,
                "usage": {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 200,
                    "cache_read_input_tokens": 500,
                    "output_tokens": 40,
                    "cache_creation": {
                        "ephemeral_1h_input_tokens": 200,
                        "ephemeral_5m_input_tokens": 0,
                    },
                },
                "modelUsage": {
                    "claude-haiku-5-5": {
                        "inputTokens": 20,
                        "cacheCreationInputTokens": 400,
                        "cacheReadInputTokens": 500,
                        "outputTokens": 90,
                        "costUSD": 0.45,
                    }
                },
            },
        ]
        request = {
            "run_id": RUN_ID,
            "provider": "claude",
            "mode": "agent",
            "task": "task",
            "prompt": "hello",
            "base": "HEAD",
            "model": "claude-haiku-5-5",
            "effort": None,
            "profile": "workspace-write",
            "inference_provider": None,
            "inference_route": None,
            "no_web": False,
            "effective_policy": {},
            "artifacts": [],
            "inputs": [],
            "timeout_seconds": None,
            "session_id": "session",
            "parent_run_id": parent_id,
            "created_at": "2026-08-30T07:09:00+00:00",
        }
        run_dir = write_run(runs, v3_result, request=request, events=events)

        counts = migrate_runs(runs)

        assert counts == {
            "request.json: current": 1,
            "result.json: cache-write-usage+claude-resume-cost+state-root (usage changed)": 1,
            "archive: copied": 1,
        }
        migrated = json.loads((run_dir / "result.json").read_text())
        assert migrated["usage"] == usage(
            710,
            500,
            40,
            0,
            cache_write=200,
            cache_write_1h=200,
        )
        assert migrated["calculated_cost"]["amount_usd"] == 0.15


class ResultsArchiveTest(AOPTestCase):
    def test_backfill_names_the_state_root_and_archives_each_result(self, tmp_path):
        state = tmp_path / "project" / ".aop"
        run_dir = write_run(
            state / "runs",
            result(
                "codex",
                usage(10, 4, 2, 1),
                usage_schema=TOKEN_USAGE_SCHEMA,
                accounting_status="complete",
                mode="agent",
                effort=None,
                time_to_first_event_seconds=None,
                time_to_first_response_seconds=None,
            ),
        )

        counts = migrate_runs(state / "runs")

        assert counts == {"result.json: state-root": 1, "archive: copied": 1}
        migrated = json.loads((run_dir / "result.json").read_text())
        assert migrated["state_root"] == str(state.resolve())
        assert migrated["migration"]["conversions"] == ["state-root"]
        archived = results_archive_dir() / f"{RUN_ID}.json"
        assert json.loads(archived.read_text()) == migrated
        assert migrate_runs(state / "runs") == {
            "result.json: current": 1,
            "archive: current": 1,
        }

    def test_archived_results_outlive_their_state_root_and_still_migrate(self, tmp_path):
        state = tmp_path / "deleted" / ".aop"
        archive = results_archive_dir()
        archive.mkdir(parents=True)
        legacy = result(
            "codex",
            usage(10, 4, 2, 1),
            usage_schema="aop-token-usage-v2",
            state_root=str(state),
        )
        (archive / f"{RUN_ID}.json").write_text(json.dumps(legacy))

        assert migrate_archive() == {"archive: v2-schema": 1}

        migrated = json.loads((archive / f"{RUN_ID}.json").read_text())
        assert migrated["usage_schema"] == TOKEN_USAGE_SCHEMA
        assert migrated["state_root"] == str(state)
        original = archive / "originals" / f"{RUN_ID}.pre-v3.json"
        assert json.loads(original.read_text()) == legacy
        assert sorted(path.name for path in archive.glob("*.json")) == [
            f"{RUN_ID}.json"
        ]
        assert migrate_archive() == {"archive: current": 1}

    def test_registry_records_each_root_once_and_compacts(self, tmp_path):
        live = tmp_path / "live" / ".aop"
        live.mkdir(parents=True)
        gone = tmp_path / "gone" / ".aop"
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(lambda _: register_state_root(live), range(64)))
        assert registered_state_roots() == [live.resolve()]

        with state_roots_file().open("a") as stream:
            stream.write(f"{live.resolve()}\n{gone}\n")
        assert compact_state_roots() == (1, 2)
        assert state_roots_file().read_text() == f"{live.resolve()}\n"
