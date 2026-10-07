from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_orchestration_process.models import (
    LEGACY_TOKEN_USAGE_SCHEMA,
    TOKEN_USAGE_SCHEMA,
    RunResult,
)
from agent_orchestration_process.pricing import TokenUsage
from agent_orchestration_process.runner import (
    AgyAdapter,
    ClaudeAdapter,
    CodexAdapter,
    CursorAdapter,
    DeepSeekHarnessAdapter,
    DevinAdapter,
    GrokAdapter,
    OpenCodeAdapter,
)


def _usage_tuple(usage: TokenUsage) -> tuple[int, int, int, int]:
    return (
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.output_tokens,
        usage.reasoning_output_tokens,
    )


def test_codex_retained_terminal_usage_is_already_normalized() -> None:
    events = [
        {
            "id": 2,
            "result": {
                "thread": {"id": "thread", "turns": []},
                "model": "gpt-5.6-sol",
            },
        },
        {
            "id": 3,
            "result": {"turn": {"id": "turn", "items": [], "status": "inProgress"}},
        },
        {
            "method": "thread/tokenUsage/updated",
            "params": {
                "threadId": "thread",
                "turnId": "turn",
                "tokenUsage": {
                    "last": {
                        "inputTokens": 34_709,
                        "cachedInputTokens": 13_056,
                        "outputTokens": 397,
                        "reasoningOutputTokens": 135,
                        "totalTokens": 35_106,
                    },
                    "total": {
                        "inputTokens": 34_709,
                        "cachedInputTokens": 13_056,
                        "outputTokens": 397,
                        "reasoningOutputTokens": 135,
                        "totalTokens": 35_106,
                    },
                },
            },
        },
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread",
                "turn": {"id": "turn", "items": [], "status": "completed"},
            },
        },
    ]

    parsed = CodexAdapter._parse_events("\n".join(map(json.dumps, events)))

    assert parsed["completed"]
    assert parsed["usage_observed"]
    assert _usage_tuple(parsed["usage"]) == (34_709, 13_056, 397, 135)
    assert [_usage_tuple(usage) for usage in parsed["request_usages"]] == [
        (34_709, 13_056, 397, 135)
    ]


def test_codex_sums_current_turn_updates_without_resumed_thread_history() -> None:
    def usage_event(turn_id: str, input_tokens: int, output_tokens: int) -> dict:
        breakdown = {
            "inputTokens": input_tokens,
            "cachedInputTokens": 0,
            "outputTokens": output_tokens,
            "reasoningOutputTokens": 0,
            "totalTokens": input_tokens + output_tokens,
        }
        return {
            "method": "thread/tokenUsage/updated",
            "params": {
                "threadId": "thread",
                "turnId": turn_id,
                "tokenUsage": {"last": breakdown, "total": breakdown},
            },
        }

    events = [
        usage_event("previous-turn", 1000, 100),
        {
            "id": 2,
            "result": {
                "thread": {"id": "thread", "turns": []},
                "model": "gpt-5.6-sol",
            },
        },
        {
            "id": 3,
            "result": {
                "turn": {"id": "current-turn", "items": [], "status": "inProgress"}
            },
        },
        usage_event("current-turn", 10, 2),
        usage_event("current-turn", 20, 3),
        usage_event("other-turn", 2000, 200),
        {
            "method": "turn/completed",
            "params": {
                "threadId": "thread",
                "turn": {
                    "id": "current-turn",
                    "items": [],
                    "status": "completed",
                },
            },
        },
    ]

    parsed = CodexAdapter._parse_events("\n".join(map(json.dumps, events)))

    assert parsed["usage_observed"]
    assert _usage_tuple(parsed["usage"]) == (30, 0, 5, 0)
    assert [_usage_tuple(usage) for usage in parsed["request_usages"]] == [
        (10, 0, 2, 0),
        (20, 0, 3, 0),
    ]


def test_claude_retained_result_adds_cache_creation_and_read_to_input() -> None:
    event = {
        "type": "result",
        "subtype": "success",
        "session_id": "session",
        "result": "answer",
        "usage": {
            "input_tokens": 2,
            "cache_creation_input_tokens": 11_802,
            "cache_read_input_tokens": 15_190,
            "output_tokens": 58,
        },
    }

    usage = ClaudeAdapter._parse(json.dumps(event))["usage"]

    assert isinstance(usage, TokenUsage)
    assert _usage_tuple(usage) == (26_994, 15_190, 58, 0)


def test_agy_usage_is_this_invocations_steps_with_disjoint_cache() -> None:
    def step(input_tokens: int, cache_read: int, output: int, thinking: int) -> dict:
        return {
            "event": "step_update",
            "step_update": {
                "conversation_id": "conversation",
                "usage": {
                    "input_tokens": input_tokens,
                    "cache_read_tokens": cache_read,
                    "output_tokens": output,
                    "thinking_tokens": thinking,
                },
            },
        }

    events = [
        step(200_000, 390_699, 40_000, 30_000),
        step(16_004, 0, 1_309, 165),
        {
            "event": "result",
            "result": {
                "conversation_id": "conversation",
                "status": "SUCCESS",
                "response": "answer",
                # A resumed conversation's running total, from earlier runs too.
                "usage": {
                    "input_tokens": 5_127_032,
                    "cache_read_tokens": 142_567_888,
                    "output_tokens": 1_603_877,
                },
            },
        },
    ]

    usage = AgyAdapter._parse("\n".join(map(json.dumps, events)))["usage"]

    assert isinstance(usage, TokenUsage)
    assert _usage_tuple(usage) == (606_703, 390_699, 41_309, 30_165)


def test_v2_agy_results_load_with_usage_from_their_own_steps(tmp_path) -> None:
    from agent_orchestration_process.runner import RunStore

    store = RunStore(tmp_path)
    raw = _v2_result("agy")
    raw.update(
        session_id="conversation",
        usage={
            "input_tokens": 147_694_920,
            "cached_input_tokens": 142_567_888,
            "output_tokens": 1_603_877,
            "reasoning_output_tokens": 0,
        },
        calculated_cost={
            "amount_usd": 99.0,
            "currency": "USD",
            "model": "gemini-3.7-flash",
            "priced_as": "gemini-3.7-flash",
            "pricing_version": "models-dev-2026-08-29",
            "pricing_source": "https://models.dev/api.json",
            "long_context_pricing": False,
        },
    )
    run_dir = tmp_path / raw["run_id"]
    run_dir.mkdir()
    (run_dir / "result.json").write_text(json.dumps(raw))
    step = {
        "event": "step_update",
        "step_update": {
            "usage": {
                "input_tokens": 6_476,
                "cache_read_tokens": 220_102,
                "output_tokens": 2_504,
                "thinking_tokens": 426,
            }
        },
    }
    (run_dir / "events.jsonl").write_text(json.dumps(step) + "\n")

    result = store.load_result(raw["run_id"])

    assert result.usage_schema == TOKEN_USAGE_SCHEMA
    assert _usage_tuple(result.usage) == (226_578, 220_102, 2_504, 426)
    assert result.calculated_cost is None
    assert result.accounting_status == "complete"

    (run_dir / "events.jsonl").unlink()
    without_events = store.load_result(raw["run_id"])
    assert without_events.usage is None
    assert without_events.accounting_status == "unavailable"


def test_v2_results_of_other_providers_load_unchanged() -> None:
    raw = _v2_result("codex")

    result = RunResult.from_dict(raw)

    assert result.usage_schema == TOKEN_USAGE_SCHEMA
    assert result.usage.to_dict() == raw["usage"]


def _v2_result(provider: str) -> dict:
    return {
        "run_id": "f1e2d3c4-0000-4000-8000-000000000001",
        "provider": provider,
        "mode": "agent",
        "task": "task",
        "model": "model",
        "effort": None,
        "session_id": "session",
        "command": [provider],
        "started_at": "2026-08-30T07:09:00+00:00",
        "finished_at": "2026-08-30T07:09:09+00:00",
        "duration_seconds": 9.0,
        "exit_code": 0,
        "timed_out": False,
        "error": None,
        "final_message": "answer",
        "usage": {
            "input_tokens": 10,
            "cached_input_tokens": 4,
            "output_tokens": 2,
            "reasoning_output_tokens": 1,
        },
        "calculated_cost": None,
        "provider_reported_cost": None,
        "billing": {"route": "subscription"},
        "accounting_status": "complete",
        "usage_schema": "aop-token-usage-v2",
    }


def test_cursor_retained_result_adds_cache_read_and_write_to_input() -> None:
    event = {
        "type": "result",
        "subtype": "success",
        "session_id": "session",
        "result": "answer",
        "usage": {
            "inputTokens": 8_125,
            "cacheReadTokens": 66_529,
            "cacheWriteTokens": 11,
            "outputTokens": 2_294,
        },
    }

    usage = CursorAdapter._parse(json.dumps(event))["usage"]

    assert isinstance(usage, TokenUsage)
    assert _usage_tuple(usage) == (74_665, 66_529, 2_294, 0)


def test_devin_retained_atif_metrics_use_prompt_total_and_cached_subset(
    tmp_path: Path,
) -> None:
    export = tmp_path / "trajectory.json"
    export.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "session_id": "session",
                "steps": [
                    {"source": "user", "message": "prompt"},
                    {
                        "source": "agent",
                        "message": "answer",
                        "metrics": {
                            "prompt_tokens": 15_356,
                            "cached_tokens": 292,
                            "completion_tokens": 758,
                        },
                    },
                ],
            }
        )
    )

    usage = DevinAdapter._parse_export(export, "prompt")["usage"]

    assert isinstance(usage, TokenUsage)
    assert _usage_tuple(usage) == (15_356, 292, 758, 0)


def test_devin_fresh_export_rejects_a_foreign_user_prompt(tmp_path: Path) -> None:
    export = tmp_path / "trajectory.json"
    export.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.6",
                "session_id": "foreign-session",
                "steps": [
                    {"source": "user", "message": "another run"},
                    {"source": "agent", "message": "foreign answer"},
                ],
            }
        )
    )

    parsed = DevinAdapter._parse_export(
        export,
        "current run",
        allow_missing_prompt=True,
    )

    assert parsed["error"] == (
        "Devin trajectory export did not contain the current prompt"
    )


def test_opencode_retained_step_adds_cache_to_input_and_reasoning_to_output() -> None:
    event = {
        "type": "step_finish",
        "sessionID": "session",
        "part": {
            "type": "step-finish",
            "tokens": {
                "total": 30_560,
                "input": 6_498,
                "cache": {"read": 18_891, "write": 0},
                "output": 5_160,
                "reasoning": 11,
            },
        },
    }

    usage = OpenCodeAdapter._parse(json.dumps(event), "model")["usage"]

    assert isinstance(usage, TokenUsage)
    assert _usage_tuple(usage) == (25_389, 18_891, 5_171, 11)


def test_opencode_steps_are_separate_timed_requests() -> None:
    def step(timestamp: int, tokens: dict[str, object]) -> str:
        return json.dumps(
            {
                "type": "step_finish",
                "timestamp": timestamp,
                "sessionID": "session",
                "part": {"type": "step-finish", "tokens": tokens},
            }
        )

    first = {"input": 1, "output": 2, "reasoning": 1, "cache": {"read": 3, "write": 4}}
    second = {"input": 40, "output": 5, "reasoning": 1, "cache": {"read": 7}}
    parsed = OpenCodeAdapter._parse(
        "\n".join([step(1_789_000_000_000, first), step(1_789_000_000_500, second)]),
        "model",
    )

    assert parsed["requests"] == [
        (TokenUsage(8, 3, 3, 1), "2026-09-10T00:26:40+00:00"),
        (TokenUsage(47, 7, 6, 1), "2026-09-10T00:26:40.500000+00:00"),
    ]
    assert parsed["usage"] == TokenUsage(55, 10, 9, 2)
    assert parsed["duration_seconds"] == 0.5


def test_grok_terminal_usage_has_total_output_and_disjoint_input_cache() -> None:
    event = {
        "type": "end",
        "sessionId": "session",
        "usage": {
            "input_tokens": 40,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 5,
            "output_tokens": 20,
            "reasoning_tokens": 7,
            "total_tokens": 75,
        },
    }

    usage = GrokAdapter._parse(json.dumps(event), "grok-4.5")["usage"]

    assert isinstance(usage, TokenUsage)
    assert _usage_tuple(usage) == (55, 10, 20, 7)


def test_dsh_disjoint_harness_usage_is_normalized() -> None:
    event = {
        "type": "aop.dsh.result",
        "session_id": "session",
        "model": "model",
        "final_message": "answer",
        "usage": {
            "input_tokens": 64,
            "cached_input_tokens": 7_424,
            "cache_write_input_tokens": 3,
            "output_tokens": 4,
            "reasoning_output_tokens": 2,
        },
        "completed": True,
        "error": None,
    }

    usage = DeepSeekHarnessAdapter._parse(json.dumps(event))["usage"]

    assert isinstance(usage, TokenUsage)
    assert _usage_tuple(usage) == (7_491, 7_424, 4, 2)


def _result_dict(
    provider: str,
    usage: dict[str, int],
    *,
    inference_provider: str | None = None,
) -> dict[str, object]:
    return {
        "run_id": "run",
        "provider": provider,
        "mode": "agent",
        "task": "task",
        "model": "model",
        "effort": None,
        "session_id": "session",
        "command": [],
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "duration_seconds": 1.0,
        "time_to_first_event_seconds": None,
        "time_to_first_response_seconds": None,
        "exit_code": 0,
        "timed_out": False,
        "error": None,
        "final_message": "answer",
        "usage": usage,
        "calculated_cost": None,
        "inference_provider": inference_provider,
    }


@pytest.mark.parametrize(
    ("provider", "inference_provider", "expected"),
    [
        ("agy", None, (40, 30, 20, 7)),
        ("claude", None, (10, 8, 20, 7)),
        ("codex", None, (10, 8, 20, 7)),
        ("cursor", None, (40, 30, 20, 7)),
        ("devin", None, (10, 8, 20, 7)),
        ("dsh", "deepseek-official", (40, 30, 20, 7)),
        ("dsh", "anthropic", (40, 30, 20, 7)),
        ("grok", None, (10, 8, 20, 7)),
        ("hermes", "xai-oauth", (10, 8, 20, 7)),
        ("opencode", None, (10, 8, 27, 7)),
    ],
)
def test_unversioned_results_load_with_centralized_legacy_normalization(
    provider: str,
    inference_provider: str | None,
    expected: tuple[int, int, int, int],
) -> None:
    raw = _result_dict(
        provider,
        {
            "input_tokens": 10,
            "cached_input_tokens": 30 if provider in {"agy", "cursor", "dsh"} else 8,
            "output_tokens": 20,
            "reasoning_output_tokens": 7,
        },
        inference_provider=inference_provider,
    )

    result = RunResult.from_dict(raw)

    assert result.usage_schema == TOKEN_USAGE_SCHEMA
    assert _usage_tuple(result.usage) == expected


def test_versioned_result_round_trips_without_renormalization() -> None:
    raw = _result_dict(
        "agy",
        {
            "input_tokens": 40,
            "cached_input_tokens": 30,
            "output_tokens": 20,
            "reasoning_output_tokens": 7,
        },
    )
    raw["usage_schema"] = TOKEN_USAGE_SCHEMA
    raw["accounting_status"] = "complete"

    result = RunResult.from_dict(raw)
    serialized = result.to_dict()
    round_tripped = RunResult.from_dict(serialized)

    assert serialized["usage_schema"] == TOKEN_USAGE_SCHEMA
    assert round_tripped == result
    assert _usage_tuple(round_tripped.usage) == (40, 30, 20, 7)


def test_versioned_complete_result_preserves_measured_zero_usage() -> None:
    raw = _result_dict("codex", {})
    raw["usage_schema"] = TOKEN_USAGE_SCHEMA
    raw["accounting_status"] = "complete"

    result = RunResult.from_dict(raw)

    assert result.accounting_status == "complete"
    assert result.usage == TokenUsage()
    assert result.to_dict()["usage"] == {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
    }


def test_legacy_timed_out_zero_usage_loads_as_unavailable() -> None:
    raw = _result_dict("codex", {})
    raw["usage_schema"] = LEGACY_TOKEN_USAGE_SCHEMA
    raw["timed_out"] = True
    raw["exit_code"] = -15

    result = RunResult.from_dict(raw)

    assert result.usage_schema == TOKEN_USAGE_SCHEMA
    assert result.accounting_status == "unavailable"
    assert result.usage is None
    assert result.calculated_cost is None


def test_unknown_usage_schema_is_rejected() -> None:
    raw = _result_dict("codex", {})
    raw["usage_schema"] = "future"

    with pytest.raises(ValueError, match="unsupported token usage schema"):
        RunResult.from_dict(raw)
