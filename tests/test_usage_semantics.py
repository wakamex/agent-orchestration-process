from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_orchestration_process.models import TOKEN_USAGE_SCHEMA, RunResult
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
    event = {
        "type": "turn.completed",
        "usage": {
            "input_tokens": 34_709,
            "cached_input_tokens": 13_056,
            "output_tokens": 397,
            "reasoning_output_tokens": 135,
        },
    }

    _, _, usage, completed = CodexAdapter._parse_events(json.dumps(event))

    assert completed
    assert _usage_tuple(usage) == (34_709, 13_056, 397, 135)


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


def test_agy_retained_result_adds_disjoint_cache_to_input() -> None:
    event = {
        "event": "result",
        "result": {
            "conversation_id": "conversation",
            "status": "SUCCESS",
            "response": "answer",
            "usage": {
                "input_tokens": 216_004,
                "cache_read_tokens": 390_699,
                "output_tokens": 41_309,
                "thinking_tokens": 30_165,
                "total_tokens": 257_313,
            },
        },
    }

    usage = AgyAdapter._parse(json.dumps(event))["usage"]

    assert isinstance(usage, TokenUsage)
    assert _usage_tuple(usage) == (606_703, 390_699, 41_309, 30_165)


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

    result = RunResult.from_dict(raw)
    serialized = result.to_dict()
    round_tripped = RunResult.from_dict(serialized)

    assert serialized["usage_schema"] == TOKEN_USAGE_SCHEMA
    assert round_tripped == result
    assert _usage_tuple(round_tripped.usage) == (40, 30, 20, 7)


def test_unknown_usage_schema_is_rejected() -> None:
    raw = _result_dict("codex", {})
    raw["usage_schema"] = "future"

    with pytest.raises(ValueError, match="unsupported token usage schema"):
        RunResult.from_dict(raw)
