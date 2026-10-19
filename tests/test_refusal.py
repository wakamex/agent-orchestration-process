"""Claude refusals in the normalized result, from Claude Code's own events."""

from __future__ import annotations

import json

from support import AOPTestCase

from agent_orchestration_process.models import RunResult
from agent_orchestration_process.runner import ClaudeAdapter

INIT = {"type": "system", "subtype": "init", "session_id": "s", "model": "claude-opus-5-5"}
FALLBACK = {
    "type": "system",
    "subtype": "model_refusal_fallback",
    "trigger": "refusal",
    "original_model": "claude-opus-5-5",
    "fallback_model": "claude-opus-4-8",
    "api_refusal_category": "cyber",
    "api_refusal_explanation": "first explanation",
    "retracted_message_uuids": ["assistant-1"],
}
NO_FALLBACK = {
    "type": "system",
    "subtype": "model_refusal_no_fallback",
    "original_model": "claude-opus-4-8",
    "api_refusal_category": "cyber",
    "api_refusal_explanation": "final explanation",
}


def result(stop_reason: str | None, text: str, *, is_error: bool) -> dict:
    return {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "stop_reason": stop_reason,
        "terminal_reason": "api_error" if is_error else "completed",
        "session_id": "s",
        "result": text,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def parse(*events: dict) -> dict:
    return ClaudeAdapter._parse("\n".join(json.dumps(event) for event in events))


class RefusalTest(AOPTestCase):
    def test_a_refused_turn_records_category_models_and_where_it_began(self) -> None:
        tool_use = {
            "type": "assistant",
            "uuid": "assistant-1",
            "message": {"content": [{"type": "tool_use", "name": "Read"}]},
        }
        parsed = parse(
            INIT,
            tool_use,
            FALLBACK,
            NO_FALLBACK,
            result("refusal", "API Error: safeguards flagged this message", is_error=True),
        )

        assert parsed["refusal"] == {
            "category": "cyber",
            "explanation": "final explanation",
            "models": ["claude-opus-5-5", "claude-opus-4-8"],
            "retracted_messages": ["assistant-1"],
            "event_line": 3,
        }
        assert parsed["error"]

    def test_a_refusal_with_only_a_stop_reason_still_counts(self) -> None:
        parsed = parse(INIT, result("refusal", "API Error: refused", is_error=True))

        assert parsed["refusal"] == {
            "category": None,
            "explanation": None,
            "models": [],
            "retracted_messages": [],
            "event_line": 2,
        }

    def test_a_fallback_that_answers_and_an_api_error_are_not_refusals(self) -> None:
        recovered = parse(INIT, FALLBACK, result("end_turn", "answer", is_error=False))
        outage = parse(INIT, result(None, "API Error: 529 overloaded", is_error=True))

        assert recovered["refusal"] is None
        assert recovered["error"] is None
        assert outage["refusal"] is None
        assert outage["error"]

    def test_results_round_trip_the_refusal_and_older_records_have_none(self) -> None:
        record = {
            "run_id": "r", "provider": "claude", "mode": "agent", "task": "t",
            "model": "m", "effort": None, "session_id": "s", "command": ["claude"],
            "started_at": "2026-10-19T00:00:00+00:00",
            "finished_at": "2026-10-19T00:00:01+00:00", "duration_seconds": 1.0,
            "time_to_first_event_seconds": None, "time_to_first_response_seconds": None,
            "exit_code": 1, "timed_out": False, "error": "refused", "final_message": None,
            "usage": None, "calculated_cost": None, "accounting_status": "unavailable",
            "usage_schema": "aop-token-usage-v3", "state_root": "/p/.aop",
        }
        assert RunResult.from_dict(record).refusal is None
        refusal = {"category": "bio", "explanation": None, "models": ["m"],
                   "retracted_messages": [], "event_line": 4}
        loaded = RunResult.from_dict({**record, "refusal": refusal})
        assert RunResult.from_dict(loaded.to_dict()).refusal == refusal
