"""Serializable runner contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .pricing import EstimatedCost, TokenUsage


@dataclass(frozen=True)
class RunRequest:
    run_id: str
    provider: str
    task: str
    prompt: str
    base: str
    model: str | None
    effort: str | None
    sandbox: str
    timeout_seconds: float | None
    session_id: str | None
    parent_run_id: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunRequest:
        return cls(**value)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    provider: str
    task: str
    model: str | None
    effort: str | None
    session_id: str | None
    command: list[str]
    started_at: str
    finished_at: str
    duration_seconds: float
    time_to_first_event_seconds: float | None
    time_to_first_response_seconds: float | None
    exit_code: int
    timed_out: bool
    error: str | None
    final_message: str | None
    usage: TokenUsage
    api_equivalent_cost: EstimatedCost | None

    @property
    def succeeded(self) -> bool:
        return (
            self.exit_code == 0
            and not self.timed_out
            and self.error is None
            and self.session_id is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "succeeded": self.succeeded,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunResult:
        fields = dict(value)
        fields.pop("succeeded", None)
        fields.setdefault("model", None)
        fields.setdefault("effort", None)
        fields.setdefault("time_to_first_event_seconds", None)
        fields.setdefault("time_to_first_response_seconds", None)
        fields["usage"] = TokenUsage.from_dict(fields.get("usage"))
        fields["api_equivalent_cost"] = EstimatedCost.from_dict(
            fields.get("api_equivalent_cost")
        )
        return cls(**fields)
