"""Serializable runner contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .pricing import CalculatedCost, TokenUsage


@dataclass(frozen=True)
class InputFile:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class InputSnapshot:
    source_path: str
    mounted_path: str
    kind: str
    size_bytes: int
    sha256: str
    files: tuple[InputFile, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InputSnapshot:
        fields = dict(value)
        fields["files"] = tuple(InputFile(**item) for item in fields.get("files", ()))
        return cls(**fields)


@dataclass(frozen=True)
class RunRequest:
    run_id: str
    provider: str
    mode: str
    task: str
    prompt: str
    base: str
    model: str | None
    inference_provider: str | None
    effort: str | None
    profile: str
    effective_policy: dict[str, Any]
    timeout_seconds: float | None
    session_id: str | None
    parent_run_id: str | None
    artifacts: tuple[str, ...]
    inputs: tuple[InputSnapshot, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunRequest:
        fields = dict(value)
        fields.setdefault("mode", "agent")
        fields.setdefault("inference_provider", None)
        fields.setdefault("effective_policy", {})
        fields["artifacts"] = tuple(fields.get("artifacts", ()))
        fields["inputs"] = tuple(
            InputSnapshot.from_dict(item) for item in fields.get("inputs", ())
        )
        return cls(**fields)


@dataclass(frozen=True)
class RunArtifact:
    logical_path: str
    archive_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class BillingProvenance:
    route: str = "unknown"
    credential_source: str | None = None
    detected_by: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> BillingProvenance:
        if value is None:
            return cls()
        fields = dict(value)
        fields.pop("actual_cost_known", None)
        return cls(**fields)


@dataclass(frozen=True)
class ProviderReportedCost:
    amount_usd: float
    currency: str
    source: str

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> ProviderReportedCost | None:
        return cls(**value) if value is not None else None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    provider: str
    mode: str
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
    calculated_cost: CalculatedCost | None
    provider_reported_cost: ProviderReportedCost | None = None
    billing: BillingProvenance = BillingProvenance()
    inference_provider: str | None = None
    artifacts: tuple[RunArtifact, ...] = ()
    inputs: tuple[InputSnapshot, ...] = ()
    provider_duration_seconds: float | None = None

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
        fields.setdefault("mode", "agent")
        fields.setdefault("model", None)
        fields.setdefault("effort", None)
        fields.setdefault("time_to_first_event_seconds", None)
        fields.setdefault("time_to_first_response_seconds", None)
        fields.setdefault("provider_duration_seconds", None)
        fields["usage"] = TokenUsage.from_dict(fields.get("usage"))
        legacy_cost = fields.pop("api_equivalent_cost", None)
        legacy_billing = fields.get("billing")
        legacy_reported = (
            isinstance(legacy_billing, dict)
            and legacy_billing.get("actual_cost_known") is True
        )
        if "calculated_cost" not in fields and not legacy_reported:
            fields["calculated_cost"] = legacy_cost
        fields["calculated_cost"] = CalculatedCost.from_dict(
            fields.get("calculated_cost")
        )
        fields.setdefault("provider_reported_cost", None)
        if fields["provider_reported_cost"] is None and legacy_reported:
            fields["provider_reported_cost"] = {
                "amount_usd": legacy_cost["amount_usd"],
                "currency": legacy_cost.get("currency", "USD"),
                "source": legacy_cost.get("pricing_source", "legacy run result"),
            }
        fields["provider_reported_cost"] = ProviderReportedCost.from_dict(
            fields.get("provider_reported_cost")
        )
        fields["billing"] = BillingProvenance.from_dict(legacy_billing)
        fields.setdefault("inference_provider", None)
        fields["artifacts"] = tuple(
            RunArtifact(**artifact) for artifact in fields.get("artifacts", ())
        )
        fields["inputs"] = tuple(
            InputSnapshot.from_dict(item) for item in fields.get("inputs", ())
        )
        return cls(**fields)
