"""Serializable runner contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .pricing import CalculatedCost, TokenUsage


TOKEN_USAGE_SCHEMA = "aop-token-usage-v3"
MIGRATE_HINT = "run `aop migrate-results` to convert older run records"
ACCOUNTING_STATUSES = frozenset({"complete", "partial", "unavailable"})


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
class InferenceRoute:
    provider: str
    native_provider: str
    endpoint: str
    wire_api: str
    credential_env: str
    authenticated: bool
    inventory_retrieved_at: str
    inventory_sha256: str
    inventory_models: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> InferenceRoute | None:
        if value is None:
            return None
        fields = dict(value)
        fields["inventory_models"] = tuple(fields.get("inventory_models", ()))
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
    inference_route: InferenceRoute | None
    effort: str | None
    profile: str
    no_web: bool
    effective_policy: dict[str, Any]
    timeout_seconds: float | None
    session_id: str | None
    parent_run_id: str | None
    artifacts: tuple[str, ...]
    inputs: tuple[InputSnapshot, ...]
    created_at: str
    # Replacement for the harness's base system prompt, and its exact
    # built-in tool set; None keeps the harness default.
    system_prompt: str | None = None
    tools: tuple[str, ...] | None = None
    # How the harness returns the model's thinking, such as "summarized".
    thinking_display: str | None = None
    # Base URL the harness sends inference requests to, instead of its default.
    provider_base_url: str | None = None
    # Extra URLs a --no-web run may reach through its egress proxy.
    allowed_urls: tuple[str, ...] = ()
    # A sealed run's private, initially empty /workspace is writable.
    writable_workspace: bool = False
    # How `aop migrate-results` converted an older record, if it did.
    migration: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunRequest:
        fields = dict(value)
        fields["inference_route"] = InferenceRoute.from_dict(
            fields.get("inference_route")
        )
        fields["artifacts"] = tuple(fields["artifacts"])
        if fields.get("tools") is not None:
            fields["tools"] = tuple(fields["tools"])
        fields["allowed_urls"] = tuple(fields.get("allowed_urls", ()))
        fields["inputs"] = tuple(
            InputSnapshot.from_dict(item) for item in fields["inputs"]
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
        return cls(**value)


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
    usage: TokenUsage | None
    calculated_cost: CalculatedCost | None
    provider_reported_cost: ProviderReportedCost | None = None
    billing: BillingProvenance = BillingProvenance()
    inference_provider: str | None = None
    inference_route: InferenceRoute | None = None
    artifacts: tuple[RunArtifact, ...] = ()
    inputs: tuple[InputSnapshot, ...] = ()
    provider_duration_seconds: float | None = None
    provider_status: str | None = None
    provider_error: str | None = None
    accounting_status: str = "complete"
    usage_schema: str = TOKEN_USAGE_SCHEMA
    # How `aop migrate-results` converted an older record, if it did.
    migration: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.usage_schema != TOKEN_USAGE_SCHEMA:
            raise ValueError(
                f"unsupported token usage schema {self.usage_schema!r}; {MIGRATE_HINT}"
            )
        if self.accounting_status not in ACCOUNTING_STATUSES:
            raise ValueError(f"unsupported accounting status: {self.accounting_status}")
        evidence = (
            self.usage is not None
            or self.calculated_cost is not None
            or self.provider_reported_cost is not None
        )
        if self.accounting_status == "unavailable" and evidence:
            raise ValueError("unavailable accounting cannot contain measurements")
        if self.accounting_status == "partial" and not evidence:
            raise ValueError("partial accounting requires a measurement")
        if self.accounting_status == "complete" and self.usage is None:
            raise ValueError("complete accounting requires token usage")
        if self.calculated_cost is not None and self.usage is None:
            raise ValueError("calculated cost requires token usage")

    @property
    def execution_completed(self) -> bool:
        return (
            self.exit_code == 0 and not self.timed_out and self.session_id is not None
        )

    @property
    def response_available(self) -> bool:
        return self.final_message is not None

    @property
    def status(self) -> str:
        if self.succeeded:
            return "succeeded"
        if (
            self.execution_completed
            and self.response_available
            and self.provider_error is not None
            and self.error is None
        ):
            return "response_available_with_provider_error"
        return "failed"

    @property
    def succeeded(self) -> bool:
        return (
            self.execution_completed
            and self.response_available
            and self.error is None
            and self.provider_error is None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "execution_completed": self.execution_completed,
            "response_available": self.response_available,
            "status": self.status,
            "succeeded": self.succeeded,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunResult:
        fields = dict(value)
        for derived in (
            "execution_completed",
            "response_available",
            "status",
            "succeeded",
        ):
            fields.pop(derived, None)
        if fields.get("usage_schema") != TOKEN_USAGE_SCHEMA:
            raise ValueError(
                f"unsupported token usage schema {fields.get('usage_schema')!r}; "
                f"{MIGRATE_HINT}"
            )
        usage = fields["usage"]
        fields["usage"] = None if usage is None else TokenUsage.from_dict(usage)
        fields["calculated_cost"] = CalculatedCost.from_dict(fields["calculated_cost"])
        fields["provider_reported_cost"] = ProviderReportedCost.from_dict(
            fields.get("provider_reported_cost")
        )
        fields["billing"] = BillingProvenance.from_dict(fields.get("billing"))
        fields["inference_route"] = InferenceRoute.from_dict(
            fields.get("inference_route")
        )
        fields["artifacts"] = tuple(
            RunArtifact(**artifact) for artifact in fields.get("artifacts", ())
        )
        fields["inputs"] = tuple(
            InputSnapshot.from_dict(item) for item in fields.get("inputs", ())
        )
        return cls(**fields)
