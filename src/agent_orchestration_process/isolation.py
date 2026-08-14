"""Semantic execution profiles and their resolved isolation contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .worktrees import AOPError


PROFILES = ("edit", "review", "sealed", "host")


@dataclass(frozen=True)
class Profile:
    name: str
    workspace_access: str
    host_access: str
    inputs: str
    instructions: str
    environment: str
    state: str
    network: str
    identity: str


@dataclass(frozen=True)
class ResolvedPolicy:
    schema_version: int
    profile: str
    repository: dict[str, object]
    workspace: dict[str, object]
    host: dict[str, object]
    inputs: dict[str, object]
    writable_paths: tuple[str, ...]
    writable_path_scopes: dict[str, str]
    working_directory: str
    environment: dict[str, object]
    instructions: dict[str, str]
    state: str
    network: dict[str, object]
    identity: str
    namespaces: tuple[str, ...]
    capabilities: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_PROFILE_DEFINITIONS = {
    "edit": Profile(
        name="edit",
        workspace_access="write",
        host_access="runtime-only",
        inputs="snapshot",
        instructions="project-and-user",
        environment="filtered",
        state="resumable",
        network="native",
        identity="normal",
    ),
    "review": Profile(
        name="review",
        workspace_access="read",
        host_access="runtime-only",
        inputs="snapshot",
        instructions="project-and-user",
        environment="filtered",
        state="resumable",
        network="native",
        identity="normal",
    ),
    "sealed": Profile(
        name="sealed",
        workspace_access="none",
        host_access="runtime-only",
        inputs="snapshot",
        instructions="none",
        environment="filtered",
        state="resumable",
        network="native",
        identity="opaque",
    ),
    "host": Profile(
        name="host",
        workspace_access="write",
        host_access="native",
        inputs="snapshot",
        instructions="project-and-user",
        environment="native",
        state="resumable",
        network="native",
        identity="normal",
    ),
}


def profile(name: str) -> Profile:
    try:
        return _PROFILE_DEFINITIONS[name]
    except KeyError as error:
        raise AOPError(f"unknown execution profile: {name}") from error


def resolve_policy(
    name: str,
    *,
    workspace_host_path: Path | None = None,
    input_names: tuple[str, ...] = (),
) -> ResolvedPolicy:
    selected = profile(name)
    workspace_visible = selected.workspace_access != "none"
    workspace = {
        "access": selected.workspace_access,
        "guest_path": "/workspace",
        "content": "task-checkout" if workspace_visible else "neutral-empty",
        "writable": selected.workspace_access == "write",
        "host_path": (
            str(workspace_host_path) if workspace_host_path is not None else None
        ),
    }
    repository_access = (
        "none" if name == "sealed" else "native" if name == "host" else "read"
    )
    writable = ["/output", "/scratch", "/state", "/cache", "/tmp"]
    if selected.workspace_access == "write":
        writable.insert(0, "/workspace")
    if selected.host_access == "native":
        writable = ["native host filesystem"]
    writable_scopes = {
        "/workspace": "task-private",
        "/output": "run-private",
        "/scratch": "session-private" if name == "sealed" else "task-private",
        "/state": "session-private" if name == "sealed" else "task-private",
        "/cache": "session-private" if name == "sealed" else "shared",
        "/tmp": "process-private",
    }
    if selected.host_access == "native":
        writable_scopes = {"native host filesystem": "native"}
    return ResolvedPolicy(
        schema_version=1,
        profile=name,
        repository={
            "access": repository_access,
            "guest_path": "/repository" if repository_access == "read" else None,
        },
        workspace=workspace,
        host={
            "access": selected.host_access,
            "runtime_paths": (
                ["/usr", "/bin", "/lib", "/lib64", "/etc runtime files"]
                if selected.host_access == "runtime-only"
                else ["/"]
            ),
        },
        inputs={
            "mode": selected.inputs,
            "guest_root": (
                "/inputs"
                if selected.inputs == "snapshot" and selected.host_access != "native"
                else None
            ),
            "names": list(input_names),
            "source_paths_visible": selected.host_access == "native",
        },
        writable_paths=tuple(writable),
        writable_path_scopes={path: writable_scopes[path] for path in writable},
        working_directory="/workspace",
        environment={
            "mode": selected.environment,
            "ambient_values_inherited": selected.environment == "native",
            "allowed_names": (
                []
                if selected.environment == "native"
                else [
                    "HOME",
                    "LANG",
                    "LC_ALL",
                    "LC_CTYPE",
                    "PATH",
                    "PWD",
                    "SSL_CERT_DIR",
                    "SSL_CERT_FILE",
                    "TERM",
                    "TZ",
                    "http_proxy",
                    "https_proxy",
                    "no_proxy",
                    "AOP_CACHE_DIR",
                    "AOP_INPUT_DIR",
                    "AOP_INPUT_MANIFEST",
                    "AOP_OUTPUT_DIR",
                    "AOP_PROVIDER_STATE_DIR",
                    "AOP_SCRATCH_DIR",
                    "CODEX_HOME",
                    "HERMES_HOME",
                    "XDG_CACHE_HOME",
                    "XDG_CONFIG_HOME",
                    "XDG_DATA_HOME",
                    "XDG_STATE_HOME",
                ]
            ),
            "credential_names": (
                []
                if selected.environment == "native"
                else [
                    "ANTHROPIC_API_KEY",
                    "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY",
                    "AWS_SESSION_TOKEN",
                    "AZURE_OPENAI_API_KEY",
                    "CURSOR_API_KEY",
                    "DEVIN_API_KEY",
                    "GEMINI_API_KEY",
                    "GOOGLE_API_KEY",
                    "NOUS_API_KEY",
                    "OPENAI_API_KEY",
                    "OPENCODE_API_KEY",
                    "XAI_API_KEY",
                ]
            ),
            "credential_exposure": "provider-process-readable",
            "recorded_values": "redacted",
        },
        instructions={
            "inherited_local": selected.instructions,
            "provider_builtin": "present",
            "aop_generated": "input-and-artifact-contract-when-needed",
        },
        state=selected.state,
        network={
            "mode": selected.network,
            "isolation": "none",
            "possible_context_channels": [
                "local-network-services",
                "provider-account-state",
            ],
        },
        identity=selected.identity,
        namespaces=(
            () if selected.host_access == "native" else ("mount", "pid", "ipc", "uts")
        ),
        capabilities="native" if selected.host_access == "native" else "none",
    )


def explain_profile(name: str) -> dict[str, object]:
    return resolve_policy(name).to_dict()
