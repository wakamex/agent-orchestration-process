# Harness isolation and runtime state

This page documents the provider-specific behavior behind AOP's common run and resume interface.
Most users only need the sandbox and limitation summaries in the main README.

## Sandbox boundary

AOP runs providers inside `bwrap` unless `danger-full-access` is selected. The boundary controls
repository writes, while the provider retains access to its configured authentication and user
instructions.

| Mode | Repository | Task worktree | Scratch and cache |
| --- | --- | --- | --- |
| `workspace-write` | Read-only | Writable | Writable |
| `scratch-write` | Read-only | Read-only | Writable |
| `danger-full-access` | Native access | Native access | Native access |

The worktree's `.git` pointer and the shared Git directory remain read-only in both safe modes.
Provider permission prompts are bypassed because this mount boundary decides whether repository
writes can succeed.

## Task-private provider state

Private state lives under `.aop/provider-state/<task>/` and is reused for exact resume. Cleanup
removes it with the task while preserving immutable run records. AOP copies only the configuration
and credentials needed to start the provider, leaving existing sessions, logs, and caches in the
source profile.

| Harness | Private state | Applies in danger mode | Important behavior |
| --- | --- | --- | --- |
| Codex | `codex/home` | Yes | Global sessions and databases are never changed |
| Agy | `agy/gemini` | Yes | Resume fails if the reported conversation ID changes |
| Cursor | `cursor` | No | Sandboxed runs use private project and chat state |
| Devin | `devin` | Yes | The installed CLI bundle stays shared and read-only |
| OpenCode | `opencode` | No | Plugin dependencies are shared read-only; downloads use the shared cache |
| Hermes | `hermes/home` | Yes | Rotating OAuth credentials are coordinated across tasks |

Claude currently uses its configured native profile. Its repository access is still constrained by
the selected sandbox mode.

### Codex

AOP seeds authentication, root configuration, user rules and skills, and the model catalog. It does
not copy sessions, history, databases, logs, caches, or generated system skills. Set
`AOP_CODEX_SOURCE_HOME` when the authenticated source is not `${CODEX_HOME:-~/.codex}`.

### Agy

AOP seeds configuration and credentials from `~/.gemini`, but not conversations, history, caches,
logs, scratch files, or databases. An exact resume must report the requested conversation ID or the
run fails closed. Set `AOP_AGY_SOURCE_DIR` for a nonstandard source profile.

### Cursor Agent

Sandboxed runs copy CLI configuration, skills, plugins, policies, and authentication without copying
IDE extensions. Project metadata, chats, tracking data, and Cursor-created worktrees start empty.
`danger-full-access` uses Cursor's native global state. Set `AOP_CURSOR_HOME` or
`AOP_CURSOR_CONFIG_DIR` for nonstandard source locations.

### Devin CLI

AOP seeds authentication, configuration, and MCP state, excluding installed versions, sessions,
transcripts, logs, and databases. It invokes Devin's non-interactive dangerous permission mode
inside the outer AOP sandbox. On SSH, authenticate with
`devin auth login --force-manual-token-flow`. Use `AOP_DEVIN_DATA_DIR` or
`AOP_DEVIN_CONFIG_DIR` for nonstandard source locations.

Every invocation creates a fresh ATIF trajectory. AOP validates its prompt, terminal response, and
resume identity, records its token metrics, and archives it as `provider-result.json`.

### OpenCode

Sandboxed runs seed small configuration and authentication files while keeping sessions, logs,
model state, refreshed tokens, generated metadata, and downloads private. Existing generated plugin
dependencies are mounted read-only. `danger-full-access` uses native global state. Set
`AOP_OPENCODE_CONFIG_DIR` or `AOP_OPENCODE_DATA_DIR` for nonstandard source locations.

### Hermes

AOP seeds configuration, skills, hooks, and memories while keeping new sessions, logs, databases,
and caches private. Some Hermes OAuth refresh tokens rotate after use, so AOP maintains the freshest
credentials under `.aop/shared-provider-state/hermes/auth.json`. A repository lock serializes Hermes
turns to prevent concurrent tasks from consuming the same refresh token. Other harnesses remain
concurrent.

## Executable overrides

The following variables select alternate executables:

```text
AOP_CODEX_BIN
AOP_CLAUDE_BIN
AOP_CURSOR_BIN
AOP_DEVIN_BIN
AOP_OPENCODE_BIN
AOP_AGY_BIN
AOP_HERMES_BIN
AOP_BWRAP_BIN
```
