# Harness isolation and runtime state

This page documents the provider-specific behavior behind AOP's common run and resume interface.
Most users only need the profile and limitation summaries in the main README.

## Execution boundary

AOP runs providers inside an empty `bwrap` mount namespace unless `host` is selected. It adds only
the selected workspace, repository view, immutable input snapshots, output, private provider state,
cache, scratch, provider executable, operating-system runtime files, and a fresh `/proc`, `/dev`,
and `/tmp`. The child environment is cleared and rebuilt from an allowlist.

| Profile | Repository | Task workspace | Other host paths | Instructions |
| --- | --- | --- | --- | --- |
| `edit` | Read-only | Writable | Runtime allowlist | Project and user |
| `review` | Read-only | Read-only | Runtime allowlist | Project and user |
| `sealed` | Hidden | Empty read-only directory | Runtime allowlist | No inherited local instructions |
| `host` | Native | Native | Native | Native |

The worktree's `.git` pointer and shared Git directory remain read-only under `edit` and `review`.
The repository's `.aop` directory is masked from its `/repository` view so one task cannot read run
records, provider state, caches, or sibling worktrees. Provider permission prompts are bypassed
because the mount boundary decides whether filesystem writes can succeed.

Repeatable `--input` paths are copied into controller-owned per-run snapshots before dispatch. AOP
hashes the copies and mounts them read-only beneath `/inputs` without mounting their original host
paths. The source-path-free child manifest and prompt use stable guest paths. Controller records
retain the original source path for audit. An exact sealed resume snapshots the parent's immutable
copies again by default, so changed or deleted source files cannot alter the session. Explicit new
`--input` arguments replace the inherited snapshots.

All profiles currently use native host networking. The effective policy reports that no network
isolation is present and names local services and provider-side account state as possible context
channels. Sealed is a filesystem and local-state boundary, not complete information isolation.

## Task-private provider state

Private state lives under `.aop/provider-state/<task>/` for repository tasks and beneath an opaque
run key for sealed tasks. It is reused for exact resume. Cleanup
removes it with the task while preserving immutable run records. AOP copies only the configuration
and credentials needed to start the provider, leaving existing sessions, logs, and caches in the
source profile.

Sealed cache state lives under a separate opaque `.aop/sealed-cache/<run-key>/` directory. It is
mounted only into that sealed session, reused on exact resume, and removed by sealed cleanup. It is
never the shared `.aop/cache` used by non-sealed tasks.

| Harness | Private state | Important behavior |
| --- | --- | --- |
| Codex | `codex/home` | Global sessions and databases are never changed |
| Claude | `claude/home` | Runtime history and project state start private |
| Agy | `agy/gemini` | Resume fails if the reported conversation ID changes |
| Cursor | `cursor` | Project and chat state start private |
| Devin | `devin` | The installed CLI bundle is not copied into task state |
| OpenCode | `opencode` | Plugin dependencies are copied into private state |
| Grok Build | `grok/home` | Global sessions, logs, caches, and memory are never copied |
| Hermes | `hermes/home` | Rotating OAuth credentials are coordinated across tasks |
| DeepSeek Harness | `dsh/home` | AOP supplies a structured, resumable runner as a final dsh patch layer |

Claude uses the same filtered environment and filesystem profile as the other adapters. Its
repository access is constrained by the selected AOP profile.

### Codex

AOP seeds authentication, root configuration, user rules and skills, and the model catalog for
`edit` and `review`. `sealed` seeds authentication only. It does
not copy sessions, history, databases, logs, caches, or generated system skills. Set
`AOP_CODEX_SOURCE_HOME` when the authenticated source is not `${CODEX_HOME:-~/.codex}`.

### Agy

AOP seeds configuration and credentials from `~/.gemini`, but not conversations, history, caches,
logs, scratch files, or databases. An exact resume must report the requested conversation ID or the
run fails closed. `sealed` omits user configuration and retains authentication material only. Set
`AOP_AGY_SOURCE_DIR` for a nonstandard source profile.

### Cursor Agent

`edit` and `review` copy CLI configuration, skills, plugins, policies, and authentication without copying
IDE extensions. Project metadata, chats, tracking data, and Cursor-created worktrees start empty.
`sealed` copies authentication only.
`host` uses Cursor's native global state. Set `AOP_CURSOR_HOME` or
`AOP_CURSOR_CONFIG_DIR` for nonstandard source locations.

### Devin CLI

AOP seeds authentication and configuration, excluding installed versions, sessions,
transcripts, logs, and databases. It invokes Devin's non-interactive dangerous permission mode
inside the outer AOP profile. `sealed` removes instruction, MCP, plugin, hook, skill, rule, and
memory keys from its minimal JSON configuration. On SSH, authenticate with
`devin auth login --force-manual-token-flow`. Use `AOP_DEVIN_DATA_DIR` or
`AOP_DEVIN_CONFIG_DIR` for nonstandard source locations.

Every invocation creates a fresh ATIF trajectory. AOP validates its prompt, terminal response, and
resume identity, records its token metrics, and archives it as `provider-result.json`.

### OpenCode

`edit` and `review` seed small configuration and authentication files while keeping sessions, logs,
model state, refreshed tokens, generated metadata, and downloads private. Existing generated plugin
dependencies are copied into private state. `sealed` omits user configuration and instructions.
`host` uses native global state. Set
`AOP_OPENCODE_CONFIG_DIR` or `AOP_OPENCODE_DATA_DIR` for nonstandard source locations.

### Grok Build

AOP seeds Grok authentication, configuration, user rules, skills, agents, commands, hooks,
personas, plugins, and workflows into task-private `GROK_HOME` state for `edit` and `review`.
Existing sessions, logs, caches, cross-session memory, plugin data, model caches, crash reports,
trace exports, worktrees, and installed binaries are not copied. `sealed` retains only
`auth.json`. Set `AOP_GROK_SOURCE_HOME` when the authenticated source is not
`${GROK_HOME:-~/.grok}`.

The adapter invokes Grok's native single-turn `streaming-json` interface with always-approve mode,
no plan mode, no shared leader, no auto-update, verbatim prompts, and its nested sandbox disabled inside AOP's outer
profile boundary. It records the terminal session ID, response, model, disjoint prompt cache
buckets, the reasoning-token subset, timing, cost evidence, and exact resume identity. Grok defaults to `grok-build` and
accepts native reasoning levels from `none` through `max`.

`GROK_STORAGE_MODE` is part of the filtered child environment only when the user set it. AOP passes
`local` or `writeback` through unchanged and does not create a default, so an unset value retains
Grok's native configuration and remote-setting behavior.

### Hermes

AOP seeds configuration, skills, hooks, and memories for `edit` and `review` while keeping new sessions, logs, databases,
and caches private. Some Hermes OAuth refresh tokens rotate after use, so AOP maintains the freshest
credentials under `.aop/shared-provider-state/hermes/auth.json`. A repository lock serializes Hermes
turns to prevent concurrent tasks from consuming the same refresh token. Other harnesses remain
concurrent. `aop run --provider NAME --model MODEL` overrides the profile's inference provider for
the task, records that choice, and reuses it on resume.

Under `sealed`, Hermes receives credentials but not skills, hooks, memories, plugins, optional MCPs,
or other user extension directories.

### DeepSeek Harness

AOP invokes the official `dsh --profile headless` command directly. It does not use a personal
`dsh` wrapper or assume a DeepSeek Harness source checkout. Because the released headless runner
creates only fresh sessions and prints unstructured final text, AOP replaces only that runner row
through the documented `--patch` interface. The packaged AOP runner drives dsh's public Agent
Registry, chooses a stable session ID, supports exact cross-process resume, and emits one JSON result
with the final response and per-turn token buckets. DeepSeek Harness continues to own its provider,
agent loop, tools, prompt assembly, and JSONL session persistence.

Every task receives a private `DSH_HOME` under `dsh/home`. AOP reads the source `settings.yaml` and
projects only the selected route: `llm-deepseek` for `deepseek-official`, or the exact
`llm-pi-ai.providers.<provider>` profile selected by `--provider`. It then reads that profile's
explicit `apiKeyEnv` and projects only the matching entry from dsh's managed `.credentials.yaml`.
The direct DeepSeek adapter defaults that reference to `DEEPSEEK_API_KEY`, matching dsh itself.
AOP does not derive a key name from a provider label.

If a pi-ai profile omits `apiKeyEnv`, AOP preserves the omission and delegates authentication to
that provider's native ambient environment discovery. Inside `edit`, `review`, and `sealed`, the
environment contains only the selected credential reference or the known ambient variables for the
selected provider family. Another provider's key is excluded. `host` retains its documented native
host environment.

AOP does not copy unrelated settings, profiles, sessions, logs, plugins, credentials, or the
user-level `.env`. This makes cleanup own the generated profile and session log. Set
`AOP_DSH_SOURCE_HOME` when the dsh source state is not `${DSH_HOME:-~/.dsh}`. AOP sets
`DSH_TELEMETRY_DISABLED=1` for every dsh run.

An inherited environment value matching the selected `apiKeyEnv` still wins according to dsh's
native credential precedence. AOP does not create a second credential source or ask users to
authenticate again for isolated work. Credential values are never placed in AOP's request, result,
command record, or prompt.

AOP disables dsh's optional `session-title-llm` row. Its auxiliary title request does not contribute
to the agent turn or its reported token buckets, while the deterministic fallback title is enough
for AOP's private session record.

The adapter defaults to dsh's bundled `deepseek-v4-flash`, also accepts
`deepseek-v4-pro`, and maps AOP effort `none` to native `off`. With `--provider`, it accepts an exact
model ID and the reasoning levels dsh exposes across its pi-ai adapters. The headless command has no
provider-aware model-list operation, so `aop models --agent dsh` reports the bundled DeepSeek
defaults rather than claiming an account-derived inventory.

## Executable overrides

The following variables select alternate executables:

```text
AOP_CODEX_BIN
AOP_CLAUDE_BIN
AOP_CURSOR_BIN
AOP_DEVIN_BIN
AOP_OPENCODE_BIN
AOP_AGY_BIN
AOP_GROK_BIN
AOP_HERMES_BIN
AOP_DSH_BIN
AOP_BWRAP_BIN
```

For dsh installed under a user npm prefix, AOP mounts the installed `node_modules` runtime at its
original absolute path so Node can follow the package's normal dependency links. System and
Homebrew installations use their existing runtime mounts. `AOP_DSH_BIN` should identify the
official packaged entry point, not a personal wrapper or a source-checkout script.

## Live input-snapshot smoke test

The default suite tests the shared mount contract without contacting providers. An opt-in matrix
checks whether real authenticated harnesses can read a declared path:

```sh
AOP_LIVE_INPUT_AGENTS=agy,codex,devin,grok,hermes uv --no-config run --locked pytest tests/test_live_read_paths.py
```

Use `all` to select every adapter. `AOP_LIVE_<AGENT>_MODEL` and
`AOP_LIVE_<AGENT>_EFFORT` select harness-specific test options.
`AOP_LIVE_HERMES_PROVIDER` selects the Hermes inference provider, and
`AOP_LIVE_INPUT_TIMEOUT` changes the default 300-second bound. The test uses `sealed`, verifies a
random nonce from an immutable input snapshot, confirms the source was unchanged, and retains normal
AOP run evidence. These tests spend provider tokens and are never part of the default suite.
