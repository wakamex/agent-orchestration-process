# Execution profiles

AOP profiles define the complete execution boundary, not only where writes are allowed. The three
isolated profiles build an empty `bwrap` mount namespace and explicitly add the selected workspace,
repository view, read-only input snapshots, output, provider runtime, private state, cache, scratch,
required operating-system runtime paths, and fresh `/proc`, `/dev`, and `/tmp`. They do not inherit
the host root filesystem.

Inside the sandbox, bubblewrap's own reaper is process 1 and collects orphaned processes, and the harness runs as process 2. AOP passes bubblewrap's options through a private file descriptor, so process 1's command line, which every process in the sandbox can read, names no mount or host path. When `--timeout` expires, AOP sends SIGTERM to the harness itself, so it can save its session and flush its output, and stops the sandbox after a five-second grace period.

| Profile | Primary repository | Task workspace | Other host paths | Instructions | Writable guest paths |
| --- | --- | --- | --- | --- | --- |
| `edit` | Read-only at `/repository` | Read-write at `/workspace` | Runtime allowlist only | Project and user | `/workspace`, `/output`, `/scratch`, `/state`, `/cache`, `/tmp` |
| `review` | Read-only at `/repository` | Read-only at `/workspace` | Runtime allowlist only | Project and user | `/output`, `/scratch`, `/state`, `/cache`, `/tmp` |
| `sealed` | Not mounted | Empty `/workspace`, read-only unless `--writable-workspace` | Runtime allowlist only | No inherited local instructions | `/output`, `/scratch`, `/state`, `/cache`, `/tmp`, and `/workspace` with `--writable-workspace` |
| `host` | Native | Native | Native | Native | Native |

The worktree's `.git` pointer and shared Git directory remain read-only under `edit` and `review`.
The repository's `.aop` directory is masked from `/repository`, so a task cannot read run records,
provider state, caches, or sibling worktrees. Provider permission prompts are bypassed because this
outer mount boundary decides whether filesystem writes can succeed.

`host` skips `bwrap` and environment filtering. Use it only when the native host filesystem and
environment are part of the intended boundary.

## Sealed runs

`sealed` does not require a Git repository. It uses an opaque agent-visible identity, canonical
guest paths, a filtered environment, and minimal private provider configuration. It does not
inherit project or user rules, skills, plugins, hooks, MCP configuration, memories, repository
contents, controller paths, or human task labels.

Under `sealed`, `/cache`, `/scratch`, and `/state` are private to one session and reused only by its
exact resumes. `/output` is private to one invocation. The compiled policy records these scopes
explicitly.

`--writable-workspace` makes a sealed run's `/workspace` writable, so the model can write and run code where it naturally works. The workspace still starts empty, belongs to that session alone, keeps its files across the session's resumes, and is removed by `aop cleanup`. The request records `writable_workspace`, and the effective policy marks `/workspace` writable.

Account-level customizations arrive with a login rather than from local files, so a private home alone does not exclude them. Claude Code downloads a claude.ai account's connectors as `mcp__claude_ai_*` tools and its claude.ai skills into the home it runs with. Sealed Claude runs therefore pass `--safe-mode`, `--strict-mcp-config`, and an empty `--mcp-config`, which keep out MCP servers, connectors, skills, and other customizations. Claude Code's own built-in tools and skills remain.

A sealed run records itself under the directory it was started from, even inside a Git repository. `aop resume` and `aop cleanup` look for the run in the repository's state first and then in the current directory's, so run them from the directory the run started in.

Without `--no-web`, every profile retains native host networking because the supported provider CLIs need it. Local network services and provider-side account state can therefore remain context channels. With `--no-web`, the sandboxed profiles get their own network namespace with only an allowlisted egress proxy, as the CLI guide describes.
`sealed` is a filesystem and local-state boundary, not complete information isolation.

`--no-web` adds a separate model-capability boundary. It leaves provider inference traffic enabled
but denies model-controlled external retrieval through native web tools, browsers, extensions,
subagents, and shell processes. AOP records the adapter-specific enforcement and effective tool
allowlist in `request.json`. Unsupported adapters fail before dispatch rather than receiving a
prompt-only prohibition.

## Inspect the effective profile

Inspect a declared profile or preview a run without dispatching it:

```sh
aop profile explain sealed
aop profile explain sealed --agent codex --json
aop run study-arm --profile sealed --prompt "Answer using only declared inputs" --dry-run
```

Every dispatched run persists its compiled boundary in `request.json`. It records repository and
workspace access, guest paths, writable-path scope, input mode, environment and credential
exposure, inherited and generated instruction policy, namespaces, network limitations, provider
executable hash, and controller-owned state locations.
The record also includes the requested and effective model capabilities and their enforcement
mechanisms.

Credential values are redacted from recorded commands and results, but the selected provider and
its shell tools can read credentials exposed through the environment or private `/state`. A sealed
profile limits which credentials enter the process; it does not make those credentials unreadable
to that process.

## Immutable inputs and declared outputs

Repeatable `--input` paths are copied into private, durable per-run snapshots beneath
`.aop/snapshots`. Isolated profiles mount them read-only beneath `/inputs`. Their original host paths
are not mounted or included in the child command, environment, or prompt. Stable guest paths such
as `/inputs/ledger.json` do not reveal controller directory names. The `host` profile uses the same
copies but does not enforce a read-only mount because its filesystem access is intentionally native.
Sealed runs mount an opaque temporary projection of the durable snapshot so Linux mount metadata
does not expose the controller path. AOP removes the projection after execution and retains the
snapshot.

Each invocation receives a fresh writable `/output`, exposed in `AOP_OUTPUT_DIR`. AOP validates and
archives only paths declared with `--artifact`; it never copies artifacts into the main worktree.
See the [CLI guide](cli.md#inputs-and-artifacts) for declaration and validation details.

## Provider runtime state

Isolated provider state lives beneath `.aop/provider-state/` for repository tasks and beneath an
opaque key for sealed sessions. AOP seeds only the configuration, credentials, and extensions
allowed by the selected profile. Existing provider sessions, logs, caches, and unrelated credentials
are excluded. Exact resume reuses the task-private state associated with that session.

The exact seeded paths and adapter-specific exceptions are documented in
[Harness adapters](harnesses.md).
