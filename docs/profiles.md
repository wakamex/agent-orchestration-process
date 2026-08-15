# Execution profiles

AOP profiles define the complete execution boundary, not only where writes are allowed. The three
isolated profiles build an empty `bwrap` mount namespace and explicitly add the selected workspace,
repository view, immutable inputs, output, provider runtime, private state, cache, scratch, required
operating-system runtime paths, and fresh `/proc`, `/dev`, and `/tmp`. They do not inherit the host
root filesystem.

| Profile | Primary repository | Task workspace | Other host paths | Instructions | Writable guest paths |
| --- | --- | --- | --- | --- | --- |
| `edit` | Read-only at `/repository` | Read-write at `/workspace` | Runtime allowlist only | Project and user | `/workspace`, `/output`, `/scratch`, `/state`, `/cache`, `/tmp` |
| `review` | Read-only at `/repository` | Read-only at `/workspace` | Runtime allowlist only | Project and user | `/output`, `/scratch`, `/state`, `/cache`, `/tmp` |
| `sealed` | Not mounted | Empty read-only `/workspace` | Runtime allowlist only | No inherited local instructions | `/output`, `/scratch`, `/state`, `/cache`, `/tmp` |
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

All profiles currently retain native host networking because the supported provider CLIs need it.
Local network services and provider-side account state can therefore remain context channels.
`sealed` is a filesystem and local-state boundary, not complete information isolation.

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

Credential values are redacted from recorded commands and results, but the selected provider and
its shell tools can read credentials exposed through the environment or private `/state`. A sealed
profile limits which credentials enter the process; it does not make those credentials unreadable
to that process.

## Immutable inputs and declared outputs

Repeatable `--input` paths are copied into controller-owned per-run snapshots and mounted read-only
beneath `/inputs`. Their original host paths are not mounted or included in the child command,
environment, or prompt. Stable guest paths such as `/inputs/ledger.json` do not reveal controller
directory names.

Each invocation receives a fresh writable `/output`, exposed in `AOP_OUTPUT_DIR`. AOP validates and
archives only paths declared with `--artifact`; it never copies artifacts into the main worktree.
See the [CLI guide](cli.md#inputs-and-artifacts) for declaration and validation details.

## Provider runtime state

Isolated provider state lives beneath `.aop/provider-state/` for repository tasks and beneath an
opaque key for sealed sessions. AOP seeds only the configuration, credentials, and extensions
allowed by the selected profile. Existing provider sessions, logs, caches, and unrelated credentials
are excluded. Exact resume reuses the task-private state associated with that session.

The exact seeded paths and adapter-specific exceptions are documented in
[Provider adapters](harnesses.md).
