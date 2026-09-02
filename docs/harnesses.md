# Harness adapters

AOP provides one `run` and `resume` interface over installed agent harnesses. Install and
authenticate the harness you want to use first. AOP uses its native credentials, models, tools, and
session mechanism while giving every run the same AOP access controls and result format.

## Supported harnesses

| Harness | `--agent` | Default | Model and effort selection |
| --- | --- | --- | --- |
| [Codex](https://github.com/openai/codex) | `codex` | Native default | Accepts native model IDs and effort levels. Z.AI Coding Plan is available as the `zai-coding-plan` inference provider. |
| [Claude Code](https://code.claude.com/docs/en/overview) | `claude` | Native default | Accepts normal model aliases and effort from `low` through `max`. |
| [Cursor Agent](https://docs.cursor.com/en/cli/overview) | `cursor` | `composer-2.5` | Reasoning and fast variants are part of the model ID, so separate effort is rejected. |
| [Devin CLI](https://docs.devin.ai/cli/index) | `devin` | `swe-1-7` | Reasoning variants are part of the model ID, so separate effort is rejected. |
| [OpenCode](https://github.com/anomalyco/opencode) | `opencode` | `opencode/deepseek-v4-flash` | Short model names are qualified with `opencode/`; effort selects the native variant. |
| [Antigravity](https://github.com/google-antigravity/antigravity-cli) | `agy` | Native default | Accepts exact names printed by `agy models`; effort is `low`, `medium`, or `high`. |
| [Grok Build](https://github.com/xai-org/grok-build) | `grok` | `grok-build` | Accepts native effort from `none` through `max`. |
| [Hermes](https://github.com/NousResearch/hermes-agent) | `hermes` | `deepseek/deepseek-v4-flash-0731` | Uses the configured inference provider unless both `--provider` and `--model` override it. |
| [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) | `dsh` | `deepseek-v4-flash` through `deepseek-official` | Also supports `deepseek-v4-pro`; another configured route requires both `--provider` and `--model`. |

Use `aop models` to inspect installed or configured inventories and see whether each entry came from
the authenticated harness, an installed default, or AOP's catalog. Exact resume retains the original
harness, model, effort, inference provider, mode, and execution profile.

Every adapter supports bounded execution, exact resume, all four execution profiles, input
snapshots, validated artifacts, normalized terminal results and token usage, and retained
provenance. Harness-native automation surfaces determine the remaining feature differences:

| Harness | Enforced no-web | Participant | Model inventory | Provider override | Calculated cost | Provider-reported cost |
| --- | --- | --- | --- | --- | --- | --- |
| Codex | Yes | No | Account or authenticated route | Z.AI Coding Plan | API-equivalent | No |
| Claude Code | Yes | No | Catalog | No | CLI-calculated | No |
| Cursor Agent | No | No | Account | No | No | No |
| Devin CLI | No | No | Account | No | No | No |
| OpenCode | Yes | No | Account | No | API-equivalent | When emitted |
| Antigravity | No | No | Account | No | API-equivalent | No |
| Grok Build | Isolated profiles only | No | Account | No | API-equivalent | When emitted |
| Hermes | Yes | Experimental | Configured route | Yes | API-equivalent or CLI-calculated | When emitted |
| DeepSeek Harness | No | No | Bundled defaults | Yes | Known provider routes | No |

Calculated cost is an API-equivalent comparison unless the table identifies a native CLI
calculation. Provider-reported cost remains separate and is retained only when the harness exposes
monetary evidence for the selected billing route. Hermes turns are serialized when shared rotating
OAuth credentials require it; other harnesses remain concurrent.

## Enforced no-web support

`aop run --no-web` keeps inference traffic available while denying model-controlled external
retrieval. Enforcement differs because the harnesses expose different native control surfaces.

| Harness | Effective no-web policy |
| --- | --- |
| Codex | Native web search and tool network access disabled; local shell and file tools remain available inside Codex's network-disabled workspace sandbox. Native routes use authentication-only state; selected external routes receive an AOP-generated provider-only config. |
| Claude Code | Local file-tool allowlist; safe mode, Chrome, and MCP restrictions enabled. |
| OpenCode | All model tool calls denied; external plugins disabled. |
| Grok Build | Isolated profiles only; local file-tool allowlist; native web, subagent, shell, fetch, and MCP routes denied; authentication-only state. |
| Hermes | `file` and `todo` toolsets only; safe mode disables customizations and MCP. |
| Cursor Agent | Unsupported; the run fails before dispatch. |
| Devin CLI | Unsupported; the run fails before dispatch. |
| Antigravity | Unsupported; the run fails before dispatch. |
| DeepSeek Harness | Unsupported; the run fails before dispatch. |

The complete effective policy and adapter mechanisms are retained in each run's `request.json`.
Exact resumes inherit the policy. AOP never substitutes a prompt instruction for missing
enforcement.

## Authentication and private state

Authenticate with the native harness before using it through AOP. For isolated profiles, AOP copies
only the credentials, configuration, and extensions allowed by the selected profile into private
state beneath `.aop/provider-state/`. Existing global sessions, logs, and caches are not reused.
Exact resume reuses the same private state so credential updates and harness session data remain
available to that session.

`sealed` uses an opaque session identity and omits local instructions and extensions. It receives
only the authentication material required by the selected harness. Its private state and cache are
reused only by exact resumes.

Cleanup removes mutable harness state with the task while preserving retained requests, results,
events, logs, inputs, and artifacts. See [Execution profiles](profiles.md) for the surrounding
filesystem and environment boundary.

## Harness-specific behavior

### Codex

For native routes under `edit` and `review`, AOP seeds authentication, root configuration, user rules and skills, and the model catalog. It does not copy sessions, history, databases, logs, caches, or generated system skills. Native `sealed` runs receive authentication only.

AOP executes new Codex turns and exact resumes through Codex app-server. The native protocol supplies thread identity, terminal messages, per-turn token usage, effective model identity, and exact resume without AOP reading Codex session history. When a run reaches its AOP deadline, AOP sends `turn/interrupt`, allows a bounded five-second accounting and teardown grace outside the requested run time, and then terminates the provider process if it does not stop. A timed-out turn remains failed even when native interruption returns terminal accounting evidence.

Under `--no-web`, AOP gives app-server authentication-only state, disables native web search, and selects a named Codex permission profile that mirrors the AOP filesystem profile while disabling tool network access. Local shell and file tools remain usable, `review` and `sealed` keep `/workspace` read-only, and `edit` can modify its task worktree without permitting tool-controlled network retrieval. Named permission profiles require Codex 0.150.1 or newer.

Z.AI Coding Plan remains a Codex inference route rather than a separate harness. Configure a native Codex provider for `https://api.z.ai/api/v1` with `wire_api = "responses"`, `requires_openai_auth = false`, and one environment-key credential. AOP reads the effective layered Codex configuration for the task working directory through app-server `config/read`, records the stable route identity as `zai-coding-plan`, and separately preserves the native Codex provider key.

Use both `--provider zai-coding-plan` and `--model MODEL` for an explicit override. When native Codex configuration already selects the validated route, AOP records its effective model and route without requiring an override. AOP rejects ambiguous route matches, noncanonical endpoints, mixed authentication, missing credentials, and models absent from the fresh authenticated inventory.

For isolated Z.AI runs, AOP creates route-private Codex state containing only the selected provider, pinned authenticated model catalog, and allowed rules or skills for the profile. It projects only the configured credential environment variable and never copies ChatGPT authentication, `.env`, unrelated providers, or unrelated API keys. Exact resume retains the original route and inventory snapshot while rereading the pinned credential variable and verifying current model availability.

Inspect the current route inventory with:

```sh
aop models --agent codex --provider zai-coding-plan --json
```

Set `AOP_CODEX_SOURCE_HOME` when the authenticated source is not
`${CODEX_HOME:-~/.codex}`.

### Claude Code

For `edit` and `review`, AOP seeds Claude authentication and configuration into a private home while
leaving runtime history, project state, logs, and caches behind. `sealed` receives authentication
only. Repository access is constrained by the selected AOP profile.

### Cursor Agent

For `edit` and `review`, AOP seeds CLI configuration, skills, plugins, policies, and authentication
without copying IDE extensions. Project metadata, chats, tracking data, and Cursor-created worktrees
start empty. `sealed` receives authentication only, while `host` uses Cursor's native global state.

Set `AOP_CURSOR_HOME` or `AOP_CURSOR_CONFIG_DIR` for nonstandard source locations.

### Devin CLI

AOP seeds authentication and configuration without copying installed versions, sessions,
transcripts, logs, or databases. It invokes Devin's noninteractive permission mode inside the AOP
execution boundary. `sealed` removes instruction, MCP, plugin, hook, skill, rule, and memory settings
from its minimal configuration.

On SSH, authenticate with `devin auth login --force-manual-token-flow`. Set `AOP_DEVIN_DATA_DIR` or
`AOP_DEVIN_CONFIG_DIR` for nonstandard source locations. Each invocation must produce a valid Agent
Trajectory Interchange Format (ATIF) export containing its terminal response and resume identity.

### OpenCode

For `edit` and `review`, AOP seeds configuration, authentication, and existing generated plugin
dependencies while keeping sessions, logs, model state, refreshed tokens, generated metadata, and
downloads private. `sealed` omits user configuration and instructions, while `host` uses native
global state.

Set `AOP_OPENCODE_CONFIG_DIR` or `AOP_OPENCODE_DATA_DIR` for nonstandard source locations.

### Antigravity

AOP seeds configuration and credentials from `~/.gemini`, but not conversations, history, caches,
logs, scratch files, or databases. `sealed` retains authentication only. An exact resume fails if
Antigravity reports a different conversation ID.

When native Agy settings select `modelProvider: "gemini"`, AOP preserves `GEMINI_API_KEY` and the
optional `GOOGLE_GEMINI_BASE_URL` override. Isolated state omits unrelated OAuth credentials, and
`sealed` retains only the provider selection. Results record the metered API route and API-key
credential source without recording either environment value.

Set `AOP_AGY_SOURCE_DIR` for a nonstandard source profile.

### Grok Build

For `edit` and `review`, AOP seeds authentication, configuration, rules, skills, agents, commands,
hooks, personas, plugins, and workflows into a private `GROK_HOME`. It does not copy sessions, logs,
caches, cross-session memory, plugin data, model caches, crash reports, traces, worktrees, or
installed binaries. `sealed` receives `auth.json` only.

AOP runs one noninteractive Grok turn and disables Grok's nested sandbox inside the outer AOP
profile. It retains the exact session identity, normalized usage, timing, cost evidence, and final
response needed by the common result interface.

If `GROK_STORAGE_MODE` is exported in the environment that starts AOP, its value is passed to Grok
unchanged. AOP neither validates nor defaults it. Set `AOP_GROK_SOURCE_HOME` when the authenticated
source is not `${GROK_HOME:-~/.grok}`.

### Hermes

For `edit` and `review`, AOP seeds configuration, skills, hooks, and memories while keeping sessions,
logs, databases, and caches private. `sealed` receives credentials without extensions or other user
instruction sources.

Some Hermes OAuth refresh tokens rotate after use. AOP keeps the freshest credentials in
repository-private shared state and serializes Hermes turns so concurrent tasks cannot consume the
same token. Other harnesses remain concurrent. Supplying both `--provider NAME` and `--model MODEL`
overrides the configured inference route and retains it on resume.

Hermes alone supports experimental participant mode:

```sh
aop run player --agent hermes --mode participant --profile review --prompt "Submit one move"
```

AOP requests one turn without model tools, but Hermes 0.20 does not have a supported no-tools
option. Participant mode is therefore not a durable tool-security boundary. The selected AOP
profile remains the filesystem boundary.

### DeepSeek Harness

AOP invokes the official `dsh --profile headless` interface and adds the structured result and exact
resume support required by AOP. DeepSeek Harness still owns its agent loop, tools, prompt assembly,
inference route, and session persistence.

Each task receives a private `DSH_HOME`. AOP reads the native `settings.yaml`, copies only the
selected inference route and its matching managed credential, and leaves unrelated settings,
profiles, sessions, plugins, credentials, and user-level `.env` files behind. If a configured route
does not name a credential environment variable, its native ambient authentication discovery still
applies. AOP disables dsh telemetry for every run.

The default route is `deepseek-official` with `deepseek-v4-flash`; `deepseek-v4-pro` is also
supported. Another route must exist under `llm-pi-ai.providers` and be selected with both
`--provider` and `--model`. The dsh headless interface cannot list models by provider, so
`aop models --agent dsh` reports its bundled DeepSeek defaults.

Set `AOP_DSH_SOURCE_HOME` when the source state is not `${DSH_HOME:-~/.dsh}`.

## Current limitations

| Harness | Current limitation |
| --- | --- |
| Codex | No participant mode. |
| Claude Code | Model inventory is catalog-only; no participant mode. |
| Cursor Agent | Effort is part of the model ID; no participant mode. |
| Devin CLI | Effort is part of the model ID; successful runs require a valid ATIF export; no participant mode. |
| OpenCode | No participant mode. |
| Antigravity | Exact resume fails if the conversation ID changes; no participant mode. |
| Grok Build | No participant mode. |
| Hermes | Participant mode is experimental; rotating OAuth credentials serialize Hermes turns. |
| DeepSeek Harness | Developer-preview CLI; inventory lists bundled defaults rather than the selected provider; no participant mode. |

Antigravity integration requires Agy 1.1.16 or newer. AOP checks the installed version before
dispatch and model discovery so an older or unrecognized interface fails before task state is
created.

Cursor Agent and Devin CLI do not report enough information for AOP to calculate an API-equivalent
cost. See [Token usage and pricing](token-usage.md) for the values available from each harness.

## Executable overrides

Use these environment variables when a supported executable is not on the normal `PATH` or AOP
should invoke a specific installation:

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

In isolated profiles, overrides should normally point to an installed harness. AOP supports system
and Homebrew installations, self-contained executables, dsh installed through npm, and Hermes
installed for development from a local source directory. A custom launcher may not work if it needs
other files that are not visible inside the isolated run.
