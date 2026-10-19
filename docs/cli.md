# CLI guide

The AOP CLI creates isolated tasks, runs bounded harness turns, retains their results, and keeps
checkpointing and integration explicit. Each installed harness continues to own its authentication.

## Initialize a repository and run a task

From an existing Git repository:

```sh
aop init
aop run task-a --agent claude --model sonnet --effort high \
  --prompt "Implement the parser and its tests"
```

`aop init` adds `/.aop/` to `.gitignore`. Commit that change before integrating tasks so the main
worktree is clean.

`aop run` creates the task worktree automatically at `HEAD`. Use `--base REF` to start a new task at
another commit. The final message goes to stdout, while the AOP run ID, harness session ID, and
retained run directory go to stderr.

Resume the exact harness session with the AOP run ID:

```sh
aop resume <run-id> --prompt "Address the review findings"
```

Use `--prompt-file` for a prompt stored on disk, `--prompt-parts` for a Claude prompt split into cacheable blocks, and `--timeout` to set a wall-clock deadline. Model,
effort, inference-provider, and resume behavior vary by harness. See
[Harness adapters](harnesses.md).

AOP preserves OpenCode's native `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX` environment setting in sealed runs. OpenCode owns validation and the model-capacity bound. This can raise its default output ceiling for reasoning-heavy tasks without changing reasoning effort; record the selected value alongside experiment inputs. For example, prefix an invocation with `OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX=64000`.

## Execution profiles

Select the access boundary with `--profile`:

- `edit` gives the harness a writable isolated task worktree.
- `review` exposes the task worktree read-only.
- `sealed` exposes only declared input snapshots and does not require a Git repository.
- `host` uses the native host filesystem and environment without AOP isolation.

Inspect a profile or preview a run without starting the harness:

```sh
aop profile explain review
aop run task-a --agent claude --profile review --dry-run \
  --prompt "Review the parser"
```

See [Execution profiles](profiles.md) for the exact filesystem, instruction, credential,
environment, input, output, and network boundaries.

## Disable model-controlled web access

Use `--no-web` when a run must not retrieve external information:

```sh
aop run study-arm --agent claude --profile sealed --no-web \
  --input source.md --prompt "Answer using only the declared source"
```

This is an enforced capability policy, not a prompt instruction. AOP disables native search and
fetch tools, browser and extension routes, subagents that could restore those routes, and networked
shell execution. The harness keeps the inference network access needed to call its model provider.
If the selected adapter cannot prove the complete boundary, AOP rejects the run before creating a
worktree or provider state.

In the `edit`, `review`, and `sealed` profiles, AOP also enforces the boundary at the network level. The sandbox gets its own network namespace with only a loopback interface, and its one way out is an HTTP CONNECT proxy that AOP runs outside the sandbox and reaches through a Unix socket. The proxy opens connections only to the endpoints the harness needs for inference and for keeping its login valid, plus any URL named with `--allow-url` or `--provider-base-url`. Inside the sandbox, `HTTPS_PROXY` points at the proxy, and each allowed loopback URL, such as a local request recorder, is reachable at its own address. A harness that ignores the proxy settings cannot reach the network at all. Every connection attempt is logged with its decision in the run's `network.jsonl`, and the allowed endpoints are recorded in `request.json` under `effective_policy.network`:

```sh
aop run --agent claude --profile sealed --no-web \
  --provider-base-url http://127.0.0.1:18977 --prompt-file prompt.txt
aop run study-arm --agent codex --profile sealed --no-web \
  --allow-url https://api.example.com --prompt "..."
```

`--allow-url` requires `--no-web` and a sandboxed profile, accepts `https` URLs and loopback `http` URLs, and is recorded in `request.json` as `allowed_urls`; resumes reuse it. These are the endpoints AOP allows for each harness, and the evidence for each:

| Harness | Allowed endpoints | Evidence |
| --- | --- | --- |
| Claude | `api.anthropic.com:443` for inference, `platform.claude.com:443` for OAuth token refresh | A sealed `--no-web` run with the OAuth login answered; the refresh host is Claude Code's `TOKEN_URL` |
| Codex | `chatgpt.com:443`; AOP refreshes Codex logins outside the sandbox | A sealed `--no-web` run with the ChatGPT login answered |
| OpenCode | `opencode.ai:443` | A sealed `--no-web` run answered |
| Grok | `auth.x.ai:443`, `cli-chat-proxy.grok.com:443` | A sealed `--no-web` run reached both hosts and got the same server reply as an unrestricted run |
| DeepSeek Harness on its `deepseek-official` route | `api.deepseek.com:443` | A sealed `--no-web` run reached the API, which reported the account out of credit; the run's tool roster had no web or shell tools |
| Devin | `server.codeium.com:443` | A sealed `--no-web` run answered; `web_search` and shell calls were refused and a `webfetch` to an outside site was blocked by the proxy |
| Hermes with the `xai-oauth` provider | `auth.x.ai:443`, `api.x.ai:443` | A run reached `auth.x.ai` as an unrestricted run did; `api.x.ai` is the provider's configured base URL |

A harness or Hermes provider without an entry reaches only the URLs the run names. The network boundary cannot stop retrieval that a provider performs on its own servers through an allowed inference endpoint, such as a server-side search tool, and a shell holding the run's credentials could call such a tool directly. The harness-level restrictions above therefore still apply, and harnesses whose web search runs server-side also lose their shell tools:

- DeepSeek Harness: AOP's patch disables the `web`, `web-search-deepseek`, `web-deepseek-api`, `web-fetch-http`, `tool-web`, `tool-bash`, `tool-pwsh`, `tool-jobs`, and `tool-workflow` plugins, because its `web_search` is a server-side tool call through the DeepSeek API host. File, subagent, skill, todo, and goal tools remain.
- Devin: AOP adds the permission deny rules `exec` and `web_search` to the run's private Devin config, because `web_search` runs on Devin's API server. Its `webfetch` tool runs on the client, where the proxy refuses it. When a deny rule refuses a call, Devin ends the turn without a final message, so a run whose model insists on searching fails instead of continuing offline.

Both require a sandboxed profile, because their remaining tools rely on the network namespace. The `host` profile has no sandbox and keeps only the harness-level restrictions.

The effective policy is stored in `request.json` under `model_capabilities`. Some harnesses require a stronger restriction than the name suggests. OpenCode currently runs without any model-visible tools, while Codex, Claude, Grok, and Hermes retain local-only tools. Cursor and Antigravity currently reject `--no-web`: Antigravity's `search_web` and `read_url_content` run on Google's servers through its inference backend, and neither its `disabledTools` setting nor its permission rules removed them in a real run. Grok also rejects the `host` profile because host-visible global customizations cannot be excluded with its current automation interface.

Exact resumes inherit the original `--no-web` policy and private provider state. For batch
manifests, set `no_web = true` on a task.

Codex runs use app-server for new turns and exact resumes. When `--timeout` expires, AOP requests native turn interruption and allows up to five additional seconds only for terminal accounting and process teardown. That grace does not extend the requested work budget, and the result remains timed out.

## Sealed runtime storage

For long-running sealed workloads that would exhaust `/run/user` tmpfs, set `AOP_SEALED_RUNTIME_ROOT` to an absolute disk-backed directory. AOP creates it with mode 0700 and requires current-user ownership with no group or other permissions. Keep the same setting when resuming those runs. `aop cleanup RUN_ID` removes the finished run's disposable runtime while retaining its archived run records, and finds that runtime from the paths recorded in the run's request, whatever `AOP_SEALED_RUNTIME_ROOT` is set to at cleanup.

## Machine-readable results

Add `--json` to `run` or `resume` for the stable machine interface. Stdout then contains only the
normalized result object, including its run ID, harness session ID, status, timing, token usage,
final message, and artifacts. See [Token usage and pricing](token-usage.md) for the usage contract.

## Inputs and artifacts

Declare files or directories as input snapshots and require outputs as retained artifacts:

```sh
aop run analysis \
  --agent agy \
  --profile sealed \
  --input /data/source-material \
  --input /data/ledger.json \
  --artifact report.md \
  --prompt "Analyze the declared sources and write the report"
```

AOP copies accepted inputs into private controller-owned storage beneath `.aop/snapshots`, hashes
them, and mounts the snapshots read-only beneath `/inputs` for isolated profiles. The `host` profile
uses the same point-in-time copies but retains native host access. The harness does not receive the
inputs' original host paths.

AOP checks every input before it creates any run state, and `--dry-run` runs the same check, so a dry run that passes will not fail later on an input. It prints each input's file count, size, excluded paths and symlinks. Missing paths, special files, an input that is itself a symlink, and duplicate basenames fail the check. Inside an input directory:

- A symlink whose target lies inside the same input, such as `CLAUDE.md -> AGENTS.md`, is copied as its target's contents.
- A symlink that points outside the input, or nowhere, fails the run. `--skip-external-links` leaves such links out instead.
- `--input-exclude GLOB` leaves matching paths out of every input directory, and can be repeated. A pattern without `/`, such as `*.pt`, matches names at any depth; one with `/`, such as `runs/*/net-*.pt`, matches paths from the input's root.
- AOP's own state directories, such as `.aop` when the input is the directory AOP runs from, are always left out, so a snapshot never copies itself.

Each snapshot records its followed and skipped links and its excluded paths in `input-manifest.json`, and the run's `request.json` keeps `input_excludes` and `skip_external_links` so resumes apply them again.

While a harness runs, its output grows in `events.jsonl` and `stderr.log` in the run directory, so `tail -f .aop/runs/RUN_ID/events.jsonl` shows its tool calls and messages as they happen. Codex and ZCode, which AOP drives through their own protocols, write these files when the run ends.

An exact `sealed` resume inherits the parent run's snapshots unless new `--input` arguments replace
them. Other profiles snapshot the recorded source paths again, so they receive the current files.

Each `--artifact` is relative to the fresh output directory exposed as `AOP_OUTPUT_DIR`. AOP accepts
nonempty regular files and directories, archives them beneath the retained run directory, and
records their paths, sizes, and SHA-256 values. Artifact declarations apply to one invocation, so a
resume must declare its own required artifacts.

## Models

Inspect models exposed by installed harnesses and their current comparison prices:

```sh
aop models
aop models --agent codex --agent opencode
aop models --agent codex --provider zai-coding-plan --json
aop models --agent hermes --json
aop models --refresh
```

`models --provider` currently selects a Codex inference route and therefore requires exactly `--agent codex`. Z.AI Coding Plan inventory is fetched from its authenticated route on every listing and includes the canonical inference provider, retrieval time, normalized inventory hash, and authenticated status. Listed prices are model-prices API-equivalent rates in effect now, and the output records their `pricing_basis`. `--refresh` also downloads the newest published price data; a failed download or stale data appears under `warnings.prices`.

Availability can come from an authenticated harness inventory, an installed harness default, or a
catalog fallback. The output identifies the source instead of implying every model was verified
against the current account.

## Batch runs

Run independent tasks concurrently from a TOML manifest:

```toml
[[tasks]]
id = "parser"
agent = "claude"
prompt_file = "tasks/parser.md"
model = "sonnet"
effort = "high"
timeout = 1800
artifacts = ["parser-report.md"]
profile = "review"
no_web = true
inputs = ["fixtures"]

[[tasks]]
id = "tests"
agent = "hermes"
prompt = "Add adversarial parser tests"
model = "deepseek/deepseek-v4-flash-0731"
provider = "nous"
effort = "high"
```

```sh
aop batch tasks.toml --jobs 4
```

Prompt files and inputs are resolved relative to the manifest. Each task can select its harness,
base, model, inference provider, effort, mode, profile, timeout, artifacts, and inputs. The scheduler
runs at most `--jobs` tasks concurrently and retains successful results when a sibling fails.

Each batch writes `.aop/batches/<batch-id>.json` with results in task order. The command exits
nonzero unless every task is a clean success. Terminal provider errors are recorded separately from
execution failures. If the provider also returned a response, the task status is
`response_available_with_provider_error` and the batch summary counts it separately from runs with
no response.

## Checkpoint and integrate

Commit a completed task and integrate its commits into the current main branch:

```sh
aop checkpoint task-a -m "Implement parser"
aop integrate task-a --remove-worktree
```

Integration validates and rebases the task before fast-forwarding main. If conflicts occur, AOP
resumes the original authoring session to resolve and validate them. See [Integration](integration.md)
for the complete safety contract.

## Cleanup and lower-level commands

AOP reads only current run records. Convert records written by older AOP versions once with `aop migrate-results`, which keeps each original beside it; see [migrating older run records](token-usage.md#migrating-older-run-records).

When you no longer need to resume or integrate a task, clean it up using any of its run IDs:

```sh
aop cleanup <run-id>
```

Cleanup removes the task worktree, scratch data, overlays, and private harness state. It retains the
request, input snapshots, normalized result, logs, final message, and archived artifacts. Repeating
cleanup is safe, but an active task cannot be cleaned while it holds its execution lock. Cleanup prints each runtime directory it removes to stderr.

AOP never removes runtime state on its own, because a task stays resumable until it is cleaned up. Each sealed run keeps a private harness home, scratch, and cache directory until then, so a caller that starts many sealed runs should clean up each task once it no longer needs to resume it. Run AOP, including cleanup, from the caller's own project directory: run records live under that directory's `.aop/runs`, and sealed runtime directories are named after it.

Inspect or explicitly manage task worktrees when needed:

```sh
aop worktree list
aop worktree path task-a
aop worktree create task-b --base REF
aop worktree remove task-b
```

Explicit creation is only needed when a worktree must exist before its first `aop run`. Removing a
dirty worktree requires `--force`.

Run another command inside a task worktree with `aop exec`. Repeat `--overlay PATH` to give large
ignored build or data directories private copy-on-write views:

```sh
aop exec task-a -- <command>
aop exec task-a --overlay target --overlay cache -- <command>
```

Overlays require `fuse-overlayfs`, persist across `exec` calls, and are deleted with the task.

## Retained results

AOP keeps controller-owned state beneath the ignored `.aop/` directory. Each
`.aop/runs/<run-id>/` contains the request, normalized result, harness events and logs, final
message, input evidence, and accepted artifacts available for that invocation. Batch, checkpoint,
and integration records live in their corresponding `.aop/` directories.

AOP creates this state with user-only permissions. Treat the directory as sensitive because
provider configuration, prompts, logs, and artifacts may contain credentials or private data.
