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

Use `--prompt-file` for a prompt stored on disk and `--timeout` to set a wall-clock deadline. Model,
effort, inference-provider, and resume behavior vary by harness. See
[Harness adapters](harnesses.md).

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

## Machine-readable results

Add `--json` to `run` or `resume` for the stable machine interface. Stdout then contains only the
normalized result object, including its run ID, harness session ID, status, timing, token usage,
final message, and artifacts. See [Token usage and pricing](token-usage.md) for the usage contract.

## Inputs and artifacts

Declare files or directories as immutable input snapshots and require outputs as retained artifacts:

```sh
aop run analysis \
  --agent agy \
  --profile sealed \
  --input /data/source-material \
  --input /data/ledger.json \
  --artifact report.md \
  --prompt "Analyze the declared sources and write the report"
```

AOP copies accepted inputs into controller-owned storage, hashes them, and mounts the snapshots
read-only beneath `/inputs`. The harness does not receive their original host paths. Missing paths,
symlinks, special files, and duplicate basenames are rejected before launch.

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
aop models --agent hermes --json
aop models --refresh
```

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
nonzero if any task fails.

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

When you no longer need to resume or integrate a task, clean it up using any of its run IDs:

```sh
aop cleanup <run-id>
```

Cleanup removes the task worktree, scratch data, overlays, and private harness state. It retains the
request, input snapshots, normalized result, logs, final message, and archived artifacts. Repeating
cleanup is safe, but an active task cannot be cleaned while it holds its execution lock.

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
