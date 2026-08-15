# CLI guide

The AOP CLI creates isolated tasks, runs bounded provider turns, retains their evidence, and keeps
checkpointing and integration explicit. Provider authentication remains owned by each installed
agent CLI.

## Initialize a repository

```sh
cd /path/to/project
aop init
aop worktree create task-a
```

`aop init` adds `/.aop/` to `.gitignore`. Commit that change before integrating tasks so the main
worktree is clean. Task worktrees are detached at their selected base commit, so one worker cannot
move another worker's branch.

List and inspect task worktrees with:

```sh
aop worktree list
aop worktree path task-a
```

## Run and resume

```sh
aop run task-a --agent claude --model sonnet --effort high \
  --prompt "Implement the parser and its tests"
```

`aop run` prints the final message to stdout. The AOP run ID, provider session ID, and run artifact
directory go to stderr. Resume the exact provider session with the AOP run ID:

```sh
aop resume <run-id> --prompt "Address the review findings"
```

Add `--json` to `run` or `resume` for a stable machine interface. Stdout then contains only the
normalized result object, including its run ID, provider session ID, terminal status, metrics, final
message, and artifacts. See [Token usage and pricing](token-usage.md) for the result's usage
contract.

Use `--prompt-file` for a prompt stored on disk and `--timeout` to set the process deadline. Model,
effort, provider, and resume behavior vary by adapter, as documented in
[Provider adapters](harnesses.md).

## Inputs and artifacts

Declare files or directories as immutable input snapshots:

```sh
aop run analysis \
  --agent agy \
  --profile sealed \
  --input /data/source-material \
  --input /data/ledger.json \
  --artifact report.md \
  --prompt "Analyze the declared sources and write the report"
```

AOP rejects missing paths, symlinks, special files, and duplicate basenames before launch. It
copies accepted inputs into retained controller-owned storage, hashes them, marks them read-only,
and mounts only those snapshots beneath `/inputs`. The provider does not receive their original
host paths. `input-manifest.json` records their sizes and SHA-256 values.

For `sealed`, an exact resume copies the parent run's retained snapshots unless new `--input`
arguments replace them. Changed or deleted source files therefore cannot alter inherited sealed
input. Other profiles snapshot the recorded source paths again so they receive current project
inputs.

Each `--artifact` is a path relative to the fresh output directory named by `AOP_OUTPUT_DIR`.
Artifacts may be nonempty regular files or directories. AOP rejects missing declarations,
overlapping paths, symlinks, special files, empty files, and path escapes. Accepted files are copied
to `.aop/runs/<run-id>/artifacts/`, and the result records each logical path, archived path, size,
and SHA-256.

Artifact declarations apply to one invocation. A resume gets a fresh output directory and validates
only the artifacts declared on that `aop resume` command. See [Execution profiles](profiles.md) for
the surrounding filesystem and instruction boundary.

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

Prompt files and inputs are resolved relative to the manifest. Each task can select its agent,
base, model, provider, effort, profile, timeout, artifacts, and inputs. The scheduler keeps at most
`--jobs` tasks active. On interruption it launches no additional tasks and waits for active tasks to
finish.

Each batch writes `.aop/batches/<batch-id>.json` with task-order-preserving run IDs, session IDs,
durations, exit codes, and errors. A batch exits nonzero if any task fails without discarding
successful sibling results.

## Models

Inspect the models exposed by installed agent CLIs and their current comparison prices:

```sh
aop models
aop models --agent codex --agent opencode
aop models --agent hermes --json
aop models --refresh
```

Availability can come from an authenticated provider inventory, an installed harness default, or a
catalog fallback. The output records that distinction instead of implying every model was verified
against the current account.

## Cleanup and lower-level commands

When a run is no longer resumable or integrable, clean it up by run ID:

```sh
aop cleanup <run-id>
```

Cleanup removes the disposable worktree, scratch directory, overlays, and private runtime state. It
retains the request, input snapshots, result, logs, final message, and archived artifacts. Repeating
cleanup is safe, but an active task cannot be cleaned while it holds its execution lock.

Run another command inside a task worktree with `aop exec`. Private copy-on-write overlays can reuse
large ignored build or data directories without an up-front copy:

```sh
aop exec task-a -- <agent-command>
aop exec task-a --overlay target --overlay cache -- <evaluator-command>
aop worktree remove task-a
```

`aop exec` supplies `AOP_ROOT`, `AOP_TASK`, `AOP_WORKTREE`, and `AOP_CACHE_DIR` to the child.
Overlays require `fuse-overlayfs`; their upper layers persist across exec calls and are deleted with
the task. Dirty worktrees require an explicit `--force` to remove.

Use [Integration](integration.md) to turn completed task changes into reviewed commits on the main
branch.

## Retained state

Runtime state lives under the ignored `.aop/` directory:

```text
.aop/
├── batches/            structured batch summaries
├── cache/              shared cache root for non-sealed tasks
├── sealed-cache/       session-private caches for sealed runs and exact resumes
├── checkpoints/        task checkpoint records
├── integrations/       successful integration audit records
├── locks/              per-task execution and checkpoint locks
├── overlays/<task>/    private copy-on-write upper layers for aop exec
├── provider-state/     task-local mutable provider profiles
├── runs/<run-id>/      request, result, logs, final message, and accepted artifacts
├── shared-provider-state/ repository-local credential state shared across tasks
├── tasks/               recorded task bases and worktree paths
├── worktrees/          one isolated checkout per task
├── integration.lock    single-writer main-branch integration lock
└── worktrees.lock      lifecycle-operation lock
```

AOP creates state and run directories with user-only permissions because provider profiles,
prompts, logs, and artifacts may contain sensitive information.
