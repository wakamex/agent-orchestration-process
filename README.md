# Agent Orchestration Process (AOP)

AOP is a reusable operating model for running autonomous CLI agents as bounded workers while a
separate judgment layer decides what is worth trying, what the evidence means, and when the search
should change direction.

It is intended for iterative work where a candidate can be proposed, checked, evaluated, and
reverted in one transaction. Examples include code optimization, model or prompt search, data
pipeline tuning, test generation, and other empirical engineering tasks.

AOP is a methodology and a set of interfaces, not a universal orchestration library. The control
frame is portable; the evaluator, candidate format, mutation mechanism, and domain guards belong to
each project.

## Reference implementation

This repository includes a dependency-free Python CLI for running Codex, Claude Code, Cursor Agent,
OpenCode, Antigravity (`agy`), and Hermes concurrently in isolated Git worktrees. Each adapter
records a normalized result and resumes the exact provider session associated with an earlier run.

Install the CLI from this checkout:

```sh
uv tool install /code/aop
```

For development, let uv create and synchronize the local environment. The default `dev` dependency
group contains pytest and Ruff:

```sh
uv sync
uv run pytest
uv run ruff check src tests
```

Prepare a Git repository and create two isolated tasks:

```sh
cd /path/to/project
aop init
aop worktree create task-a
aop worktree create task-b
```

`aop init` adds `/.aop/` to `.gitignore`. Commit that project-level change before integrating
tasks so the main worktree is clean.

Run an agent in a new or existing task worktree:

```sh
aop run task-a --prompt-file task.md --timeout 1800
aop run task-b --agent claude --model sonnet --effort high \
  --prompt "Implement the parser and its tests"
aop run task-cursor --agent cursor \
  --prompt "Refactor the parser without changing behavior"
aop run task-opencode --agent opencode --effort high \
  --prompt "Fix the parser and run its tests"
aop run task-c --agent agy --model gemini-3.5-flash --effort low \
  --prompt "Add adversarial parser tests"
aop run task-d --agent hermes --model deepseek/deepseek-v4-flash-0731 --effort high \
  --prompt "Investigate and fix the failing benchmark"
```

`aop run` prints the agent's final message to stdout and its AOP run ID, provider session ID, and
run artifact directory to stderr. Resume the exact session using the AOP run ID:

```sh
aop resume <run-id> --prompt "Address the review findings"
```

Callers that need a stable machine interface can add `--json` to `run` or `resume`. AOP then prints
only the normalized result object to stdout, including its run ID, provider session ID, terminal
status, metrics, final message, and artifacts.

When a run is no longer resumable or integrable, discard its task worktree by run ID:

```sh
aop cleanup <run-id>
```

Cleanup force-removes that task's disposable worktree, scratch directory, and overlays, but retains
the immutable request, result, logs, and archived artifacts under `.aop/runs/`. Repeating cleanup is
safe. An active task cannot be cleaned while it holds its execution lock.

All providers run inside `bwrap` by default, with their own permission prompts bypassed because the
OS mount boundary is the enforcement layer. `workspace-write` mounts the main repository read-only,
rebinds only the isolated task worktree and shared `AOP_CACHE_DIR` writable, and keeps the
worktree's `.git` pointer read-only. `scratch-write` leaves the task read-only and rebinds only its
`scratch/` directory plus the shared cache writable; use it for agents that need working space but
must not edit the repository. `danger-full-access` explicitly skips `bwrap`. Configured
authentication and user instructions are preserved. `AOP_CODEX_BIN`, `AOP_CLAUDE_BIN`,
`AOP_CURSOR_BIN`, `AOP_OPENCODE_BIN`, `AOP_AGY_BIN`, `AOP_HERMES_BIN`, and `AOP_BWRAP_BIN` may
override their respective executables.

Every Agy task uses a persistent private Gemini profile under
`.aop/provider-state/<task>/agy/gemini`, including with `danger-full-access`. AOP initializes it
once from the authenticated `~/.gemini` profile, copying configuration and credentials but not
conversations, history, caches, logs, scratch files, or databases. Agy writes all new runtime state
to the private profile. `aop resume` requires the terminal Agy result to report exactly the requested
conversation ID; a missing or different ID fails closed and is not resumable. Removing the task
worktree removes its private profile while preserving run records. Set `AOP_AGY_SOURCE_DIR` only
when the authenticated source profile is somewhere other than `~/.gemini`.

For Cursor Agent in either non-danger sandbox, AOP seeds a persistent task-local Cursor profile and
authentication directory under `.aop/provider-state/<task>/cursor`. Project metadata, chats,
tracking data, and Cursor-created worktrees start empty and remain private to the AOP task; existing
CLI configuration, skills, plugins, policies, and authentication are copied without duplicating IDE
extensions. Cursor caches use the shared AOP cache. This lets the first turn and exact resume work
when the surrounding home directory is read-only, without modifying global Cursor state. Cleanup
removes the private profile with the task. `danger-full-access` retains Cursor's native global state.

For OpenCode in either non-danger sandbox, AOP seeds small user configuration and authentication
files into a persistent task-local XDG profile under `.aop/provider-state/<task>/opencode`.
OpenCode's session database, logs, model state, token refreshes, generated config metadata, and
downloaded tools stay outside the global profile. Existing generated plugin dependencies are
mounted read-only instead of copied, while downloads use `.aop/cache/opencode`, shared by all tasks.
This permits a first turn and exact resume when the surrounding home directory is read-only without
duplicating the plugin tree per task or changing global OpenCode state. Cleanup removes the private
profile with the task. `danger-full-access` retains OpenCode's native global state. Set
`AOP_OPENCODE_CONFIG_DIR` or `AOP_OPENCODE_DATA_DIR` only when the authenticated source profile is
outside the standard XDG locations.

For Hermes in either non-danger sandbox, AOP seeds a persistent task-local Hermes home under
`.aop/provider-state/<task>/` from the authenticated profile. Hermes can read the existing
configuration, credentials, skills, hooks, and memories even when their host filesystem is
read-only, while new session databases, logs, token refreshes, and caches stay isolated from the
global Hermes home. The same state is reused by `aop resume` and removed with the task worktree;
it requires neither root access nor filesystem overlay support, and no preparation beyond normal
Hermes authentication. `danger-full-access` keeps Hermes's native filesystem behavior.

For a file-producing task, declare each expected path relative to the run's output directory:

```sh
aop run extraction \
  --agent agy \
  --model gemini-3.5-flash \
  --sandbox scratch-write \
  --artifact paper.md \
  --artifact assets \
  --prompt "Extract the source document as Markdown"
```

AOP gives the agent a fresh, prompt-visible `AOP_OUTPUT_DIR` for every invocation. Stdout and stderr
remain logs; deliverables belong in that output directory. After a successful provider exit, AOP
accepts each declared artifact as either a file or directory. Files must be nonempty and regular;
directories are collected recursively in deterministic path order and may be empty. AOP rejects
missing declarations, overlapping declarations, symlinks, special files, empty files, and path
escapes, then copies accepted files into `.aop/runs/<run-id>/artifacts/` with their hierarchy intact.
The normalized result records each collected file's logical path, archived path, byte size, and
SHA-256. A failed validation makes the run unsuccessful, and AOP never copies artifacts into the
main worktree.

Artifact declarations are per invocation. A resume keeps the exact provider session and reusable
task scratch, but receives a fresh output directory and only validates artifacts declared on that
`aop resume` command.

Claude accepts its normal model aliases and effort levels `low`, `medium`, `high`, `xhigh`, and
`max`. Agy accepts its native model names and effort levels `low`, `medium`, and `high`.

Hermes uses the provider selected in the Hermes profile and AOP does not override it. Install
Hermes 0.20 or newer and authenticate the desired provider once. For example, on a headless host:

```sh
hermes auth add nous --type oauth --no-browser
hermes auth add xai-oauth --type oauth --no-browser
```

Hermes defaults to `deepseek/deepseek-v4-flash-0731`, which expects a compatible configured
provider such as Nous. Pass another model ID unchanged with `--model`; AOP passes `--effort` through
as Hermes's native reasoning level. Supported levels are
`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, and `ultra`. AOP uses Hermes's quiet
programmatic mode, tags sessions as tool-created, and resumes the exact session without restoring
its old working directory. It reasserts the original model and reasoning level on resume
because Hermes otherwise falls back to its current configured defaults.

Hermes can run a bounded conversational turn with the experimental participant mode:

```sh
aop run player \
  --agent hermes \
  --mode participant \
  --sandbox scratch-write \
  --prompt "Submit one move"
```

Participant mode persists across `aop resume` and can be selected in a batch task with
`mode = "participant"`. AOP records the mode in the request, normalized result, and batch summary.
It is currently supported only by the Hermes adapter.

Hermes 0.20 does not have a supported no-tools option, so AOP temporarily combines `--safe-mode`,
`--max-turns 1`, and an intentionally nonexistent `--toolsets __aop_no_tools__` allowlist. In the
current Hermes implementation, an explicit unknown toolset resolves to zero model tool schemas.
Safe mode also omits user configuration, repository rules, plugins, hooks, skills, and MCP servers.
AOP reapplies the same flags on every resume and does not pass coding-agent permission flags such as
`--yolo` or `--accept-hooks`. It also removes inherited internal no-tools and Kanban-worker mode
variables so ambient Hermes process state cannot change the declared invocation mode or add worker
tools.

This unknown-toolset behavior is undocumented and may emit a warning outside quiet mode. It is not
a durable security contract and could change in a future Hermes release, so the `bwrap` sandbox
remains the filesystem boundary. AOP should replace the workaround with the proposed official
`--no-tools` flag once
[NousResearch/hermes-agent#78262](https://github.com/NousResearch/hermes-agent/pull/78262) is merged
and available in the installed Hermes release.

The agy default is `gemini-3.5-flash` at `medium` effort.

For example, `--model gemini-3.5-flash --effort low` is passed directly to agy. An exact model
printed by `agy models` can instead be passed without `--effort`.

Cursor Agent uses `composer-2.5` by default. Pass any model ID printed by `agent models` through
`--model`. Cursor encodes reasoning effort and fast variants in its model IDs, so its adapter rejects
a separate `--effort`. AOP disables Cursor's native sandbox and permission prompts inside the outer
`bwrap` boundary, records its structured token and timing metrics, and resumes the exact chat ID.

OpenCode defaults to the paid OpenCode Zen model `opencode/deepseek-v4-flash`. A short model name
such as `deepseek-v4-flash` is automatically qualified with `opencode/`; pass a full
`provider/model` ID to use another configured provider. AOP maps `--effort` to OpenCode's native
`--variant`, reasserts both model and variant on exact-session resume, and uses OpenCode's automatic
permission mode inside the outer `bwrap` boundary. Authenticate OpenCode normally before the first
AOP run, for example with `opencode providers login`.

Inspect the models exposed by the installed agent CLIs and their current comparison prices:

```sh
aop models
aop models --agent codex --agent opencode
aop models --agent hermes --json
aop models --refresh
```

Codex, Cursor, Agy, and OpenCode results are queried from their non-interactive model interfaces.
Hermes follows its configured provider: Nous results come from the live Nous inference endpoint,
while providers without a live listing use their matching catalog entries. Claude has no
non-interactive model listing, so its rows come from the Anthropic catalog and are marked `catalog`
rather than account-verified. The `availability` and `price_scope` fields keep those distinctions explicit.
`api-equivalent` prices are standard API comparison rates and do not describe subscription billing;
`provider` prices are rates reported by that provider endpoint.

Every run records wall-clock time and time to first event and agent response. Adapters also record
input, cached-input, output, and reasoning-output tokens when the provider exposes them. When
`--model` names a priced OpenAI model, the result also contains an estimated standard API-equivalent
USD cost.
This is a comparison metric for subscription runs, not an amount billed to the account. Reasoning
tokens are reported separately but are already included in output tokens and are not charged twice.

Before dispatching any AOP command, the CLI verifies that its global models.dev catalog cache is
less than 24 hours old. A stale or missing cache is refreshed under a process lock and replaced
atomically. If refresh fails, AOP fails closed instead of reporting or using expired prices. Set
`AOP_MODEL_CATALOG_CACHE` to relocate the cache; its default is
`${XDG_CACHE_HOME:-~/.cache}/aop/models-dev.json`. `aop models --refresh` refreshes immediately.
Every estimated cost records the catalog URL, retrieval time, and content hash-derived version.
models.dev is a community-maintained machine-readable registry, not an official provider price
guarantee, and the provenance is retained so results remain auditable.

The refreshed price data includes direct long-context tiers. AOP displays cache-write rates when
the source provides them, but Codex currently reports cached reads without identifying cache writes,
so run-cost estimates do not add cache-write charges. If the model is implicit or unknown, token and
timing metrics remain available while cost is reported as `n/a`. Claude's result stream supplies its
resolved model, token usage, and CLI-reported USD API-equivalent cost. Hermes's session accounting
supplies its resolved model, cache and reasoning token buckets, and provider-aware estimated or
actual cost; AOP records the delta for each invocation so resumed turns are not double-counted. If
Hermes reports `cost_source: none`, AOP uses the fresh catalog for the session's billing provider
and records an API-equivalent estimate instead of treating Hermes's zero-initialized counter as a
real zero-dollar cost. A zero-token failed turn remains unpriced. Agy
currently supplies timing and token usage from its structured result stream. Cursor, OpenCode, and
Agy each report provider duration separately from AOP's wall-clock duration. OpenCode also reports
per-step billed USD cost; AOP sums it for the invocation and records it as an actual CLI-reported
cost instead of applying AOP's API price table. Cursor and Agy do not report API-equivalent cost, so
that field remains `n/a`.

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

[[tasks]]
id = "tests"
agent = "hermes"
prompt = "Add adversarial parser tests"
model = "deepseek/deepseek-v4-flash-0731"
effort = "high"
```

```sh
aop batch tasks.toml --jobs 4
```

Prompt-file paths are resolved relative to the manifest. Each task may set `agent`, `base`, `model`,
`effort`, `sandbox`, `timeout`, and an `artifacts` array; unspecified values use the same defaults as
`aop run`. The scheduler keeps at most `--jobs` tasks active, prints only concise lifecycle status,
and stores full agent output in the normal per-run directories. On interruption it launches no
additional tasks and waits for already-active tasks to finish.

Every batch writes `.aop/batches/<batch-id>.json` with task-order-preserving run IDs, session IDs,
durations, exit codes, and errors. A batch exits nonzero if any task fails, without discarding
successful sibling results. Its terminal summary compares task, agent, model, effort, wall time, total
tokens, and estimated API-equivalent cost.

Checkpoint a completed task, then integrate its commits onto the branch currently checked out in
the main worktree:

```sh
aop checkpoint task-a -m "Implement parser"
aop integrate task-a
```

`checkpoint` commits all tracked, staged, and untracked changes in the task worktree. It refuses
unresolved conflicts, whitespace errors, empty changes, missing Git author identity, and any task
that still has an active AOP agent run. The checkpoint record includes its base and parent commits,
the resulting commit, and successful AOP run IDs associated with the task.

`integrate` rebases exactly the task commits onto current main. AOP owns the privileged, mechanical
Git operations: starting and continuing the rebase, recording any final validation edits, and
fast-forwarding main. When a commit conflicts, AOP resumes the task's latest authoring session in its
original sandbox. The author resolves file content and runs relevant tests inside its isolated
worktree; AOP then stages the resolution and continues. This repeats for every conflicting commit.
After the rebase, the author gets one final sandboxed validation turn before AOP fast-forwards main.

AOP serializes the operation with task and integration locks and verifies the recorded base,
linear history, clean starting state, unchanged main branch, and fast-forward ancestry. Author
continuations retain the sandbox selected by the original run—normally `workspace-write`—and are
explicitly denied responsibility for Git metadata or main. Use `--timeout` to override the
authoring run's timeout. A successful integration updates the task's recorded base and writes an
audit record linking original commits, rebased commits, conflict-resolution runs, and the final
validation run. Keep the task worktree by default for further work, or remove it only after success
with:

```sh
aop integrate task-a --remove-worktree
```

AOP never stashes changes, force-updates a branch, or decides conflict content. Conflict judgment
belongs to the sandboxed authoring agent, and a task is never deleted after a failed integration.

Run any other command in a task worktree with the lower-level escape hatch. Large ignored build or
data directories can be exposed as private copy-on-write overlays: reads reuse the main worktree's
files, while writes remain task-local without an up-front copy.

```sh
aop exec task-a -- <agent-command>
aop exec task-a --overlay target --overlay cache -- <evaluator-command>
aop worktree list
aop worktree path task-a
aop worktree remove task-a
```

Task worktrees are detached at the selected base commit, so one worker cannot move another worker's
branch. `aop exec` supplies `AOP_ROOT`, `AOP_TASK`, `AOP_WORKTREE`, and the shared `AOP_CACHE_DIR` to
the child process. Overlays require `fuse-overlayfs`; their private upper layers persist across exec
calls and are deleted with the task. A command may resume the task's agent while it runs; AOP treats
that nested resume as part of the already locked exec transaction. Dirty worktrees cannot be removed
unless `--force` is explicit.

Runtime state lives under the ignored `.aop/` directory:

```text
.aop/
├── batches/            structured batch summaries
├── cache/              shared cache root for future build and runner adapters
├── checkpoints/        task checkpoint records
├── integrations/       successful integration audit records
├── locks/              per-task execution/checkpoint locks
├── overlays/<task>/    private copy-on-write upper layers for aop exec
├── provider-state/     task-local mutable provider profiles
├── runs/<run-id>/      request, result, logs, final message, and accepted artifacts
├── tasks/               recorded task bases and worktree paths
├── worktrees/          one isolated checkout per task
├── integration.lock    single-writer main-branch integration lock
└── worktrees.lock      lifecycle-operation lock
```

The current CLI does not configure language-specific build caches. That interface will be added
when a real project needs it.

### Roadmap

- Replace Hermes participant mode's unknown-toolset workaround with the official `--no-tools` flag
  proposed in [NousResearch/hermes-agent#78262](https://github.com/NousResearch/hermes-agent/pull/78262)
  after it is merged and released. Until then, participant mode is experimental rather than a hard
  no-tools guarantee.
- Extend provider-neutral participant mode only when each adapter can enforce the full contract.
  Claude has suitable safe-mode and empty-tool controls but is not wired yet. Codex has no exposed
  option to disable all tools and replace its coding-agent context. Agy has no exposed option to
  disable tools and project context; plan mode is not tool-free. Unsupported adapters fail before
  creating a task worktree.

## 1. Core contract

An AOP project follows five rules:

1. **One cycle is one transaction.** Propose, parse, guard, apply, evaluate, decide, restore, and
   record as a single recoverable operation.
2. **The evaluator is the referee.** Define and lock it before autonomous search begins. Workers may
   propose evaluator changes but may not apply them.
3. **Workers generate evidence; the judgment layer interprets it.** Agents do not decide the
   objective, quietly relax constraints, declare an era complete, or promote their own result.
4. **State lives in inspectable artifacts.** The diary, search ledger, evaluation governance log,
   result records, and version-control history are the shared memory.
5. **Every mutation is reversible.** A checked-in baseline or equivalent restoration mechanism is
   the source of truth, and cleanup also runs after failures and interrupts.

If a project cannot provide a deterministic candidate boundary, an informative evaluation, or safe
restoration, it is not ready for an autonomous loop. Use supervised, deterministic runs until those
conditions exist.

## 2. Architecture

AOP separates execution into four layers.

### 2.1 Runner

The runner exposes one interface for every supported agent and hides model-specific CLI details.
It is responsible for:

- model aliases and invocation;
- sandbox and permission policy;
- timeouts and process cleanup;
- session creation, capture, and resumption;
- normalized stdout, stderr, and exit status.

AOP supplies this environment contract to child processes:

```text
AOP_ROOT        main Git worktree
AOP_TASK        stable task identifier
AOP_WORKTREE    isolated task worktree
AOP_CACHE_DIR   shared cache root
AOP_PROVIDER_STATE_DIR  task-local provider runtime state
AOP_RUN_ID      current structured run identifier (model runs only)
```

By default AOP stores its state under the primary Git worktree. Set `AOP_STATE_ROOT` to the absolute
path of another registered worktree before invoking AOP when that linked worktree, rather than the
primary checkout, must own `.aop/`, candidate worktrees, run records, and provider state. AOP rejects
a configured path that Git does not report as a worktree.

Set `AOP_HIDE_PATHS` to an `os.pathsep`-separated list of existing absolute directories that must be
masked with empty temporary filesystems inside sandboxed provider processes. This is intended for
same-user control sockets and similarly scoped host interfaces that a worker must not reach. AOP
rejects symlinks, missing paths, and required runtime directories. It has no effect in explicit
`danger-full-access` mode.

Model session identifiers, parent runs, timeouts, and terminal status live in each run's JSON
artifacts instead of mutable environment variables.

The other layers should not need to know which model or CLI is running.

### 2.2 Loop

The loop is deliberately mechanical. It owns:

- the cycle counter;
- model rotation;
- explore/exploit cadence;
- the interval between cycles;
- invocation of one step per cycle;
- append-only operational logging;
- per-cycle version-control checkpoints where appropriate;
- shutdown traps that terminate the whole child process tree.

The loop contains no evaluation or search intelligence. Keep it small enough to audit at a glance.
Treat a running loop as immutable; stop and restart it after editing its script.

Suggested controls:

```text
AOP_MODELS          ordered model aliases
AOP_EXPLORE_EVERY   mark every Nth cycle as exploratory; 0 disables
AOP_INTERVAL        seconds between cycles
AOP_MAX_CYCLES      optional cycle limit
```

### 2.3 Step

The step performs exactly one candidate transaction:

```text
build prompt
  -> obtain proposal
  -> parse
  -> run static guards
  -> apply candidate
  -> run dynamic guards
  -> evaluate
  -> classify result
  -> restore baseline
  -> record evidence
  -> checkpoint
```

The step re-reads its configuration and knowledge artifacts from disk on every invocation. This
makes prompts, ledgers, and step logic hot-editable between cycles.

### 2.4 Judgment layer

The judgment layer is a person, or a strong agent explicitly acting on a person's behalf. It owns:

- the objective and evaluation policy;
- search-ledger curation;
- review of promising results;
- deterministic experiments and attribution runs;
- search-era opening and closure;
- evaluator and infrastructure changes;
- adversarial review of untested assumptions;
- allocation of models, time, and quota.

This is a required role, not an optional dashboard. Autonomous workers are good at producing
candidates; they are not reliable referees of the search process that rewards them.

## 3. Project-owned interfaces

Each adoption must define four interfaces before implementing the loop.

### 3.1 Candidate

Specify the smallest mutation the worker may propose. Define:

- the exact output grammar;
- the identifier and deduplication key;
- editable files or regions;
- allowed dependencies and operations;
- the single swept parameter, if search includes a sweep;
- the maximum search grid;
- how the candidate is applied and removed.

Prefer a narrow machine-readable contract. For example:

```text
NAME: <stable identifier>
PARAMETER: <name> = <comma-separated values>
BEGIN_CANDIDATE
<candidate body>
END_CANDIDATE
```

Parse with anchored rules. Save malformed output verbatim for diagnosis and skip it. Do not guess
what an invalid proposal meant.

### 3.2 Evaluator

The evaluator converts an applied candidate into a verdict record. Define:

- an objective in absolute, hard-to-game units;
- a development partition used for selection;
- one or more held-out partitions used only for confirmation;
- a noise floor below which changes are treated as indistinguishable;
- hard disqualifiers such as safety, correctness, latency, or resource breaches;
- promotion criteria and tie-breaking rules;
- required diagnostics for a promoted result;
- reproducibility inputs: seeds, data version, toolchain, and environment.

Selection occurs only on the development partition. Evaluate the selected variant once against the
held-out policy. If repeated access to a holdout can influence future proposals, it is no longer a
holdout; rotate or redesign it.

Lock evaluator code and configuration by permissions, hashes, ownership boundaries, or a
combination. Record proposed changes separately and review them outside active result promotion.

### 3.3 Guards

Guards are ordered from cheapest to most expensive and fail closed. A typical chain is:

1. duplicate identifier or candidate hash;
2. forbidden token, dependency, file, or operation;
3. syntax and schema validation;
4. build, typecheck, or unit tests;
5. domain invariants;
6. smoke evaluation;
7. behavioral/result deduplication;
8. full evaluation.

Every rejection writes a reason to the diary. A crash or interrupt still restores the baseline.
Seed behavioral deduplication with all incumbents so rediscovery is recognized immediately.

### 3.4 Verdict

Use a small, explicit verdict vocabulary. At minimum:

```text
INVALID       proposal could not enter evaluation
FAIL          valid candidate did not clear the bar
INCONCLUSIVE  observed change is within uncertainty or evidence is incomplete
PASS          candidate cleared preregistered criteria and awaits audit
PROMOTED      audited candidate became an incumbent
```

`PASS` is not `PROMOTED`. Promotion is a judgment-layer action after reproduction, comparison with
the incumbent, and required diagnostics.

## 4. Step protocol

### 4.1 Build the prompt from live state

Construct each prompt with:

- current baseline metrics, recomputed rather than copied from prose;
- the candidate and evaluator contracts;
- hard prohibitions with their causal reasons;
- a short tail of recent attempts and findings;
- the active search-era brief;
- an exact output contract.

On exploratory cycles, use the diary to prevent repetition, not as an optimization target. Direct
the worker toward an untried family rather than asking it to improve the current leader.

### 4.2 Capture a pre-result hypothesis

Before exposing scores, ask the same agent session to record:

- the expected mechanism;
- the expected direction and affected metric;
- the main failure mode;
- what outcome would falsify the idea.

This separates a real prediction from an explanation invented after the result.

### 4.3 Apply and evaluate

Apply the candidate only after parsing and static guards pass. Assert that the intended mutation
actually occurred. Then run dynamic guards and the evaluator.

When sweeping a parameter:

- keep the grid small and preregistered;
- choose the best value using development evidence only;
- apply the confirmation policy once to that selected value;
- preserve all cells, including failures, in the result record.

### 4.4 Capture a post-result reflection

Resume the same session after scoring and ask:

- what the result rules out;
- whether the proposed mechanism survived;
- the narrowest justified next question.

Store the response as evidence, not as the verdict.

### 4.5 Restore and record

Restore modified targets from the checked-in baseline in both the normal path and a cleanup trap.
Verify restoration before the step exits.

Append a structured record containing:

- cycle, timestamp, model, and session identifier;
- candidate identifier and content hash;
- pre-result hypothesis;
- guard outcomes;
- evaluator version and reproducibility inputs;
- complete metrics and diagnostics;
- verdict and machine-readable reason;
- post-result reflection;
- paths to raw proposal and logs;
- repository revision before and after the cycle.

Commit only durable evidence and intentional state changes. Temporary mutations and build products
must never enter the checkpoint.

### 4.6 Deterministic mode

The step must accept a proposal from a file and send it through the identical parser, guards,
evaluator, restoration, and recording path:

```text
step file:<proposal-path> <cycle>
```

This is the fast path for harness tests, precisely specified hypotheses, reproductions, and
attribution experiments. It must not be a privileged path with weaker checks.

## 5. Knowledge artifacts

Keep durable knowledge in four small, reviewable documents.

### 5.1 Experiment diary

The diary is append-only and records every attempted candidate, including malformed and rejected
ones. A compact entry template is:

```markdown
## <cycle>: <candidate name> — <verdict>

- Time / model / session:
- Candidate hash:
- Era / family:
- Hypothesis:
- Guard outcomes:
- Development result:
- Held-out result:
- Diagnostics:
- What this rules out:
- Audit note:
- Raw artifacts:
```

Automated steps write factual fields. The judgment layer may append clearly attributed audit notes
but does not rewrite history.

### 5.2 Search ledger

The search ledger is the primary steering channel. Maintain exactly one active era, with two
sections:

```markdown
## Active era: <name>

Goal:
Promotion bar:
Constraints and carried exclusions:
Era exit conditions:

### Untried

- Family:
  - Mechanism:
  - Candidate boundary:
  - Grid or cases:
  - Pass criterion:
  - Stop condition:

### Exhausted

- Family:
  - Attempts:
  - Evidence:
  - Mechanism-level conclusion:
  - Reopen only if:
```

Remove or archive stale `Untried` entries when an era closes. Otherwise workers will treat obsolete
text as a live instruction.

### 5.3 Evaluation governance log

The governance log is append-only and owned by the judgment layer:

```markdown
## <date>: <proposed change>

- Status: proposed | accepted | rejected | superseded
- Motivation:
- Threat to validity addressed:
- Risk introduced:
- Effect on comparability with earlier results:
- Decision and approver:
- Activation revision / evaluator version:
```

Workers may submit proposals here only through a designated channel. Evaluator changes never take
effect silently, and results from incompatible evaluator versions are not directly ranked.

### 5.4 Patterns log

Use a short patterns log for lessons that apply across candidates or eras: known confounders,
minimum evidence for broad claims, recurring failure signatures, and rules for interpreting each
verdict type. Promote a lesson into the process only after repeated evidence.

## 6. Search and steering doctrine

### Explore and exploit

Exploit cycles may refine a promising family inside its preregistered boundary. Explore cycles must
choose from `Untried`, avoid building on the incumbent, and seek a different mechanism. A fixed
cadence is usually easier to audit than letting the worker decide when to explore.

### Steer through artifacts

When workers circle, sharpen the ledger entry: specify a mechanism, candidate boundary, cases, pass
criterion, and stop condition. Do not accumulate ad hoc prompt edits that are invisible to future
reviewers.

### Audit every pass

For each `PASS`, the judgment layer should:

1. reproduce it from the recorded candidate and environment;
2. compare it directly with the current incumbent;
3. classify it as a real improvement, trade-off/frontier alternative, or rediscovery;
4. run the preregistered diagnostics and concentration or robustness checks;
5. check for evaluator gaming and confounded comparisons;
6. append an audit note;
7. promote, retain as a frontier candidate, or reject it.

### Escalate to deterministic experiments

Stop autonomous search and use deterministic mode when:

- the next hypothesis is already precisely specified;
- workers repeatedly miss the intended intervention;
- a result needs reproduction or component attribution;
- the claim requires several coordinated evaluations;
- the cost or risk makes trial-and-error inappropriate.

### Manage eras explicitly

Open an era with a goal, baseline, promotion bar, carried exclusions, and preregistered exit
conditions. Close it only when those conditions are met, then archive the queue and summarize the
mechanism-level findings.

When the evaluator, data, objective, or baseline changes incompatibly, start a new era and version
boundary. Numerical rankings may not cross that boundary; durable mechanism-level findings may.

## 7. Match orchestration to verdict cost

Classify claim types before assigning them to a loop:

- **Cheap-verdict claims** receive a trustworthy verdict from one bounded evaluation. They are
  suitable for autonomous cycles.
- **Expensive-verdict claims** require a battery of tests, long observation, rare-event evidence,
  subjective review, or substantial human interpretation. Run them as preregistered deterministic
  experiments with agents as instruments.

The distinction is about the cost of deciding, not the cost of generating a candidate. A cheap
proposal with an expensive verdict still belongs outside the autonomous loop.

## 8. Monitoring and resource allocation

Routine failures should be visible in logs without demanding constant intervention. Alert or pause
on material events:

- `PASS` or promotion candidates;
- guard or restoration breaches;
- evaluator hash or version mismatch;
- stale artifacts or suspected no-op mutations;
- repeated rediscovery;
- cross-model convergence;
- abrupt changes in failure patterns;
- quota, timeout, or infrastructure exhaustion.

Monitor failure signatures as well as successes. Preserve raw output for any parser, build, or
evaluator failure.

Use abundant or expiring model quota for candidate search. Reserve scarce, durable, or
high-capability quota for audits, era decisions, adversarial review, and deterministic experiment
design. Independent convergence by different models is evidence, provided their prompts and search
histories were sufficiently independent.

Avoid automatic in-loop synthesis when the judgment layer is already curating the ledger. It spends
resources, creates competing writers, and can overwrite more carefully reviewed state.

## 9. Operational safety

These rules apply across domains:

- Kill child processes by captured PID or process group. Pattern-based process killing can match
  the command performing the kill.
- With `pipefail`, remember that commands such as a no-match search can produce valid output while
  returning a nonzero status. Handle expected no-match cases explicitly.
- Never hide output from a build or evaluator you depend on. Assert that expected artifacts were
  freshly produced before consuming them.
- Treat environment variables as untyped interfaces: validate names and prove that each control
  changes behavior with a binding test before running a sweep.
- Assert the match count before any text replacement or source splice. A silent no-op is a failed
  candidate application.
- Coordinate dependent jobs using output marker files containing run identity and status, not
  process-name matching.
- Write logs and result files atomically, using a temporary file followed by rename.
- Hold a per-track lock so two steps cannot mutate the same target concurrently.
- Pin or record evaluator inputs, dependencies, toolchain, locale, timezone, and random seeds.
- Keep secrets outside prompts, artifacts, and child environments unless explicitly required.
- Test cleanup by interrupting the step during application, build, and evaluation.

## 10. Suggested repository shape

Names may change, but ownership boundaries should remain obvious:

```text
.
├── AGENTS.md                         project-specific worker rules
├── aop/
│   ├── run_agent.sh                  normalized runner
│   ├── loop.sh                       mechanical scheduler
│   ├── step                          one candidate transaction
│   └── lib/                          parsing, guards, records, cleanup
├── evaluator/                        locked, project-owned referee
├── baselines/                        restoration sources
├── proposals/                        deterministic candidate files
├── knowledge/
│   ├── diary.md                      append-only attempt history
│   ├── search-ledger.md              active steering state
│   ├── evaluation-governance.md      evaluator change control
│   └── patterns.md                   cross-run lessons
├── results/                          structured immutable result records
├── logs/                             operational and raw agent output
└── scratch/                          ignored disposable state
```

At minimum, version the process code, evaluator, baselines, knowledge documents, proposal fixtures,
and durable result records. Ignore credentials, mutable locks, temporary candidate splices, build
products, and session caches.

## 11. Bootstrap a new project

Complete these steps in order:

1. **Define the referee.** Preregister the objective, partitions, uncertainty/noise policy,
   disqualifiers, diagnostics, and promotion rule in the governance log.
2. **Triage claim types.** Separate cheap-verdict work that can enter a loop from expensive-verdict
   work that requires deterministic, supervised experiments.
3. **Define the candidate boundary.** Write its grammar, mutation scope, guards, deduplication keys,
   and restoration path.
4. **Implement the runner and step.** Keep model handling in the runner and the full transaction in
   the step. Add deterministic file mode from the start.
5. **Create the knowledge artifacts.** Open the first era with its goal, bar, exclusions, candidate
   families, and exit conditions.
6. **Seed three to five deterministic candidates.** Include one valid candidate, one parse failure,
   one guard failure, one evaluator failure, and one interrupted run where practical. These are the
   harness acceptance tests and the diary's first evidence.
7. **Verify recovery.** After every fixture, confirm the baseline hash, evaluator hash, result
   record, raw logs, and exit status.
8. **Staff the judgment layer.** Name the reviewer, define alert conditions, establish the
   audit-every-pass routine, and reserve time for adversarial review.
9. **Start with bounded operation.** Use a small cycle cap and conservative timeout. Review the
   first batch before enabling unattended operation.

Do not start the loop until a human-designed deterministic candidate completes the exact end-to-end
path that autonomous candidates will use.

## 12. Readiness and review checklists

### Before a run

- The active era, baseline, evaluator version, and promotion bar are explicit.
- Every queued family has a mechanism, boundary, pass criterion, and stop condition.
- Candidate mutation is restricted and restoration has been interrupt-tested.
- Evaluator inputs are pinned and its integrity check passes.
- Deterministic fixtures pass through the production step path.
- Duplicate prevention includes current and historical incumbents.
- Alerts include failures, stale artifacts, and cleanup breaches.
- Resource and time limits are set.

### After a run batch

- Every cycle has a terminal record or an explained infrastructure gap.
- Baseline and evaluator integrity checks pass.
- Promising results were reproduced and audited before promotion.
- Repeated failures were converted into ledger exclusions or sharper briefs.
- Stale queue entries were removed.
- The era exit conditions were checked without rewriting them after seeing results.
- Any evaluator change is recorded and versioned.

## 13. What transfers and what does not

Portable across projects:

- the runner, loop, step, and judgment-layer separation;
- the transactional step and restoration invariant;
- strict proposal contracts and ordered guard hooks;
- deterministic file mode;
- diary, search ledger, governance log, and patterns log;
- explore/exploit cadence, preregistration, pass auditing, and era discipline;
- verdict-cost triage, monitoring conventions, and operational safety rules.

Project-specific and intentionally not standardized:

- the evaluator and its data partitions;
- objectives, thresholds, noise policy, and diagnostics;
- candidate syntax and source splice targets;
- domain invariants and forbidden operations;
- prompt content tied to local interfaces;
- any constant that cannot be re-derived in the adopting project.

Reuse the discipline and interfaces first. Extract shared code only after at least two projects have
demonstrated that the same abstraction is real rather than an accident of the first implementation.
