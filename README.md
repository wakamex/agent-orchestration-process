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

This repository includes a dependency-free Python CLI for running concurrent tasks in isolated Git
worktrees. Its first normalized model adapter runs Codex non-interactively, records the full result,
and resumes the exact Codex session associated with an earlier run.

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

Run Codex in a new or existing task worktree:

```sh
aop run task-a --prompt-file task.md --timeout 1800
aop run task-b --prompt "Implement the parser and its tests"
```

`aop run` prints the agent's final message to stdout and the AOP run ID, Codex session ID, and
artifact directory to stderr. Resume the exact session using the AOP run ID:

```sh
aop resume <run-id> --prompt "Address the review findings"
```

The default Codex sandbox is `workspace-write`, scoped to the isolated task worktree. The configured
Codex model, reasoning effort, authentication, and user instructions are preserved unless `--model`
or `--effort` is supplied. `AOP_CODEX_BIN` may override the Codex executable for testing or a custom
installation.

Run independent tasks concurrently from a TOML manifest:

```toml
[[tasks]]
id = "parser"
prompt_file = "tasks/parser.md"
model = "<model-id>"
effort = "xhigh"
timeout = 1800

[[tasks]]
id = "tests"
prompt = "Add adversarial parser tests"
effort = "high"
```

```sh
aop batch tasks.toml --jobs 4
```

Prompt-file paths are resolved relative to the manifest. Each task may set `base`, `model`,
`effort`, `sandbox`, and `timeout`; unspecified values use the same defaults as `aop run`. The
scheduler keeps at most `--jobs` tasks active, prints only concise lifecycle status, and stores full
agent output in the normal per-run directories. On interruption it launches no additional tasks and
waits for already-active tasks to finish.

Every batch writes `.aop/batches/<batch-id>.json` with task-order-preserving run IDs, session IDs,
durations, exit codes, and errors. A batch exits nonzero if any task fails, without discarding
successful sibling results.

Run any other command in a task worktree with the lower-level escape hatch:

```sh
aop exec task-a -- <agent-command>
aop worktree list
aop worktree path task-a
aop worktree remove task-a
```

Task worktrees are detached at the selected base commit, so one worker cannot move another worker's
branch. `aop exec` supplies `AOP_ROOT`, `AOP_TASK`, `AOP_WORKTREE`, and the shared `AOP_CACHE_DIR` to
the child process. Dirty worktrees cannot be removed unless `--force` is explicit.

Runtime state lives under the ignored `.aop/` directory:

```text
.aop/
├── batches/            structured batch summaries
├── cache/              shared cache root for future build and runner adapters
├── runs/<run-id>/      request, result, JSONL events, stderr, and final message
├── worktrees/          one isolated checkout per task
└── worktrees.lock      lifecycle-operation lock
```

The current CLI does not merge task commits, configure language-specific build caches, or launch
providers other than Codex. Those interfaces will be added when a real project needs them.

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
AOP_RUN_ID      current structured run identifier (model runs only)
```

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
