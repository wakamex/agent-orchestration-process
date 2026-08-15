# Agent Orchestration Process (AOP)

AOP runs autonomous CLI agents as bounded workers through one provider-independent interface. It
creates isolated task worktrees, enforces execution profiles and deadlines, retains run evidence,
resumes exact provider sessions, and keeps checkpointing and integration explicit.

It is intended for unattended jobs such as parallel code changes, evaluation pipelines, test
generation, and autoresearch loops. Authentication, model execution, tools, and session persistence
remain owned by each installed agent CLI.

## Guiding principles

- Describe complete semantic boundaries. Write access, visibility, identity, instructions,
  environment, credentials, and network access are separate facts and must be enforced and recorded
  explicitly.
- Delegate harness-native behavior. Authentication, model execution, tools, session persistence,
  and provider protocols stay with the harness unless its automation interface is missing a required
  capability.
- Project the least authority needed. A task receives only the selected provider credential and
  required runtime state, never an entire multi-provider credential store merely because copying it
  is easier.
- Do not make users maintain parallel authentication. AOP reuses harness-managed credentials and
  preserves native environment overrides instead of inventing an AOP-specific secret store.
- Keep controller policy outside model control. Isolation, deadlines, evidence, cleanup, and
  integration remain AOP responsibilities even when provider permission prompts are bypassed.
- Prefer small native extension points over wrappers and forks. When a harness lacks a bounded
  automation feature, use its documented plugin or protocol surface and keep the added layer narrow.
- Fail closed and record what ran. Ambiguous identity, stale pricing, invalid credentials, incomplete
  terminal output, and unenforced policy are failures, while effective policy and provenance remain
  inspectable after the run.
- Optimize for the long-term product, not compatibility with accidental local setup. A clean break is
  acceptable before users exist when it removes ambiguity or foreseeable rewrites.

## Why use AOP when ACP exists?

AOP is built for bounded automation, not as a general interactive agent host. Its provider adapters
deliberately invoke the installed CLIs directly for one turn at a time and normalize only the
contract needed by an automated worker: model and effort selection, a final result, exact resume
identity, timing and usage, durable logs, process termination, and declared artifacts. Worktrees,
filesystem isolation, deadlines, and integration remain AOP responsibilities outside the adapters.

[Agent Client Protocol (ACP)](https://agentclientprotocol.com/) standardizes communication between
coding agents and interactive clients such as editors and IDEs. Use ACP when a person needs to chat
with different agents through one client, approve actions, see live progress, and steer a session.

Use AOP when software needs to run bounded agent jobs unattended. AOP was designed to support
autoresearch loops across multiple agent harnesses without reimplementing each harness's model
selection, session resume, output parsing, permissions, runtime state, and failure handling in every
research project. It starts disposable CLI turns, runs jobs concurrently in isolated worktrees,
enforces deadlines, validates declared artifacts, records durable evidence and usage, resumes exact
sessions, cleans up runtime state, and keeps integration explicit. It is suited to fanout,
independent experiment cycles, evaluation pipelines, and other automation where the result matters
more than an interactive interface.

The two are complementary. An interactive client can use ACP while an automation system uses AOP
to schedule and govern bounded work.

## Requirements and installation

The reference implementation is a Python CLI for running Codex, Claude Code, Cursor Agent, Devin
CLI, OpenCode, Antigravity (`agy`), Grok Build, Hermes, and DeepSeek Harness (`dsh`) as bounded
workers. See [Provider adapters](docs/harnesses.md) for authentication, model selection, runtime
state, and current limitations.

Requirements:

- Linux with `bwrap`; Git is required for `edit` and `review` profiles but not `sealed`
- Python 3.11 or newer, managed through uv
- At least one installed and authenticated supported agent CLI

Install the CLI from [PyPI](https://pypi.org/project/agent-orchestration-process/):

```sh
uv tool install agent-orchestration-process
```

For development, clone the repository and let uv create and synchronize the local environment. The
default `dev` dependency group contains pytest and Ruff:

```sh
git clone https://github.com/wakamex/agent-orchestration-process.git
cd agent-orchestration-process
uv sync
uv run pytest
uv run ruff check src tests
```

## First run

Initialize a Git repository and create one isolated task worktree:

```sh
cd /path/to/project
aop init
aop worktree create task-a
```

`aop init` adds `/.aop/` to `.gitignore`. Commit that project-level change before integrating
tasks so the main worktree is clean.

Run one bounded agent turn:

```sh
aop run task-a --agent claude --model sonnet --effort high \
  --prompt "Implement the parser and its tests"
```

`aop run` prints the agent's final message to stdout and its AOP run ID, provider session ID, and
run artifact directory to stderr. Resume the exact session using the AOP run ID:

```sh
aop resume <run-id> --prompt "Address the review findings"
```

AOP retains the request, result, logs, final message, and declared artifacts under `.aop/`. See the
[CLI guide](docs/cli.md) for inputs, artifacts, batch runs, machine-readable results, cleanup, model
discovery, and lower-level worktree commands. See [Integration](docs/integration.md) when a task's
changes are ready to checkpoint and merge.

## Execution profiles

AOP profiles describe the complete execution boundary rather than only where writes are allowed.
The isolated profiles expose only the declared workspace, inputs, provider runtime, private state,
cache, scratch, output, and required operating-system paths. `host` runs without that isolation.

| Profile | Use it when |
| --- | --- |
| `edit` | The agent should modify an isolated task worktree. |
| `review` | The agent should inspect a task worktree without modifying it. |
| `sealed` | The agent should see only explicit input snapshots, with no repository or inherited instructions. |
| `host` | The command needs the native host environment and filesystem. |

Inspect a declared profile or preview the intended policy without compiling or dispatching it:

```sh
aop profile explain sealed
aop profile explain sealed --agent codex --json
aop run study-arm --profile sealed --prompt "Answer using only declared inputs" --dry-run
```

All profiles currently retain native network connectivity because the supported provider CLIs need
it. See [Execution profiles](docs/profiles.md) for the exact filesystem, instruction, environment,
credential, input, and output boundaries.

## Results and usage

Every adapter records the same result shape, including exact resume identity, timing, token usage,
cost evidence, logs, and declared artifacts. See [Token usage and pricing](docs/token-usage.md) for
the normalized token contract, provider mappings, and cost calculation rules.
