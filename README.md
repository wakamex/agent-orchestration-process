# Agent Orchestration Process (AOP)

AOP runs autonomous CLI agents as bounded workers through one provider-independent interface. It
creates isolated task worktrees, enforces execution profiles and configured deadlines, retains run
evidence, resumes exact provider sessions, and keeps checkpointing and integration explicit.

It is intended for unattended jobs such as parallel code changes, evaluation pipelines, test
generation, and autoresearch loops. Authentication, model execution, tools, and session persistence
remain owned by each installed agent CLI.

## Why AOP?

AOP gives automation one CLI for controlling different agent harnesses without requiring callers to
learn each harness's command flags, session format, output schema, runtime state, or resume behavior:

- Run Codex, Claude Code, Cursor Agent, Devin CLI, OpenCode, Antigravity (`agy`), Grok Build,
  Hermes, and DeepSeek Harness (`dsh`) through the same `run` and `resume` commands. See
  [Provider adapters](docs/harnesses.md) for current model support and limitations.
- Reuse each harness's existing authentication, model execution, tools, and conversations instead
  of maintaining a separate AOP credential store.
- Apply the same isolated worktrees, inputs-only sealed runs, configured deadlines, process cleanup,
  immutable inputs, and validated artifacts across harnesses.
- Retain consistent results, logs, timing, token usage, and exact provider session identity when
  available.
- Keep cleanup, checkpointing, and Git integration outside agent control.

[Agent Client Protocol (ACP)](https://agentclientprotocol.com/) standardizes interactive
communication between agents and clients such as editors and IDEs. Use ACP when a person needs to
chat, approve actions, watch progress, and steer a session.

Use AOP when software needs to run bounded jobs unattended across different agent CLIs. The two are
complementary: ACP serves interactive clients, while AOP serves automation.

## Requirements and installation

Requirements:

- Linux
- Python 3.11 or newer
- `bwrap` for `edit`, `review`, and `sealed`; `host` does not use it
- Git for `edit`, `review`, and `host`; `sealed` does not require a repository
- uv for the installation and development commands below
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
uv --no-config sync --locked
uv --no-config run --locked pytest
uv --no-config run --locked ruff check src tests
```

## First run

Initialize AOP in an existing Git repository:

```sh
cd /path/to/project
aop init
```

`aop init` adds `/.aop/` to `.gitignore`. Commit that project-level change before integrating
tasks so the main worktree is clean.

Run one bounded turn with Codex. AOP creates the isolated task worktree when it does not already
exist:

```sh
aop run task-a --agent codex --prompt "Implement the parser and its tests"
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

Every adapter records the same result shape, including provider session identity when available,
timing, token usage, cost evidence, logs, and declared artifacts. See
[Token usage and pricing](docs/token-usage.md) for the normalized token contract, provider mappings,
and cost calculation rules.
