# Agent Orchestration Process (AOP)

AOP gives automation one CLI for running bounded, non-interactive jobs across different agent
harnesses without requiring callers to learn each harness's command flags, session format, output
schema, runtime state, or resume behavior:

- Run [Codex](https://github.com/openai/codex),
  [Claude Code](https://code.claude.com/docs/en/overview),
  [Cursor Agent](https://docs.cursor.com/en/cli/overview),
  [Devin CLI](https://docs.devin.ai/cli/index),
  [OpenCode](https://github.com/anomalyco/opencode),
  [Antigravity (`agy`)](https://github.com/google-antigravity/antigravity-cli),
  [Grok Build](https://github.com/xai-org/grok-build),
  [Hermes](https://github.com/NousResearch/hermes-agent), and
  [DeepSeek Harness (`dsh`)](https://github.com/deepseek-ai/deepseek-harness) through the same
  `run` and `resume` commands. See [Harness adapters](docs/harnesses.md) for current model support
  and limitations.
- Reuse each harness's existing authentication, model execution, tools, and conversations instead
  of maintaining a separate AOP credential store.
- Apply the same isolated worktrees, inputs-only sealed runs, configured deadlines, process cleanup,
  snapshotted inputs, and validated artifacts across harnesses.
- Retain consistent results, logs, timing, token usage, and exact provider session identity when
  available.
- Keep cleanup, checkpointing, and Git integration outside agent control.

Codex runs can also use [Z.AI Coding Plan](docs/harnesses.md#codex) as the `zai-coding-plan` inference provider rather than as a separate harness. AOP resolves the effective native Codex configuration, validates the authenticated model inventory, projects only the selected credential into isolated runs, and preserves the exact route for resume. List the currently available models with:

```sh
aop models --agent codex --provider zai-coding-plan
```

[Agent Client Protocol (ACP)](https://agentclientprotocol.com/) standardizes interactive
communication between agents and clients such as editors and IDEs. Use ACP when a person needs to
chat, approve actions, watch progress, and steer a session.

Use AOP when software needs to run bounded jobs unattended across different agent CLIs. The two are
complementary: ACP serves interactive clients, while AOP serves automation.

## Install and run

AOP requires Linux and Python 3.11 or newer. The `edit`, `review`, and `sealed` profiles require
`bwrap`. Git is required for `edit`, `review`, and `host`. Install and authenticate at least one
supported agent harness.

Install AOP from [PyPI](https://pypi.org/project/agent-orchestration-process/):

```sh
uv tool install agent-orchestration-process
```

Run a sealed task from any directory without exposing local files or requiring a Git repository:

```sh
aop run --agent codex --profile sealed --prompt "Explain the difference between TCP and UDP"
```

Run a task in an existing Git repository:

```sh
aop init
aop run task-a --agent codex --prompt "Implement the parser and its tests"
```

`aop init` adds `/.aop/` to `.gitignore`; commit that change before integrating tasks. AOP creates
the task worktree automatically and prints the run ID needed to resume the exact provider session:

```sh
aop resume <run-id> --prompt "Address the review findings"
```

See the [CLI guide](docs/cli.md) for other harnesses, inputs, artifacts, batch runs, cleanup, and
machine-readable results. See [Integration](docs/integration.md) when a task's changes are ready to
checkpoint and merge.

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

Use `--no-web` to deny model-controlled external retrieval while retaining the harness's inference
connection. AOP records the effective tool policy and rejects harnesses that cannot enforce the
boundary. See [CLI guide](docs/cli.md#disable-model-controlled-web-access) for current support.

## Results and usage

Every adapter records the same result shape, including provider session identity when available,
timing, token usage, cost evidence, logs, and declared artifacts. See
[Token usage and pricing](docs/token-usage.md) for the normalized token contract, provider mappings,
and cost calculation rules.

Results distinguish complete accounting from partial or unavailable deadline accounting. Codex deadlines use native app-server interruption to retain per-turn usage and an API-equivalent cost when available. If any harness terminates before reporting usage, `usage` and cost fields remain null instead of being reported as measured zero.

## Development

Clone the repository and synchronize its locked development environment:

```sh
git clone https://github.com/wakamex/agent-orchestration-process.git
cd agent-orchestration-process
uv --no-config sync --locked
uv --no-config run --locked pytest
uv --no-config run --locked ruff check src tests
```
