# Agent Orchestration Process (AOP)

AOP is a command-line tool for running coding agents such as Claude Code and Codex as unattended jobs. It is for automation: CI jobs, batch experiments, benchmarks, and programs that hand work to an agent and need a dependable result back. One `aop run` command drives eleven different agent CLIs, called harnesses here, and gives every run the same sandbox, deadline, and result record, so a caller does not need a separate adapter for each CLI's flags, sessions, and output.

- Run [Codex](https://github.com/openai/codex), [Claude Code](https://code.claude.com/docs/en/overview), [Cursor Agent](https://docs.cursor.com/en/cli/overview), [Devin CLI](https://docs.devin.ai/cli/index), [OpenCode](https://github.com/anomalyco/opencode), [Antigravity (`agy`)](https://github.com/google-antigravity/antigravity-cli), [Grok Build](https://github.com/xai-org/grok-build), [Hermes](https://github.com/NousResearch/hermes-agent), [DeepSeek Harness (`dsh`)](https://github.com/deepseek-ai/deepseek-harness), Zcode, and [Muse Code](https://dev.meta.ai/) through the same `run` and `resume` commands. Each keeps its own login, models, and tools.
- Isolate each run in a Git worktree or an empty sealed workspace, inside a [bubblewrap](https://github.com/containers/bubblewrap) sandbox, with optional network-level blocking of web access.
- Get one JSON result per run: the answer, timing, token usage, an API-equivalent cost, the harness session ID for exact resume, and whether the provider refused.
- Keep cleanup, checkpointing, and Git integration outside the agent's control.

[Agent Client Protocol (ACP)](https://agentclientprotocol.com/) connects agents to editors for interactive sessions, where a person chats, approves actions, and steers. AOP serves the opposite case: software running bounded jobs with nobody watching. The two are complementary.

## Requirements

- Linux. AOP's sandbox is built on Linux namespaces.
- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).
- `bwrap` (bubblewrap) for the sandboxed profiles, with unprivileged user namespaces allowed. On Fedora and Debian, install the `bubblewrap` package. Ubuntu 24.04 and later also need `sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0`.
- Git, for runs in a repository.
- At least one harness, installed and signed in. Codex needs version 0.150.1 or newer and Antigravity 1.1.16 or newer; [Harness adapters](docs/harnesses.md#supported-harnesses) lists the versions each adapter was last tested with.

## Quickstart

Install AOP from [PyPI](https://pypi.org/project/agent-orchestration-process/):

```sh
uv tool install agent-orchestration-process
```

Run a sealed task. It sees no repository and no local files, only what you pass with `--input`:

```sh
aop run --agent claude --profile sealed --prompt "Reply with only the word copper."
```

The answer goes to stdout; the run's details go to stderr:

```text
aop: run_id=74a91b9f-d64e-45cd-9c56-b308c8840d58 session_id=fba843cd-3050-4672-a39b-1deaa63ef5d9 time=2.13s tokens=16887 accounting=complete calculated_cost=$0.000171 provider_reported_cost=n/a billing=subscription artifacts=/home/me/.aop/runs/74a91b9f-d64e-45cd-9c56-b308c8840d58
copper
```

AOP records the run in a `.aop` folder in the directory you ran it from. Add `--json` for the full result as one object, abridged here:

```json
{
  "run_id": "5f668bbc-cad7-4852-b1b3-00b021952dd2",
  "provider": "claude",
  "model": "claude-haiku-5-5",
  "status": "succeeded",
  "final_message": "copper",
  "session_id": "24ec7778-cd70-450f-a1f1-9090ae51b889",
  "duration_seconds": 2.18,
  "error": null,
  "usage": {"input_tokens": 16883, "cached_input_tokens": 11227, "output_tokens": 4},
  "accounting_status": "complete",
  "calculated_cost": {"amount_usd": 0.00124527, "currency": "USD", "priced_as": "claude-haiku-5-5"},
  "billing": {"route": "subscription", "credential_source": "claude-oauth"},
  "refusal": null
}
```

Continue the same harness session from any directory with the run ID:

```sh
aop resume 5f668bbc-cad7-4852-b1b3-00b021952dd2 --prompt "Now reply with silver."
```

To work on a repository, run from inside it. AOP creates a worktree for the task and leaves your checkout alone:

```sh
aop init
aop run task-a --agent codex --prompt "Implement the parser and its tests"
```

`aop init` adds `/.aop/` to `.gitignore`; commit that before integrating tasks. See [Integration](docs/integration.md) for merging a task's changes back.

## Safety defaults

AOP is built to run agents unattended, so know what a run can do before you start one:

> [!WARNING]
> - Harness permission prompts are always bypassed. The agent runs its tools without asking; the AOP sandbox and profile are the boundary.
> - The network is open unless you pass `--no-web`. Without it, an agent can reach any host, including to send data out.
> - The agent can read the credentials of its own harness, which AOP copies into the run so the harness can sign in.
> - The `host` profile has no sandbox: the agent has your full user account and filesystem.
> - AOP limits a run only by wall-clock time with `--timeout`. It has no spending, token, or resource cap.

[Running unattended](docs/unattended.md) covers what the sandbox does and does not stop, every result status and exit code, recovery after a crash, and cleanup.

## Execution profiles

`--profile` sets what a run can see and change. The default is `edit`.

| Profile | Use it when |
| --- | --- |
| `edit` | The agent should modify an isolated task worktree. |
| `review` | The agent should inspect a task worktree without modifying it. |
| `sealed` | The agent should see only explicit input snapshots, with no repository or inherited instructions. |
| `host` | The command needs the native host environment and filesystem, with no sandbox. |

Preview a run's boundary without starting it:

```sh
aop run --agent codex --profile sealed --prompt "Answer using only declared inputs" --dry-run
```

See [Execution profiles](docs/profiles.md) for exact filesystem, instruction, environment, credential, input, and output boundaries, and [CLI guide](docs/cli.md#disable-model-controlled-web-access) for `--no-web`.

## Documentation

- [CLI guide](docs/cli.md): running and resuming tasks, inputs and artifacts, batches, models, and cleanup.
- [Execution profiles](docs/profiles.md): what each profile exposes.
- [Running unattended](docs/unattended.md): safety, statuses, exit codes, and recovery.
- [Harness adapters](docs/harnesses.md): per-harness support, versions, and options.
- [Token usage and pricing](docs/token-usage.md): the usage and cost fields, and upgrading old run records.
- [Integration](docs/integration.md): checkpointing and merging task changes.
- [Glossary](docs/glossary.md): the terms these pages use.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setting up a checkout and running the tests, including the opt-in tests against installed harnesses.
