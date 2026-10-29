# Running unattended

AOP runs agents with nobody watching, so this page collects what a caller needs to rely on that: what the sandbox does and does not stop, how to tell how a run ended, and how to recover and clean up.

## What a run can do

Harness permission prompts are always bypassed: an agent runs its tools without asking. The boundary is the AOP profile, not the harness's own approvals.

| Concern | `edit`, `review`, `sealed` | `host` |
| --- | --- | --- |
| Files | Only the workspace, input snapshots, and the run's private directories; see [Execution profiles](profiles.md) | Your whole user account |
| Network | Open, unless `--no-web` | Open, unless `--no-web` |
| Harness credentials | Readable by the agent, which needs them to sign in | Readable by the agent |
| Other processes | A separate process, IPC, and hostname namespace | Visible |

The network is the largest gap. Without `--no-web`, an agent can reach any host, so instructions planted in a file, web page, or tool result can make it send data out, including the credentials it can read. `--no-web` closes the network to everything except the harness's own inference and login endpoints, in the sandboxed profiles for every harness that supports it; see [CLI guide](cli.md#disable-model-controlled-web-access). It cannot stop what a provider retrieves on its own servers, which is why AOP also disables server-side search tools under `--no-web`.

The sandbox is [bubblewrap](https://github.com/containers/bubblewrap) with fresh user, mount, process, IPC, and hostname namespaces and all capabilities dropped. AOP adds no seccomp filter, so the agent can make any system call its user may, and a kernel vulnerability reachable from an unprivileged user is not contained.

AOP limits a run only by wall-clock time, with `--timeout`. It sets no spending, token, CPU, memory, disk, or process limit; use the harness's own account limits for spending, and a systemd unit or cgroup around `aop` for resources.

## How a run ended

Every result has three fields that say how it ended:

| `status` | Meaning |
| --- | --- |
| `succeeded` | The harness finished, reported its session, and returned an answer with no error. |
| `response_available_with_provider_error` | The harness returned an answer but its provider also reported an error, for example a failed tool call late in the turn. Read `final_message` and `provider_error`. |
| `failed` | Anything else: the harness exited with an error, timed out, or returned no answer, or AOP rejected the result. An empty answer counts as none unless the run declared artifacts, and so does an answer that is a tool call the model wrote as text instead of making it. `error` says why. |

- `timed_out` is true when `--timeout` ended the run.
- `refusal` is set when the provider refused the input, with its category and the models that refused. A refusal never succeeds on retry; an outage, quota limit, or expired login has `refusal: null` and may.
- `accounting_status` says whether token usage is `complete`, `partial` (a timed-out run that reported some usage), or `unavailable`. Usage and cost are null rather than zero when unmeasured.

`aop run` and `aop resume` exit with:

| Exit code | Meaning |
| --- | --- |
| 0 | The run succeeded. |
| 124 | The run timed out. |
| 1 to 125 | The harness's own non-zero exit code, passed through, or 1 for any other failure. |
| 2 | AOP refused the request, usually before starting the harness: a bad option, an unsupported combination, a failed input check, missing credentials, or a refused admission. |
| 127 | A command AOP itself needed was not found. A harness that is missing when its run starts gives a failed result and exit code 1. |
| 130 | Interrupted with Ctrl-C. |

`aop batch` exits 0 only if every task succeeded, and 1 otherwise.

## Admission before a run starts

An admission command decides whether a run may start, for example by checking a subscription's remaining quota. AOP keeps no quota logic of its own: the command makes the decision. Every `aop run`, `aop resume`, batch task, and integration resume asks the first of these that is set:

1. `--admit-command CMD` on `run` or `resume`;
2. the `AOP_ADMIT_COMMAND` environment variable;
3. `admit_command` in `~/.config/aop/config.toml` (or under `XDG_CONFIG_HOME`), which applies to every `aop` you run, from any shell, job, or directory:

```toml
admit_command = "/path/to/admit-check"
```

An empty value at a level turns admission off from there, and the run records the opt-out. A config file that cannot be read, or that sets anything else, refuses every run until it is fixed.

The command is split like a shell command line but run without a shell. It receives the run on standard input as one JSON object, and the same fields as `AOP_ADMIT_*` environment variables, each cut to 4,096 characters:

| Field | Meaning |
| --- | --- |
| `run_id` | The ID the run will have |
| `parent_run_id`, `session_id` | For a resume, the run and harness session it continues |
| `agent`, `model`, `effort`, `inference_provider` | The harness and model selection |
| `profile`, `task`, `workdir` | Where the run will execute |
| `credential_home` | The harness home its credentials come from, such as `~/.claude` or `CODEX_HOME`; the run's environment is a copy of the caller's, so this is the home the run uses |
| `backend` | The inference provider the run spends, where AOP can tell: Hermes's configured or chosen provider, the provider prefix of an OpenCode model, DeepSeek Harness's route, or a Codex `--provider`; null otherwise |
| `timeout_seconds` | The run's deadline |

Exit status 0 admits the run. Its `admission` record, in `request.json`, `result.json`, and the results archive, holds `by` (the flag, variable, or config file that set the command), `command`, `exit_code`, `stdout`, and `stderr`; an opt-out is `{"skipped": true, "by": ...}`, and `admission` is null only when no admission command is configured anywhere. Any other exit status refuses it, and so does a command that cannot be started or does not finish within 60 seconds: no task state is created, the harness never starts, and `aop` exits 2 and prints one JSON line on standard output, `{"admitted": false, "reason": ..., "command": ..., "exit_code": ..., "stdout": ..., "stderr": ...}`. A run's standard output otherwise carries its answer, so that line tells a refusal apart from a run.

## Long commands inside a run

A run's only deadline is its `--timeout`. A harness may also limit each tool call, and a long check such as a full build or test suite can outlast one call. The supported way to run it is the harness's own shell tool: in the foreground when the command fits in one call, and otherwise in that tool's own background mode, checking on it with later calls. Everything started that way belongs to the run and ends with it.

Every run has `AOP_RUN_ID` in its environment, so instructions and tools can tell that they are inside a run. Helpers that hand work to a terminal tab or a service manager, such as a user's job launcher or `systemd-run --user`, are not part of a run: in the sandboxed profiles they cannot reach the user's service manager at all, which AOP's tests check. With `--profile host` nothing stops them, and a unit they start outlives the run, so tell executors not to use them.

A sentence that states this in a brief: "Run long commands, such as builds and test suites, inside this run with your own shell tool, in the foreground or in that tool's background mode, and never start services, timers, units, or detached processes that outlive the run."

## Deadlines

When `--timeout` expires, AOP sends SIGTERM to the harness itself, so it can save its session and flush its output, waits five seconds, and then stops the sandbox. Codex receives a native turn interruption instead, which also keeps its usage for the turn. The grace period does not extend the work budget, and the result stays `timed_out`.

## Crashes and recovery

The sandbox is started with bubblewrap's `--die-with-parent`, so if AOP itself is killed, the harness and everything it started are killed too. Task and execution locks are kernel file locks, which are released when the process holding them exits, so no stale lock survives a crash.

A run whose AOP process died has a `request.json` but no `result.json`, and cannot be resumed; start a new run. Its worktree or sealed state stays until `aop cleanup <run-id>`. Read the partial `events.jsonl` in its run directory to see how far it got.

## Cleanup and disk use

AOP never deletes run state on its own, because a task stays resumable until it is cleaned up. Each task keeps its worktree or private sealed workspace, harness home, scratch, and cache until `aop cleanup <run-id>`, so a caller that starts many runs should clean up each one once it no longer needs to resume it, or run `aop cleanup --all`, which skips tasks that are still running. Cleanup keeps the run's request, input snapshots, result, logs, and archived artifacts.

AOP also keeps a copy of every result, about 10 KB each, in `~/.local/state/aop/results/`. It is never pruned.

## Integration with nobody watching

`aop integrate` rebases a task's commits onto main and fast-forwards main. When a commit conflicts, AOP resumes the task's agent to resolve it inside the worktree, then gives the agent one final validation turn before fast-forwarding. AOP checks the Git history at each step but does not review what the agent decided, so an integration run unattended merges the agent's conflict resolution as it stands. Review the integration's audit record, or integrate only after reading the diff, when that matters. See [Integration](integration.md).
