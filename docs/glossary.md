# Glossary

The terms these pages use, in the sense AOP gives them.

- Harness: a coding agent's own command-line program, such as Claude Code (`claude`) or Codex (`codex`). AOP runs the harness and keeps its login, models, tools, and sessions; it does not reimplement any of them. `--agent` selects one.
- Run: one invocation of a harness, by `aop run` or `aop resume`. It gets a UUID run ID and a directory `.aop/runs/<run-id>/` holding its request, result, and logs.
- Task: a named piece of work in a repository, such as `task-a`. Each task has its own Git worktree, and its runs share it. Sealed runs need no task name.
- Session: the harness's own conversation. A run records its harness's session ID.
- Exact resume: `aop resume` continues the same harness session with the same harness, model, effort, provider, and profile as the run it resumes, rather than starting a new conversation with similar settings.
- Profile: the boundary a run gets, chosen with `--profile`: `edit` (default), `review`, `sealed`, or `host`. See [Execution profiles](profiles.md).
- Sealed run: a run in an empty private workspace that sees only the files passed with `--input`, and no repository or user instructions.
- Input snapshot: a read-only copy of a file or folder passed with `--input`, taken when the run starts and mounted at `/inputs/<name>`.
- Artifact: a file the run must produce in its output directory, declared with `--artifact`. AOP checks it exists and archives it with the run.
- State root: a `.aop` folder that holds run records. Each run's result names its state root.
- Inference provider and route: the service a harness sends model requests to, and the endpoint and credential it uses. Most harnesses have one; Codex, Hermes, and DeepSeek Harness can be pointed at others with `--provider`.
- Projected credential: the single login AOP copies into a run's private harness home, out of everything the harness has stored, so the run can sign in.
- `--no-web`: the policy that denies the model web access while keeping its inference connection; see [CLI guide](cli.md#disable-model-controlled-web-access).
- Participant mode: an experimental Hermes mode for one turn in a structured exchange, such as a game move, rather than open-ended work.
- API-equivalent cost: what a run's tokens would cost at the provider's public API prices, from the [model-prices](https://pypi.org/project/model-prices/) data. It is a comparison figure, not your bill: subscription runs cost nothing extra per token.
- Accounting status: whether a run's token usage is `complete`, `partial`, or `unavailable`; see [Token usage and pricing](token-usage.md).
