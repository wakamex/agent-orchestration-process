# Contributing

Clone the repository and set up its locked environment with [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/wakamex/agent-orchestration-process.git
cd agent-orchestration-process
uv sync --locked
uv run --locked python -m unittest discover -s tests
uv run --locked ruff check src tests
```

The sandbox tests need `bwrap` with unprivileged user namespaces; see the README's requirements.

## Tests against installed harnesses

These tests run a real installed harness and are skipped unless its variable names the binary. None needs quota or credit: they use a local model server or stop before the first model request.

| Variable | Checks |
| --- | --- |
| `AOP_TEST_DSH_BIN=/path/to/dsh` | DeepSeek Harness run and exact resume under all four profiles against a local Messages API server, and the tools a `--no-web` run offers |
| `AOP_TEST_GROK_BIN=/path/to/grok` | the tools a sealed `--no-web` Grok run offers |
| `AOP_TEST_HERMES_BIN=/path/to/hermes` | Hermes run and exact resume in the `sealed` and `host` profiles against a local OpenAI-compatible server; `AOP_TEST_HERMES_HOME` names the Hermes home whose dependency store and tools to use, by default `~/.hermes` |
| `AOP_TEST_AGY_BIN=/path/to/agy` | Antigravity starts inside the `sealed`, `edit`, and `host` profiles with its tools and working directory, using your signed-in Antigravity home; it sends no prompt |
| `AOP_TEST_MUSE_BIN=/path/to/muse` | Muse Code run and exact resume in the `sealed`, `edit`, and `host` profiles with Muse's offline echo provider; it needs no Meta login |

```sh
AOP_TEST_DSH_BIN=$(command -v dsh) AOP_TEST_HERMES_BIN=$(command -v hermes) AOP_TEST_AGY_BIN=$(command -v agy) \
  uv run --locked python -m unittest discover -s tests -p 'test_*native.py'
```

Run them after updating a harness, before relying on it through AOP.

## Documentation

The site at [agentorchestrationprocess.com](https://agentorchestrationprocess.com/) is built from `README.md` and `docs/*.md`; see `docs-site/README.md`.
