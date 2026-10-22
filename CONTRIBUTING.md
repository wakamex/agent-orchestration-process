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

```sh
AOP_TEST_DSH_BIN=$(command -v dsh) uv run --locked python -m unittest discover -s tests -p 'test_*native.py'
```

Run them after updating a harness, before relying on it through AOP.

## Documentation

The site at [agentorchestrationprocess.com](https://agentorchestrationprocess.com/) is built from `README.md` and `docs/*.md`; see `docs-site/README.md`.
