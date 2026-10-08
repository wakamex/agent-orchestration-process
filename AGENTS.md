# Repository guidance

## Design principles

- Describe complete semantic boundaries. Write access, visibility, identity, instructions,
  environment, credentials, and network access are separate facts. Enforce and record each one
  explicitly.
- Delegate harness-native behavior. Authentication, model execution, tools, session persistence,
  and provider protocols stay with the harness unless its automation interface lacks a required
  capability.
- Project the least authority needed. A task receives only the selected credentials and runtime
  state, not an entire multi-provider credential store.
- Do not create parallel authentication for users. Reuse harness-managed credentials and preserve
  native environment overrides instead of introducing an AOP-specific secret store.
- Keep controller policy outside model control. Isolation, deadlines, evidence, cleanup, and
  integration remain AOP responsibilities even when harness permission prompts are bypassed.
- Use "AOP sandbox" for AOP's isolation layer and "harness sandbox" for a harness's native isolation layer. Use "outer" and "inner" only when the nesting itself matters.
- Prefer small native extension points over wrappers and forks. When a harness lacks a bounded
  automation feature, use its documented plugin or protocol surface and keep the added layer narrow.
- Fail closed and record what ran. Ambiguous identity, stale pricing, invalid credentials,
  incomplete terminal output, and unenforced policy are failures. Effective policy and provenance
  remain inspectable after the run.
- Optimize for the long-term product rather than accidental local setup. A clean break is acceptable
  before users exist when it removes ambiguity or avoids a foreseeable rewrite.
- Keep one current format for run records. Raw provider artifacts such as `events.jsonl` and
  `stderr.log` are never rewritten. Derived records such as `result.json` and `request.json` may be
  rewritten when an older format is wrong or obsolete, but only through an explicit, one-time
  migration command that is documented in `docs/`, preserves each original file beside the
  rewritten one, records the conversion it applied in the rewritten record, and is idempotent. After
  a migration, readers accept only the current format instead of converting old formats in memory.

## Release policy exceptions

- Validation uses this repository's `.github/workflows/validation.yml` instead of the shared
  `python-validation.yml`. AOP's tests run harnesses inside bubblewrap, so the job installs
  bubblewrap and allows unprivileged user namespaces before running the same
  `python -m unittest discover -s tests` command, and it also runs `ruff check`. Publishing and
  GitHub Releases use the shared actions.
