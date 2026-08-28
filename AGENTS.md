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
- Prefer small native extension points over wrappers and forks. When a harness lacks a bounded
  automation feature, use its documented plugin or protocol surface and keep the added layer narrow.
- Fail closed and record what ran. Ambiguous identity, stale pricing, invalid credentials,
  incomplete terminal output, and unenforced policy are failures. Effective policy and provenance
  remain inspectable after the run.
- Optimize for the long-term product rather than accidental local setup. A clean break is acceptable
  before users exist when it removes ambiguity or avoids a foreseeable rewrite.

## Release validation

Before creating a release tag, push the intended release commit to `main` without the tag and require a successful `Validate` push workflow for that exact commit SHA. Find the run with `gh run list --workflow validation.yml --commit "$release_sha" --event push`, wait for it with `gh run watch RUN_ID --exit-status`, and confirm `origin/main` still resolves to the same SHA. A tag-triggered validation run is not a substitute for this pre-tag gate. Create and push the annotated tag only after the gate passes.
