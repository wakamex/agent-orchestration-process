# Token usage and pricing

In brief: every result has a `usage` object with six token counters, an `accounting_status` that says whether they are complete, a `calculated_cost` that prices them at the provider's public API rates, and, when the harness reports one, a `provider_reported_cost`. The calculated cost is a comparison figure, not your bill: runs on a subscription cost nothing extra per token. Usage and cost are null, never zero, when a harness did not report them. The rest of this page defines each field exactly, for programs that read results.

AOP persists token usage under one provider-independent convention. The marker for this convention is `usage_schema: "aop-token-usage-v3"` on `result.json`. Every result records only its own invocation's usage, so consumers can sum results across runs, including runs that resume the same session.

The invariant is:

- `input_tokens` is all processed input, including cached input and cache creation or writes when
  the provider reports them.
- `cached_input_tokens` is the cache-read subset of `input_tokens`.
- `cache_write_input_tokens` is the cache-creation or cache-write subset of `input_tokens`.
- `cache_write_1h_input_tokens` is the 1-hour TTL subset of `cache_write_input_tokens`.
- `output_tokens` is all processed output, including reasoning output.
- `reasoning_output_tokens` is the reasoning subset of `output_tokens`.
- `cached_input_tokens + cache_write_input_tokens <= input_tokens`.
- `cache_write_1h_input_tokens <= cache_write_input_tokens`.
- `reasoning_output_tokens <= output_tokens`.
- `total_tokens = input_tokens + output_tokens`.
- Consumers never add any subset to `input_tokens`, `output_tokens`, or `total_tokens`.

The six persisted counters are non-negative. AOP rejects a newly constructed normalized value when
a subset exceeds its total.

## Deadline accounting

Every result records an `accounting_status` of `complete`, `partial`, or `unavailable`. A normally terminated invocation is `complete`, and its `usage` object may legitimately contain six measured zero counters. A timed-out invocation is `partial` when AOP captured provider usage or monetary evidence before terminating it. An invocation is also `partial` when the harness reported only part of its usage, such as a Muse Code run whose session log does not reconcile with its reported usage. A timed-out invocation is `unavailable` when the provider emitted no accounting evidence, in which case `usage`, `calculated_cost`, and `provider_reported_cost` are null rather than measured zero.

Calculated cost is retained for a partial result only when token usage was observed. Provider-reported cost can be retained independently because it is direct monetary evidence. These rules are adapter-neutral; each adapter only identifies whether its native event or session export contained usage evidence.

Codex app-server emits usage updates scoped to the active turn. AOP sums the `last` usage breakdown from each current-turn update, excluding cumulative totals and updates for previous turns on exact resume. At a deadline, AOP requests native `turn/interrupt` and retains any usage delivered during the bounded teardown grace. ChatGPT subscription authentication does not provide per-run monetary spend, so AOP calculates an API-equivalent comparison from the observed tokens and effective model while leaving `provider_reported_cost` null. If Codex emits no usage for the interrupted turn, accounting remains unavailable rather than falling back to session-history recovery.

Every run also records wall-clock time, time to first provider event, and time to the agent response
when the provider exposes it. Resume results contain only the current invocation's usage rather than
cumulative session totals.

## Provider boundary mappings

The mappings below describe the structured event or retained export that each adapter reads. A plus
sign means the provider reports disjoint buckets that AOP combines into a normalized total.

| Adapter | Provider input shape | Normalized input | Provider output shape | Normalized output |
| --- | --- | --- | --- | --- |
| Agy | each `step_update` carries usage where `input_tokens` is uncached and `cache_read_tokens` is disjoint | sum over this invocation's steps of input + cache read | `output_tokens` includes `thinking_tokens` | sum of output, with thinking as subset |
| Claude | input, cache read, and cache creation are disjoint | input + cache read + cache creation, with cache read and cache creation (including 1h TTL) as subsets | output is total | output |
| Codex | input already includes `cached_input_tokens` | input | output already includes reasoning | output |
| Cursor | input, cache read, and cache write are disjoint | input + cache read + cache write, with cache read and cache write as subsets | output is total | output |
| Devin | ATIF `prompt_tokens` is total and `cached_tokens` is its subset | prompt | ATIF completion is total | completion |
| DeepSeek Harness | Harness input, cache read, and cache write are disjoint for every inference provider | input + cache read + cache write, with cache read and cache write as subsets | Harness output includes reasoning | output, with reasoning as subset |
| Grok | input, cache read, and cache creation are disjoint | input + cache read + cache creation, with cache read and cache creation (including 1h TTL) as subsets | output includes reasoning | output, with reasoning as subset |
| Hermes | session input, cache read, and cache write are disjoint cumulative counters | invocation delta of their sum, with cache read and cache write deltas as subsets | session visible output and reasoning are disjoint cumulative counters | invocation delta of visible output + reasoning |
| OpenCode | input, cache read, and cache write are disjoint | input + cache read + cache write, with cache read and cache write as subsets | output and reasoning are disjoint | output + reasoning, with reasoning as subset |
| Zcode | Native `inputTokens` includes `cacheReadTokens` and `cacheWriteTokens` | input, with cache read and cache write as subsets | Native `outputTokens` includes `reasoningTokens` | output, with reasoning as subset |

Hermes exports cumulative session counters. AOP snapshots the session before and after an invocation
and persists only the component-wise invocation delta. For a Devin resume, AOP selects only agent
steps after the current user prompt. A fresh run whose run-owned ATIF export omits every user step
uses all agent steps from that new private session. The other adapters consume per-invocation
terminal events. Resuming a session therefore does not turn a new run's usage into a cumulative
session total.

Muse Code reports only its main agent's usage in `muse exec --json`. AOP sums every model call of the run, including Muse's reminder agents, from the session log in the task's private state, after checking that its main-agent calls equal the reported usage; see [Harness adapters](harnesses.md#muse-code).

Zcode uses the native terminal `usage` summary for the current turn. If the terminal summary is missing, it sums observed `model_request_completed` measurements for that turn, deduplicated by request and attempt, and marks accounting partial. Timeouts retain these measurements. Cumulative session projection totals are never substituted for per-turn usage. For the canonical Z.ai Coding Plan route, AOP calculates API-equivalent cost for each measured model request at its completion time, including compaction and lite-model calls. A cost is available only when all measured models have prices and the measurements reconcile with reported turn usage. Native credential provenance identifies the subscription route separately; provider-reported monetary cost remains unavailable.


## Pricing

Pricing receives normalized usage only. It prices `input_tokens - cached_input_tokens - cache_write_input_tokens` at the normal
input rate, `cached_input_tokens` at the cache-read rate, `cache_write_input_tokens - cache_write_1h_input_tokens` at the 5-minute cache-write rate, `cache_write_1h_input_tokens` at the 1-hour cache-write rate, and `output_tokens` at the output rate. It never
adds `reasoning_output_tokens` to output. This is algebraically equivalent to the former additive
cache calculation for Agy and direct DeepSeek Harness runs, but no provider-specific pricing flag is
needed.

DeepSeek Harness runs use the same calculation for every inference route.

Cache creation and cache writes count toward total processed input and are priced through `cache_write_input_tokens` and `cache_write_1h_input_tokens`. Provider-reported monetary cost remains separate. On resumed Claude Code runs where the CLI reports cumulative session `modelUsage` and `total_cost_usd`, AOP subtracts the parent run's reported session cost (or falls back to `model-prices` when parent event data is unavailable) so `calculated_cost` reflects only the current turn.

`calculated_cost` is an API-equivalent USD comparison based on normalized tokens and the first-party API rates that the model-prices package records as in effect when each request ran. It does not claim that a subscription account was billed that amount. `provider_reported_cost` remains null unless the harness or provider emits monetary cost evidence for the invocation. The result records billing route evidence separately as `subscription`, `provider-credits`, `metered-api`, `local`, or `unknown` rather than inferring billing from the calculated comparison.

model-prices resolves harness model names such as `opencode/grok-4.7`, `gemini-3.8-flash-medium`, or `deepseek-v4.1-flash` to the lab's own API model and prices it at that lab's rates, including sourced corrections for dates when models.dev was late or wrong and DeepSeek's peak and off-peak hours. AOP passes the adapter's pricing provider as a hint: `openai` for Codex, `zai` for Codex through `zai-coding-plan` and for Zcode, `google` for Agy, `xai` for Grok, the selected inference provider for DeepSeek Harness and Hermes, and no hint for OpenCode, whose model prefix names a reseller. Codex runs through `zai-coding-plan` therefore price the selected GLM model at Z.ai's API rates, while billing records the Z.AI Coding Plan route as `subscription`. A model model-prices does not price leaves calculated cost null rather than guessed.

Each request is priced at its own time when the harness reports one. Zcode reports a completion time for each model request and OpenCode a timestamp for each step, and AOP rejects a reported time outside the run's start and finish as a clock or parsing error. Codex, Agy, Grok, Hermes, and DeepSeek Harness report no per-request time, so AOP prices their usage only when model-prices reports a single price for the whole span from the run's start to its finish. A DeepSeek Harness run that crosses a peak boundary, or any run that spans a price change, therefore has no calculated cost. Long-context tiers apply per request when that request's whole prompt exceeds the tier size.

Every calculated cost records `pricing_basis`, the `model_prices.pricing_basis()` identifier naming the package version, models.dev commit, and a hash of all price data; `pricing_version`, the model-prices package version; `pricing_source`, the models.dev commit or correction source of each rate used; `priced_as`, the resolved `provider/model` pairs; `removed_suffix`, any text such as `-max` or `-2026-04-23` that model-prices removed from the model name to find a price; and `pricing_period`, which is `peak`, `off_peak`, or `mixed` under time-of-day pricing and null otherwise. Before `run`, `resume`, `batch`, `integrate`, and `models`, AOP asks model-prices for its newest published price data, downloading it when the cached copy is more than a day old. model-prices publishes data daily, separately from package releases, and caches it in `~/.cache/model-prices` or `MODEL_PRICES_CACHE`. A failed download never blocks a run: AOP warns and uses the newest cached or bundled data, and each cost's `pricing_basis` records which data priced it. AOP also warns when the data in use was last confirmed current more than three days ago. `aop models --refresh` downloads the newest data regardless of the cache's age.

Provider inventories and cost evidence differ. Claude can supply its own CLI-calculated comparison;
OpenCode can report billed per-step cost; Grok can report terminal cost; and Hermes can report cost
through its selected inference provider. Cursor and Devin do not currently expose enough
information for AOP to calculate an API-equivalent cost, so their calculated cost remains null.

## Results archive

Every result AOP writes to `.aop/runs/RUN_ID/result.json` is also written, with the same content, to `$XDG_STATE_HOME/aop/results/RUN_ID.json` (by default `~/.local/state/aop/results/`). The archive keeps a run's usage and cost after the directory holding its `.aop` is deleted, such as a benchmark's per-question profile. Each result's `state_root` field names the `.aop` directory the run was recorded in, exactly as it appears in `state-roots`, so a consumer can attribute an archived run to its project after that directory is gone. The archive holds one current record per run at its top level; originals that a migration replaced are kept in its `originals` subdirectory.

## Migrating older run records

AOP reads only current run records. Loading an older `result.json` or `request.json` fails with a message to run the one-time migration:

```sh
aop migrate-results --dry-run
aop migrate-results
aop migrate-results /code/project/.aop/runs ~/other/.aop/runs
```

Without arguments, the command migrates every state directory listed in `$XDG_STATE_HOME/aop/state-roots` (by default `~/.local/state/aop/state-roots`) and the current project's `.aop/runs`. AOP adds a state directory to that list whenever it records a run there, and the migration adds every directory it migrates. `--dry-run` prints, for each run directory, how many records each conversion would change, and writes nothing.

The migration also maintains the results archive. It copies each current result into the archive when the archived copy is missing or differs, and migrates archived results whose run directory no longer exists in place. Finally it rewrites `state-roots` without duplicate entries or directories that no longer exist.

For each record the migration rewrites, it keeps the original byte for byte beside it as `result.pre-v3.json` or `request.pre-v3.json` and adds a `migration` object naming the original schema, the conversions applied, the original's file name, and the migration time. Raw provider artifacts such as `events.jsonl` are never changed. Running the migration again skips records already in the current format.

Results convert as follows:

| Original record | Conversion |
| --- | --- |
| Unversioned Agy, Cursor, and DeepSeek Harness results (every inference provider) | `input_tokens += cached_input_tokens`, because these adapters persisted the provider's disjoint uncached input bucket |
| Unversioned OpenCode results | `output_tokens += reasoning_output_tokens` |
| Unversioned and `aop-token-usage-v1` results | accounting status derived from the record: a run that did not time out becomes `complete`; a timed-out run becomes `partial` when it has usage or a provider-reported cost, and `unavailable` otherwise |
| `aop-token-usage-v2` results of providers other than Agy | marker changed to v3; usage unchanged |
| Every Agy result before v3 | usage recomputed from the `step_update` usage in the run's own `events.jsonl`, because Agy's terminal `result` event reports the conversation's running total across every run that resumed it; without usage events, usage and costs become unavailable |
| Claude, Cursor, DeepSeek Harness, Grok, and OpenCode results with cache writes in `events.jsonl` | `cache_write_input_tokens` and `cache_write_1h_input_tokens` recovered from `events.jsonl` when `result.json` predates those fields |
| Resumed Claude results with cumulative `modelUsage` in `events.jsonl` | `calculated_cost` recomputed as the single-turn delta from the parent run's reported session cost (or `model-prices` when parent events are absent) |
| Results without `state_root` | `state_root` set to the `.aop` directory that holds the run directory |
| `api_equivalent_cost`, `billing.actual_cost_known`, `calculated_cost.estimated`, `read_paths` | folded into `calculated_cost` or `provider_reported_cost`, removed, or renamed to `inputs` |

When a conversion changes a result's usage, the migration reprices `calculated_cost` with model-prices at the run's own time and records the price data in `pricing_basis`. A run the model-prices data cannot price for that whole span gets a null cost. Results whose usage is unchanged keep their stored cost.

Requests convert `sandbox` to `profile` and `read_paths` to `inputs`, and fill fields added after they were written with their defaults: `mode: "agent"`, `no_web: false`, an empty `effective_policy`, `artifacts`, and `inputs`, and null inference provider and route. A `sandbox` value is a Codex sandbox mode that no execution profile resolves, so resuming such a run fails while cleanup works.

## Reading results from other programs

Programs that read AOP results should require `usage_schema: "aop-token-usage-v3"` and inspect `accounting_status` before reading usage. For `complete` or usage-bearing `partial` results, use the six counters as stored and calculate uncached input as `input_tokens - cached_input_tokens - cache_write_input_tokens` and total tokens as `input_tokens + output_tokens`. Never add cached input, cache writes, or reasoning output to `input_tokens`, `output_tokens`, or `total_tokens`. Each result records only its own invocation's usage, so results can be summed across runs. For `unavailable` results, keep usage and cost unknown rather than substituting zero. Records with other schema markers should be migrated with `aop migrate-results` rather than converted by the consumer. To read every AOP result on a machine, including runs whose state directory was deleted, read `$XDG_STATE_HOME/aop/results/*.json`, deduplicate by `run_id`, and take the project from `state_root`. `$XDG_STATE_HOME/aop/state-roots` lists the state directories that still exist, for consumers that also need run directories such as `events.jsonl`.
