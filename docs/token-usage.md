# Token usage and pricing

AOP persists token usage under one provider-independent convention. The marker for this convention
is `usage_schema: "aop-token-usage-v1"` on `result.json`.

The invariant is:

- `input_tokens` is all processed input, including cached input and cache creation or writes when
  the provider reports them.
- `cached_input_tokens` is the cache-read subset of `input_tokens`.
- `output_tokens` is all processed output, including reasoning output.
- `reasoning_output_tokens` is the reasoning subset of `output_tokens`.
- `cached_input_tokens <= input_tokens`.
- `reasoning_output_tokens <= output_tokens`.
- `total_tokens = input_tokens + output_tokens`.
- Consumers never add either subset to `input_tokens`, `output_tokens`, or `total_tokens`.

The four persisted counters are non-negative. AOP rejects a newly constructed normalized value when
a subset exceeds its total.

Every run also records wall-clock time, time to first provider event, and time to the agent response
when the provider exposes it. Resume results contain only the current invocation's usage rather than
cumulative session totals.

## Provider boundary mappings

The mappings below describe the structured event or retained export that each adapter reads. A plus
sign means the provider reports disjoint buckets that AOP combines into a normalized total.

| Adapter | Provider input shape | Normalized input | Provider output shape | Normalized output |
| --- | --- | --- | --- | --- |
| Agy | `input_tokens` is uncached and `cache_read_tokens` is disjoint | input + cache read | `output_tokens` includes `thinking_tokens` | output, with thinking as subset |
| Claude | input, cache read, and cache creation are disjoint | input + cache read + cache creation | output is total | output |
| Codex | input already includes `cached_input_tokens` | input | output already includes reasoning | output |
| Cursor | input, cache read, and cache write are disjoint | input + cache read + cache write | output is total | output |
| Devin | ATIF `prompt_tokens` is total and `cached_tokens` is its subset | prompt | ATIF completion is total | completion |
| DeepSeek Harness | Harness input, cache read, and cache write are disjoint for every inference provider | input + cache read + cache write | Harness output includes reasoning | output, with reasoning as subset |
| Grok | input, cache read, and cache creation are disjoint | input + cache read + cache creation | output includes reasoning | output, with reasoning as subset |
| Hermes | session input, cache read, and cache write are disjoint cumulative counters | invocation delta of their sum | session visible output and reasoning are disjoint cumulative counters | invocation delta of visible output + reasoning |
| OpenCode | input, cache read, and cache write are disjoint | input + cache read + cache write | output and reasoning are disjoint | output + reasoning, with reasoning as subset |

Hermes exports cumulative session counters. AOP snapshots the session before and after an invocation
and persists only the component-wise invocation delta. For a Devin resume, AOP selects only agent
steps after the current user prompt. A fresh run whose run-owned ATIF export omits every user step
uses all agent steps from that new private session. The other adapters consume per-invocation
terminal events. Resuming a session therefore does not turn a new run's usage into a cumulative
session total.

## Pricing

Pricing receives normalized usage only. It prices `input_tokens - cached_input_tokens` at the normal
input rate, `cached_input_tokens` at the cache rate, and `output_tokens` at the output rate. It never
adds `reasoning_output_tokens` to output. This is algebraically equivalent to the former additive
cache calculation for Agy and direct DeepSeek Harness runs, but no provider-specific pricing flag is
needed.

Unversioned records keep their persisted historical `calculated_cost` when loaded. AOP does not
retroactively recalculate or rewrite that evidence. New normalized DeepSeek Harness runs use the
same calculation for every inference route. This also fixes older non-DeepSeek Harness calculations
that treated its disjoint uncached bucket as though it already included cache reads.

Cache creation and cache writes count toward total processed input. AOP currently has only a cached
read subset and no separate persisted cache-write pricing bucket, so calculated cost continues to use
the normal input rate for those tokens. Provider-reported monetary cost remains separate.

`calculated_cost` is an API-equivalent USD comparison based on normalized tokens and a versioned
price schedule. It does not claim that a subscription account was billed that amount.
`provider_reported_cost` remains null unless the harness or provider emits monetary cost evidence
for the invocation. The result records billing route evidence separately as `subscription`,
`provider-credits`, `metered-api`, `local`, or `unknown` rather than inferring billing from the
calculated comparison.

Codex runs through `zai-coding-plan` price the selected GLM model against the models.dev `zai` provider. The calculated amount remains an API-equivalent comparison, while billing records the Z.AI Coding Plan route as `subscription` and the selected environment credential name without persisting its value. AOP does not report that comparison as actual spend, quota consumption, or provider credits.

Before dispatch, AOP requires its models.dev catalog cache to be less than 24 hours old. A stale or
missing cache is refreshed under a process lock and replaced atomically. If refresh fails, AOP does
not use expired prices. Set `AOP_MODEL_CATALOG_CACHE` to relocate the cache, or run
`aop models --refresh` to refresh it immediately. Every calculated cost retains its pricing source,
retrieval time when applicable, and price version.

Provider inventories and cost evidence differ. Claude can supply its own CLI-calculated comparison;
OpenCode can report billed per-step cost; Grok can report terminal cost; and Hermes can report cost
through its selected inference provider. Cursor and Devin do not currently expose enough
information for AOP to calculate an API-equivalent cost, so their calculated cost remains null.

## Loading unversioned results

Unversioned `result.json` records predate the common convention and use mixed meanings. AOP leaves
those files and their raw provider artifacts unchanged. `RunResult.from_dict()` applies this
centralized in-memory conversion and marks the loaded object as `aop-token-usage-v1`:

| Legacy provider | In-memory conversion |
| --- | --- |
| Agy | `input_tokens += cached_input_tokens` |
| Cursor | `input_tokens += cached_input_tokens` |
| DeepSeek Harness, all inference providers | `input_tokens += cached_input_tokens` |
| OpenCode | `output_tokens += reasoning_output_tokens` |
| Claude, Codex, Devin, Grok, Hermes | no arithmetic conversion |

Historical Cursor results did not retain cache-write tokens in `result.json`, and historical
DeepSeek Harness results did not retain the optional Harness cache-write bucket. Their cache-read
input is reconstructed exactly, but any unrecorded cache writes cannot be recovered without reading
raw artifacts. The loader does not consult or alter those artifacts.

## Downstream consumer rule

Consumers such as `clanker-analytics` should require
`usage_schema: "aop-token-usage-v1"`, use the four counters as stored, and calculate total tokens as
`input_tokens + output_tokens`. They must never add cached input or reasoning output. Missing or
unknown schema markers should be rejected instead of guessed.

Legacy conversion remains centralized in AOP's `RunResult` loader for unversioned records that have
not been migrated by their owner. Downstream consumers do not need to reproduce the provider table.
