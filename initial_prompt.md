# Footval — Implementation Spec

A subjective, multi-model LLM evaluation. Candidate models answer `footval/footval.prompt.md`; their responses are auto-checked and judged by an LLM panel through a **pairwise head-to-head tournament** (every pair of responses compared on two reasoning criteria — analytical and intuitive — each a forced A/B choice over its own grid row), then published to a public web page as win-percentage rankings.

---

## 1. Configuration

Put these at the top of the pipeline as editable variables.

```yaml
candidate_models:
  - claude-fable-5
  - claude-opus-5
  - gpt-5.6-sol
  - gpt-5.6-terra
  - gemini-3.1-pro-preview
  - gemini-3.6-flash
  - glm-5.2

# Pairwise judge panel — the only judging round. All three judges are candidates.
pairwise:
  judges:
    - claude-fable-5
    - gpt-5.6-sol
    - glm-5.2
  both_orders: true         # judge every pair in both presentation orders

n_samples: 1              # responses generated per candidate
gen_temperature: 1.0      # fixed for all candidates
judge_temperature: 0.0    # judges run as deterministically as possible
seed: 42                  # where the provider supports it
interval_tol: 0.10        # relative tolerance for parameter-reproduction check
```

**Thinking / effort policy.** Candidates run one notch above the middle of each provider's
effort ladder; judges run one notch above the candidates. All three judges also appear on
the candidate roster, so each model entry carries a candidate `params` block and an optional
`judge_params` block that is merged over it when that model judges.

| Provider | Ladder | Candidate | Judge |
|---|---|---|---|
| Anthropic (`output_config.effort`) | low→medium→high→xhigh→max | `high` | `xhigh` |
| OpenAI (`reasoning.effort`) | low→medium→high→xhigh→max | `high` | `xhigh` |
| Gemini (`thinking_config.thinking_level`) | minimal→low→medium→high | `HIGH` | — (no Gemini judge) |
| z.ai GLM (`reasoning_effort`) | high→max | `high` | `max` |

Pin the exact API snapshot ID for every model at run time and store it (the version strings above will drift).

---

## 2. Pipeline overview

```
Stage 1  Generate     candidate responses  (7 models x n_samples)
Stage 2  Auto-check   structural + numeric consistency of the JSON
Stage 3  Tables       consolidate the objective per-instance artifacts into tables
Pairwise Judge        3 judges compare every pair head-to-head on 2 criteria (forced A/B each)
Stage 4  Publish      web page on shaneorr.me (win-percentage rankings)
```

The pairwise judging round runs on its own batch lifecycle (Section 5), not as part of the
linear `run`. Each stage reads and writes structured artifacts (Section 7) so the run is
reproducible and the website is a pure view over stored data.

> **Revision (post-second-run).** The original spec had a Stage 2 that executed a `plot_script`
> from each response inside a Docker sandbox and rendered figures. Part 3 of the candidate
> prompt (the plotting script) was removed, so there is nothing left to execute: the sandbox
> stage, the container image, and the `exec_results` table were deleted and the later stages
> renumbered. Reinstating code execution means restoring Part 3 to the prompt first.

---

## 3. Stage 1 — Generate responses

For each candidate model, generate `n_samples` independent responses to `footval.prompt.md` at `gen_temperature`.

- Treat each (model, sample) pair as a distinct **response instance** with its own `instance_id`. All later stages operate on instances.
- Capture and store: model snapshot ID, timestamp, temperature, seed, raw text, and the parsed JSON object.
- Do not retry or repair malformed output; a malformed response is a real signal and is handled by Stage 2.
- Mark the prompt as cacheable. It is byte-identical across every call, so repeated calls
  (`n_samples > 1`, resumed runs, re-runs inside the cache TTL) should read from the
  provider's prompt cache rather than re-paying for the prefix.

API keys are in `footval/.env` and have the following names:

- Anthropic — auto-read by the anthropic SDK
`ANTHROPIC_API_KEY`

- OpenAI — auto-read by the openai SDK
`OPENAI_API_KEY`

- Google Gemini — auto-read by the google-genai SDK
`GEMINI_API_KEY`

- Z.ai (GLM) — convention only; read explicitly by the OpenAI-compatible adapter
`ZAI_API_KEY`

---

## 4. Stage 2 — Automated checks

Deterministic checks on the parsed JSON. These are objective and should not be left to judge opinion; their results are surfaced on the site, and the subset relevant to the judged criterion is passed to judges as factual context.

Per instance, record pass/fail (with detail) for:

1. **JSON valid** — parses, top-level `quantity` / `strategies` present.
2. **Grid complete** — exactly 6 strategies, one per (`analytics`/`intuition` × `game_management`/`offensive`/`defensive`) cell, with distinct titles.
3. **Interval sanity** — for each prior: `low < high` and `low ≤ most_plausible_value ≤ high`.
4. **Valid family** — `name ∈ {normal, student_t, skew_normal}` (signed-support families only).
5. **Parameter reproduction** — recompute the 95% interval from the reported parameters and check it matches the reported `interval_95` within `interval_tol`:
   - normal: `μ ± 1.96·σ`
   - student_t: `μ ± t_{0.975,ν}·σ`
   - skew_normal: `skewnorm.ppf(0.025/0.975, a=α, loc=ξ, scale=ω)`
   Also check the family's center (`μ`, or mode for skew_normal) ≈ `most_plausible_value`.

Store a per-instance check report. These are diagnostics, **not** a separate scored criterion.

Checks 3–5 concern the priors, which are no longer judged (§5.4). They still run and are
still published, but they are **withheld from judges**: only checks 1 and 2 reach the judge
prompt, so a prior-mechanics failure cannot leak into either reasoning verdict.

---

## 5. Judging — pairwise head-to-head tournament

The only judging round. Every unordered pair of **judged** response instances is compared by
the three-judge panel on **two criteria — analytical reasoning and intuitive reasoning** —
each as its own forced A/B call scoped to just the grid row that criterion covers. There is no
numeric scale and no separate per-response score. `pairwise.judged_samples` limits which
sample indices are judged (currently only `s0`); additional samples are generated, checked,
and published as unjudged artifacts so readers can see draw-to-draw variability.

The construct being measured is **reasoning quality**, not knowledge of the current NFL meta:
judges are instructed to judge the inference rather than the conclusion, and
"underutilized" is treated as a case the response must argue (why would staffs underuse
this?), not a fact the judge adjudicates from its own — possibly stale — picture of league
practice.

### 5.1 Inputs to each comparison

Each judge call shows **two anonymized responses** ("Response A" and "Response B") for the same
task, reduced to the row the call's criterion covers. Each call provides:

- `footval.prompt.md` (the task both candidates were given).
- The two **anonymized, row-scoped** response bundles (strip any model-identifying text; label
  only as "Response A" / "Response B"): for the analytical call only the `analytics`-row
  strategies, for the intuitive call only the `intuition`-row strategies, each as
  pretty-printed JSON with every strategy's `prior` block removed (§5.4) — or as raw,
  unmodified text if the response failed to parse.
- The Stage 2 automated check report for each response — restricted to the JSON-validity and
  grid facts (which describe the full six-cell grid) — as ground truth for the mechanical
  facts.

Row-scoping is what makes the two criteria separable: a strong analytics row cannot
halo-carry a weak intuition row, or vice versa.

The round is **text-only**. Which response is shown as "A" is a deterministic per-pair coin
flip from the run seed, shared by both criteria of the pair; with `both_orders: true` every
pair is also judged in the reversed order so position bias is measurable.

Prompts are assembled most-stable-first — `[system][task][bundle A][bundle B][instruction]` —
with explicit cache breakpoints after the task and after bundle A, and requests are sorted so
calls sharing a criterion (same system prompt) and the same candidate-A are adjacent. Volatile
content (a format-retry reminder) is appended after the last breakpoint so it never
invalidates the cached prefix.

### 5.2 Output from each comparison

Each call returns a verdict under its own criterion key (`analytical_reasoning` or
`intuitive_reasoning`):

```json
{
  "analytical_reasoning": {"winner": "A" | "B", "justification": "1-2 sentences"}
}
```

Forced choice: the judge **must** pick "A" or "B"; ties are not allowed. Instruct judges to use
the check reports as ground truth for the mechanical facts and to spend their judgment on
reasoning quality.

### 5.3 Quality descriptions

Give judges a stronger-vs-weaker description per dimension (no numeric anchors); the judge
picks whichever response better fits the "stronger" description. Both rubrics carry the same
spine: *judge the inference, not the conclusion* — do not penalize a conclusion the judge
disagrees with if the reasoning is sound, and do penalize invalid reasoning even when the
judge agrees with where it lands.

**Criterion 1 — Analytical reasoning** (over the `analytics` row; prompt:
`footval.judge.analytical.prompt.md`)

Five dimensions, in the priority order that decides close calls:

1. **Faithful use of evidence** — the pick is a claim public win-probability / expected-points
   work genuinely speaks to, described in terms the research would recognize, with a valid
   inference from it; misstating or overclaiming the research is the failure.
2. **Credibly win-positive mechanism** — a traced causal path to more wins with the mechanism
   named, honest about costs and conditions. Candor about a thin pick is correct behavior,
   not a defect.
3. **Situational precision** — concrete thresholds and situations, and where the edge does
   not apply; blanket advice naming no behavior change is the failure.
4. **The case for under-adoption** — the quality of the response's argument for *why*
   adoption lags the evidence (incentives, career risk, variance aversion, inertia); judges
   must not decide this from their own belief about current league practice unless the claim
   is flagrantly wrong.
5. **Engagement with the obvious counterargument.**

**Criterion 2 — Intuitive reasoning** (over the `intuition` row; prompt:
`footval.judge.intuitive.prompt.md`)

Five dimensions, in the priority order that decides close calls:

1. **Genuine intuition** — a real position on a question public data is absent, silent, or
   mixed about, with explicit reasoning why it qualifies; analytics consensus with the label
   swapped is the failure.
2. **Plausibility of the causal story** — a mechanism a coordinator would recognize that
   would survive an actual NFL Sunday.
3. **Originality without fantasy** — a specific, actionable behavior change that is neither
   the league default nor uninstallable fiction.
4. **The case for under-adoption** — same reframe and guard as the analytical criterion.
5. **Epistemic honesty** — speculation labeled as speculation, failure conditions stated,
   confidence proportional to the argument.

Judges are told not to reward length, formatting polish, confident tone, unverifiable
statistics, or exotic-but-unworkable cleverness.

> **Revision (post-third-run).** Judging previously used a single "soundness of
> recommendations" criterion over the whole response, whose top-priority dimension —
> "genuinely underutilized, judged against current adoption" — made the round a test of
> knowledge recency rather than reasoning: candidates lost for reasoning soundly from a
> premise that had aged out of contrarian status, and judges adjudicated "current adoption"
> from their own equally cutoff-limited knowledge. The criterion was replaced by the two
> row-scoped reasoning criteria above, with under-adoption reframed as an argued case. At the
> same time the panel dropped `gemini-3.5-flash` (judges are now exactly the three
> dual-roster models) and the candidate roster dropped `claude-sonnet-5`. Runs before and
> after this change are not comparable.

### 5.4 Priors are generated but not judged

The candidates still produce a Bayesian prior per strategy (Part 2 of the prompt), it is still
checked in Stage 2, and it is still published as the prior-comparison chart. It is **not** a
judged criterion. Three mechanisms keep it out of the verdicts: both judge prompts mark it
explicitly out of scope, the judge-facing check summary omits every prior-mechanics result,
and the `prior` blocks themselves are removed from the parsed-JSON bundles judges see (a
response that failed to parse is shown raw and whole, and the judge prompts say to ignore any
prior content there). Withholding the blocks — rather than instructing judges past them —
also cuts the bundles' uncached tokens on every judge call.

> **Revision (post-second-run).** Judging previously had a second criterion, "Reasonableness of
> Bayesian priors". Even after the first-run fix that demoted parameter reproduction to a
> tiebreaker, the criterion had no ground truth to appeal to — there is no true per-game points
> value for a strategy, so judges could only reward priors matching their own unverified
> intuitions, and they anchored on the automated report handed to them. The criterion was
> removed rather than repaired. Runs before and after that change are not comparable.

### 5.5 Self- and family-comparison

All three judges are candidates themselves, so a judge will sometimes compare its own or a
sibling model's response. Keep **all** verdicts; tag each with `is_self_a`/`is_self_b` (same
exact model) and `judge_family`. Handle self-preference as a diagnostic in Section 6 rather
than by dropping data here.

### 5.6 Cost design

Comparisons go through the providers' 50%-off batch APIs where available (Anthropic, OpenAI;
the Gemini batch adapter remains for a future Gemini judge) and run synchronously for z.ai/GLM
(no batch API). Prompts share byte-stable prefixes so provider prompt caching applies, with
explicit `cache_control` breakpoints on Anthropic and a stable per-criterion
`prompt_cache_key` on OpenAI. The lifecycle is submit → status → collect → (resubmit
stragglers) → csv, with `outstanding = expected − verdict files` so every step is idempotent.

---

## 6. Aggregate — win tallies and judge diagnostics

### 6.1 Win percentages and rankings

- **Comparison count** per model = number of head-to-head comparisons it appeared in (across all judges, pairs, and presentation orders).
- **Win count / win percentage** per model = wins ÷ comparisons × 100.
- **Rankings** = sort models by win percentage.

### 6.2 Judge-agreement diagnostics

Reported for the lay audience as panel-consensus readouts rather than formal reliability statistics:

- **Vote splits** — for each ordered pair and criterion, how decisively the three judges agreed (unanimous or 2–1 majority).
- **Order consistency** — whether a judge picked the same canonical winner after the two responses were swapped (A↔B); low consistency flags sensitivity to presentation order.

### 6.3 Self-preference diagnostic

For each judge, compare the win rate it awarded to **own-family** candidates vs. **other** candidates. A large positive gap flags self-preference and belongs in the limitations section.

---

## 7. Data artifacts

Store as JSON or a small SQLite DB so every stage is auditable and the website reads from it.

Objective tables (the `tables` stage):

| Table | Key fields |
|---|---|
| `responses` | instance_id, model, sample_idx, snapshot_id, ts, temperature, seed, usage (provider token counts, incl. reasoning/thinking where reported), billable_output_tokens, output_cost_usd, raw_text, parsed_json |

The tables stage also writes `outputs/data/candidate_costs.csv` — per candidate: samples,
billable output tokens (visible output + separately-reported thinking), the model's
`output_usd_per_mtok` list price from config.yaml, and the total output cost in USD. A model
with no configured price gets a blank cost, never a silent zero.
| `checks` | instance_id, json_valid, grid_complete, intervals_ok, family_ok, params_reproduce_ok, detail |

Pairwise judging artifacts (under `artifacts/pairwise/`):

| Artifact | Key fields |
|---|---|
| `manifest` | seed, both_orders, judges, instances, comparisons (custom_id → pair, order, instance/model A & B) |
| `verdicts/{judge}/{custom_id}` | judge, judge_family, model_a, model_b, order, is_self_a, is_self_b, verdicts (winner per criterion), raw_text |
| `pairwise_results.csv` | judge, model_a, model_b, criterion, winner |

---

## 8. Stage 4 — Publish (shaneorr.me → Other Writings)

Publish as a blog post under 'Other Writings' (`https://www.shaneorr.me/writings`)

Audience is non-technical. The page reads from the Section 7 artifacts.

**Sections:**

1. **Plain-language intro** — what the test asked the models to do, in one short paragraph. Explain "prior" and "uncertainty interval" with a one-line analogy (e.g., a prior is the model's best guess plus an honest statement of how unsure it is).
2. **Leaderboard** — win-percentage ranking (head-to-head wins ÷ comparisons). Alongside, show the judge-agreement diagnostics (vote-split mix and order consistency) with a plain label so the ranking isn't read as more precise than the panel's agreement justifies. Use visualizations.
3. **Per-model response viewer** — each model's full response, JSON collapsed by default.
4. **Prior comparison** — **re-plot from the reported parameters.** Parse each model's family + parameters from the JSON and redraw every prior with one canonical routine on shared axes and shared styling. Lay out as small multiples by model. This is the clean payoff of the structured footval schema. Note plainly that priors are shown but not scored.
5. **Methodology & limitations** — disclose: every judge is also a contestant (self-preference risk, with the Section 6.3 gap shown); `n_samples` is small; the reasoning rubrics reduce but cannot eliminate the pull of the judges' own football opinions; the priors the prompt also asks for are charted but not judged.

Interactive tabs/toggles are encouraged where they aid clarity (model toggle on the comparison view).

---

## 9. Decisions deliberately left open

- `n_samples` and `gen_temperature` are set as defaults; adjust to taste and budget.
- The footval prompt leaves the plausible effect-size **scale** unspecified on purpose. That was originally so the priors criterion could test whether a model picks sane magnitudes; with priors unjudged it now only affects the published prior chart. Keep it that way unless you decide to anchor it.
