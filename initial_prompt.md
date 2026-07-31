# Footval — Implementation Spec

A subjective, multi-model LLM evaluation. Candidate models answer `footval/footval.prompt.md`; their responses are auto-checked and judged by an LLM panel through a **pairwise head-to-head tournament** (every pair of responses compared on soundness, forced A/B choice), then published to a public web page as win-percentage rankings.

---

## 1. Configuration

Put these at the top of the pipeline as editable variables.

```yaml
candidate_models:
  - claude-fable-5
  - claude-opus-5
  - claude-sonnet-5
  - gpt-5.6-sol
  - gpt-5.6-terra
  - gemini-3.1-pro-preview
  - gemini-3.6-flash
  - glm-5.2

# Pairwise judge panel — the only judging round.
pairwise:
  judges:
    - claude-fable-5
    - gpt-5.6-sol
    - gemini-3.5-flash
    - glm-5.2
  both_orders: true         # judge every pair in both presentation orders

n_samples: 1              # responses generated per candidate
gen_temperature: 1.0      # fixed for all candidates
judge_temperature: 0.0    # judges run as deterministically as possible
seed: 42                  # where the provider supports it
interval_tol: 0.10        # relative tolerance for parameter-reproduction check
```

**Thinking / effort policy.** Candidates run one notch above the middle of each provider's
effort ladder; judges run one notch above the candidates. Three models appear on both
rosters, so each model entry carries a candidate `params` block and an optional
`judge_params` block that is merged over it when that model judges.

| Provider | Ladder | Candidate | Judge |
|---|---|---|---|
| Anthropic (`output_config.effort`) | low→medium→high→xhigh→max | `high` | `xhigh` |
| OpenAI (`reasoning.effort`) | low→medium→high→xhigh→max | `high` | `xhigh` |
| Gemini (`thinking_config.thinking_level`) | minimal→low→medium→high | `HIGH` | `HIGH` |
| z.ai GLM (`reasoning_effort`) | high→max | `high` | `max` |

Gemini's ladder tops out at `HIGH`, so its judge cannot sit a tier above the candidates —
a documented deviation, not an oversight.

Pin the exact API snapshot ID for every model at run time and store it (the version strings above will drift).

---

## 2. Pipeline overview

```
Stage 1  Generate     candidate responses  (8 models x n_samples)
Stage 2  Auto-check   structural + numeric consistency of the JSON
Stage 3  Tables       consolidate the objective per-instance artifacts into tables
Pairwise Judge        4 judges compare every pair head-to-head on 1 criterion (forced A/B)
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

Checks 3–5 concern the priors, which are no longer judged (§5.3). They still run and are
still published, but they are **withheld from judges**: only checks 1 and 2 reach the judge
prompt, so a prior-mechanics failure cannot leak into the soundness verdict.

---

## 5. Judging — pairwise head-to-head tournament

The only judging round. Every unordered pair of response instances is compared by the four-judge
panel on **one criterion — soundness of the recommendations** — as a forced A/B choice. There is
no numeric scale and no separate per-response score.

### 5.1 Inputs to each comparison

Each judge call shows **two anonymized responses** ("Response A" and "Response B") for the same
task. Each call provides:

- `footval.prompt.md` (the task both candidates were given).
- The two **anonymized** response bundles (strip any model-identifying text; label only as
  "Response A" / "Response B"), each as pretty-printed JSON (or raw text if it failed to parse).
- The Stage 2 automated check report for each response — restricted to the JSON-validity and
  grid facts — as ground truth for the mechanical facts.

The round is **text-only**. Which response is shown as "A" is a deterministic per-pair coin flip
from the run seed; with `both_orders: true` every pair is also judged in the reversed order so
position bias is measurable.

Prompts are assembled most-stable-first — `[system][task][bundle A][bundle B][instruction]` —
with explicit cache breakpoints after the task and after bundle A, and requests are sorted so
same-candidate-A calls are adjacent. Volatile content (a format-retry reminder) is appended
after the last breakpoint so it never invalidates the cached prefix.

### 5.2 Output from each comparison

```json
{
  "soundness": {"winner": "A" | "B", "justification": "1-2 sentences"}
}
```

Forced choice: the judge **must** pick "A" or "B"; ties are not allowed. Instruct judges to use
the check reports as ground truth for the mechanical facts and to spend their judgment on
subjective quality. Judges weigh distinct *kinds* of problems rather than the raw count of
affected cells (one idea reused across two cells is one defect, not two).

### 5.3 Quality description

Give judges a stronger-vs-weaker description (no numeric anchors); the judge picks whichever
response better fits the "stronger" description.

**Criterion 1 — Soundness of recommendations**

Five dimensions, in the priority order that decides close calls:

1. **Genuinely underutilized** — picks NFL staffs demonstrably still don't do, judged against
   *current* adoption; a familiar-but-rarely-executed idea qualifies, a league-standard one
   does not, and vague "be more aggressive" filler names no behavior change.
2. **Credibly win-positive** — a traced causal path to more wins with the mechanism named
   (points, possessions, field position, win probability, variance), honest about costs and
   conditions. Candor about a thin pick is correct behavior, not a defect.
3. **Correct bucket attribution** — analytics picks are claims the research actually speaks
   to; intuition picks stake out a position the data doesn't settle and say why. The grid
   check confirms only that a cell is *labeled*; verifying it belongs there is the judge's job.
4. **Grid discipline and distinctness** — one substantively different idea per cell, filed
   under the right phase of football.
5. **Quality of the rationale** — specific, football-literate reasoning; correct terminology;
   engages the obvious counterargument.

Judges are told not to reward length, formatting polish, confident tone, unverifiable
statistics, or exotic-but-unworkable cleverness.

### 5.4 Priors are generated but not judged

The candidates still produce a Bayesian prior per strategy (Part 2 of the prompt), it is still
checked in Stage 2, and it is still published as the prior-comparison chart. It is **not** a
judged criterion. The judge prompt marks it explicitly out of scope and the judge-facing check
summary omits every prior-mechanics result, so a judge cannot be nudged by them.

> **Revision (post-second-run).** Judging previously had a second criterion, "Reasonableness of
> Bayesian priors". Even after the first-run fix that demoted parameter reproduction to a
> tiebreaker, the criterion had no ground truth to appeal to — there is no true per-game points
> value for a strategy, so judges could only reward priors matching their own unverified
> intuitions, and they anchored on the automated report handed to them. The criterion was
> removed rather than repaired. Runs before and after this change are not comparable, and the
> `criterion` column in the results CSV now always reads `soundness` (kept so the schema
> survives a future second criterion).

### 5.5 Self- and family-comparison

All four judges are drawn from the candidate families (three are candidates themselves), so a
judge will sometimes compare its own or a sibling model's response. Keep **all** verdicts; tag
each with `is_self_a`/`is_self_b` (same exact model) and `judge_family`. Handle self-preference
as a diagnostic in Section 6 rather than by dropping data here.

### 5.6 Cost design

Comparisons go through the providers' 50%-off batch APIs where available (Anthropic, OpenAI,
Gemini) and run synchronously for z.ai/GLM (no batch API). Prompts share byte-stable prefixes so
provider prompt caching applies, with explicit `cache_control` breakpoints on Anthropic and a
stable `prompt_cache_key` on OpenAI. The lifecycle is submit → status → collect → (resubmit
stragglers) → csv, with `outstanding = expected − verdict files` so every step is idempotent.

---

## 6. Aggregate — win tallies and judge diagnostics

### 6.1 Win percentages and rankings

- **Comparison count** per model = number of head-to-head comparisons it appeared in (across all judges, pairs, and presentation orders).
- **Win count / win percentage** per model = wins ÷ comparisons × 100.
- **Rankings** = sort models by win percentage.

### 6.2 Judge-agreement diagnostics

Reported for the lay audience as panel-consensus readouts rather than formal reliability statistics:

- **Vote splits** — for each ordered pair, how decisively the four judges agreed (unanimous, 3–1 majority, or 2–2 split).
- **Order consistency** — whether a judge picked the same canonical winner after the two responses were swapped (A↔B); low consistency flags sensitivity to presentation order.

### 6.3 Self-preference diagnostic

For each judge, compare the win rate it awarded to **own-family** candidates vs. **other** candidates. A large positive gap flags self-preference and belongs in the limitations section.

---

## 7. Data artifacts

Store as JSON or a small SQLite DB so every stage is auditable and the website reads from it.

Objective tables (the `tables` stage):

| Table | Key fields |
|---|---|
| `responses` | instance_id, model, sample_idx, snapshot_id, ts, temperature, seed, raw_text, parsed_json |
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
5. **Methodology & limitations** — disclose: judges are also contestants (self-preference risk, with the Section 6.3 gap shown); `n_samples` is small; "underutilized" is a subjective, time-sensitive call partly reflecting the judges' own football knowledge; only one of the two things the prompt asks for is judged.

Interactive tabs/toggles are encouraged where they aid clarity (model toggle on the comparison view).

---

## 9. Decisions deliberately left open

- `n_samples` and `gen_temperature` are set as defaults; adjust to taste and budget.
- The footval prompt leaves the plausible effect-size **scale** unspecified on purpose. That was originally so the priors criterion could test whether a model picks sane magnitudes; with priors unjudged it now only affects the published prior chart. Keep it that way unless you decide to anchor it.
