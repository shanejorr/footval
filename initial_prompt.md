# Footval — Implementation Spec

A subjective, multi-model LLM evaluation. Candidate models answer `footval/footval.prompt.md`; their responses are executed, auto-checked, and judged by an LLM panel through a **pairwise head-to-head tournament** (every pair of responses compared on soundness and priors, forced A/B choice), then published to a public web page as win-percentage rankings.

---

## 1. Configuration

Put these at the top of the pipeline as editable variables.

```yaml
candidate_models:
  - claude-fable-5
  - claude-opus-4-8
  - claude-sonnet-5
  - claude-haiku-4-5-20251001
  - gpt-5.5-pro
  - gpt-5.5
  - gpt-5.4-mini
  - gemini-3.1-pro-preview
  - gemini-3.5-flash
  - gemini-3.1-flash-lite
  - deepseek-v4-pro
  - deepseek-v4-flash

# Pairwise judge panel — the only judging round.
pairwise:
  judges:
    - claude-opus-4-8
    - gpt-5.5
    - gemini-3.5-flash
    - deepseek-v4-pro
  both_orders: true         # judge every pair in both presentation orders

n_samples: 1              # responses generated per candidate
gen_temperature: 1.0      # fixed for all candidates
judge_temperature: 0.0    # judges run as deterministically as possible
seed: 42                  # where the provider supports it
interval_tol: 0.10        # relative tolerance for parameter-reproduction check
```
Use the highest level of thinking available for both candidate and judge models. Except use `max` for `claude-fable-5`.

Pin the exact API snapshot ID for every model at run time and store it (the version strings above will drift).

---

## 2. Pipeline overview

```
Stage 1  Generate     candidate responses  (12 models x n_samples)
Stage 2  Execute      run each plot_script in a sandbox, render plots
Stage 3  Auto-check   structural + numeric consistency of the JSON
Stage 4  Tables       consolidate the objective per-instance artifacts into tables
Pairwise Judge        4 judges compare every pair head-to-head on 2 criteria (forced A/B)
Stage 6  Publish      web page on shaneorr.me (win-percentage rankings)
```

The pairwise judging round runs on its own batch lifecycle (Section 6), not as part of the
linear `run`. Each stage reads and writes structured artifacts (Section 8) so the run is
reproducible and the website is a pure view over stored data.

---

## 3. Stage 1 — Generate responses

For each candidate model, generate `n_samples` independent responses to `footval.prompt.md` at `gen_temperature`.

- Treat each (model, sample) pair as a distinct **response instance** with its own `instance_id`. All later stages operate on instances.
- Capture and store: model snapshot ID, timestamp, temperature, seed, raw text, and the parsed JSON object.
- Do not retry or repair malformed output; a malformed response is a real signal and is handled by Stage 3.

API keys are in `footval/.env` and have the following names:

- Anthropic — auto-read by the anthropic SDK
`ANTHROPIC_API_KEY`

- OpenAI — auto-read by the openai SDK
`OPENAI_API_KEY`

- Google Gemini — auto-read by the google-genai SDK
`GEMINI_API_KEY`

- DeepSeek — convention only (see note)
`DEEPSEEK_API_KEY`

---

## 4. Stage 2 — Execute code

The footval JSON contains one `plot_script`. Execute it; do not ask a judge to predict whether it runs.

**Sandbox requirements** (the code is untrusted, model-generated):

- Run in an isolated container or subprocess: no network, ephemeral filesystem, CPU/memory limits, hard timeout (e.g., 60s).
- Force a non-interactive matplotlib backend (`Agg`) and redirect figures to files.

**Capture per instance:** exit status, `ran_ok` (bool), number of figures produced, stdout/stderr, and the rendered PNG paths.

The rendered images and `ran_ok` are inputs to both the judges and the website. Some scripts will fail — record the failure; do not let it halt the run.

---

## 5. Stage 3 — Automated checks

Deterministic checks on the parsed JSON. These are objective and should not be left to judge opinion; their results are passed to judges as factual context and surfaced on the site.

Per instance, record pass/fail (with detail) for:

1. **JSON valid** — parses, top-level `quantity` / `strategies` / `plot_script` present.
2. **Grid complete** — exactly 6 strategies, one per (`analytics`/`intuition` × `game_management`/`offensive`/`defensive`) cell, with distinct titles.
3. **Interval sanity** — for each prior: `low < high` and `low ≤ most_plausible_value ≤ high`.
4. **Valid family** — `name ∈ {normal, student_t, skew_normal}` (signed-support families only).
5. **Parameter reproduction** — recompute the 95% interval from the reported parameters and check it matches the reported `interval_95` within `interval_tol`:
   - normal: `μ ± 1.96·σ`
   - student_t: `μ ± t_{0.975,ν}·σ`
   - skew_normal: `skewnorm.ppf(0.025/0.975, a=α, loc=ξ, scale=ω)`
   Also check the family's center (`μ`, or mode for skew_normal) ≈ `most_plausible_value`.

Store a per-instance check report. These are diagnostics fed to judges, **not** a separate scored criterion.

---

## 6. Judging — pairwise head-to-head tournament

The only judging round. Every unordered pair of response instances is compared by the four-judge
panel on two criteria — **soundness** and **priors** — as a forced A/B choice. There is no numeric
scale and no separate per-response score.

### 6.1 Inputs to each comparison

Each judge call shows **two anonymized responses** ("Response A" and "Response B") for the same
task. Each call provides:

- `footval.prompt.md` (the task both candidates were given).
- The two **anonymized** response bundles (strip any model-identifying text; label only as
  "Response A" / "Response B"), each as pretty-printed JSON (or raw text if it failed to parse).
- The Stage 3 automated check report for each response, as ground truth for the mechanical facts.

The round is **text-only** (no execution result or rendered images), so any code/plot criterion is
not judged here. Which response is shown as "A" is a deterministic per-pair coin flip from the run
seed; with `both_orders: true` every pair is also judged in the reversed order so position bias is
measurable.

### 6.2 Output from each comparison

```json
{
  "soundness": {"winner": "A" | "B", "justification": "1-2 sentences"},
  "priors":    {"winner": "A" | "B", "justification": "1-2 sentences"}
}
```

Forced choice: the judge **must** pick "A" or "B" for each criterion; ties are not allowed.
Criteria are judged independently (the same response need not win both). Instruct judges to use the
check reports as ground truth for the mechanical facts and to spend their judgment on subjective
quality. Two weighting rules keep the mechanical `params_reproduce` check from dominating the
priors criterion (see the revision note in §6.3): a parameter-reproduction failure is a mechanical
conversion error that breaks ties only when the two responses are otherwise comparable on
substance, and judges weigh distinct *kinds* of problems rather than the raw count of failing
cells (one repeated conversion mistake across several cells is one defect). The judge-facing check
summary carries a `params_reproduce_summary` roll-up that states the common cause when all failing
cells share a family.

### 6.3 Quality descriptions

Give judges stronger-vs-weaker descriptions (no numeric anchors) for each criterion; the judge
picks whichever response better fits the "stronger" description.

**Criterion 1 — Soundness of recommendations**

- A **stronger** response: all six picks are genuinely underutilized *and* credibly win-positive; rationales are specific and football-literate; analytics vs. intuition buckets correctly distinguished (intuition picks are not just restated analytics consensus); picks distinct and correctly slotted in their grid cell.
- A **weaker** response: picks generic/already standard, implausible, duplicative, or misattributed; rationales vague or wrong; or buckets misfiled.

**Criterion 2 — Reasonableness of Bayesian priors**

Judge the *substance* of the beliefs first; parameter-reproduction accuracy is a secondary, mechanical matter (tiebreaker only).

- A **stronger** response (primary): football-plausible effect-size magnitudes (no single strategy swinging the per-game scoring margin by several points); the six cells genuinely differentiated rather than copy-pasted; intervals neither overconfident nor absurdly wide; weak cells handled honestly (near-zero center, wider interval); the family's shape matches the stated belief (a real left tail where downside is claimed, signed support where the effect can go either way).
- A **weaker** response (primary): grossly implausible or over-tight/over-wide magnitudes; boilerplate priors reused across differing strategies; an invalid family for signed data; or a family whose shape contradicts the stated belief.
- **Secondary (tiebreaker only):** whether reported parameters reproduce the stated interval. A reproduction failure is a conversion slip, not evidence the belief is unreasonable; do not reward playing it safe with a symmetric family over an ambitious asymmetric one whose tail quantile is slightly off.

> **Revision (post-first-run).** The original Criterion 2 listed "conversions correct / conversion slips" as a co-equal signal, and with the automated `params_reproduce` result handed to judges as ground truth, the priors criterion collapsed into "who passed more reproduction cells." One candidate (Claude Sonnet 4.6) lost 76 of 88 priors matchups almost entirely because a single wrong skew-normal tail quantile failed reproduction across its four skew-normal cells — a correlated, one-root-cause slip counted as four defects. The rubric above demotes reproduction to a tiebreaker, adds explicit weighting/common-cause ground rules (§6.2), and the check summary now emits a `params_reproduce_summary` roll-up so a repeated slip reads as one issue. This is a deliberate methodology change; runs before and after it are not directly comparable on the priors criterion.

### 6.4 Self- and family-comparison

All four judges are also candidates, so a judge will sometimes compare its own or a sibling model's
response. Keep **all** verdicts; tag each with `is_self_a`/`is_self_b` (same exact model) and
`judge_family`. Handle self-preference as a diagnostic in Section 7 rather than by dropping data here.

### 6.5 Cost design

Comparisons go through the providers' 50%-off batch APIs where available (Anthropic, OpenAI,
Gemini) and run synchronously for DeepSeek (no batch API). Prompts share byte-stable prefixes so
provider prompt caching applies. The lifecycle is submit → status → collect → (resubmit
stragglers) → csv, with `outstanding = expected − verdict files` so every step is idempotent.

---

## 7. Aggregate — win tallies and judge diagnostics

### 7.1 Win percentages and rankings

- **Comparison count** per model = number of head-to-head comparisons it appeared in (across all judges, pairs, and presentation orders), per criterion.
- **Win count / win percentage** per model = wins ÷ comparisons × 100, per criterion and overall (pooling both criteria).
- **Rankings** = sort models by win percentage (publish an overall leaderboard and a per-criterion one).

### 7.2 Judge-agreement diagnostics

Reported for the lay audience as panel-consensus readouts rather than formal reliability statistics:

- **Vote splits** — for each pair×criterion, how decisively the four judges agreed (unanimous, 3–1 majority, or 2–2 split).
- **Order consistency** — whether a judge picked the same canonical winner after the two responses were swapped (A↔B); low consistency flags sensitivity to presentation order.

### 7.3 Self-preference diagnostic

For each judge, compare the win rate it awarded to **own-family** candidates vs. **other** candidates. A large positive gap flags self-preference and belongs in the limitations section.

---

## 8. Stage 8 — Data artifacts

Store as JSON or a small SQLite DB so every stage is auditable and the website reads from it.

Objective tables (the `tables` stage):

| Table | Key fields |
|---|---|
| `responses` | instance_id, model, sample_idx, snapshot_id, ts, temperature, seed, raw_text, parsed_json |
| `exec_results` | instance_id, ran_ok, n_figures, stderr, figure_paths |
| `checks` | instance_id, json_valid, grid_complete, intervals_ok, family_ok, params_reproduce_ok, detail |

Pairwise judging artifacts (under `artifacts/pairwise/`):

| Artifact | Key fields |
|---|---|
| `manifest` | seed, both_orders, judges, instances, comparisons (custom_id → pair, order, instance/model A & B) |
| `verdicts/{judge}/{custom_id}` | judge, judge_family, model_a, model_b, order, is_self_a, is_self_b, verdicts (winner per criterion), raw_text |
| `pairwise_results.csv` | judge, model_a, model_b, criterion, winner |

---

## 9. Stage 6 — Publish (shaneorr.me → Other Writings)

Publish as a blog post under 'Other Writings' (`https://www.shaneorr.me/writings`)

Audience is non-technical. The page reads from the Stage 8 artifacts.

**Sections:**

1. **Plain-language intro** — what the test asked the models to do, in one short paragraph. Explain "prior" and "uncertainty interval" with a one-line analogy (e.g., a prior is the model's best guess plus an honest statement of how unsure it is).
2. **Leaderboard** — win-percentage ranking (head-to-head wins ÷ comparisons), with a toggle to view overall vs. each of the two criteria. Alongside, show the judge-agreement diagnostics (vote-split mix and order consistency) with a plain label so the ranking isn't read as more precise than the panel's agreement justifies. Use visualizations.
3. **Per-model response viewer** — each model's full response, with the `plot_script` **rendered as its plot image and the source code collapsed by default**. Show a clear "failed to render" state for instances where `ran_ok` is false.
4. **Prior comparison** — **re-plot from the reported parameters, not the models' scripts.** Parse each model's family + parameters from the JSON and redraw every prior with one canonical routine on shared axes and shared styling. Lay out as small multiples by grid cell (6 cells) with the ability to overlay or toggle models within a cell. This is the clean payoff of the structured footval schema; it sidesteps 12 inconsistent matplotlib styles.
5. **Methodology & limitations** — disclose: judges are also contestants (self-preference risk, with the Section 7.3 gap shown); single-vendor judge pool; `n_samples` is small; "underutilized" is a subjective, time-sensitive call partly reflecting the judges' own football knowledge.

Interactive tabs/toggles are encouraged where they aid clarity (criterion toggle on the leaderboard, model toggle on the comparison view).

---

## 10. Decisions deliberately left open

- `n_samples` and `gen_temperature` are set as defaults; adjust to taste and budget.
- The footval prompt leaves the plausible effect-size **scale** unspecified on purpose, so the priors criterion partly tests whether a model picks sane magnitudes. Keep it that way unless you decide to anchor it.
