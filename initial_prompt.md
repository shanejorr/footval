# Footbench — Implementation Spec

A subjective, multi-model LLM evaluation. Candidate models answer `footbench/footbench.prompt.md`; their responses are executed, auto-checked, scored by LLM judges on a 1–5 rubric, aggregated by mean, and published to a public web page with an interrater-reliability report.

---

## 1. Configuration

Put these at the top of the pipeline as editable variables.

```yaml
candidate_models:
  - claude-fable-5
  - claude-opus-4-8
  - claude-sonnet-4-6
  - claude-haiku-4-5-20251001
  - gpt-5.5-pro
  - gpt-5.5
  - gpt-5.4-mini
  - gemini-3.1-pro-preview
  - gemini-3.5-flash
  - gemini-3.1-flash-lite
  - deepseek-v4-pro
  - deepseek-v4-flash

judge_models:
  - claude-fable-5
  - gpt-5.5-pro
  - gemini-3.1-pro-preview
  - deepseek-v4-pro

n_samples: 1              # responses generated per candidate
gen_temperature: 1.0      # fixed for all candidates
judge_temperature: 0.0    # judges run as deterministically as possible
seed: 42                  # where the provider supports it
criteria_weights: [1, 1, 1]   # soundness, priors, code — equal weight for composite
interval_tol: 0.10        # relative tolerance for parameter-reproduction check
```
Use the highest level of thinking available for both candidate and judge models.

Pin the exact API snapshot ID for every model at run time and store it (the version strings above will drift).

---

## 2. Pipeline overview

```
Stage 1  Generate     candidate responses  (12 models x n_samples)
Stage 2  Execute      run each plot_script in a sandbox, render plots
Stage 3  Auto-check   structural + numeric consistency of the JSON
Stage 4  Judge        4 judges score each response 1-5 on 3 criteria
Stage 5  Aggregate    mean scores, rankings, interrater reliability
Stage 6  Publish      web page on shaneorr.me
```

Each stage reads and writes structured artifacts (Section 8) so the run is reproducible and the website is a pure view over stored data.

---

## 3. Stage 1 — Generate responses

For each candidate model, generate `n_samples` independent responses to `footbench.prompt.md` at `gen_temperature`.

- Treat each (model, sample) pair as a distinct **response instance** with its own `instance_id`. All later stages operate on instances.
- Capture and store: model snapshot ID, timestamp, temperature, seed, raw text, and the parsed JSON object.
- Do not retry or repair malformed output; a malformed response is a real signal and is handled by Stage 3.

API keys are in `footbench/.env` and have the following names:

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

The footbench JSON contains one `plot_script`. Execute it; do not ask a judge to predict whether it runs.

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

## 6. Stage 4 — LLM judging

### 6.1 Inputs to each judge call

One **response instance at a time** (independent scoring, not a 12-way ranking — this removes position/length bias and matches the 1–5 scale). Each call provides:

- `footbench.prompt.md` (the task the candidate was given).
- The rubric with anchors (Section 6.3).
- The **anonymized** response JSON (strip any model-identifying text; label only as "Response under review").
- The Stage 2 execution result and the rendered plot images (judges are multimodal; let them see the actual plots).
- The Stage 3 check report.

Randomize nothing across candidates is needed because each is scored in isolation, but still keep the response anonymized.

### 6.2 Output from each judge call

```json
{
  "soundness":   {"score": 1-5, "justification": "1-2 sentences"},
  "priors":      {"score": 1-5, "justification": "1-2 sentences"},
  "code":        {"score": 1-5, "justification": "1-2 sentences"}
}
```

Scores are **integers 1–5**. Instruct judges to use the execution and check reports as ground truth for the mechanical facts (does it run, is the grid complete, do parameters reproduce the interval) and to spend their judgment on the subjective quality.

### 6.3 Rubric anchors

Give judges these anchors; interpolate 4 and 2.

**Criterion 1 — Soundness of recommendations**

- **5** — All six picks are genuinely underutilized *and* credibly win-positive; rationales are specific and football-literate; analytics vs. intuition buckets correctly distinguished (intuition picks are not just restated analytics consensus); picks distinct and correctly slotted in their grid cell.
- **3** — Mixed: several solid picks, but some are mainstream rather than underutilized, weakly justified, or misfiled.
- **1** — Most picks are generic/already standard, implausible, duplicative, or misattributed; rationales vague or wrong.

**Criterion 2 — Reasonableness of Bayesian priors**

- **5** — Family matches the stated shape/tails and the signed support; most-plausible values and 95% intervals are football-realistic in magnitude (neither overconfident nor absurdly wide); conversions correct; honest treatment of weak cells (near-zero center, wider interval).
- **3** — Generally reasonable but with notable issues: an implausible magnitude or two, an over-tight or over-wide interval, a family that doesn't match the stated shape, or a conversion slip.
- **1** — Largely unreasonable: invalid/wrong family for signed data, badly mis-scaled intervals, inconsistent parameters, or identical boilerplate priors copy-pasted across differing strategies.

**Criterion 3 — Python code quality** (treat the execution result as ground truth for whether it runs)

- **5** — Runs cleanly and produces all six correct plots; code is clear, efficient, parameterized as the prompt requires, uses the correct scipy distribution per family, and does what it claims.
- **3** — Runs but rough: inefficiency, partial plotting, minor mismatch between code and reported parameters, or sloppy structure.
- **1** — Fails to run or produces wrong/empty plots, or does not implement the stated distributions/parameters.

### 6.4 Self- and family-judgment

All four judges are also candidates, so judges will sometimes score their own or a sibling model. Collect **all** scores (a complete judges × instances matrix — needed for clean reliability stats), and tag each judgment with `is_self` (same exact model) and `judge_family`. Handle the bias two ways in Stage 5 rather than by dropping data here.

---

## 7. Stage 5 — Aggregate and reliability

### 7.1 Scores and rankings

- **Instance score** per criterion = mean of the 4 judges' integer scores.
- **Model score** per criterion = mean over that model's instances (samples) and judges.
- **Composite** per model = weighted mean of the three criteria (`criteria_weights`, equal by default).
- **Rankings** = sort models by score (publish a per-criterion leaderboard and a composite leaderboard).
- **Self-excluded leaderboard** (secondary) = recompute model scores after dropping `is_self` judgments. Publish alongside the primary so readers can see the effect.

### 7.2 Interrater reliability

Compute across all judged instances, **per criterion**, with the 4 judges as raters.

- **Primary: Krippendorff's α (ordinal).** Chosen because the scores are ordinal, there are >2 raters, and it tolerates any missing cells. Interpretation: α ≥ 0.80 strong agreement; 0.667–0.80 tentative; < 0.667 weak.
- **Secondary: ICC(2,k), two-way random effects, average-measures.** This is the reliability of the 4-judge **mean** — i.e., of the number you actually report. Interpretation (Koo & Li): < 0.50 poor, 0.50–0.75 moderate, 0.75–0.90 good, > 0.90 excellent. Compute on complete cases.
- **Optional readout: Kendall's W** on each judge's derived ranking, plus pairwise Spearman between judges, for a human-readable "who agrees with whom" view.

Report each statistic with its interpretation band. Low reliability on a criterion is itself a publishable finding — it means that criterion is hard to judge, not that the pipeline is broken.

### 7.3 Bias diagnostic

For each judge, report mean score given to **own-family** candidates vs. **other** candidates, per criterion. A large positive gap flags self-preference and belongs in the limitations section.

---

## 8. Stage 8 — Data artifacts

Store as JSON or a small SQLite DB so every stage is auditable and the website reads from it.

| Table | Key fields |
|---|---|
| `responses` | instance_id, model, sample_idx, snapshot_id, ts, temperature, seed, raw_text, parsed_json |
| `exec_results` | instance_id, ran_ok, n_figures, stderr, figure_paths |
| `checks` | instance_id, json_valid, grid_complete, intervals_ok, family_ok, params_reproduce_ok, detail |
| `judgments` | instance_id, judge, judge_family, is_self, criterion, score, justification |
| `model_scores` | model, criterion, mean_score, rank, mean_score_self_excluded |
| `reliability` | criterion, krippendorff_alpha, icc2k, kendall_w |
| `bias` | judge, criterion, own_family_mean, other_mean, gap |

---

## 9. Stage 6 — Publish (shaneorr.me → Other Writings)

Publish as a blog post under 'Other Writings' (`https://www.shaneorr.me/writings`)

Audience is non-technical. The page reads from the Stage 8 artifacts.

**Sections:**

1. **Plain-language intro** — what the test asked the models to do, in one short paragraph. Explain "prior" and "uncertainty interval" with a one-line analogy (e.g., a prior is the model's best guess plus an honest statement of how unsure it is).
2. **Leaderboard** — composite ranking, with a toggle to view each of the three criteria. Show the interrater-reliability value beside each criterion with a plain label ("judges agreed strongly / moderately / weakly here") so the ranking isn't read as more precise than it is. Use visualizations.
3. **Per-model response viewer** — each model's full response, with the `plot_script` **rendered as its plot image and the source code collapsed by default**. Show a clear "failed to render" state for instances where `ran_ok` is false.
4. **Prior comparison** — **re-plot from the reported parameters, not the models' scripts.** Parse each model's family + parameters from the JSON and redraw every prior with one canonical routine on shared axes and shared styling. Lay out as small multiples by grid cell (6 cells) with the ability to overlay or toggle models within a cell. This is the clean payoff of the structured footbench schema; it sidesteps 12 inconsistent matplotlib styles.
5. **Methodology & limitations** — disclose: judges are also contestants (self-preference risk, with the Section 7.3 gap shown); single-vendor judge pool; `n_samples` is small; "underutilized" is a subjective, time-sensitive call partly reflecting the judges' own football knowledge.

Interactive tabs/toggles are encouraged where they aid clarity (criterion toggle on the leaderboard, model toggle on the comparison view).

---

## 10. Decisions deliberately left open

- `n_samples`, `gen_temperature`, and the equal criterion weights are set as defaults; adjust to taste and budget.
- The footbench prompt leaves the plausible effect-size **scale** unspecified on purpose, so the priors criterion partly tests whether a model picks sane magnitudes. Keep it that way unless you decide to anchor it.
