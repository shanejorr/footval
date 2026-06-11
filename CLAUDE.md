# Footbench — CLAUDE.md

Guidance for Claude Code working in the `footbench` folder. Keep this file current as the project evolves.

## What this is

Footbench is a subjective, multi-model LLM evaluation. Twelve candidate models answer a fixed prompt (NFL strategy recommendations + Bayesian priors + a plotting script). Their responses are executed, auto-checked, scored by four LLM judges on a 1–5 rubric, and aggregated by mean with an interrater-reliability report. Final outputs are a long-form scores CSV and two summary plots (per-judge scores; prior comparison).

## Source-of-truth documents

- `footbench.prompt.md` — the exact prompt sent to every candidate. **Do not edit casually**; changing it invalidates prior runs.
- `initial_prompt.md` — full methodology (stages, rubric anchors, IRR, data schema). This is authoritative, with one deliberate deviation: stage 6 (spec §9, a web page) was replaced by the owner's decision with static outputs — `outputs/scores.csv` (long form: one row per judge×candidate, one column per criterion), a judge×candidate score plot, and a prior-comparison plot. If code and spec disagree elsewhere, fix the code or update the spec deliberately — don't let them drift.

When asked to implement something, read the spec first; this file is only a fast orientation.

## Architecture

Six stages, each reading/writing structured artifacts so they run independently and the run is reproducible:

| Stage | Module | Does |
|---|---|---|
| 1 Generate | `footbench/generate.py` | Call each candidate `n_samples` times → response instances |
| 2 Execute | `footbench/sandbox.py` | Run each `plot_script` in a container, render plots |
| 3 Check | `footbench/checks.py` | Deterministic JSON/structure/parameter checks |
| 4 Judge | `footbench/judge.py` | 4 judges score each instance 1–5 on 3 criteria |
| 5 Aggregate | `footbench/aggregate.py` | Mean scores, rankings, interrater reliability, bias diagnostic |
| 6 Publish | `footbench/publish.py` | Build `outputs/` (scores CSV + the two plots) from artifacts |

The unit of analysis everywhere is the **response instance** = (model, sample_idx).

## Commands

```bash
make run            # full pipeline end to end
make generate       # stage 1 only (then execute / check / judge / aggregate / publish)
make sandbox-build  # build the code-execution container
make test           # pytest
make lint           # ruff check + format
```

Stages are resumable: each reads the previous stage's artifacts, so re-run a single stage without redoing the whole pipeline.

## Pairwise comparison stage

`footbench/pairwise.py` compares every unordered pair of response instances (66 pairs for 12
candidates) on criteria 1–2 only (soundness, priors) — forced A/B choice, text-only. Judges
and the `both_orders` flag live under `pairwise:` in config.yaml. Which model is shown as
"Response A" is a deterministic per-pair coin flip from the run seed; presentation order is
recorded so position bias is analyzable.

Cost design: Anthropic/OpenAI/Gemini go through their 50%-off batch APIs; DeepSeek has no
batch API and runs synchronously. Prompts share byte-stable prefixes
(`[system][task][bundle A][bundle B]`), with explicit `cache_control` breakpoints for
Anthropic and automatic prefix caching elsewhere; requests are sorted so same-candidate-A
calls are adjacent.

```bash
make pairwise-estimate   # offline: outstanding counts + token estimate
make pairwise-submit     # submit batches (resubmission IS the retry mechanism)
make pairwise-deepseek   # run the DeepSeek judge synchronously (anytime)
make pairwise-status     # poll batch states
make pairwise-collect    # fetch ended batches -> verdict files; failures stay outstanding
make pairwise-csv        # outputs/pairwise_results.csv (judge, model_a, model_b, criterion, winner)
```

State lives under `artifacts/pairwise/`: `manifest.json` (custom_id -> pair mapping; drift
hard-fails), `batches.json` (submitted batch ids + submission order), `attempts.json`
(format-failure counts; after `judging.max_format_retries` an item gets a terminal
`verdicts: null` file), `verdicts/{judge}/{custom_id}.json`, and `raw/` (verbatim batch
output for audit). `outstanding = expected − verdict files`, so every subcommand is
idempotent and safe to re-run.

## Conventions

- Python 3.12+, managed with `uv`. Type-hint public functions. Lint/format with `ruff`.
- Use **Polars** (not pandas) for tabular work.
- Tests with `pytest`; cover `checks.py` and `aggregate.py` thoroughly (they encode the methodology).
- Keep it simple and direct — no speculative abstraction. Match the spec; don't add scope.

## Critical constraints — read before writing code

- **Sandbox is mandatory.** Candidate `plot_script` code is untrusted and model-generated. Never execute it on the host. Always run it in the disposable container with **no network, a CPU/memory cap, a hard timeout, and the matplotlib `Agg` backend**. A script that fails or times out is recorded as `ran_ok=false` — it is not an error to fix.
- **Do not repair candidate output.** Malformed JSON, a missing grid cell, or a broken script is real signal. Capture it; never auto-fix, re-prompt, or hand-edit a candidate response.
- **Judges see anonymized responses.** Strip any model-identifying text before a response reaches a judge. Each judge scores one instance at a time (independent scoring, not a 12-way ranking).
- **Scoring is locked:** integers 1–5 per criterion; aggregate is the arithmetic mean; composite is the equal-weighted mean of the three criteria. Don't introduce a different scale or weighting without updating the spec.
- **Reliability:** Krippendorff's α (ordinal) is primary; ICC(2,k) secondary; both per criterion. Collect the full judges × instances matrix (including self- and same-family judgments, tagged) so the matrix stays complete.
- **Prior-comparison plots are re-rendered from the reported parameters**, not from the candidates' own scripts. Parse family + params from the JSON and draw with one canonical routine on shared axes.
- **Objective vs. subjective split:** "does it run / is the grid complete / do parameters reproduce the interval" come from Stages 2–3, not from judge opinion. Judges score only the subjective quality, using those reports as ground truth.
- **Pin model snapshots.** The version strings in `config.yaml` will drift and some may not exist yet — resolve and record the exact API snapshot ID at run time. Never hardcode API keys.

## Data & artifacts

Stored under `artifacts/` (gitignored; optionally synced to S3). Tables: `responses`, `exec_results`, `checks`, `judgments`, `model_scores`, `reliability`, `bias`. Schema is in the spec. Stage 6 reads only from these — no recomputation in the publish step.

## Secrets

API keys come from environment variables / a gitignored `.env`. Never commit keys, raw responses, or rendered images.

## Publishing (stage 6 outputs)

Written to `outputs/` (not auto-committed; owner reviews first):

- `scores.csv` — long form: columns `judge, candidate, soundness, priors, code`; one row per judge×candidate pair; empty cells where a judge failed to return valid scores.
- `scores_by_judge.png` — annotated heatmaps (one panel per criterion) of each judge's integer score for each candidate, candidates sorted by composite.
- `priors_comparison.png` — 2×3 grid (one subplot per strategy cell) overlaying every model's prior pdf, re-rendered from the **reported parameters** via the canonical routine in `distributions.py` — never from the candidates' scripts — on shared per-cell axes.
