# Footbench — CLAUDE.md

Guidance for Claude Code working in the `footbench` folder. Keep this file current as the project evolves.

## What this is

Footbench is a subjective, multi-model LLM evaluation. Twelve candidate models answer a fixed prompt (NFL strategy recommendations + Bayesian priors + a plotting script). Their responses are executed, auto-checked, and judged by a four-model panel through a **pairwise head-to-head tournament** (every pair compared on soundness and priors, forced A/B choice). The published outputs are a win-percentage tally (`outputs/data/pairwise_results.csv`) and the notebook charts built from it plus the priors. There is **no 1–5 scoring** — that round was removed.

## Source-of-truth documents

- `footbench.prompt.md` — the exact prompt sent to every candidate. **Do not edit casually**; changing it invalidates prior runs.
- `initial_prompt.md` — full methodology (stages, pairwise judging, data schema). This is authoritative, with one deliberate deviation: stage 6 (spec §9, a web page) was replaced by the owner's decision with static outputs — `outputs/data/pairwise_results.csv` (long form: one row per judge×pair×criterion with the winner) plus the notebook charts (win percentage overall/by criterion/by judge, the prior comparison, and panel-consensus diagnostics). If code and spec disagree elsewhere, fix the code or update the spec deliberately — don't let them drift.

When asked to implement something, read the spec first; this file is only a fast orientation.

## Architecture

Four core stages, each reading/writing structured artifacts so they run independently and the run is reproducible. Judging is the separate pairwise tournament (below), run via its own batch lifecycle:

| Stage | Module | Does |
|---|---|---|
| 1 Generate | `footbench/generate.py` | Call each candidate `n_samples` times → response instances |
| 2 Execute | `footbench/sandbox.py` | Run each `plot_script` in a container, render plots |
| 3 Check | `footbench/checks.py` | Deterministic JSON/structure/parameter checks |
| 4 Tables | `footbench/tables.py` | Consolidate per-instance artifacts into the objective `responses` / `exec_results` / `checks` tables |

The unit of analysis everywhere is the **response instance** = (model, sample_idx).

## Commands

```bash
make run            # generate → sandbox → check → tables
make generate       # stage 1 only (then sandbox / check / tables)
make sandbox-build  # build the code-execution container
make test           # pytest
make lint           # ruff check + format
```

Stages are resumable: each reads the previous stage's artifacts, so re-run a single stage without redoing the whole pipeline.

## Pairwise comparison stage (the judging round)

`footbench/pairwise.py` is the only judging stage: it compares every unordered pair of response
instances (66 pairs for 12 candidates) on two criteria — soundness and priors — as a forced A/B
choice, text-only, with no numeric scale (the judge prompt uses stronger-vs-weaker prose anchors).
Judges and the `both_orders` flag live under `pairwise:` in config.yaml. Which model is shown as
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
make pairwise-csv        # outputs/data/pairwise_results.csv (judge, model_a, model_b, criterion, winner)
```

State lives under `artifacts/pairwise/`: `manifest.json` (custom_id -> pair mapping; drift
hard-fails), `batches.json` (submitted batch ids + submission order), `attempts.json`
(format-failure counts; after `judging.max_format_retries` an item gets a terminal
`verdicts: null` file), `verdicts/{judge}/{custom_id}.json`, and `raw/` (verbatim batch
output for audit). `outstanding = expected − verdict files`, so every subcommand is
idempotent and safe to re-run.

## Bradley-Terry aggregation (optional ranking layer)

`footbench/bradley_terry.py` (`make bradley-terry`) reads `outputs/data/pairwise_results.csv`
and fits a hierarchical Bayesian Bradley-Terry model in PyMC, once per track
(`soundness`, `priors`, `overall`). It turns the forced-choice verdicts into a latent-strength
ranking **with 94% HDIs**, modeling the judges as raters: per-judge discrimination (`kappa`),
slot-A/position bias (`gamma`), and self-family preference (`delta`), all partially pooled — so
`mu_gamma`/`mu_delta` are panel-wide bias estimates (a principled successor to the §7.3 win-rate
gap). It is **purely additive**: it consumes the published CSV and touches no prompt or prior
artifact. Outputs to `outputs/data/`: `bradley_terry_rankings.csv` (strength + `win_prob_vs_field`
with HDIs, rank), `bradley_terry_judge_effects.csv` (per-judge `kappa`/`gamma`/`delta` + a `panel
(mean)` row), and `bradley_terry.json` (full results + sampler diagnostics).

The full posterior **traces** are persisted (via `bt.trace_path(cfg, track)`) to
`artifacts/bradley_terry/{track}.nc` (gitignored) so the notebooks can recompute any HDI /
posterior-predictive quantity and run ArviZ diagnostics without re-fitting.

Notes: PyMC/ArviZ are imported lazily so the pure data-prep helpers stay testable without the
sampling stack (tests cover only data-prep, not the MCMC). Sampler knobs are module constants.
With only 4 judges the judge-level variances are weakly identified and lean on their priors —
read `mu_gamma`/`mu_delta` as directional. On the balanced round-robin the BT ranking matches
raw win% (Spearman ≈ 1.0); the value-add is the uncertainty bands and the judge-effect
decomposition, not a different ordering. Requires `pymc` + `arviz` + `h5netcdf`/`h5py` (added to
`pyproject.toml`); the notebooks additionally need `nbconvert` + `vl-convert-python` (dev group).

## Notebooks (`outputs/`)

Both read the persisted BT traces (`import footbench`, `az.from_netcdf(bt.trace_path(...))`), so
run `make bradley-terry` first. Re-render headlessly with
`uv run jupyter nbconvert --to notebook --execute --inplace outputs/<nb>.ipynb`.

- `blog_output.ipynb` — the published charts. The leaderboards (overall + by-criterion) plot
  `win_prob_vs_field` as a point estimate with a thick **50% HDI** and thin **95% HDI** line
  (no axis rule); the by-judge line chart and the judge/candidate matrix use **posterior-predictive
  per-judge win probabilities** with HDIs. Subjective-priors, interrater-reliability, and
  order-consistency charts are unchanged (computed from `responses.json` / raw verdicts). Charts
  export to `src/content/writings/2026-06-12-footbench/charts/*.svg`.
- `model_diagnostics.ipynb` — ArviZ convergence/sampling diagnostics per track (summary tables,
  trace/rank/energy/forest/ESS plots, BFMI). Assessment-only; reports no results.

## Conventions

- Python 3.12+, managed with `uv`. Type-hint public functions. Lint/format with `ruff`.
- Use **Polars** (not pandas) for tabular work.
- Tests with `pytest`; cover `checks.py` and `pairwise.py` thoroughly (they encode the methodology).
- Keep it simple and direct — no speculative abstraction. Match the spec; don't add scope.

## Critical constraints — read before writing code

- **Sandbox is mandatory.** Candidate `plot_script` code is untrusted and model-generated. Never execute it on the host. Always run it in the disposable container with **no network, a CPU/memory cap, a hard timeout, and the matplotlib `Agg` backend**. A script that fails or times out is recorded as `ran_ok=false` — it is not an error to fix.
- **Do not repair candidate output.** Malformed JSON, a missing grid cell, or a broken script is real signal. Capture it; never auto-fix, re-prompt, or hand-edit a candidate response.
- **Judges see anonymized responses.** Strip any model-identifying text before a response reaches a judge. Judging is the pairwise tournament: each call shows two anonymized responses and forces an A/B winner per criterion (no numeric scale, no ties).
- **Judging is locked to pairwise:** two criteria (soundness, priors), forced A/B choice, no 1–5 score and no composite. Don't reintroduce a numeric scoring scale without updating the spec. Self- and same-family comparisons are tagged (`is_self_a`/`is_self_b`, `judge_family`) and kept, not dropped.
- **Objective vs. subjective split:** "does it run / is the grid complete / do parameters reproduce the interval" come from Stages 2–3, not from judge opinion. Judges weigh only the subjective quality, using those reports as ground truth.
- **Pin model snapshots.** The version strings in `config.yaml` will drift and some may not exist yet — resolve and record the exact API snapshot ID at run time. Never hardcode API keys.

## Data & artifacts

Stored under `artifacts/` (gitignored; optionally synced to S3). Objective tables (from the `tables` stage): `responses`, `exec_results`, `checks`. Pairwise judging state lives under `artifacts/pairwise/` (manifest, batches, attempts, per-verdict files, raw batch output). Schema is in the spec.

## Secrets

API keys come from environment variables / a gitignored `.env`. Never commit keys, raw responses, or rendered images.

## Publishing outputs

Written to `outputs/` (not auto-committed; owner reviews first). CSV data files live in `outputs/data/`:

- `pairwise_results.csv` — long form: columns `judge, model_a, model_b, criterion, winner`; one row per judge×pair×criterion; blank `winner` where a judge returned no valid verdict. The notebook (`outputs/blog_output.ipynb`) reads this plus `artifacts/tables/responses.json` to build the win-percentage and prior-comparison charts.
