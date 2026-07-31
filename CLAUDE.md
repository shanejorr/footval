# Footval — CLAUDE.md

Guidance for Claude Code working in the `footval` folder. Keep this file current as the project evolves.

This project used to be called `footbench` but was renamed to `footval`.

## What this is

Footval is a subjective, multi-model LLM evaluation. Eight candidate models answer a fixed prompt (NFL strategy recommendations + Bayesian priors). Their responses are auto-checked and judged by a four-model panel through a **pairwise head-to-head tournament** (every pair compared on a single criterion — soundness — forced A/B choice). The published outputs are a win-percentage tally (`outputs/data/pairwise_results.csv`) and the notebook charts built from it plus the priors. There is **no 1–5 scoring** — that round was removed.

## Source-of-truth documents

- `footval.prompt.md` — the exact prompt sent to every candidate. **Do not edit casually**; changing it invalidates prior runs.
- `footval.judge.prompt.md` — the exact system prompt sent to every pairwise judge (the soundness rubric, ground rules, verdict schema). Loaded byte-for-byte by `pairwise.judge_system()`; the same edit caution applies — changing it invalidates prior verdicts.
- `initial_prompt.md` — full methodology (stages, pairwise judging, data schema). This is authoritative, with deliberate deviations recorded in its own revision notes — chiefly that stage 6 (spec §9, a web page) was replaced by the owner's decision with static outputs (`outputs/data/pairwise_results.csv` plus the notebook charts). If code and spec disagree elsewhere, fix the code or update the spec deliberately — don't let them drift.

When asked to implement something, read the spec first; this file is only a fast orientation.

## Architecture

Three core stages, each reading/writing structured artifacts so they run independently and the run is reproducible. Judging is the separate pairwise tournament (below), run via its own batch lifecycle:

| Stage | Module | Does |
|---|---|---|
| 1 Generate | `footval/generate.py` | Call each candidate `n_samples` times → response instances |
| 2 Check | `footval/checks.py` | Deterministic JSON/structure/parameter checks |
| 3 Tables | `footval/tables.py` | Consolidate per-instance artifacts into the objective `responses` / `checks` tables |

The unit of analysis everywhere is the **response instance** = (model, sample_idx).

There is no code-execution stage. The prompt used to ask for a `plot_script` that ran in a
Docker sandbox; Part 3 was removed from the prompt, so `footval/sandbox.py`, the `sandbox/`
image, and the `exec_results` table are gone with it. Do not reintroduce them without
restoring Part 3 to the prompt.

## Commands

```bash
make run            # generate → check → tables
make generate       # stage 1 only (then check / tables)
make probe          # one cheap call per configured model; prints resolved snapshot IDs
make probe-full     # same, with full thinking/effort params (catches param-name drift)
make test           # pytest
make lint           # ruff check + format
```

Stages are resumable: each reads the previous stage's artifacts, so re-run a single stage without redoing the whole pipeline.

## Git workflow — `main` only

**This repo uses a single branch: `main`. Commit directly to it.** Do not create feature
branches, do not open pull requests, and do not branch just because a change is large.
This overrides the general "branch before committing on the default branch" default — it
is a deliberate choice for this repo, not an oversight.

The owner works from two machines, so pull before starting and push after committing.

## Rosters and thinking levels

Candidates (8) and the judge panel (4) live in `config.yaml`. Three models sit on both
rosters, so `ModelCfg` carries **two** parameter blocks: `params` (used when the model
answers) and `judge_params` (deep-merged on top when it judges, via `cfg.judge_model()`).
That is the only way to run a model at candidate effort and judge effort in the same run.

Policy: candidates run one notch above the middle of each provider's effort ladder, judges
one notch above that.

| Provider | Ladder | Candidate | Judge |
|---|---|---|---|
| Anthropic (`output_config.effort`) | low→medium→high→xhigh→max | `high` | `xhigh` |
| OpenAI (`reasoning.effort`) | low→medium→high→xhigh→max | `high` | `xhigh` |
| Gemini (`thinking_config.thinking_level`) | minimal→low→medium→high | `HIGH` | `HIGH` (ladder ends — documented deviation) |
| z.ai GLM (`reasoning_effort`) | high→max | `high` | `max` |

## Pairwise comparison stage (the judging round)

`footval/pairwise.py` is the only judging stage: it compares every unordered pair of response
instances (28 pairs for 8 candidates) on **one criterion — soundness** — as a forced A/B
choice, text-only, with no numeric scale (the judge prompt uses stronger-vs-weaker prose
anchors over five ranked rubric dimensions). Judges and the `both_orders` flag live under
`pairwise:` in config.yaml. Which model is shown as "Response A" is a deterministic per-pair
coin flip from the run seed; presentation order is recorded so position bias is analyzable.

**Priors are generated but not judged.** The candidates still produce a Bayesian prior per
strategy (Part 2 of the prompt) and the checks stage still validates them for the published
charts, but the judge prompt marks them explicitly out of scope,
`pairwise.check_summary()` withholds the prior-mechanics results (interval sanity, family
validity, parameter reproduction), and `pairwise.strip_priors()` removes the `prior` blocks
from the parsed bundles judges see (unparsed raw text is shown whole), so they cannot
contaminate the soundness call — and every judge call skips the priors' token cost.

Cost design: Anthropic/OpenAI/Gemini go through their 50%-off batch APIs
(`providers.BATCH_PROVIDERS`); z.ai has no batch API and runs synchronously. Prompts share
byte-stable prefixes (`[system][task][bundle A][bundle B]`) and requests are sorted so
same-candidate-A calls are adjacent.

```bash
make pairwise-estimate   # offline: outstanding counts + token estimate
make pairwise-submit     # submit batches (resubmission IS the retry mechanism)
make pairwise-sync       # run the non-batch judges (GLM) synchronously (anytime)
make pairwise-status     # poll batch states
make pairwise-collect    # fetch ended batches -> verdict files; failures stay outstanding
make pairwise-csv        # outputs/data/pairwise_results.csv (judge, model_a, model_b, criterion, winner)
```

State lives under `artifacts/pairwise/`: `manifest.json` (custom_id -> pair mapping; drift
hard-fails), `batches.json` (submitted batch ids + submission order), `attempts.json`
(format-failure counts; after `pairwise.max_format_retries` an item gets a terminal
`verdicts: null` file), `verdicts/{judge}/{custom_id}.json`, and `raw/` (verbatim batch
output for audit). `outstanding = expected − verdict files`, so every subcommand is
idempotent and safe to re-run.

## Prompt caching

Caching is declared on the request, not at the call site. `providers.LLMRequest` carries
`cache_part_idxs` (content parts that close a byte-stable prefix — Anthropic turns each into
an explicit `cache_control` breakpoint, and the system prompt always gets one) and
`cache_key` (OpenAI's `prompt_cache_key`; suppressed for other OpenAI-compatible endpoints
such as z.ai, which would reject the unknown field). Because the breakpoints live on the
request, the batch body and the synchronous body are byte-identical — a resubmitted
straggler caches the same way the batch did.

- Judges use `CACHE_PART_IDXS = (0, 1)`: part 0 (the task prompt) closes the prefix shared by
  *every* comparison; part 1 (candidate A's bundle) closes the prefix shared by every
  comparison with the same candidate in slot A — which is why `submit` sorts by `instance_a`.
  With the system prompt that is 3 of Anthropic's 4 allowed breakpoints.
- Candidates mark the prompt itself cacheable, which pays off for `n_samples > 1`, resumed
  runs, and re-runs inside the cache TTL.
- Gemini and z.ai cache matching prefixes automatically; ordering is what makes it work.

If you add a content part, keep the ordering most-stable-first — anything volatile must sit
after the last breakpoint, or caching silently stops paying.

## Bradley-Terry aggregation (optional ranking layer)

`footval/bradley_terry.py` (`make bradley-terry`) reads `outputs/data/pairwise_results.csv`
and fits a hierarchical Bayesian Bradley-Terry model in PyMC. With a single judged criterion
there is exactly one track (`soundness`), so `TRACKS = ("soundness",)`. It turns the
forced-choice verdicts into a latent-strength ranking **with 94% HDIs**, modeling the judges
as raters: per-judge discrimination (`kappa`), slot-A/position bias (`gamma`), and
self-family preference (`delta`), all partially pooled — so `mu_gamma`/`mu_delta` are
panel-wide bias estimates (a principled successor to the §7.3 win-rate gap). It is **purely
additive**: it consumes the published CSV and touches no prompt or prior artifact. Outputs to
`outputs/data/`: `bradley_terry_rankings.csv` (strength + `win_prob_vs_field` with HDIs,
rank), `bradley_terry_judge_effects.csv` (per-judge `kappa`/`gamma`/`delta` + a `panel
(mean)` row), and `bradley_terry.json` (full results + sampler diagnostics).

The full posterior **trace** is persisted (via `bt.trace_path(cfg, track)`) to
`artifacts/bradley_terry/soundness.nc` (gitignored) so the notebooks can recompute any HDI /
posterior-predictive quantity and run ArviZ diagnostics without re-fitting.

Notes: PyMC/ArviZ are imported lazily so the pure data-prep helpers stay testable without the
sampling stack (tests cover only data-prep, not the MCMC). Sampler knobs are module constants.
With only 4 judges the judge-level variances are weakly identified and lean on their priors —
read `mu_gamma`/`mu_delta` as directional. On the balanced round-robin the BT ranking matches
raw win% (Spearman ≈ 1.0); the value-add is the uncertainty bands and the judge-effect
decomposition, not a different ordering. Requires `pymc` + `arviz` + `h5netcdf`/`h5py` (added to
`pyproject.toml`); the notebooks additionally need `nbconvert` + `vl-convert-python` (dev group).

## Notebooks (`outputs/`)

Both read the persisted BT trace (`import footval`, `az.from_netcdf(bt.trace_path(...))`), so
run `make bradley-terry` first. Re-render headlessly with
`uv run jupyter nbconvert --to notebook --execute --inplace outputs/<nb>.ipynb`.

- `blog_output.ipynb` — the published charts. The leaderboard plots `win_prob_vs_field` as a
  point estimate with a thick **50% HDI** and thin **95% HDI** line (no axis rule); the
  by-judge line chart and the judge/candidate matrix use **posterior-predictive per-judge win
  probabilities** with HDIs. Subjective-priors, interrater-reliability, and order-consistency
  charts are computed from `responses.json` / raw verdicts. Charts export to
  `outputs/charts/*.svg`. There is no by-criterion leaderboard — with one criterion it would
  duplicate the overall one.
- `model_diagnostics.ipynb` — ArviZ convergence/sampling diagnostics for the track (summary
  tables, trace/rank/energy/forest/ESS plots, BFMI). Assessment-only; reports no results.

Both notebooks still carry stored outputs from the retired 12-model / two-criterion run;
they are refreshed by re-executing after the next full run.

## Conventions

- Python 3.12+, managed with `uv`. Type-hint public functions. Lint/format with `ruff`.
- Use **Polars** (not pandas) for tabular work.
- Tests with `pytest`; cover `checks.py` and `pairwise.py` thoroughly (they encode the methodology).
- Keep it simple and direct — no speculative abstraction. Match the spec; don't add scope.

## Critical constraints — read before writing code

- **Do not repair candidate output.** Malformed JSON or a missing grid cell is real signal. Capture it; never auto-fix, re-prompt, or hand-edit a candidate response.
- **Judges see anonymized responses.** Strip any model-identifying text before a response reaches a judge (`parsing.redact` covers the vendor list, including GLM/z.ai). Judging is the pairwise tournament: each call shows two anonymized responses and forces an A/B winner (no numeric scale, no ties).
- **Judging is locked to one criterion:** soundness, forced A/B choice, no 1–5 score and no composite. Don't reintroduce a numeric scoring scale or a priors criterion without updating the spec. Self- and same-family comparisons are tagged (`is_self_a`/`is_self_b`, `judge_family`) and kept, not dropped.
- **Objective vs. subjective split:** "did it parse / is the grid complete" comes from Stage 2, not from judge opinion. Judges weigh only the subjective quality, using those reports as ground truth.
- **Pin model snapshots.** The version strings in `config.yaml` will drift and some may not exist yet — resolve and record the exact API snapshot ID at run time (`make probe`). Never hardcode API keys.

## Data & artifacts

Stored under `artifacts/` (gitignored; optionally synced to S3). Objective tables (from the `tables` stage): `responses`, `checks`. Pairwise judging state lives under `artifacts/pairwise/` (manifest, batches, attempts, per-verdict files, raw batch output). Schema is in the spec.

## Secrets

API keys come from environment variables / a gitignored `.env`: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ZAI_API_KEY`. Never commit keys, raw responses, or rendered images.

## Publishing outputs

Written to `outputs/` (not auto-committed; owner reviews first). CSV data files live in `outputs/data/`:

- `pairwise_results.csv` — long form: columns `judge, model_a, model_b, criterion, winner`; one row per judge×pair (the `criterion` column is always `soundness`, kept so the schema survives a future second criterion); blank `winner` where a judge returned no valid verdict. The notebook (`outputs/blog_output.ipynb`) reads this plus `artifacts/tables/responses.json` to build the win-percentage and prior-comparison charts.
