# Footval — a football-coaching evaluation for large language models

Footval is a small, opinionated test of how well today's leading AI models can think
**like a football coach**. Every model is handed the exact same assignment: play the role of
an NFL head coach, recommend smart-but-underused strategies, and — crucially — say honestly
how *sure* it is about each one. The models' answers are then graded, partly by computer and
partly by a panel of other AI models acting as judges.

This document explains what the test asks, who takes it, who grades it, and how the grading
works. It does **not** report who won — it only describes the evaluation itself.

> **A note on jargon.** Two words show up a lot:
> - A **prior** is just a best guess stated *before* you've seen the outcome — a coach's
>   honest hunch about how much a strategy will help.
> - An **uncertainty interval** is the range the model is fairly confident the true answer
>   falls inside. A narrow range means "I'm pretty sure"; a wide range means "I really don't
>   know." Forcing each model to state this range is how Footval measures *confidence*, not
>   just opinion.

---

## The assignment, in one paragraph

Each model is told: *"You are a modern NFL head coach known for bucking convention and
accepting calculated risk. Your record is .500. Your only goal is to win more games."* From
there it must recommend exactly **six** underutilized strategies — laid out on a strict grid
(explained below) — and for each one attach a number describing how many points per game it
thinks the strategy is worth, **plus** an honest statement of how uncertain it is about that
number. Finally, it must write a short Python program that draws all six of those
uncertainty curves. The full, word-for-word prompt every model receives lives in
[`footval.prompt.md`](footval.prompt.md).

---

## Types of questions

Footval deliberately mixes three different *kinds* of thinking. A model can be great at one
and weak at another, and the test is designed to pull those apart.

### 1. Reasoning with evidence (the "analytics" row)

Half the recommendations must be **backed by public football analytics** — win-probability
models, expected-points research, published findings. The model is asked to surface ideas
that the data supports but that real coaches still underuse (the classic example being
"go for it on 4th down more often"). This rewards models that actually know the football
analytics literature and can reason from it.

### 2. Reasoning on intuition (the "intuition" row)

The other half must come from the coach's **own judgment in areas where public analytics is
absent, silent, or mixed**. The model has to briefly justify *why* the idea qualifies as
intuition rather than settled analytics — and it is explicitly told **not** to just restate
the analytics consensus in different words. This tests whether a model can form an original,
defensible opinion when the data won't hand it the answer.

### 3. Measuring confidence in its beliefs (the "priors")

For every one of the six strategies, the model must quantify its uncertainty as a **Bayesian
prior** over a single, precisely defined quantity:

> the change in the team's **expected single-game scoring margin** (your points minus the
> opponent's), in **points**, caused by adopting the strategy versus not — on average, not in
> any one game. Positive means it helps; negative means it backfires.

To express that belief the model must provide:

- a **most-plausible value** (its single best guess, and it's told *not* to default to round
  numbers);
- a **95% interval** — the low/high range it's ~95% sure contains the true effect;
- the **shape** of its belief (symmetric, or skewed toward the upside or downside) and how
  **heavy the tails** are (how much chance it gives to a surprisingly large effect);
- a matching **probability distribution family** — `normal`, `student_t`, or `skew_normal` —
  with the actual parameters, *and the math showing how it converted its plain-English belief
  into those parameters.*

This is the heart of the test: it's easy to have opinions, harder to attach calibrated,
internally consistent confidence to them. A weak idea is supposed to be reported *honestly* —
a near-zero best guess with a wide interval — not dressed up to look strong.

---

## Categories of questions

Cutting across those two rows are three **phases of football**. The model must produce a
recommendation in each phase for *both* the evidence row and the intuition row — so the six
strategies form a rigid **2 × 3 grid**, one idea per cell, all six distinct:

|                                  | **Game management** | **Offense** | **Defense** |
|----------------------------------|:------------------:|:-----------:|:-----------:|
| **Analytics** (reasoning w/ evidence) | ✔ | ✔ | ✔ |
| **Intuition** (reasoning on judgment) | ✔ | ✔ | ✔ |

- **Game management** — clock and timeout use, challenges, 4th-down and 2-point decisions, and
  similar in-game choices.
- **Offense** — play-calling, scheme, personnel groupings, tempo.
- **Defense** — play-calling, scheme, coverage, personnel.

If a particular cell only yields a weak idea, the model is told to **pick it anyway and be
honest about the weakness** in its prior, rather than inflate a mediocre pick to fill the
grid. Filling the grid without cheating is itself part of what's being measured.

---

## What the prompt actually asks for

The prompt is split into three parts and a strict output format. In full it asks each model to:

**Part 1 — Strategy recommendations.** Exactly six strategies, one per grid cell, all
distinct. For each: a 1–3 sentence recommendation and a rationale tied directly to winning
games. Analytics picks must be genuinely supported by public analytics and underused;
intuition picks must be the coach's own judgment, with a short note on why it isn't just
analytics consensus.

**Part 2 — A prior over each strategy's effect.** For all six, the belief summaries, the
distribution family, the parameters with the conversion math shown, a self-consistency check
(best guess sits inside the interval; the distribution allows negative values), and an honesty
note. The prompt gives the exact conversion formulas — for example, for a normal distribution,
σ = (high − low) / 3.92 — so a model's arithmetic can be checked.

**Part 3 — One plotting script.** A single self-contained Python script (using only `numpy`,
`scipy.stats`, and `matplotlib`) that holds all six strategies' parameters in an editable list,
draws each distribution's curve, marks the most-plausible value, shades the 95% interval it
computes *from the distribution itself*, labels the x-axis "Change in own scoring margin
(points)," and prints each strategy's family and parameters.

**Output format.** Everything must come back as a single JSON object with a fixed shape — a
`quantity` block, a `strategies` array of six objects (each with `bucket`, `type`, `title`,
`recommendation`, `rationale`, and a structured `prior`), and the `plot_script` as one escaped
string. Requiring a rigid structure is what makes hundreds of answers comparable and machine-
checkable.

---

## Candidate models (the test-takers)

Twelve models take the test — four "families" of three or four models each, spanning flagship,
mid-tier, and lightweight options:

| Family | Models |
|---|---|
| Anthropic (Claude) | `claude-fable-5`, `claude-opus-4-8`, `claude-sonnet-5`, `claude-haiku-4-5-20251001` |
| OpenAI (GPT) | `gpt-5.5-pro`, `gpt-5.5`, `gpt-5.4-mini` |
| Google (Gemini) | `gemini-3.1-pro-preview`, `gemini-3.5-flash`, `gemini-3.1-flash-lite` |
| DeepSeek | `deepseek-v4-pro`, `deepseek-v4-flash` |

Every candidate answers the **identical** prompt, with the highest level of reasoning ("thinking")
each model supports turned on, at a fixed creativity setting (temperature 1.0). Each model
answers once. Whatever it returns is taken as-is: **malformed or broken answers are never
repaired or re-requested** — a botched answer is treated as real signal about the model, not a
bug to fix.

---

## Judge models (the graders)

Footval is a **subjective** evaluation, so the grading is done by other AI models acting as
judges. A single four-model panel renders every head-to-head comparison (described under
[Scoring](#scoring)):

`claude-opus-4-8`, `gpt-5.5`, `gemini-3.5-flash`, `deepseek-v4-pro`.

Two facts about the judges are important and are disclosed as limitations:

- **The judges are also contestants.** Every judge is itself one of the candidate families, so
  a judge will sometimes grade its own or a sibling model's answer. Rather than throw that data
  away, Footval *keeps* it, *labels* it ("is this the judge's own answer? its own family?"),
  and separately measures whether judges favor their own family.
- **Judges never see who wrote an answer.** Before any answer reaches a judge, every brand and
  model name (Claude, GPT, Gemini, DeepSeek, etc.) is automatically stripped out and replaced
  with `[redacted]`, so judging is blind.

Judges run as deterministically as their APIs allow (temperature 0).

---

## Scoring

### Objective checks come first (no judging required)

Some things are simply facts, not opinions, so a computer settles them before any judge weighs
in. Footval runs two automated stages:

- **Execution.** Each model's Python plotting script is actually run — inside a locked-down
  sandbox with no internet, capped memory and CPU, and a hard 60-second timeout (the code is
  untrusted, model-written code). Whether it ran, and the images it produced, are recorded.
- **Structural checks.** The JSON is checked for: valid format; a complete 6-cell grid with
  distinct titles; sane intervals (low < best guess < high); a valid distribution family; and —
  the clever one — **parameter reproduction**: Footval recomputes the 95% interval from the
  model's own reported parameters and confirms it matches the interval the model claimed.

These objective results are handed to the judges as **ground truth**, so judges don't have to
guess whether code ran or whether the math is consistent — they spend their judgment on quality.

### How the judges grade — pairwise (head-to-head) competitions

The grading is a **tournament**: every model's answer is compared directly against every other
model's answer, and judges pick a winner each time. There are **no 1–5 ratings** — only
head-to-head wins.

- **Every unordered pair** of the 12 answers is compared. That's **66 pairs**.
- Comparisons cover **two criteria — soundness and priors** (the round is text-only, so a
  code/plots criterion isn't judged). For each, judges are given a plain description of what a
  *stronger* versus a *weaker* answer looks like and pick the better one — no numeric scale.
- Each comparison is a **forced choice**: for each criterion the judge *must* pick "A" or "B" —
  **ties are not allowed**. Judges are told to judge each criterion independently, so the same
  answer needn't win both.
- **Position bias is controlled for.** Which answer is labeled "A" versus "B" is decided by a
  fixed, reproducible coin-flip tied to the run's random seed. On top of that, every pair is
  judged in **both orders** (A-vs-B *and* B-vs-A), and the presentation order is recorded — so
  any tendency to favor whichever answer is shown first can be detected and measured.
- The **four-model judge panel** (listed above) each renders every comparison.

Putting that together: 66 pairs × 2 presentation orders × 4 judges = **528 head-to-head
verdicts per criterion.** Judges see only anonymized answers and are handed the objective check
report as ground truth for the mechanical facts.

The output is a simple tally — for each model, its **win percentage** (head-to-head wins ÷
comparisons), overall and split by soundness and priors. Two judge-agreement diagnostics ride
alongside it: how decisively the four-judge panel agreed on each comparison (unanimous / 3–1 /
2–2 split), and whether each judge kept the same winner when the two answers were swapped. A
**self-preference** read — each judge's win rate for its own family versus the rest — flags
conflicts of interest.

> **Cost note.** Because this round is large, comparisons are sent through the model providers'
> 50%-off batch processing where available (Anthropic, OpenAI, Google), and run synchronously for
> DeepSeek, which has no batch option. The prompts are also built to be byte-stable so providers'
> prompt caching kicks in.

---

## How a run flows

The pipeline runs in stages, each reading and writing structured files so the whole run is
reproducible and any single stage can be re-run on its own:

| Stage | What it does |
|---|---|
| 1. Generate | Each candidate answers the prompt |
| 2. Execute | Each plot script runs in the sandbox; images are rendered |
| 3. Check | The objective structural/numeric checks run |
| 4. Tables | The objective per-instance results are consolidated into tables |
| (judge) Pairwise | The head-to-head tournament: every pair compared on soundness and priors |
| Publish | Win-percentage results and charts are written out |

---

## Where things live

- [`footval.prompt.md`](footval.prompt.md) — the exact prompt every candidate receives.
- [`initial_prompt.md`](initial_prompt.md) — the full methodology specification.
- [`config.yaml`](config.yaml) — the candidate list, the pairwise judge panel, and per-model settings.
- [`CLAUDE.md`](CLAUDE.md) — orientation for developers working on the code.
- `footval/` — the Python implementation, one module per stage.
- `artifacts/` — the structured per-stage outputs (not committed).
- `outputs/data/` — the final results (`pairwise_results.csv`), plus an optional hierarchical
  Bayesian Bradley-Terry ranking with uncertainty bands and judge-bias estimates
  (`bradley_terry_rankings.csv`, `bradley_terry_judge_effects.csv`, `bradley_terry.json`).
- `outputs/blog_output.ipynb` — builds the published charts (Bayesian leaderboards with 50% / 95%
  HDIs, posterior-predictive per-judge charts, and the prior/agreement diagnostics).
- `outputs/model_diagnostics.ipynb` — ArviZ convergence diagnostics for the Bradley-Terry fit.
- `artifacts/bradley_terry/*.nc` — the persisted PyMC posterior traces both notebooks read (not committed).

---

## Limitations, gaps, and ways this could mislead

Footval is a fun, carefully built experiment — but it is an experiment, and a skeptical
reader should treat its rankings as *suggestive, not authoritative*. The honest case against
taking the numbers at face value is below, roughly in order of how much it should worry you.

### 1. The judges are the contestants — the whole thing is somewhat circular

Every judge is itself one of the model families being tested, and there is **no human, no
expert, and no external ground truth anywhere in the loop**. That creates two distinct
problems:

- **Self-preference.** A model grading (a disguised copy of) its own answer has an obvious
  conflict of interest. Footval mitigates this — judging is blind, every pair is judged in both
  presentation orders, and a per-judge own-family-vs-others win-rate gap is reported — but
  mitigation isn't elimination. With only four judges, two of which come from the largest
  families, a shared family lean can still tilt results.
- **Shared blind spots masquerading as agreement.** If the judges absorbed the same popular
  football takes from the same internet during training, they will agree *with each other* and
  *with candidates that echo those takes* — even if those takes are wrong. High inter-rater
  agreement would then look like reliability when it's really shared bias. **Agreement is not
  the same as correctness.** Footval measures the former and silently hopes it implies the
  latter.

### 2. "Blind" judging is only partly blind

Anonymization is a regular-expression scrub of brand and model names (`claude`, `gpt`,
`gemini`, `deepseek`, etc.). It cannot remove a model's **stylistic fingerprint** — house JSON
formatting, characteristic phrasings, the way it hedges, even its favorite distribution family.
Frontier models are fairly good at recognizing each other's writing, so a judge may *infer*
authorship (including its own) despite the scrub. The redaction also can't catch an identity
the regex doesn't list, and could over-redact ordinary football words that happen to match.

### 3. There is no ground truth for the thing actually being judged

Both judged criteria reward judgments that **cannot be checked against reality**:

- **"Soundness / underutilized."** Whether a strategy is genuinely underused *and*
  win-positive is a contested, opinion-laden call. Tellingly, the four-judge panel reached
  unanimous verdicts least often on this criterion (the most 3–1 and 2–2 splits) — so the
  soundness ranking is the noisiest part of the whole exercise, and narrow win-percentage gaps
  there are close to meaningless.
- **"Reasonable priors."** There is no true number for "how many points per game does adopting
  this strategy add." Judges reward priors whose *magnitude feels football-realistic to them* —
  i.e., priors that match the judges' own (unverified) intuitions. A model that is genuinely
  well-calibrated to reality could be marked down by judges with miscalibrated intuitions, and
  vice versa. Notably, the panel agreed most often on priors — but that may partly reflect judges
  anchoring on the same auto-check report (see §7) rather than independently assessing realism.

In short, Footval can tell you which answers *other LLMs like*, not which answers are
*actually good coaching*.

### 4. "Underutilized" drifts with time and training data

What counts as underused in the NFL moves every season — fourth-down aggression was a daring
analytics pick a decade ago and is closer to mainstream now. A model with a later training
cutoff has a different picture of "the current meta" than one with an earlier cutoff, both as a
*candidate* (what it proposes) and as a *judge* (what it considers underused). The evaluation
therefore partly rewards recency of training data, which has nothing to do with reasoning
quality.

### 5. One sample, at maximum randomness

Each model answers **exactly once**, at temperature 1.0 (high creativity/randomness). A single
sample is one noisy draw from the model's distribution of possible answers, not its best or its
typical answer — and there is no measure of within-model variance. A re-run with a different
draw could reorder the leaderboard, especially among closely matched models — and the whole
head-to-head tournament is built on those same single answers.

### 6. The models are not run under equal conditions

Despite the "identical prompt" framing, the playing field is not perfectly level:

- **Thinking budgets differ by necessity.** The spec asks for "the highest thinking available,"
  but that means different things per model. Some models run with adaptive/maximum reasoning
  effort; Sonnet had to be capped at a fixed thinking budget because higher settings made it
  think past its output limit and return *nothing*; Haiku uses yet another setting. So part of
  what's being compared is reasoning *budget and configuration*, not just model quality.
- **Temperature isn't actually uniform.** Reasoning models that reject a temperature setting
  simply don't get the 1.0 the others do, so "all at temperature 1.0" is aspirational.

### 7. The auto-checks are treated as ground truth but aren't infallible

Judges are explicitly told to trust the execution and structural-check reports as fact and not
re-litigate them. That's good for objectivity — but it means **any error or arbitrary choice in
those checks propagates straight into the verdicts**. The parameter-reproduction check in
particular relies on a 10%-of-interval tolerance and, for skew-normal priors, on an approximate
mode computation the spec itself flags as approximate; it flips to "fail" for most candidates,
and judges anchor on that verdict. A miscalibrated tolerance would systematically punish or
excuse the same models across every judge.

### 8. Instruction-following is blended into "reasoning"

The prompt is long and demanding: exact JSON, an exact six-cell grid, explicit conversion math,
and a Python script encoded as an escaped string. A model with brilliant football ideas but
sloppy formatting can fail to parse and crater across *all* criteria, while a mediocre thinker
with immaculate JSON sails through the mechanical checks. The benchmark therefore conflates
**format compliance** with **domain reasoning**, and partly rewards the former.

### 9. The grid is artificial — and partly gameable

Forcing exactly one pick per (analytics/intuition × phase) cell is a clean design but an
unnatural one: real coaching priorities aren't evenly distributed across a 2×3 matrix, and the
format can force a model to manufacture a weak pick to fill a cell. Worse, the
analytics-vs-intuition label is **self-assigned** — the automated grid check only confirms that
each cell is *labeled*, not that the strategy truly belongs there. A model can park an analytics
consensus pick in an "intuition" cell, and catching that mislabel is left entirely to judges who
may not.

### 10. The judging covers only part of the task, and forced choices add noise

- The pairwise tournament covers **only soundness and priors** — it's text-only, so the Python
  code/plot work the prompt demands is not judged at all. Two-thirds of what the prompt asks for
  therefore drives the entire ranking, and code quality goes unscored.
- Pairwise forbids ties. When two answers are genuinely equal, the judge is forced to flip a
  coin, and that coin flip is recorded as a real "win." Across a field of closely clustered
  frontier models, a meaningful share of the head-to-head record can be essentially noise dressed
  up as signal. (Judging both presentation orders detects *position* bias, but doesn't fix the
  tie problem.)

### 11. The summary statistics are themselves shaky

- Win percentages come from **12 candidates and 4 judges** on a single set of answers — a small
  sample whose own uncertainty is wide and isn't reported. Read the ordering loosely; narrow gaps
  are within the noise.
- The judge-agreement diagnostics (vote splits, order consistency) are panel-consensus readouts,
  **not** formal reliability statistics — and high agreement can reflect shared bias rather than
  accuracy (see §1).
- Overall win percentage pools soundness and priors equally; code quality isn't judged at all
  (see §10). A different weighting of the two judged criteria would move the leaderboard.

### 12. Narrow scope and single configuration

Everything runs from **one random seed, one prompt, in one narrow domain (NFL strategy), in
English**. Results say nothing about other sports, other tasks, or robustness to rephrasing the
prompt or reshuffling the A/B coin flips. Football knowledge on the public internet is abundant,
so models may be **recalling popular analytics discourse rather than reasoning** — the intuition
cells push against this, but can't guarantee it.

### 13. Operational fragility

The run depends on live, drifting third-party APIs and paid balances — during setup, DeepSeek
briefly failed with an "insufficient balance" error before being topped up and included. Model
snapshots move, batch APIs change, and a vendor outage on the wrong day could quietly shrink the
field. The pipeline is carefully made reproducible *given the stored artifacts*, but the
*generation* of those artifacts is at the mercy of external services.

### Bottom line

Footval is best read as **"which answers did a small panel of frontier LLMs prefer, on one
quirky football task, on one day"** — an interesting lens on model behavior and a nice showcase
of structured-output evaluation, not a definitive verdict on which model is the better reasoner.
The most trustworthy outputs are the *objective* ones (did the code run, is the JSON
well-formed, do the parameters reproduce the interval); the *subjective* rankings deserve the
most skepticism, and the design's own weak agreement on "soundness" is the clearest evidence of
that.
