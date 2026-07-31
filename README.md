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
>   know." Forcing each model to state this range is how Footval captures *confidence*, not
>   just opinion.

---

## The assignment, in one paragraph

Each model is told: *"You are a modern NFL head coach known for bucking convention and
accepting calculated risk. Your record is .500. Your only goal is to win more games."* From
there it must recommend exactly **six** underutilized strategies — laid out on a strict grid
(explained below) — and for each one attach a number describing how many points per game it
thinks the strategy is worth, **plus** an honest statement of how uncertain it is about that
number. The full, word-for-word prompt every model receives lives in
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

> **These priors are collected and charted, but they are not graded.** An earlier version of
> Footval asked the judge panel to score how *reasonable* each model's priors were; that
> criterion was removed. There is no true "points per game" number for a coaching strategy, so
> judges could only reward priors that matched their own unverified hunches. The priors are
> still interesting to look at side by side — they just no longer move the leaderboard. See
> [Scoring](#scoring).

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

The prompt is split into two parts and a strict output format. In full it asks each model to:

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

**Output format.** Everything must come back as a single JSON object with a fixed shape — a
`quantity` block and a `strategies` array of six objects (each with `bucket`, `type`, `title`,
`recommendation`, `rationale`, and a structured `prior`). Requiring a rigid structure is what
makes dozens of answers comparable and machine-checkable.

> **An earlier version also asked for a Part 3:** a self-contained Python script that plotted
> all six prior distributions. Footval ran that script in a locked-down container and recorded
> whether it worked. That part was removed from the prompt, and with it the whole
> code-execution stage — no model is asked to write code any more, and none is judged on it.

---

## Candidate models (the test-takers)

Eight models take the test — four "families" spanning flagship and mid-tier options:

| Family | Models |
|---|---|
| Anthropic (Claude) | `claude-fable-5`, `claude-opus-5`, `claude-sonnet-5` |
| OpenAI (GPT) | `gpt-5.6-sol`, `gpt-5.6-terra` |
| Google (Gemini) | `gemini-3.1-pro-preview`, `gemini-3.6-flash` |
| Z.ai (GLM) | `glm-5.2` |

Every candidate answers the **identical** prompt at a fixed creativity setting (temperature
1.0, where the API accepts it) and at a deliberately **high — but not maximum — reasoning
setting**: one notch above the middle of whatever "how hard should I think" ladder each
provider exposes. Each model answers once. Whatever it returns is taken as-is: **malformed or
broken answers are never repaired or re-requested** — a botched answer is treated as real
signal about the model, not a bug to fix.

---

## Judge models (the graders)

Footval is a **subjective** evaluation, so the grading is done by other AI models acting as
judges. A single four-model panel renders every head-to-head comparison (described under
[Scoring](#scoring)):

`claude-fable-5`, `gpt-5.6-sol`, `gemini-3.5-flash`, `glm-5.2`.

Judges deliberate at **one reasoning tier above the candidates** — the graders are given more
thinking room than the test-takers. (Gemini's ladder tops out at the candidates' level, so its
judge is the one exception.) Judges run as deterministically as their APIs allow
(temperature 0).

Two facts about the judges are important and are disclosed as limitations:

- **The judges are mostly contestants.** Three of the four judges are themselves on the
  candidate roster, and the fourth is a sibling of two candidates, so a judge will sometimes
  grade its own or a stablemate's answer. Rather than throw that data away, Footval *keeps*
  it, *labels* it ("is this the judge's own answer? its own family?"), and separately measures
  whether judges favor their own family.
- **Judges never see who wrote an answer.** Before any answer reaches a judge, every brand and
  model name (Claude, GPT, Gemini, GLM, etc.) is automatically stripped out and replaced
  with `[redacted]`, so judging is blind.

---

## Scoring

### Objective checks come first (no judging required)

Some things are simply facts, not opinions, so a computer settles them before any judge weighs
in. Footval runs a structural check on every answer: valid JSON format; a complete 6-cell grid
with distinct titles; sane intervals (low < best guess < high); a valid distribution family;
and — the clever one — **parameter reproduction**: Footval recomputes the 95% interval from the
model's own reported parameters and confirms it matches the interval the model claimed.

The first two of those (did it parse, is the grid complete and distinct) are handed to the
judges as **ground truth**, so judges don't have to re-derive mechanical facts. The last three
concern the priors, which are no longer judged — those results are published but deliberately
**withheld from the judges**, and the prior blocks themselves are **removed from the answers
the judges read** (an answer that failed to parse is shown raw and whole), so a stray
arithmetic slip in a prior can't quietly drag down a model's grade for its coaching ideas.

### How the judges grade — pairwise (head-to-head) competitions

The grading is a **tournament**: every model's answer is compared directly against every other
model's answer, and judges pick a winner each time. There are **no 1–5 ratings** — only
head-to-head wins.

- **Every unordered pair** of the 8 answers is compared. That's **28 pairs**.
- Comparisons cover **one criterion — soundness of the recommendations**: are the six picks
  genuinely underused, credibly win-positive, correctly filed in the grid, and defended with
  real football reasoning? Judges get a five-dimension rubric with an explicit priority order
  for close calls, and plain descriptions of what a *stronger* versus a *weaker* answer looks
  like — no numeric scale.
- Judges are told what **not** to reward: length, formatting polish, confident tone,
  statistics they can't verify, and exotic ideas that wouldn't survive an actual NFL Sunday.
- Each comparison is a **forced choice**: the judge *must* pick "A" or "B" — **ties are not
  allowed**.
- **Position bias is controlled for.** Which answer is labeled "A" versus "B" is decided by a
  fixed, reproducible coin-flip tied to the run's random seed. On top of that, every pair is
  judged in **both orders** (A-vs-B *and* B-vs-A), and the presentation order is recorded — so
  any tendency to favor whichever answer is shown first can be detected and measured.
- The **four-model judge panel** (listed above) each renders every comparison.

Putting that together: 28 pairs × 2 presentation orders × 4 judges = **224 head-to-head
verdicts.** Judges see only anonymized answers — with the unjudged prior blocks stripped
out — and are handed the relevant objective check results as ground truth.

The output is a simple tally — for each model, its **win percentage** (head-to-head wins ÷
comparisons). Two judge-agreement diagnostics ride alongside it: how decisively the four-judge
panel agreed on each comparison (unanimous / 3–1 / 2–2 split), and whether each judge kept the
same winner when the two answers were swapped. A **self-preference** read — each judge's win
rate for its own family versus the rest — flags conflicts of interest.

> **Cost note.** Because this round is large, comparisons are sent through the model providers'
> 50%-off batch processing where available (Anthropic, OpenAI, Google), and run synchronously for
> GLM, which has no batch option. The prompts are also built to be byte-stable, and carry explicit
> cache markers, so providers' prompt caching kicks in.

---

## How a run flows

The pipeline runs in stages, each reading and writing structured files so the whole run is
reproducible and any single stage can be re-run on its own:

| Stage | What it does |
|---|---|
| 1. Generate | Each candidate answers the prompt |
| 2. Check | The objective structural/numeric checks run |
| 3. Tables | The objective per-instance results are consolidated into tables |
| (judge) Pairwise | The head-to-head tournament: every pair compared on soundness |
| Publish | Win-percentage results and charts are written out |

---

## Where things live

- [`footval.prompt.md`](footval.prompt.md) — the exact prompt every candidate receives.
- [`footval.judge.prompt.md`](footval.judge.prompt.md) — the exact system prompt every pairwise judge receives (the soundness rubric, ground rules, verdict schema).
- [`initial_prompt.md`](initial_prompt.md) — the full methodology specification.
- [`config.yaml`](config.yaml) — the candidate list, the pairwise judge panel, and per-model settings (including the separate candidate/judge reasoning levels).
- [`CLAUDE.md`](CLAUDE.md) — orientation for developers working on the code.
- `footval/` — the Python implementation, one module per stage.
- `artifacts/` — the structured per-stage outputs (not committed).
- `outputs/data/` — the final results (`pairwise_results.csv`), plus an optional hierarchical
  Bayesian Bradley-Terry ranking with uncertainty bands and judge-bias estimates
  (`bradley_terry_rankings.csv`, `bradley_terry_judge_effects.csv`, `bradley_terry.json`).
- `outputs/blog_output.ipynb` — builds the published charts (Bayesian leaderboard with 50% / 95%
  HDIs, posterior-predictive per-judge charts, and the prior/agreement diagnostics).
- `outputs/model_diagnostics.ipynb` — ArviZ convergence diagnostics for the Bradley-Terry fit.
- `artifacts/bradley_terry/soundness.nc` — the persisted PyMC posterior trace both notebooks read (not committed).

---

## Limitations, gaps, and ways this could mislead

Footval is a fun, carefully built experiment — but it is an experiment, and a skeptical
reader should treat its rankings as *suggestive, not authoritative*. The honest case against
taking the numbers at face value is below, roughly in order of how much it should worry you.

### 1. The judges are the contestants — the whole thing is somewhat circular

Every judge is drawn from the model families being tested — three of the four are candidates
themselves — and there is **no human, no expert, and no external ground truth anywhere in the
loop**. That creates two distinct problems:

- **Self-preference.** A model grading (a disguised copy of) its own answer has an obvious
  conflict of interest. Footval mitigates this — judging is blind, every pair is judged in both
  presentation orders, and a per-judge own-family-vs-others win-rate gap is reported — but
  mitigation isn't elimination. With only four judges, all of them contestants or siblings of
  contestants, a shared family lean can still tilt results.
- **Shared blind spots masquerading as agreement.** If the judges absorbed the same popular
  football takes from the same internet during training, they will agree *with each other* and
  *with candidates that echo those takes* — even if those takes are wrong. High inter-rater
  agreement would then look like reliability when it's really shared bias. **Agreement is not
  the same as correctness.** Footval measures the former and silently hopes it implies the
  latter.

### 2. "Blind" judging is only partly blind

Anonymization is a regular-expression scrub of brand and model names (`claude`, `gpt`,
`gemini`, `glm`, etc.). It cannot remove a model's **stylistic fingerprint** — house JSON
formatting, characteristic phrasings, the way it hedges, even its favorite distribution family.
Frontier models are fairly good at recognizing each other's writing, so a judge may *infer*
authorship (including its own) despite the scrub. The redaction also can't catch an identity
the regex doesn't list, and could over-redact ordinary football words that happen to match.

### 3. There is no ground truth for the thing actually being judged

The single judged criterion rewards a judgment that **cannot be checked against reality**.
Whether a strategy is genuinely underused *and* win-positive is a contested, opinion-laden
call. In the previous run the four-judge panel reached unanimous verdicts *least* often on
exactly this criterion (the most 3–1 and 2–2 splits) — so what is now the whole leaderboard was
the noisiest part of the old one, and narrow win-percentage gaps are close to meaningless.

Footval can tell you which answers *other LLMs like*, not which answers are *actually good
coaching*.

### 4. "Underutilized" drifts with time and training data

What counts as underused in the NFL moves every season — fourth-down aggression was a daring
analytics pick a decade ago and is closer to mainstream now. A model with a later training
cutoff has a different picture of "the current meta" than one with an earlier cutoff, both as a
*candidate* (what it proposes) and as a *judge* (what it considers underused). The judge prompt
tells judges to grade against *current* adoption rather than novelty, but that instruction can
only work as well as the judge's own knowledge of the current league. The evaluation therefore
partly rewards recency of training data, which has nothing to do with reasoning quality.

### 5. One sample, at maximum randomness

Each model answers **exactly once**, at temperature 1.0 (high creativity/randomness). A single
sample is one noisy draw from the model's distribution of possible answers, not its best or its
typical answer — and there is no measure of within-model variance. A re-run with a different
draw could reorder the leaderboard, especially among closely matched models — and the whole
head-to-head tournament is built on those same single answers.

### 6. The models are not run under equal conditions

Despite the "identical prompt" framing, the playing field is not perfectly level:

- **Thinking ladders aren't comparable across vendors.** Every candidate runs one notch above
  the middle of its own provider's reasoning ladder, but "high" on one vendor's scale is not
  calibrated against "high" on another's, and Gemini's ladder has four rungs where Anthropic's
  and OpenAI's have five. So part of what's being compared is reasoning *budget and
  configuration*, not just model quality.
- **The judge tier is uneven too.** Judges are meant to sit one tier above the candidates;
  Gemini's judge can't, because its ladder already tops out at the candidate level.
- **Temperature isn't actually uniform.** Reasoning models that reject a temperature setting
  simply don't get the 1.0 the others do, so "all at temperature 1.0" is aspirational.

### 7. The auto-checks are treated as ground truth but aren't infallible

Judges are explicitly told to trust the structural-check report as fact and not re-litigate it.
That's good for objectivity — but it means **any error or arbitrary choice in those checks
propagates straight into the verdicts**. This is much less exposed than it used to be: only the
JSON-validity and grid checks now reach judges, and the fragile ones (notably the
parameter-reproduction check, which relies on a 10%-of-interval tolerance and an approximate
skew-normal mode) are withheld. But a wrong grid verdict would still propagate to every judge.

### 8. Instruction-following is blended into "reasoning"

The prompt is long and demanding: exact JSON, an exact six-cell grid, explicit conversion math.
A model with brilliant football ideas but sloppy formatting can fail to parse and lose across
the board, while a mediocre thinker with immaculate JSON sails through the mechanical checks.
The benchmark therefore conflates **format compliance** with **domain reasoning**, and partly
rewards the former.

### 9. The grid is artificial — and partly gameable

Forcing exactly one pick per (analytics/intuition × phase) cell is a clean design but an
unnatural one: real coaching priorities aren't evenly distributed across a 2×3 matrix, and the
format can force a model to manufacture a weak pick to fill a cell. Worse, the
analytics-vs-intuition label is **self-assigned** — the automated grid check only confirms that
each cell is *labeled*, not that the strategy truly belongs there. A model can park an analytics
consensus pick in an "intuition" cell; the judge rubric explicitly asks judges to catch that,
but catching it is left entirely to judges who may not.

### 10. The judging covers only part of the task, and forced choices add noise

- The tournament judges **only the six recommendations**. The Bayesian priors are half of what
  the prompt demands, and they are collected, checked, and charted but never scored — so the
  entire ranking rests on one half of the assignment. (This is a deliberate change: the priors
  criterion was removed because there was no defensible standard to grade it against. The
  honest consequence is narrower coverage, not better coverage.)
- Pairwise forbids ties. When two answers are genuinely equal, the judge is forced to flip a
  coin, and that coin flip is recorded as a real "win." Across a field of closely clustered
  frontier models, a meaningful share of the head-to-head record can be essentially noise dressed
  up as signal. (Judging both presentation orders detects *position* bias, but doesn't fix the
  tie problem.)

### 11. The summary statistics are themselves shaky

- Win percentages come from **8 candidates and 4 judges** on a single set of answers — a small
  sample whose own uncertainty is wide. The optional Bradley-Terry layer reports that
  uncertainty as HDIs; the raw win-percentage tally does not. Read the ordering loosely; narrow
  gaps are within the noise.
- The judge-agreement diagnostics (vote splits, order consistency) are panel-consensus readouts,
  **not** formal reliability statistics — and high agreement can reflect shared bias rather than
  accuracy (see §1).

### 12. Narrow scope and single configuration

Everything runs from **one random seed, one prompt, in one narrow domain (NFL strategy), in
English**. Results say nothing about other sports, other tasks, or robustness to rephrasing the
prompt or reshuffling the A/B coin flips. Football knowledge on the public internet is abundant,
so models may be **recalling popular analytics discourse rather than reasoning** — the intuition
cells push against this, but can't guarantee it.

### 13. Operational fragility

The run depends on live, drifting third-party APIs and paid balances. Model snapshots move,
batch APIs change, and a vendor outage on the wrong day could quietly shrink the field. The
pipeline is carefully made reproducible *given the stored artifacts*, but the *generation* of
those artifacts is at the mercy of external services.

### Bottom line

Footval is best read as **"which answers did a small panel of frontier LLMs prefer, on one
quirky football task, on one day"** — an interesting lens on model behavior and a nice showcase
of structured-output evaluation, not a definitive verdict on which model is the better reasoner.
The most trustworthy outputs are the *objective* ones (is the JSON well-formed, is the grid
complete, do the parameters reproduce the interval); the *subjective* ranking deserves the most
skepticism, and the design's own weak agreement on "soundness" is the clearest evidence of that.
