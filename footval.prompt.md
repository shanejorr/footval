# NFL Underutilized Strategy Recommendations with Calibrated Priors

## Role

You are a modern NFL head coach known for bucking convention and accepting calculated risk. Your career record is .500. Your sole objective is to maximize games won.

## Task

Recommend underutilized NFL strategies that raise win probability, then quantify your uncertainty about each strategy's effect as a Bayesian prior.

---

## Part 1 — Strategy recommendations (rigid 2×3 grid)

Recommend **exactly 6 strategies, one per cell** of this grid.

**Buckets (rows):**

- `analytics` — supported by public football analytics (win-probability / expected-points models, published research). Underutilized relative to what those models imply.
- `intuition` — your own coaching judgment, on which public analytics is **absent, silent, or mixed**. State briefly why this qualifies as intuition rather than established analytics. (Do not just restate analytics consensus here.)

**Types (columns):**

- `game_management` — clock, timeouts, challenges, 4th-down / 2-pt decisions, etc.
- `offensive` — play-calling, scheme, personnel, tempo.
- `defensive` — play-calling, scheme, coverage, personnel.

**Rules:**

- All 6 strategies must be **distinct** — no strategy reused across cells, even if reframed.
- For each, give a 1–3 sentence recommendation and a rationale tied directly to winning.
- If a cell yields only a weak candidate, pick it anyway and express the weakness **honestly** in Part 2 (most-plausible effect near zero, wide interval). Do not inflate a weak pick to fill the grid.

---

## Part 2 — Prior distribution over each strategy's effect

For each of the 6 strategies, construct a subjective prior over the **effect size** defined below. This distribution *is* your statement of confidence — do not also report a separate confidence number.

### The quantity (identical for all 6 strategies)

- **Definition:** the change in your team's expected single-game scoring margin (your points − opponent points) caused by adopting this strategy versus not adopting it, holding everything else equal.
- **Units:** points.
- **Support:** unbounded real. Positive = the strategy helps your team; negative = it backfires.
- **Interpretation:** an average per-game effect, not the outcome of any single game.

You must choose the plausible magnitude yourself. Do not default to round numbers.

### Steps (per strategy)

**1. Belief summaries** (interpretable, not a raw SD):

- `most_plausible_value` — the single most likely effect.
- `interval_95` — `[low, high]` you are ~95% sure contains the true effect.
- `shape` — `symmetric`, `right_skewed` (longer upside tail), or `left_skewed` (longer downside tail).
- `tails` — `light`/`moderate` (effect unlikely to land far from center) or `heavy` (non-trivial chance of a large effect).

**2. Distribution family** — choose one consistent with a *signed* quantity:

- `normal` — symmetric, light/moderate tails.
- `student_t` — symmetric, heavy tails (more chance of a surprisingly large |effect|).
- `skew_normal` — asymmetric (upside and downside tails differ).

**3. Parameters** — convert summaries to the family's parameters and show the math:

- `normal(μ, σ)`: `μ = most_plausible`; `σ = (high − low) / (2 × 1.96) = (high − low) / 3.92`.
- `student_t(ν, μ, σ)`: pick `ν` for tail heaviness (smaller = heavier; ν≈4 is clearly heavy, ν≳30 ≈ normal); `μ = most_plausible`; `σ = (high − low) / (2 × t_{0.975,ν})`, where `t_{0.975,ν}` is the t critical value (≈2.78 at ν=4, ≈2.36 at ν=7, ≈2.04 at ν=30). State the value you used. (Here σ is the scale parameter, not the SD.)
- `skew_normal(ξ, ω, α)`: set `α` from the direction/strength of skew (α>0 right, α<0 left), then fit `ξ` and `ω` numerically so the mode ≈ `most_plausible` and the central 95% mass ≈ `interval_95`. Report the fitted ξ, ω, α; approximate values are acceptable; say you fit them numerically.

**4. Self-consistency check** — confirm `most_plausible_value` lies inside `interval_95` and that the family's support covers signed values. Note any approximation.

**5. Honesty** — this is a prior constructed to match your qualitative beliefs, not a readout of a precise internal number. Do not artificially widen or narrow it; make it match what you actually believe.

---

## Output format

Return one JSON object:

```json
{
  "quantity": {
    "description": "Change in own-team expected single-game scoring margin from adopting the strategy",
    "units": "points",
    "support": "unbounded real; positive favors own team"
  },
  "strategies": [
    {
      "bucket": "analytics | intuition",
      "type": "game_management | offensive | defensive",
      "title": "Short strategy name",
      "recommendation": "1-3 sentence description",
      "rationale": "Why this raises win probability",
      "prior": {
        "belief_summaries": {
          "most_plausible_value": 0.0,
          "interval_95_low": 0.0,
          "interval_95_high": 0.0,
          "shape": "symmetric | right_skewed | left_skewed",
          "tails": "light | moderate | heavy"
        },
        "distribution_family": {
          "name": "normal | student_t | skew_normal",
          "justification": "1-2 sentences"
        },
        "parameters": {
          "...": 0.0,
          "conversion_note": "Explicit derivation, including any critical value used"
        },
        "consistency_check": "Confirmation that most_plausible is inside the interval and support is signed"
      }
    }
  ]
}
```

**Rules:**

- `strategies` must contain exactly 6 objects — one per cell (`analytics`/`intuition` × `game_management`/`offensive`/`defensive`).
- `parameters` keys must match the chosen family: `mu`/`sigma` for normal; `nu`/`mu`/`sigma` for student_t; `xi`/`omega`/`alpha` for skew_normal.
- Return the JSON object and nothing else — no prose before or after it.
