# Footval Bradley-Terry Model — A Complete Explanation

This document explains, end to end, the statistical model implemented in
[`footval/bradley_terry.py`](../footval/bradley_terry.py). It assumes an
intermediate familiarity with Bayesian modeling (priors, likelihoods,
posteriors, MCMC, partial pooling) but it explains every modeling decision,
every parameter, and every line of the PyMC code so that a reader can fully
reconstruct and defend the model.

The goal of the stage is narrow and specific: take the raw forced-choice
verdicts from the pairwise judging tournament and turn them into a **latent
strength ranking of the candidate models, with honest uncertainty bands**,
while simultaneously estimating and correcting for the biases of the judges
doing the rating.

---

## 1. The problem this model solves

### 1.1 What the data looks like

The input is `outputs/data/pairwise_results.csv`, the long-form output of the
pairwise tournament. One row per **judge × pair × criterion**, with `criterion`
either `analytical_reasoning` or `intuitive_reasoning`:

| judge | model_a | model_b | criterion | winner |
|---|---|---|---|---|
| claude-fable-5 | gemini-3.6-flash | claude-opus-5 | analytical_reasoning | claude-opus-5 |
| gpt-5.6-sol | gemini-3.6-flash | claude-opus-5 | intuitive_reasoning | gemini-3.6-flash |
| … | … | … | … | … |

Each row is a single **forced binary comparison**: a judge looked at two
anonymized responses (shown as "Response A" and "Response B") and was forced to
pick a winner on one criterion. There is no numeric score, no tie, no 1–5
scale. The `winner` is always either `model_a` or `model_b` (a blank means the
judge returned no valid verdict, and that row is dropped).

This is exactly the data shape that the **Bradley-Terry model** was designed
for: a pile of paired comparisons, and a desire to recover a single
one-dimensional "strength" per competitor that best explains who beat whom.

### 1.2 Why not just count wins?

A raw win-percentage tally already exists and is published. On a fully balanced
round-robin it even produces nearly the same ordering (Spearman ≈ 1.0 with the
BT rank). So why fit a model at all? Three reasons, and they are the entire
value proposition of this stage:

1. **Principled uncertainty.** A win% of 70% from 10 comparisons and a win% of
   70% from 1,000 comparisons are wildly different levels of evidence, but a raw
   tally reports them identically. The Bayesian model produces a full posterior
   for every strength, so each model gets a **94% HDI** (highest-density
   interval) instead of a bare point estimate. Differences between adjacent
   models can be read as "real" or "within noise."
2. **Rater modeling.** Judges are not neutral measuring instruments. They have a
   **position bias** (a tendency to favor whichever response is shown first),
   varying **discrimination** (some judges separate strong from weak responses
   crisply, others are noisy), and a **self-family preference** (a judge built by
   the same lab as a candidate may favor it). The model estimates each of these
   and *removes* them from the strength estimates, rather than letting them
   contaminate the leaderboard.
3. **Partial pooling.** With only three judges, estimating each judge's biases
   independently would be noisy and overconfident. A hierarchical structure
   shares information across judges, shrinking extreme per-judge estimates toward
   the panel average in proportion to how much data supports them.

---

## 2. Bradley-Terry, from first principles

### 2.1 The classical model

The Bradley-Terry model (Bradley & Terry, 1952) assigns each competitor *i* a
positive "strength" parameter. The probability that *i* beats *j* in a single
comparison is its share of the pair's total strength. If we write strengths on a
**log scale** as $\beta_i$ (so $\text{strength}_i = e^{\beta_i}$), the model has
the clean logistic form:

$$
P(i \text{ beats } j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}}
= \frac{1}{1 + e^{-(\beta_i - \beta_j)}}
= \sigma(\beta_i - \beta_j)
$$

where $\sigma$ is the logistic (sigmoid) function. This is the key insight that
makes Bradley-Terry tractable: **a paired comparison is just logistic
regression where the only predictor is the difference of the two competitors'
latent strengths.** The outcome is Bernoulli; the linear predictor (the
"logit") is $\beta_i - \beta_j$.

Two immediate consequences:

- Only **differences** of strengths are identified, not their absolute level.
  Adding a constant to every $\beta$ leaves all win probabilities unchanged. We
  must pin down the location somehow (see §4.1).
- If two competitors have equal strength, $\beta_i - \beta_j = 0$ and
  $\sigma(0) = 0.5$ — a coin flip, as it should be.

### 2.2 What "strength" means here

$\beta_i$ is a **latent quality on the log-odds scale**. A one-unit gap in
$\beta$ means the stronger model is $e^1 \approx 2.7\times$ more likely (in odds
terms) to win a head-to-head against the weaker one, under a neutral judge.
Concretely (illustrative numbers, not a result): if the strongest candidate sits
at $\beta \approx 1.92$ and the weakest at $\beta \approx -1.48$, the gap of ≈3.4
implies

$$
\sigma(1.92 - (-1.48)) = \sigma(3.40) \approx 0.968,
$$

i.e. the stronger model would win ~97% of the time before any judge effects.

---

## 3. The footval model: Bradley-Terry with rater effects

The footval model extends classical Bradley-Terry by recognizing that the
outcome of each comparison depends not only on *which two models* are being
compared but also on *who is judging* and *how the pair was presented*. The full
linear predictor for a single observation is:

$$
\eta = \underbrace{\kappa_j \,(\beta_a - \beta_b)}_{\text{strength gap, scaled by judge acuity}}
\;+\; \underbrace{\gamma_j}_{\text{judge's slot-A bias}}
\;+\; \underbrace{\delta_j \cdot s}_{\text{judge's self-family pull}}
$$

$$
\text{won\_a} \sim \text{Bernoulli}\big(\text{logit}^{-1}(\eta)\big)
$$

The outcome being modeled is **`won_a`**: 1 if the response in slot A won, 0 if
slot B won. (Encoding the outcome by *slot* rather than by *model* is what makes
the position-bias term $\gamma_j$ identifiable — see §3.3.) Each term:

### 3.1 `beta_a − beta_b` — the Bradley-Terry core

`beta` (β) is the vector of latent model strengths, one entry per candidate
model. `beta[a_idx]` and `beta[b_idx]` pull out the strengths of the two models
in this particular comparison. Their difference is the classical Bradley-Terry
logit from §2.1. Everything else in the equation is a correction applied on top
of this core.

### 3.2 `kappa_j` — judge discrimination (the rating "gain")

$\kappa_j > 0$ is a **per-judge multiplier on the strength gap**. It captures how
sharply a given judge separates strong from weak responses:

- $\kappa_j = 1$ is the neutral, "calibrated" judge: they see the strength gap at
  face value.
- $\kappa_j > 1$ is a **decisive** judge: they amplify differences, so even a
  small true quality gap produces a near-certain verdict.
- $\kappa_j < 1$ is a **noisy/lenient** judge: they squash differences toward a
  coin flip, as if they can barely tell the responses apart.

In psychometrics this is exactly the **discrimination parameter** of an item in
item-response theory — it controls the steepness of the logistic curve. Here the
"items" are the judges. Multiplying the strength gap by $\kappa_j$ lets the model
**down-weight the votes of an indiscriminate judge** automatically: a judge whose
verdicts barely track quality contributes a flatter likelihood and thus moves
the strength estimates less.

This is also why the strength scale needs to be fixed externally (§4.1): the
likelihood only ever sees the *product* $\kappa_j \beta$, so $\kappa$ and $\beta$
trade off against each other and cannot both be free.

### 3.3 `gamma_j` — position / slot-A bias

$\gamma_j$ is an **additive offset to the log-odds that slot A wins**, regardless
of which models occupy the slots. It is the per-judge tendency to favor whichever
response is presented first (or second, if negative). LLM judges are
well-documented to have exactly this bias.

This term is identifiable precisely because the outcome is coded by **slot**, not
by model, and because which model is shown as "Response A" is a deterministic
per-pair coin flip recorded in the data. If slot assignment were correlated with
model identity, $\gamma$ and $\beta$ would be confounded; because it is
randomized, a systematic slot-A win rate that is *not* explained by strength gets
absorbed into $\gamma_j$ instead of distorting the $\beta$'s.

The panel-level mean $\mu_\gamma$ (see §3.5) is the headline "does this panel have
position bias?" estimate, reported with its own HDI.

### 3.4 `delta_j · s` — self-family preference

$\delta_j$ is the **per-judge same-family bias**, and it is multiplied by a signed
indicator $s$ (`self_sign`) that encodes whether the judge shares a corporate
"family" (anthropic, openai, google, zai) with the models in the pair:

$$
s = \mathbb{1}[\text{fam}(a) = \text{fam}(j)] - \mathbb{1}[\text{fam}(b) = \text{fam}(j)]
\in \{-1, 0, +1\}
$$

- $s = +1$: only the slot-A model is from the judge's family → if $\delta_j > 0$,
  this *adds* to slot A's log-odds (the judge favors its sibling).
- $s = -1$: only the slot-B model is from the judge's family → the bias pushes the
  other way.
- $s = 0$: both or neither share the judge's family → the effect cancels and the
  term drops out.

So $\delta_j$ measures, in log-odds, how much a judge tilts toward a response made
by its own lab. The panel mean $\mu_\delta$ answers "is there systematic
in-group favoritism across the whole panel?" The self/same-family comparisons are
deliberately **kept, not dropped** — the point is to measure the bias, not hide
from it.

### 3.5 The hierarchy — partial pooling of judge effects

Each of the three judge effects is given a **hierarchical (multilevel) prior**
so that judges share statistical strength. Take $\gamma$ as the template:

$$
\mu_\gamma \sim \mathcal{N}(0, 0.5), \qquad
\sigma_\gamma \sim \text{HalfNormal}(0.5), \qquad
\gamma_j = \mu_\gamma + \sigma_\gamma\, z_{\gamma,j}, \quad z_{\gamma,j}\sim\mathcal{N}(0,1)
$$

- $\mu_\gamma$ is the **panel-wide average** position bias.
- $\sigma_\gamma$ is the **between-judge spread** — how much judges differ from
  one another.
- Each $\gamma_j$ is the panel mean plus a judge-specific deviation.

This is partial pooling: if the data don't strongly distinguish a judge, its
$\gamma_j$ shrinks toward $\mu_\gamma$. With only three judges this shrinkage is
substantial and intentional — it prevents wild per-judge estimates from one or
two unusual comparisons. (Caveat in §7: with three judges the variance parameters
are only weakly identified and lean on their priors, so the per-judge numbers
should be read as directional.)

$\delta$ follows the identical structure with its own $\mu_\delta,\sigma_\delta$.
$\kappa$ is hierarchical too but parameterized differently because it must stay
positive (§4.3).

---

## 4. The crucial modeling decisions

Three identifiability decisions are what separate a model that *samples cleanly*
from one that wanders and diverges. Each is a deliberate choice in the code.

### 4.1 Fixing the location: `ZeroSumNormal` on `beta`

As noted in §2.1, Bradley-Terry strengths are only identified up to an additive
constant. The code resolves this with a **sum-to-zero constraint**:

```python
beta = pm.ZeroSumNormal("beta", sigma=1.0, dims="model")
```

`ZeroSumNormal` is a normal prior restricted to the hyperplane where the
components sum to zero. This means $\sum_i \beta_i = 0$ by construction, so the
strengths are interpreted as **deviations from the average model**: a positive
$\beta$ is above-average, negative is below-average, and the whole vector can't
drift up or down as a block. This is cleaner than the common alternative of
pinning one model to $\beta = 0$ as a reference, because it keeps the parameter
space symmetric and avoids privileging an arbitrary baseline competitor.

### 4.2 Fixing the scale: `sigma=1.0` on `beta`

There is a *second* identifiability problem unique to this rater-aware model.
The likelihood sees $\kappa_j (\beta_a - \beta_b)$ — only the **product** of the
discrimination and the strength gap. You could halve every $\beta$ and double
every $\kappa$ and the model would be unchanged. To break this $\kappa$–$\beta$
redundancy, the **scale of $\beta$ is fixed** by the prior, `sigma=1.0`. With the
strength scale anchored at 1, the $\kappa_j$'s are then free to be read as
multipliers *relative to a standard-deviation-1 strength scale* — i.e., $\kappa$
truly means "this judge's gain relative to a calibrated judge," and is no longer
just soaking up the arbitrary units of $\beta$.

The `sigma=1.0` prior does double duty: besides fixing the scale, it **weakly
regularizes against separation**. In a small paired-comparison dataset it is
possible for one model to win every one of its comparisons; unconstrained MLE
would then send its strength to $+\infty$ (the classic "perfect separation"
pathology of logistic regression). A proper $\mathcal{N}(0,1)$-style prior keeps
the estimate finite and produces a sensible posterior even for an undefeated
model.

### 4.3 Non-centered parameterization (the `z_*` trick)

Every hierarchical effect is written in **non-centered** form. Instead of drawing
$\gamma_j \sim \mathcal{N}(\mu_\gamma, \sigma_\gamma)$ directly, the code draws a
standard normal $z_{\gamma,j} \sim \mathcal{N}(0,1)$ and *reconstructs*
$\gamma_j = \mu_\gamma + \sigma_\gamma z_{\gamma,j}$.

The two are mathematically equivalent but geometrically very different for the
sampler. In the centered form, when $\sigma_\gamma$ is small the per-judge
$\gamma_j$ are squeezed into a thin region that depends on $\sigma_\gamma$,
creating a pinched "funnel" in the posterior that Hamiltonian Monte Carlo (the
NUTS sampler PyMC uses) struggles to traverse — it produces divergences and
biased estimates. The non-centered form decouples the $z$'s from the variance, so
the geometry the sampler explores is a clean unit normal regardless of how small
$\sigma_\gamma$ turns out to be. With only three judges (so the variances *are*
small and poorly informed), this reparameterization is essentially mandatory for
clean sampling.

$\kappa$ uses the same idea but on the **log scale** to enforce positivity:

```python
sigma_kappa = pm.HalfNormal("sigma_kappa", 0.5)
z_kappa     = pm.Normal("z_kappa", 0.0, 1.0, dims="judge")
kappa       = pm.Deterministic("kappa", pm.math.exp(sigma_kappa * z_kappa), dims="judge")
```

Here $\log \kappa_j = \sigma_\kappa z_{\kappa,j}$, so $\kappa_j = e^{\sigma_\kappa
z_{\kappa,j}}$ is always positive and is **centered at 1** when $z = 0$ (since
$e^0 = 1$). That center of 1 is exactly the "neutral judge" baseline from §3.2.
Note there is no $\mu_\kappa$ free parameter: the log-mean is pinned at 0
(i.e. $\kappa$ centered at 1) on purpose, because a free multiplicative mean on
$\kappa$ would re-introduce the very scale redundancy that §4.2 just removed. Only
the spread $\sigma_\kappa$ is estimated, governing how much judges differ in
acuity.

---

## 5. The PyMC code, line by line

```python
def _fit(design: Design, seed: int):
    import pymc as pm

    coords = {"model": design.models, "judge": design.judges, "obs": np.arange(design.n_obs)}
    with pm.Model(coords=coords):
```

**`coords`** registers named dimensions ("model", "judge", "obs") so every
parameter and every posterior array is labeled — `beta` carries model names,
`kappa`/`gamma`/`delta` carry judge names, and the likelihood is indexed by
observation. This is what lets the downstream summary code address parameters by
name and emit tidy, labeled CSVs. `import pymc` is **inside** the function on
purpose: PyMC and ArviZ are heavy, so they're imported lazily, keeping the pure
data-prep helpers importable and unit-testable without the sampling stack
installed.

```python
        beta = pm.ZeroSumNormal("beta", sigma=1.0, dims="model")
```

Latent model strengths, sum-to-zero (location fixed, §4.1) and scale-1 (scale
fixed + regularized, §4.2). One entry per model.

```python
        sigma_kappa = pm.HalfNormal("sigma_kappa", 0.5)
        z_kappa = pm.Normal("z_kappa", 0.0, 1.0, dims="judge")
        kappa = pm.Deterministic("kappa", pm.math.exp(sigma_kappa * z_kappa), dims="judge")
```

Judge discrimination, non-centered on the log scale, centered at 1, strictly
positive (§4.3). `pm.Deterministic` records `kappa` in the trace as a derived
quantity (so its posterior and HDI can be summarized directly) without making it
a separately-sampled parameter.

```python
        mu_gamma = pm.Normal("mu_gamma", 0.0, 0.5)
        sigma_gamma = pm.HalfNormal("sigma_gamma", 0.5)
        z_gamma = pm.Normal("z_gamma", 0.0, 1.0, dims="judge")
        gamma = pm.Deterministic("gamma", mu_gamma + sigma_gamma * z_gamma, dims="judge")
```

Per-judge position bias, partially pooled and non-centered (§3.5, §4.3).
`mu_gamma` is the panel-wide position bias — one of the two headline bias numbers
reported. The $\mathcal{N}(0, 0.5)$ prior on $\mu_\gamma$ says: before seeing
data, expect little position bias, and a half-unit of log-odds is already a
sizeable effect (a 0.5 logit shifts a coin flip to ≈62%).

```python
        mu_delta = pm.Normal("mu_delta", 0.0, 0.5)
        sigma_delta = pm.HalfNormal("sigma_delta", 0.5)
        z_delta = pm.Normal("z_delta", 0.0, 1.0, dims="judge")
        delta = pm.Deterministic("delta", mu_delta + sigma_delta * z_delta, dims="judge")
```

Per-judge self-family preference, same hierarchical/non-centered structure.
`mu_delta` is the panel-wide in-group favoritism estimate.

```python
        eta = (
            kappa[design.j_idx] * (beta[design.a_idx] - beta[design.b_idx])
            + gamma[design.j_idx]
            + delta[design.j_idx] * design.self_sign
        )
        pm.Bernoulli("won_a", logit_p=eta, observed=design.won_a, dims="obs")
```

This is the heart of the model and a **vectorized assembly of the §3 equation
across all observations at once**. The `*_idx` arrays are integer indices (built
in data prep) that gather the right parameter for each row:

- `beta[design.a_idx]` and `beta[design.b_idx]` — strengths of the two models in
  each comparison.
- `kappa[design.j_idx]`, `gamma[design.j_idx]`, `delta[design.j_idx]` — the
  effects of the judge who rendered each verdict.
- `design.self_sign` — the $\{-1,0,+1\}$ family indicator per row.

`pm.Bernoulli(..., logit_p=eta, observed=design.won_a)` is the likelihood:
passing `logit_p` (rather than `p`) tells PyMC that `eta` is already on the
log-odds scale, so it applies the sigmoid internally in a numerically stable way.
`observed=design.won_a` attaches the actual outcomes, turning this from a
generative description into something the sampler can condition on.

```python
        idata = pm.sample(
            draws=DRAWS, tune=TUNE, chains=CHAINS,
            target_accept=TARGET_ACCEPT, random_seed=seed, progressbar=False,
        )
    return idata
```

Runs NUTS (the default gradient-based MCMC sampler) and returns an ArviZ
`InferenceData` object containing the posterior draws plus sampler diagnostics.

### 5.1 Sampler settings and why

```python
DRAWS = 2000          # kept post-warmup draws, per chain
TUNE  = 2000          # warmup/adaptation steps, discarded
CHAINS = 4            # independent chains (enables R-hat, parallelism)
TARGET_ACCEPT = 0.95  # NUTS target acceptance probability
HDI_PROB = 0.94       # credible-interval mass for all reported HDIs
```

- **4 chains × 2000 draws = 8000 posterior samples** per parameter — comfortably
  enough to estimate means and 94% HDIs stably. Multiple chains started from
  different points are what make the **R-hat** convergence diagnostic meaningful
  (it compares within-chain to between-chain variance).
- **2000 tuning steps** give NUTS ample time to adapt its step size and mass
  matrix before collecting samples; these warmup draws are thrown away.
- **`target_accept = 0.95`** (above the 0.8 default) forces NUTS to take smaller,
  more careful steps. Higher target acceptance → smaller step size → fewer
  divergences, at the cost of speed. On this small, slightly funnel-prone
  hierarchical panel, the comment notes it's tuned to keep **divergences at
  zero**.
- **`HDI_PROB = 0.94`** is the credible level for every reported interval. 94%
  (rather than the round 95%) is an ArviZ convention popularized by *Statistical
  Rethinking*, chosen partly to discourage treating the interval as a
  null-hypothesis test the way 95% invites. A 94% HDI is the **narrowest**
  interval containing 94% of the posterior mass.
- **`random_seed=seed`** (from config) makes the fit reproducible.

A separate model is fit **per track**, one track per judged criterion, since each
criterion is a distinct notion of quality and pooling them would blur
criterion-specific strengths. Judging covers two criteria, so there are two tracks
(`analytical_reasoning`, `intuitive_reasoning`) and no pooled `overall` fit — each
leaderboard stands on its own. (Earlier versions fit `soundness` alone, and before
that `soundness` + `priors` + a pooled `overall`; both designs were retired.)

---

## 6. From posterior to leaderboard

Sampling yields a posterior; `_summarize` turns it into the published numbers.

### 6.1 Strength rankings

For each model the code reports the posterior **mean** of $\beta$, its **standard
deviation**, and its **94% HDI** (via `az.hdi`). Ranks are assigned by
posterior-mean strength:

```python
ranks = (-beta_mean).argsort().argsort() + 1   # rank 1 = highest mean strength
```

The double-`argsort` is the idiomatic NumPy way to convert values into rank
positions; negating `beta_mean` makes the largest strength rank 1.

### 6.2 `win_prob_vs_field` — a human-readable strength

A raw $\beta$ of 1.92 is not intuitive. So each model's strength is also
expressed as its **expected win probability against a uniformly random opponent,
under a neutral judge** ($\kappa=1, \gamma=\delta=0$):

$$
\text{wpf}_m = \frac{1}{N-1} \sum_{n \neq m} \sigma(\beta_m - \beta_n)
$$

Crucially this is computed **per posterior draw and then summarized**, not from
the posterior-mean strengths. In `_win_prob_vs_field`, the full
`(model, sample)` array of $\beta$ draws is broadcast into an
all-pairs difference tensor, pushed through the sigmoid, averaged over opponents
(excluding self via the off-diagonal mask), and *then* reduced to a posterior
mean and HDI. Propagating the uncertainty through the nonlinear sigmoid this way
gives a correctly-calibrated interval on the win probability itself — which is
what the blog leaderboards plot (point estimate + thick 50% HDI + thin 95% HDI).

### 6.3 Judge effects

For every judge the code reports posterior mean + HDI of $\kappa$ (discrimination,
centered on 1), $\gamma$ (position bias), and $\delta$ (self-preference), plus a
`panel (mean)` row carrying $\mu_\gamma$ and $\mu_\delta$ — the panel-wide bias
estimates that are the principled successor to a raw win-rate-gap diagnostic.

### 6.4 Convergence diagnostics

```python
diagnostics = {
    "max_r_hat":    float(summ["r_hat"].max()),
    "min_ess_bulk": float(summ["ess_bulk"].min()),
    "n_divergences": int(idata.sample_stats["diverging"].to_numpy().sum()),
}
```

Three standard MCMC health checks, and the console flags the track for review if
any look off (`max_r_hat > 1.01` or any divergence):

- **R-hat (Gelman-Rubin):** compares between-chain and within-chain variance.
  ≈1.00 means the chains have mixed and agree; > 1.01 warns of non-convergence.
- **ESS (effective sample size, bulk):** how many *effectively independent* draws
  the autocorrelated chain is worth. A low ESS means the posterior summaries are
  noisier than the raw draw count suggests.
- **Divergences:** NUTS transitions that broke the Hamiltonian — the telltale of
  unexplored funnel geometry. Zero is the goal (and the reason for the
  non-centered parameterization and the high `target_accept`).

The full posterior **trace** is persisted to `artifacts/bradley_terry/{track}.nc`
(NetCDF, written atomically via a temp-file swap to dodge HDF5 read-locks) so the
diagnostics and blog notebooks can recompute any HDI or posterior-predictive
quantity and run full ArviZ checks **without re-fitting**.

---

## 7. Honest caveats

These are real limitations, stated plainly:

- **Four judges is a small panel.** The between-judge variances
  ($\sigma_\kappa, \sigma_\gamma, \sigma_\delta$) are only weakly identified by
  four data points and therefore **lean heavily on their priors**. Read the
  per-judge $\kappa/\gamma/\delta$ and the panel means $\mu_\gamma/\mu_\delta$ as
  **directional**, not precise.
- **The ranking is not the value-add.** On a balanced round-robin the BT order
  matches raw win% almost exactly (Spearman ≈ 1.0). The model earns its keep
  through (a) the **uncertainty bands** and (b) the **judge-effect
  decomposition** — separating true strength from position bias and in-group
  favoritism — not by reordering anyone.
- **It is purely additive.** This stage consumes the already-published
  `pairwise_results.csv` and touches no prompt, no candidate response, and no
  prior-run artifact. It can be re-run or skipped without affecting any other
  stage.

---

## 8. One-paragraph summary

Footval aggregates a round-robin of forced A/B judge verdicts with a
**hierarchical Bayesian Bradley-Terry model**. Each candidate gets a latent
log-strength $\beta$ (sum-to-zero and scale-1 for identifiability); the
probability that the slot-A response wins is
$\sigma\big(\kappa_j(\beta_a-\beta_b) + \gamma_j + \delta_j s\big)$, where the
judge contributes a discrimination gain $\kappa_j$ (how sharply they separate
quality), a position bias $\gamma_j$ (slot-A favoritism), and a self-family pull
$\delta_j$ (favoring their own lab), all **partially pooled** across the four
judges via non-centered hierarchical priors. PyMC's NUTS sampler (4 chains, 2000
draws, `target_accept=0.95`) returns a posterior from which the code reports each
model's strength and a human-readable **win-probability-vs-field**, each with a
**94% HDI**, plus the panel-wide bias estimates $\mu_\gamma, \mu_\delta$ and full
convergence diagnostics. The payoff over a raw win-percentage tally is not a
different ranking but **calibrated uncertainty** and an explicit, bias-corrected
account of how the judges behaved.
