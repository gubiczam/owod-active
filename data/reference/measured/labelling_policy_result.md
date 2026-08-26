# What annotating a region actually buys — experiment #5

The 2026-08-25 consultation's first question, and the one that distorts every other
measurement if it is answered wrongly: **the score points at one proposal, but the
annotator is handed an image.** Three ways to resolve that, measured at equal paid cost.

Machinery: `owodtail/labelling.py`. Contract tests: `tests/test_labelling.py`. Rows:
`data/labelling_policy_rows.csv`. No detector run — the frozen pool only.

## The three policies

| policy | the selected region | the image's other unknowns | the image's knowns |
|---|---|---|---|
| `full_image` | named | **all named** | reused, free |
| `selected_box_ignore_unknown` | named | **withheld — no gradient** | reused, free |
| `selected_box_background_rest` | named | **taught as background** | reused, free |

The third is the negative control: it is what a naive implementation does, and it
teaches the head that a real object is background.

## The cost basis

Counted in *questions the oracle had to answer that we did not already have*. Task-1
known boxes are free — the benchmark already carries them. A selected-box policy asks
exactly one question per image, whatever the answer turns out to be. `full_image` asks
one per not-yet-revealed unknown object in the image, and at least one.

The repository's original basis, every annotated object including knowns, is still the
default (`cost_basis='annotated_objects'`), so no committed number moves. The regression
is asserted, not assumed: `mult_prior_shrunk`, seed 0, round 2 still gives 287 images,
1800 objects, 134 tail objects, 479 isolated — bit-identical to `frozen_cycle_arms.csv`.

## What came out

Budget 600 paid questions, 2 rounds, seeds 0/1/2, mean ± sd. Selection endpoints only;
no head is trained, so nothing here is an AP claim.

| selector | policy | paid | tail objects | unknowns taught as background |
|---|---|---:|---:|---:|
| random | `full_image` | 366 ± 4 | **125.3 ± 22.5** | 0 |
| random | `ignore_unknown` | 600 | 4.3 ± 1.2 | 0 |
| random | `background_rest` | 600 | 4.3 ± 1.2 | **75** |
| objectness_prior | `full_image` | 461 | **173.0 ± 0.0** | 0 |
| objectness_prior | `ignore_unknown` | 600 | 29.0 ± 0.0 | 0 |
| objectness_prior | `background_rest` | 600 | 29.0 ± 0.0 | **66** |
| mult_prior_shrunk | `full_image` | 454 ± 1 | **225.0 ± 4.9** | 0 |
| mult_prior_shrunk | `ignore_unknown` | 600 | 37.0 ± 2.2 | 0 |
| mult_prior_shrunk | `background_rest` | 600 | 37.0 ± 2.2 | **87** |

**1. Full-image annotation wins, and not narrowly.** Paired by seed, `full_image` finds
5.5–6.4× more tail objects than the selected-box policies for `objectness_prior` and
`mult_prior_shrunk`, and 23–45× for `random`. Every seed, same sign. It also spends
*less*: 454 of 600 available questions, because the marginal cost of naming the other
objects in an image you are already looking at is small, while the marginal yield is not.

**2. The half-labelling bug is real and quantified.** `background_rest` teaches 66–87
genuine unknown objects as background per run. `ignore_unknown` holds that at exactly
zero, by construction, and the test asserts it. The two arms differ in nothing else:
identical regions named, identical paid cost, identical tail discovery.

**3. The acquisition ordering survives the policy change.** Under `full_image`,
`mult_prior_shrunk` (225) > `objectness_prior` (173) > `random` (125). The policy
question and the selection question are separable, which is what lets the remaining six
experiments be run on a fixed policy.

## The decision

**`full_image` is the fixed basis for experiments #3, #1, #2, #7, #4 and #6.**
`selected_box_ignore_unknown` is kept as the safe fallback if a future setting makes
whole-image annotation unaffordable; `selected_box_background_rest` is kept only as the
named negative control, and no result may be reported on it.

## What this does not say

The tail counts are *discovery* — objects put in front of the annotator — not detection
quality. No head was trained for this table, so nothing here speaks to mAP, forgetting,
or retention. `docs/real_forgetting_result.md` already records that the frozen-feature
surrogate ranks acquisition arms in the **reverse** order on forgetting, so the head
endpoints must be measured separately and flagged as proxy.

The 5.5–6.4× is also a statement about cost accounting, not about annotator effort in
seconds. If drawing four boxes in one image genuinely costs four times as much wall-clock
as drawing one, the advantage shrinks to roughly parity. That exchange rate is not
measurable from this pool, and it is the honest caveat to state to the supervisor.

## The head endpoint: not decidable at this budget

`notebooks/owod_labelling_policy.ipynb` also trains the proxy head on all three policies,
`mult_prior_shrunk`, 3 seeds, 600 paid questions. It answers the consultation's second
half — *"ez mennyit ront az osztályozón?"* — with a clear **no result**:

| policy | new-class mAP50 (PROXY) | task-1 retention (PROXY) | unknowns taught as background |
|---|---:|---:|---:|
| `full_image` | 0.174 ± 0.065 | 30.88 ± 4.86 | 0 |
| `ignore_unknown` | 0.251 ± 0.028 | 29.82 ± 3.06 | 0 |
| `background_rest` | 0.312 ± 0.110 | 30.79 ± 4.96 | 87 |

The cleanest contrast in the whole experiment is `ignore_unknown` versus
`background_rest`: identical regions named, identical paid cost, and the only difference
is what happens to the image's other unknowns. Paired by seed, the difference is
`+0.029 / −0.034 / −0.179` on new-class mAP50 and `+0.61 / +0.25 / −3.78` on retention.
**The sign flips.** The spread is larger than the effect.

That is not evidence the half-labelling bug is harmless. It is evidence that *this*
measurement cannot see it, for a reason the repository already recorded: over frozen
features almost nothing learns at all — every arm sits between 0.17 and 0.31 mAP50, close
enough to zero that a 87-object supervision error has nothing to move. The selection
number is the usable result; the head number is a null that must be reported as a null.

If the cost of half-labelling is to be measured, it needs the real detector updating its
own weights, which is a GPU experiment outside this pool.

## Why the head endpoint is dead, and what replay does fix

The null above has a measured cause, and it is not forgetting and not missing replay.
`owodtail/ceiling.py` fits a one-versus-rest linear probe per class **given every label in
the candidate pool** — a budget no campaign would have — and reports the AP50 it reaches:

| | reachability (median) | probe AP50 (median) | classes above 1.0 AP50 |
|---|---:|---:|---:|
| task-1 known classes | 84.1% | **40.31** | 18 / 18 |
| unknown classes | 11.0% | **0.03** | 2 / 41 |

PROB's proposals do not overlap 89% of the unknown objects, and what they do overlap is
not separable in the frozen feature. With unlimited supervision and no forgetting
possible, the unknown-class ceiling is three hundredths of a point. Every arm in this
study already sits at or above it. **That endpoint cannot be measured on this pool, ever**,
and no acquisition rule, labelling policy or replay schedule changes it.

Retention is different: the known-class ceiling is 40 AP50 per class, the head starts at
52, and it falls to 30. That is real headroom, so replay is measurable there — and it
works. `mult_prior_shrunk`, 600 paid questions, 3 seeds, 900 exemplars:

| policy | replay | task-1 retention mAP50 (PROXY) | new-class mAP50 |
|---|---|---:|---:|
| `full_image` | none | 30.88 ± 4.86 | 0.174 |
| `full_image` | uniform | **47.30 ± 2.62** | 0.213 |
| `full_image` | tail | 47.24 ± 1.50 | 0.234 |
| `ignore_unknown` | none | 29.82 ± 3.06 | 0.251 |
| `ignore_unknown` | uniform | 46.79 ± 1.43 | 0.247 |
| `ignore_unknown` | tail | 47.29 ± 0.37 | 0.511 |
| `background_rest` | none | 30.79 ± 4.96 | 0.312 |
| `background_rest` | uniform | **47.91 ± 1.04** | 0.406 |
| `background_rest` | tail | 47.44 ± 2.98 | 0.390 |

**Replay is decisive: +16.8 retention points on average, and all eighteen paired
seed-contrasts point the same way.** Uniform and tail allocation are indistinguishable at
this memory size, which is a question for consultation item #4 rather than this one.

**The labelling rule does not touch forgetting.** `ignore_unknown` minus
`background_rest` flips sign in every replay setting. That is not a weak measurement —
it is the expected answer, and it says something useful: forgetting is driven by whether
the task-1 boxes survive, and *all three policies reuse them for free*. The half-labelling
error damages the new-class side, which is exactly the side this pool cannot measure.

So the labelling question splits cleanly in two, and only one half is answerable here:

* **which regions get in front of the annotator** — answered, 6.1× for `full_image`;
* **what half-labelling costs the detector** — needs the real detector. The design is
  `docs/labelling_policy_real_detector_runbook.md`.

## The Colab run, and what the raw rows add

Run on a free Colab CPU, 2026-08-25. **Every number reproduced the local run exactly** —
selection counts, retention means, standard deviations, all of it. Pure numpy, fixed
seeds, no framework: the same numbers on different hardware. Rows:
`data/labelling_policy_colab_rows.csv`.

Three things sit in the per-seed rows that the printed summary did not surface.

### 1. `full_image` finds more tail *classes*, not just more objects

| selector | `full_image` | `ignore_unknown` |
|---|---:|---:|
| random | **21.3 ± 0.5** | 5.7 ± 0.5 |
| objectness_prior | **22.0 ± 0.0** | 10.0 ± 0.0 |
| mult_prior_shrunk | **19.0 ± 0.0** | 10.7 ± 0.5 |

This is the stronger form of the argument, and it should replace the object count when the
case is made out loud. **A class the oracle never named cannot be detected at any budget**,
so distinct-class coverage is the endpoint that gates everything downstream. Naming one box
per image costs half to three quarters of the classes.

It also complicates the arm ranking. On tail *objects* `mult_prior_shrunk` wins (225 vs
173). On tail *classes* `objectness_prior` wins (22.0 vs 19.0), on every seed, with zero
variance. `mult_prior_shrunk` concentrates the budget on more instances of fewer classes.
Which is better depends on the endpoint, and the free objectness control is still not beaten
on the endpoint that matters most.

### 2. Unknown recall moves the *other* way, and this must be reported

| policy | replay: none | uniform | tail |
|---|---:|---:|---:|
| `full_image` | 3.70 ± 0.60 | 4.84 ± 1.40 | 4.64 ± 1.18 |
| `ignore_unknown` | **7.03 ± 0.32** | **7.40 ± 0.35** | **7.49 ± 0.36** |
| `background_rest` | 7.24 ± 0.08 | 7.44 ± 0.19 | 7.44 ± 0.17 |

Paired by seed, `full_image` is lower in all nine comparisons. This is roughly a factor of
two on a headline OWOD metric, and it goes against the policy the selection endpoint
endorses.

The mechanism is not mysterious and it is not a defect in `full_image`: U-Recall counts
objects the model flags as *unknown*, and `full_image` names twice as many classes, so
those objects are no longer unknown to it. A policy that names almost nothing keeps almost
everything in the unknown slot and scores well by doing less. **It is a bookkeeping
artefact of the metric's denominator, not open-world ability** — but it is a real number, a
supervisor will ask about it, and it should be presented with the explanation rather than
omitted.

### 3. Retention saturates, and the free known boxes explain why

`ignore_unknown` buys 600 images and collects **1772** free task-1 boxes; `full_image` buys
418 and collects **1317**. The selected-box policy gets *more* free replay, yet both land at
the same retention. Combined with §"Why the head endpoint is dead" above, retention is
saturated at this budget: past some threshold more task-1 supervision buys nothing, which is
why the labelling rule cannot move it.

### A note on the AP column

`ignore_unknown` + tail replay reads 0.511 ± 0.285. The per-seed values are 0.313, **0.914**,
0.307 — one seed, three times the others. That single row is the mean. It is the clearest
illustration in the study of why the new-class AP column must not be read on this pool.
