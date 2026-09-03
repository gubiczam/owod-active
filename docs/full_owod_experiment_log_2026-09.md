# Research log — full OWOD active-selection benchmark, September 2026

One entry per decision that could have gone another way. Each records the date,
the reason, **the evidence available at the time**, and whether any final
endpoint had been inspected. The last column is the one that matters: a change
made after seeing a detector endpoint is a tuned change and must be labelled as
one.

Protocol: `docs/full_owod_active_benchmark_v1_protocol_2026-09-03.md`.

---

## 2026-09-03 — mode change: from component diagnostics to the full chain

**Reason.** Four days to the supervisor meeting, and three completed
experiments' worth of component diagnostics with no end-to-end OWOD result.
Method V3's own post-hoc audit concluded that its isolated `t1 -> t2` design
could not have answered the question it was asked.

**Evidence at the time.** `docs/method_v3_posthoc_audit_2026-09-03.md`:
the acquirable population held **2** instances of the only class that becomes
declarable, so `new_class_AP50 ~ 0` was a supply result rather than a learning
result; the 600-*region* budget delivered 972 supervised boxes to one arm and
2 027 to another (2.09x).

**Endpoints inspected?** Yes — Method V3's, which are frozen and closed. No
endpoint of this benchmark existed.

---

## 2026-09-03 — the task chain is the repository's own, one class per task

**Decision.** `t1 -> t2 -> t3 -> t4` declaring `traffic light`, `fire hydrant`,
`stop sign`. **Not** the published S-OWODB 19/21/20/20 split.

**Reason.** The instruction was not to invent task splits, and
`owl.protocol.build_chain` is the repository's canonical chain. A 21-class step
is unaffordable at any annotation budget we can pay for, and the published split
would make the new-class endpoint a 21-way average in which no acquisition
decision is visible.

**What it buys.** The tail band **grows** along the chain: `{bear}` at t2,
`{bear, fire hydrant}` at t3, `{bear, fire hydrant, stop sign}` at t4. So
`mAP50_tail` at t4 is a function of what the selector acquired, which is the
long-tail claim, and it is visible only across a sequence.

**Cost.** No number here may be compared against a published S-OWODB result.
Stated in the protocol, in the manifest, in every summariser run and in the
notebook's own header.

**Endpoints inspected?** None existed.

---

## 2026-09-03 — annotation cost becomes the oracle answer

**Decision.** `cost(image) = max(1, annotated objects on it)`, full-image
labelling, budget 3 000 answers per task. `budget_unit` is a new
`owl.runner.CycleConfig` field; `"regions"` remains the default so every
committed result stays reproducible and the completed Replay-V3 workspaces stay
resumable.

**Evidence at the time.** Measured on the committed pool at the real budget,
before any training: region budget -> 1.62 boxes per region for `admissibility`
against 3.38 for `entropy` (2.09x); image budget -> 1.65 against 7.9 (4.8x);
answer budget -> **2 644 – 2 809 labelled boxes across all five arms**, matched
to within 6 %, and 140 – 233 trainable images, within 17 %.

**Known residual, recorded rather than hidden.** The *declared* share of those
matched labels is not matched — 480 supervised boxes for `admissibility` against
1 812 for `entropy` at t2 in the same simulation. Matching both is impossible:
which fraction of a labelled image is currently learnable is not under the
selector's control. It is reported as an outcome; a supervision-matched
sensitivity run for the two finalists is Phase 2.

**Endpoints inspected?** No detector endpoint of this benchmark existed. The
simulation is CPU, on the committed pool, with no detector.

---

## 2026-09-03 — raw objectness gets a measurement, not a trajectory

**Question asked.** Are plain objectness and `A(x) = objectness * sqrt(area)`
actually distinct, or would a separate arm waste four GPU hours on a duplicate?

**Measured.** Spearman **0.281**; top-600 prefixes **disjoint** (Jaccard 0.000)
— genuinely distinct. But raw objectness's first 600 picks hold **2** real
annotated objects against `A`'s **284**: it prefers tiny high-confidence boxes.

**Decision.** Distinct *and* degenerate, so it is reported as a measurement.
Pinned in `tests/test_active_selection.py::test_admissibility_is_not_raw_objectness`.
Note that `owl.selection.ARMS['objectness']` already computes `A`; that older
registry's name is a misnomer.

**Endpoints inspected?** None.

---

## 2026-09-03 — the proposed method drops D, R, C and U

**Decision.** Proposed-v1 is **A-gated semantic k-center**: gate on the top 30 %
by `A`, then farthest-first traversal in frozen DINOv2 space against the
labelled reference. No `lambda`, no `gamma`, no `mu`, no exponent, no threshold.

**Reason, component by component.** `D` failed as an unknown-vs-background
separator (`D_NO_GO`) and is kept only as *coverage*, which is a different
quantity. `R` failed (`R_NO_GO`) and is removed; no rarity is estimated at all.
`C` passed its component gate and failed downstream
(`C_DOWNSTREAM_NOT_SUPPORTED`) and is removed. `U` was the weakest unknown-finder
in Method V3 and is removed from the method, surviving as its own baseline arm.
`A` was the strongest thing this project has measured and is kept, as the gate.

**Why no hyperparameter.** Every weight the original additive score needed would
have to be given a value, and a value chosen after seeing a detector endpoint
makes the result a tuned one. k-center has nothing to choose.

**Why DINOv2 only inside the gate.** DINOv2 was measured *not* to separate
object from background; PROB's objectness was. Each representation is used for
what it was measured to be good at.

**Endpoints inspected?** None of this benchmark's. The component verdicts it
reacts to are frozen and closed.

---

## 2026-09-03 — one round, and it is not called iterative active learning

**Decision.** `rounds_per_task = 1`.

**Reason.** The repository's `rounds_per_task` recomputes the *score* against a
grown labelled pool; it does not re-run the detector. For `random`, `entropy`
and `admissibility` the score does not depend on the labelled pool, so six
rounds return provably the same prefix as one — Method V3 measured exactly that.
The two coverage arms are already sequential: their reference grows at every
pick. Declaring six rounds would buy nothing and would be the mislabelling the
2026-08-25 consultation warned against.

**Genuine iterative acquisition** — detector rescoring every 100 answers — needs
6 predicts and 6 trains per task instead of 1, about 6x the cost. Out of reach
this week; it is Phase 2 and the one-shot-versus-`6x100` question is **not**
answered here.

**Endpoints inspected?** None.

---

## 2026-09-03 — four arms in session 1, `coreset` first in session 2

**Decision.** Keep 5 epochs and 2 000 candidate images; run
`random, admissibility, proposed` then `entropy` in session 1, and `coreset` at
the start of session 2 before any replication seed.

**Evidence at the time.** `tools/plan_full_owod_benchmark.py` priced the top rung
from the project's own measured basis: **44.7 – 58.0 min** per arm-task,
**9.23 h** for four arms and **12.13 h** for five, against a 10-hour ceiling.

**Why not reduce epochs instead.** Under-training is the failure mode that
produced `new_class_AP50 ~ 0` in Method V3. A second session is recoverable; a
weakened training schedule is not. The epoch ladder `5 -> 3 -> 2` stays fixed in
the protocol for the case where even four arms do not fit.

**Why the order is what it is.** So that a session cut short still yields the
primary contrast and its reference. It is **not** a licence to drop an arm
because of its numbers; the only reasons an arm may be abandoned are in the
protocol's stopping rules.

**Endpoints inspected?** None. The estimate is arithmetic over a measured cost
basis.

---

## 2026-09-03 — the candidate pool rises from 1 200 to 2 000 images

**Decision.** `candidate_images_per_task = 2000`, the value the project's own
completed six-task GPU chain used.

**Reason.** The detector pass costs 4 min per thousand images and training costs
twenty-five, so a larger pool is the cheapest way to put more of the rare
declared class within the selector's reach. At 2 000 the pool is expected to
hold ~149 traffic-light, ~63 fire-hydrant and ~62 stop-sign images; at 1 200,
~90, ~38 and ~37.

**Cost.** +0.4 h on the four-arm session (8.82 h -> 9.23 h).

**Endpoints inspected?** None. The supply figures are counts over the committed
candidate index.

---

## 2026-09-03 — PROB's seed is passed

**Decision.** `tools/run_full_owod_benchmark.py` constructs `Bridge(seed=seed)`.

**Reason.** Method V3's audit found `--seed` left at its default of 0 in all
twelve of its trajectories, because that launcher never passed one; its seed
varied only the 400-object rehearsal set. Pinned in
`tests/test_full_benchmark_chain.py::test_the_prob_seed_is_actually_passed`.

**What is still not paired.** The exemplar pool excludes the images a task just
bought, so arms that bought different images necessarily draw different
exemplars — one changed acquired image was measured to move 20 of 400. Common
random numbers across arms are impossible by construction and no claim here may
assume them.

**Endpoints inspected?** None.

---

## Entries to add before the supervisor meeting

* the outcome of the seed-0 session, and whether any stopping rule fired;
* whether `new_class_AP50` was readable at t3 and t4, against the supply stated
  in the protocol's section 7.1;
* if a `Proposed-v2` is defined after seeing seed 0, it must be versioned as
  such here, with the seed-0 numbers that motivated it, and **not** presented as
  having been pre-registered.
