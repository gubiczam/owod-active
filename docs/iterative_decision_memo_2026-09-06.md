# Banking forensic, and whether an iterative D/R/C experiment is justified

2026-09-06. CPU, read-only. No detector was trained or run, no PROB or DINOv2
pass was repeated, no ledger altered, no banking rule patched, no gate changed.

---

## 1. `distribution_aware_v1` is NO-GO, and stays NO-GO

Under the gates frozen in `docs/distribution_aware_decision_memo_2026-09-06.md`
§6, before any oracle diagnostic for the method existed:

| gate | seed 0 | seed 1 | threshold | |
|---|---:|---:|---|---|
| `jaccard_vs_entropy` | 0.0843 | 0.1095 | < 0.60 | PASS |
| **`tail_per_image_vs_entropy`** | **0.5132×** | **0.8460×** | ≥ 1.25× | **FAIL** |
| `tail_per_image_vs_cost_aware` | 5.1152× | 4.4153× | ≥ 1.25× | PASS |
| `background_share_vs_entropy` | 0.0 | 0.0 | ≤ +0.10 | PASS |
| `breadth_vs_entropy` | 0.9602× | 1.0000× | ≥ 0.75× | PASS |
| `t2_not_collapsed` | 0.8876× | 0.7402× | ≥ 0.75× | PASS |

**Verdict: NO-GO**, on `tail_per_image_vs_entropy`. Not trained, not tuned, no
threshold softened, no clusterer re-chosen. Preserved as the third negative
result of this line, after Proposed-v1 and Proposed-v2.

Two things the passes are worth stating for, because they are real findings and
not consolation. `tail_per_image_vs_cost_aware` at **5.1× and 4.4×** says the
allocator massively beats the cost-aware control on the normalised endpoint —
the confound the ceiling audit warned about is *not* what produced the numbers,
and the control earned its place. And `jaccard ≈ 0.09` says the arms buy almost
disjoint sets: this was a real contrast, not two spellings of one selector.

---

## 2. Part A — the banking forensic

### 2.1 The finding that settles it, before any table

**The frozen endpoint never applied banking.** In
`tools/run_distribution_aware_diagnostic.py`, `held` is accumulated as

```python
for image in picked.images:
    for name, n in counts.get(str(image), {}).items():
        held[name] = held.get(name, 0) + int(n)
```

— every annotated object on every opened image, carried across tasks, with **no
`reuse_deferred_labels` rule anywhere in the driver or in
`owl/active_selection/diagnostic.py`** (grep: no `deferred`, no `trained_on`, no
`banking`). So

```
held_at_declaration  =  objects of the declared class opened at this task
                      + objects of that class opened at ANY earlier task
```

which is exactly **purchased-cumulative**. The name is a misnomer against what
banking V1 would deliver, and that misnomer is what makes H2 look plausible.

**Therefore H2, as stated, is false for this endpoint.** The hypothesis is that
banking discarded early future-class purchases *so that HELD under-reports the
mechanism*. HELD credited every early purchase in full, at zero banking loss.
`distribution_aware_v1` was measured on the most favourable possible accounting
of its own intended mechanism and still returned 0.51× and 0.85× of entropy.

This is a code-reading result, not an inference from numbers, and it does not
depend on any artefact I could not reach.

### 2.2 What is and is not recoverable

**Not reachable from this machine.** `diagnostic_rows.csv` and `verdict.json`
live in `MyDrive/OWL/results/distribution_aware_diagnostic/`; the Drive mount
here still exposes only `OWL/checkpoints` and `OWL/work`. The per-task opened
image lists were never persisted at all — `opened_by_task` lives in memory and
only the aggregate row reaches disk.

| quantity asked for | recoverable? | from what |
|---|---|---|
| 1. images opened | **yes** | `images_opened` in the CSV |
| 2. answers spent | **yes** | `answers_spent` |
| 3. current-declared objects at purchase | **yes** | `current_new_objects` |
| 4. future-class objects at purchase | **yes, pooled** | `future_new_objects` — sums all later-declared classes, **not** split per class |
| 5. of those, held vs lost | **partly** | `banked_from_earlier` at t3/t4 gives *held* per class; *lost* needs the opened lists |
| 6. per named class | **held only** | `banked_from_earlier` is per declared class by construction |
| 7. per-class purchased / lost / survival | **no** | needs the opened image ids |
| 8. normalised by images and answers | **yes for 1–4** | both denominators are in the CSV |
| mixed-image structure, per arm | **no** | needs the opened image ids |

**I have not inferred any of the missing values.** What follows is the
population structure they would sit inside, computed exactly from the committed
candidate index (28,800 images) and the frozen chain — arm-independent by
construction, and offered as structure, not as a substitute for a measurement.

### 2.3 Fate of a pre-declaration purchase — population, exact

Banking V1's rule, from `docs/banking_defect_forensics_2026-09-04.md` §1: for an
image `i` opened at task `k`, `T(i) = k` if `i` holds any class declared by `k`,
else the earliest later declaration on it; a class `c` with `d(c) > k` is
**recovered** iff `d(c) == T(i)` and **lost** otherwise.

| class | bought at | objects in index | recovered | lost |
|---|---|---:|---:|---:|
| `traffic light` | — | 6,703 | *not exposed* — declared at the first purchase task | |
| `fire hydrant` | t2 | 997 | 312 (31.3 %) | **685 (68.7 %)** |
| `stop sign` | t2 | 1,021 | 326 (31.9 %) | **695 (68.1 %)** |
| `stop sign` | t3 | 1,021 | 326 (31.9 %) | **695 (68.1 %)** |

Independently recomputed here; it reproduces the committed 2026-09-04 table.

### 2.4 Mixed-image structure

| | purchase at t2 | purchase at t3 |
|---|---:|---:|
| future classes | `fire hydrant`, `stop sign` | `stop sign` |
| images holding one | 1,752 | 900 |
| **also holding a class declared by then** | **1,165 (66.5 %)** | **601 (66.8 %)** |
| future objects on them | 2,018 | 1,021 |
| **future objects at risk** | **1,371 (67.9 %)** | **695 (68.1 %)** |
| median cost, mixed vs clean | **7 vs 1** answers | **7 vs 1** answers |

Two thirds of the tail supply sits on images that would be trained at purchase
and never re-offered. Mixedness is high, as the earlier forensic said.

### 2.5 Banking loss is almost entirely a function of image cost

Survival of a t2 purchase of `fire hydrant` / `stop sign`, by the cost of the
image it sits on:

| image cost | images | future objects | survives |
|---|---:|---:|---:|
| 1 answer | 517 | 517 | **517 (100.0 %)** |
| 1–4 | 957 | 1,018 | 604 (59.3 %) |
| 4–9 | 489 | 611 | 48 (7.9 %) |
| 9–42 | 460 | 592 | **4 (0.7 %)** |

A sparse image has nothing else on it to trigger early training; a dense one
almost always does. **Banking V1 systematically favours arms that open cheap
images** — which is a statement about `cost_aware` versus everything else, and
it matters for any future detector comparison.

### 2.6 How far apart are ledger and delivered, in practice?

Ceiling policies from `tools/audit_acquisition_ceiling.py`, **not** the arms:
`cheapest*` is the ceiling of the cost-aware mechanism, `stratified*` the
ceiling of cluster rarity. Survival = delivered ÷ ledger-held at declaration.

| policy | t3 `fire hydrant` | t4 `stop sign` |
|---|---:|---:|
| `random` | 80.0 % / 71.4 % | **48.3 % / 63.2 %** |
| `cheapest*` | 80.7 % / 75.8 % | 74.6 % / 79.3 % |
| `stratified*` | 85.3 % / 83.1 % | 76.1 % / 78.4 % |

*(seed 0 / seed 1)*

The population figure is 31 % because it describes *only* the early-purchase
fraction; a task's ledger also contains objects bought at the declaring task
itself, which cannot be lost. For policies resembling our arms the real gap is
**15–25 %**, not 69 %.

### 2.7 A1–A4

**A1 — did it purchase more future t3/t4 objects early than entropy?**
**Not recoverable from what I can reach**, and I will not guess. The answer is
one column: `future_new_objects` at t2 and t3 in `diagnostic_rows.csv`, which is
on Drive. Paste those six numbers and it is answered exactly.

**A2 — what fraction of its early future-tail purchases were lost?**
**Not recoverable per arm.** The population bound is that **68.7 % / 68.1 %** of
*early* tail purchases are lost, and §2.6 puts the loss on the endpoint quantity
at **15–25 %** for policies of this kind. Neither is a measurement of this arm.

**A3 — would the ranking on PURCHASED differ from the ranking on HELD?**
**No. They cannot differ, because they are the same quantity.** §2.1. HELD is
purchased-cumulative with no banking applied. There is no second ranking to
compute.

**A4 — is banking large enough to plausibly mask the intended mechanism?**
**No — not in this endpoint**, and the reason is not a magnitude argument but a
definitional one: the endpoint never applied banking, so it cannot have hidden
anything. Even entertaining the counterfactual does not rescue the gate: seed 0
sits at 0.51×, and reaching 1.25× would need entropy's survival to be 2.45×
worse than the method's, where the widest spread observed across whole policies
is about 1.7× and the direction of the cost effect (§2.5) is unknown for these
arms. **I am not calling that causal proof of anything; I am declining to call
banking an explanation.**

**Where banking does matter** is a different question, and the answer there is
yes: a downstream *detector* comparison would deliver 15–25 % less tail
supervision than the ledger says, unevenly across arms, in a direction set by
image cost. That is a protocol question, treated in §5.

---

## 3. Part B — is an iterative experiment justified?

**Yes**, and for one reason only: **`distribution_aware_v1` tested a degenerate
case of the method as specified.**

`R_k = log((1 + n_cand,k)/(1 + n_ref,k))` is defined against a reference that
*grows with what has been bought* — that is what makes it novelty relative to
already-labelled knowledge rather than relative to a fixed anchor. One-shot
selection froze that reference for the whole task, so the feedback loop the plan
draws was never engaged. The 2026-08-25 consultation raised one-shot versus
rounds explicitly (point 7). That is on the register from before this NO-GO, so
testing it is completing the pre-registration, not reacting to a result.

> **CORRECTION, 2026-09-07.** This paragraph originally said rounds "help exactly
> the arms with something to update (`consult` 26 → 36, **+38 %**)". Read back
> from `data/results/selection_arms.csv`, the actual figures for `consult` at
> 600×1 → 6×100 are **25 → 34, 30 → 32, 37 → 40** across seeds 0/1/2 — **+36 %,
> +7 %, +8 %** — and `prior_consult_batch`, the arm that combines a prior with
> batch diversity, gets **worse** on every seed (91 → 82, 81 → 73, 90 → 73). The
> no-op for `entropy`, `objectness` and `plan` is exact and does hold. So the
> prior that rounds help is far weaker than stated, and it supports the adverse
> prediction in §3.1 rather than the case for running the experiment. The
> decision in §7 is unchanged — it rests on v1 having tested a degenerate form of
> the specification, not on this figure — but the figure was wrong and the
> justification is weaker than it read.

### 3.1 A prediction, recorded before the experiment exists

**I expect it to fail, and I expect it to fail worse.** Iteration adds this
task's purchases to the reference between rounds, which *raises* `n_ref` for
clusters just bought and *lowers* their quota — so budget moves toward clusters
not yet visited. That is **more** spreading across the ~61 unknown classes, and
spreading across 61 classes when one of them is declared next is precisely the
mechanism that has now failed three times: Proposed-v1 (0.00 mean new-class
AP50), Proposed-v2 (0.06 against a 3.56 floor), and v1's 0.51×/0.85×.

This is written down so that a failure cannot be re-narrated afterwards as
expected-all-along, and so that a *pass* would be genuinely surprising and
therefore genuinely informative. It does not change a threshold or a definition.

---

## 4. `distribution_aware_iterative_v1`

**Exactly `distribution_aware_v1`, with the reference recomputed between
rounds.** Nothing else moves: same `A` gate, same DINOv2 representation and crop,
same HDBSCAN family, same `min_cluster_size` rule, same `C`, same `R`, same
quota rule, same `U`, same candidate population, same task chain, same total
budget, same image-level cost, same evaluation endpoints. No λ, no γ, no
temperature, no traversal, no new representation, no oracle-tuned parameter, no
class label and no future declaration inside selection.

### 4.1 Round semantics, frozen

**6 rounds of 500 answers, with the unspent remainder carried forward.** 6 is the
consultation's own declared ablation; 500 follows from this protocol's 3,000-answer
budget. 6×100 belonged to the 600-region protocol and is not reused.

At round *r*, over the images this trajectory has not yet opened:

1. eligible set = the `A`-gated population `G` restricted to unopened images;
2. embeddings: **the same DINOv2 matrix**, subset by row — not re-embedded;
3. fit HDBSCAN on the eligible rows, `min_cluster_size` from the same
   budget-derived rule applied to the round's own eligible count;
4. `C(x) = 0` for noise, `1` otherwise — unchanged;
5. `n_ref,k` against REF-T1 **plus every image bought in any earlier round or
   task** — this is the only quantity that moves;
6. `R_k = log((1 + n_cand,k)/(1 + n_ref,k))`, `R⁺ = max(R, 0)` — unchanged;
7. `q_k = B_r · R⁺_k / Σ R⁺` where `B_r` is this round's allowance — unchanged rule;
8. within cluster, descending `U` — unchanged;
9. spend through the existing `spend_ranking`;
10. add the newly opened images to the reference; go to *r+1*.

No oracle class, box or declaration enters steps 1–9.

**Underspend.** The existing rule is preserved exactly: within a round, an image
that does not fit ends that round rather than being skipped, because skipping
biases a campaign's tail toward sparse images. What is *added* is that the
unspent remainder is **carried into the next round's allowance**, so round *r*
may spend `500·r − (spent so far)`. This is not a new spend semantics; it is the
only carry rule under which the schedule conserves the frozen 3,000-answer
budget rather than quietly reducing it.

### 4.2 Comparators — your preference confirmed, and it is free

**Yes: every comparator runs the same 6-round schedule.** You are right that
otherwise one-shot versus iterative changes both the method and its interaction
with the image-level cost budget.

The implementation makes this cost nothing, and the reason is provable rather
than assumed. For a *static* ranking the order never changes between rounds, so
with the carry rule the schedule opens the identical images in the identical
sequence. Measured on a 4,000-image pool:

```
one-shot 3000    : 317 images, 2997 answers
6 x 500 + carry  : 317 images, 2997 answers   identical sequence: True
6 x 500 no carry : 314 images                 identical: False
```

So `entropy` and `cost_aware` **already are** their 6-round results: their rows
from the completed diagnostic are reused verbatim, and no detector pass is
repeated for them. Note the contrast — *without* carry the schedule silently
loses three images to six separate stopping decisions, which is exactly the
method/budget confound you were guarding against.

### 4.3 Frozen gates — identical, all six

`jaccard_vs_entropy < 0.60` · `tail_per_image_vs_entropy ≥ 1.25×` ·
`tail_per_image_vs_cost_aware ≥ 1.25×` · `background_share_vs_entropy ≤ +10 pp` ·
`breadth_vs_entropy ≥ 0.75×` · `t2_not_collapsed ≥ 0.75×`. Same aggregation:
first three per seed, last three on the mean over seeds. Same seeds (0, 1).

**No gate definition changes, and none needs to.** Rounds alter *when* images are
chosen, not the unit any gate is measured in: Jaccard is over the set of opened
images, the tail endpoint is per opened image, background share is a fraction of
opened images, breadth is a count of acquired classes. Every denominator is
end-of-task and unaffected by how the task was subdivided.

---

## 5. Banking: **OPTION 1 for this experiment**

**Keep banking V1, and declare the limitation.** Reasons, in order:

1. Part A shows banking is not an explanation for the v1 endpoint, so fixing it
   would not be rescuing a method — it would be changing an unrelated variable.
2. This is a *candidate-side* diagnostic. It never trains, so the banking rule
   has no effect on any number it produces at all.
3. Comparability: `entropy` and `cost_aware` were measured under V1, and §4.2's
   reuse of their rows depends on nothing else having moved.
4. Your own instruction, which is the right one: do not mix a banking fix and an
   iterative selector into one first experiment.

**Protocol V2, defined but not implemented.**

*The fix.* `deferred = ledger − trained_on − opened` in `owl/runner.py` treats a
trained image as spent. Replace the image-level bookkeeping with class-level: an
image re-enters training at every task that declares a class present on it and
not yet supervised, i.e. defer on `(image, class)` rather than on `image`.

*Why it is a new protocol and not a patch.* It changes what PROB is handed at t3
and t4 for every arm, so every completed number under V1 becomes incomparable.
Seeds 0, 1 and 2 of `random`, `admissibility` and `entropy`, and both negative
Proposed results, were all measured under V1.

*Must the baselines be re-run?* **Yes**, all of them, for any V2 claim — a V1
number and a V2 number cannot appear in the same table. That is 9 baseline
trajectories minimum.

*Does repeated-image rehearsal become a confound?* **Yes, and it is the main
scientific cost.** Under V2 a mixed image is trained at t2 *and* again at t3,
so arms differ in how many times the same pixels are seen — an uncontrolled
rehearsal signal correlated with image density, i.e. with the very cost
structure §2.5 shows already separates the arms. V2 would need to record
per-image training multiplicity and report it, or the forgetting numbers become
uninterpretable.

*Minimum fair comparison set.* `random`, `admissibility`, `entropy` at seeds 0
and 1 under V2, plus whichever proposed arm is under test — 8 trajectories,
about 18 T4-hours. That is why it is a decision for the supervisor and not one
to take inside this experiment.

---

## 6. Cost and implementation

**Reuse, and what cannot be reused.** DINOv2 exports are keyed on
`row_fingerprint(image_ids, boxes)` of the rows handed to `semantic.cached`. The
iterative arm is `A`-gated like v1, so **for a given pool it embeds the identical
`G` and hits the identical cache** — provided the implementation embeds full `G`
once per task and subsets the matrix in memory per round, which it must. What
does *not* carry over: at t3 and t4 the iterative arm has opened a different set
at earlier tasks, so its candidate pool differs and it needs its own detector
pass and its own DINOv2 export there.

| item | new GPU work | estimate |
|---|---|---|
| `entropy`, `cost_aware` rows | **none** — provably identical under §4.2 | 0 |
| iterative t2 (both seeds) | none — pool identical to v1's, both caches hit | 0 |
| iterative t3, t4 (both seeds) | 4 detector passes over 2,000 images, 4 DINOv2 exports | ~45 min |
| candidate JPEGs for those pools | mostly already fetched | ~15 min |
| 36 HDBSCAN refits on shrinking sets | CPU | ~30 min |
| **total** | | **~1.5 h** |

**Plan.** (1) `owl/active_selection/allocation.py` gains
`distribution_aware_iterative_order(...)` — the same functions called in a loop
over rounds, with the reference growing; no existing function changes behaviour.
(2) One registry entry, appended to `ORDER`, marked development-seed-informed.
(3) `ROUNDS` and the carry rule frozen in `owl/active_selection/diagnostic.py`
beside the gates. (4) The driver gains `--rounds`, defaulting to 1 so every
existing arm is bit-identical. (5) Tests: the carry rule is a no-op for static
rankings; the reference strictly grows; no oracle reachable; the order is a
permutation; DINOv2 rows are the full `G`. (6) The notebook reuses the existing
Drive workspace and adds the one arm.

---

## 7. Decision

> ## IMPLEMENT ITERATIVE CHEAP DIAGNOSTIC

with the §3.1 prediction on the record: **I expect it to fail, and to fail worse
than v1**, because iteration pushes budget toward unvisited clusters and
spreading is the mechanism that has already failed three times. It is worth
running anyway because v1 tested a degenerate form of the specified method,
because the test was pre-registered before this NO-GO, and because it costs
~1.5 hours and no baseline re-run.

If it fails, it is preserved as the fourth negative and **this D/R/C
formulation stops there** — no v3, no tail-specific heuristic, no banking fix
used to rescue it. Contribution B is not launched.

---

## 8. Correction to §2.2, and the state of A1 / A2

**Recorded as a correction rather than an edit, because §2.2 was wrong about
what the file can do.** It said per-class early purchase was unavailable. It is
**exactly recoverable**, and the conclusions in §1–§7 are unaffected.

No column is a per-class counter, but three are linear in the quantities wanted.
Writing `FH2` for fire hydrants on images opened at t2 and `SS2`, `SS3` for stop
signs opened at t2 and t3, then from `oracle_row`:

* `future_new_objects` sums the **later-declared** classes: `FH2 + SS2` at t2,
  `SS3` at t3, `0` at t4;
* `banked_from_earlier` is `held[declared class]` read **before** the task's own
  purchases are added: `0` at t2, `FH2` at t3, `SS2 + SS3` at t4.

So `FH2 = banked(t3)`, `SS3 = future_new(t3)`, `SS2 = banked(t4) − future_new(t3)`
— and the system is over-determined, which yields a free audit:

```
future_new(t2)  ==  banked(t3) + banked(t4) - future_new(t3)
```

`tools/close_part_a.py` performs the reconstruction and **refuses to report if
that identity fails**, rather than printing a plausible number.

| | status |
|---|---|
| **A1** — did it buy more future t3/t4 objects early than entropy? | **open.** Exactly answerable; the file is on Drive and unreachable from the machine this was written on. One command closes it. |
| **A2** — what fraction of early purchases banking lost, per arm | **NOT RECOVERABLE FROM PERSISTED ARTEFACTS.** Banking survival is a property of *which* images were opened, and the opened image identities were never written. Not estimated. |

```bash
python tools/close_part_a.py \
  /content/drive/MyDrive/OWL/results/distribution_aware_diagnostic/diagnostic_rows.csv
```

**No A1 outcome can change the decision in §7, and this was settled before the
numbers were seen.** If it bought *more* early, its discovery mechanism works
and the failure is in the per-image normalisation — which leaves the NO-GO
standing and, if anything, strengthens the case for testing whether rounds
convert discovery into supply. If it bought *fewer*, that is the selector
failing at discovery, which is §3.1's prediction and also leaves the NO-GO
standing. Both readings say run the iterative test once. A1 is worth closing
because it decides *which* negative we are looking at, not whether to look.
