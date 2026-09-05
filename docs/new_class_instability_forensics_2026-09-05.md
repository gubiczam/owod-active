# Why new-class learning is unstable across t2/t3/t4

2026-09-05. CPU, read-only. No detector was trained or run, no benchmark
behaviour modified, no method proposed.

---

## 0. What I could and could not read

**Could not:** the seed-0 trajectory artefacts — `per_task_metrics.csv`,
`acquisition.csv`, `supervision_cost.csv`, `per_class_ap.csv`, and the per-task
`state.json` ledgers. They live in
`MyDrive/OWL/results/full_owod_active_benchmark_v1/`. The Google Drive mount
reachable from this machine still holds only `OWL/checkpoints` and `OWL/work`
(an older experiment) and **has no `results/` directory**. Anything below marked
*tool-required* needs `audit_new_class_chain.py`, supplied separately.

**Could:** the committed benchmark annotations — every ground-truth box of the
candidate pool (28 800 images) and of the evaluation split — plus the frozen
chain, the candidate index, and the arm-level numbers reported to me.

**The headline answer does not depend on the missing files.** It is settled by
the annotations, and it is two different failures at two ends of the chain.

---

## 1. The causal chain: AVAILABLE → ACQUIRED → SUPERVISED → TRAINED → AP

### 1.1 AVAILABLE — measured, complete

| task | declared | objects in candidate pool | images in pool | **objects in the shared eval split** | **eval images** | one eval object is worth |
|---|---|---:|---:|---:|---:|---:|
| t2 | `traffic light` | 6 703 | 2 159 | **534** | 160 | 0.19 % of recall |
| t3 | `fire hydrant` | 997 | 911 | **101** | 86 | 0.99 % of recall |
| t4 | `stop sign` | 1 021 | 900 | **75** | 69 | **1.33 % of recall** |

And the class that dominates the tail metric:

| | objects in pool | **objects in the eval split** | one eval object |
|---|---:|---:|---:|
| `bear` | 41 | **2** | **50 % of recall** |

### 1.2 Object geometry — the fact that explains t2

Median instance size, on the shared evaluation split, in absolute pixels, with
COCO's own size convention (`small` < 32×32):

| class | eval objects | median area | ≈ median side | **% COCO-small** | % medium | % large |
|---|---:|---:|---:|---:|---:|---:|
| **`traffic light`** | 534 | **282 px²** | **16.8 px** | **78.5 %** | 18.0 % | 3.6 % |
| `fire hydrant` | 101 | 10 564 px² | 102.8 px | 22.8 % | 22.8 % | 54.5 % |
| `stop sign` | 75 | 3 720 px² | 61.0 px | 33.3 % | 26.7 % | 40.0 % |
| `bear` | 2 | 317 298 px² | 563.3 px | 0 % | 0 % | 100 % |
| `person` | 1 927 | 2 585 px² | 50.8 px | 35.0 % | 34.7 % | 30.3 % |
| `car` | 821 | 868 px² | 29.5 px | 54.6 % | 35.4 % | 10.0 % |

`traffic light` is **37× smaller by area than `fire hydrant`** and 13× smaller
than `stop sign`. Nearly four in five instances are below the threshold at which
COCO stops expecting detectors to do well.

### 1.3 ACQUIRED → SUPERVISED → TRAINED — *tool-required*

The per-arm columns you asked for (`acquired this task`, `supervised this task`,
`from banked supervision`, `in replay`, `total training instances`,
`training_iterations`) are reconstructable **exactly** from each trajectory's
`state.json` and `train/labelled_ids.txt` crossed with the candidate index — no
inference needed. That is what `audit_new_class_chain.py` does. The one value I
was given, `entropy` at t2 acquiring **146** `traffic light` objects, is used
below.

---

## 2. The four anomalies

### A. t2 `traffic light`: AP ≈ 0 even at 146 acquired objects

**Not a supply failure. A learning failure, and the cause is object size.**

Supply at t2 is the *best* of the three tasks by a wide margin. A random
300-image draw is expected to contain ~70 `traffic light` boxes — **6.0 % of all
declared boxes, 5.3× the median declared class**. `entropy` acquired **146**,
roughly twice that. Method V3's audit measured ≈0 AP from 23–101 instances and
concluded supply was the binding constraint; at 146 instances and 6 % of the
training set, that explanation is exhausted.

What is left is the target itself: a **17-pixel** object, 78.5 % of instances
below COCO-small, being learned by a Deformable-DETR fine-tuned for **5 epochs**
on a few hundred images with **`freeze_prob_model=True`** — PROB's own
procedure, which holds the objectness head fixed. The detector is not being
asked to learn a new category so much as to detect a category it has no
resolution for.

The zero is also **well measured**: 534 test objects on 160 images. It is a
trustworthy zero, unlike the t3/t4 numbers below.

### B. t3 `fire hydrant`: `random` 6.82, everyone else 0

**The honest answer is that the evidence cannot distinguish signal from noise,
and the measurement design is why.**

`fire hydrant` has **101 objects on 86 images** in the evaluation split, so one
detection is worth 0.99 % of recall. A structural draw supplies ~10 hydrant
boxes to any arm at t3 — **0.88 % of declared boxes, 1:76 against `person`, and
*below* the median declared class**. Every arm is in a ten-instance regime.

So `random` at 6.82 and three arms at exactly 0.00 is the difference between
"a handful of detections landed" and "none did", on a ~10-example training
signal evaluated against 101 objects. Section 5 treats this at length. **No
causal claim is supportable without a measured nondeterminism floor, which this
benchmark has never provided.**

### C. t4 `stop sign`: `admissibility` and `entropy` ≈ 20–22, others ≈ 0

Same regime, worse: **75 test objects on 69 images**, one worth **1.33 %** of
recall, and ~11 acquired boxes expected (0.90 % of declared boxes, 1:74 against
`person`). But `stop sign` is a much easier target than `traffic light` —
median 61 px side, 40 % of instances COCO-large — which is why anything is
learned here at all and nothing is at t2.

The **bimodality is the striking part**: across t3 and t4 the per-task outcomes
are ≈0 or ≈20, with nothing between. That is the signature of a *few-shot
threshold* — a class either catches or it does not — measured on a test set
small enough that catching moves AP 20 points at once. Which arm lands on which
side of the threshold at which task is, on present evidence, not shown to be a
property of the selector.

### D. Proposed-v2: 4 101 supervised boxes, highest U-Recall, 0.06 new-class AP

*Determining what those boxes are is tool-required* — it needs v2's opened-image
ids. What the structure already says: at **0.88–0.90 %** of declared boxes for
the t3/t4 classes and **6 %** for t2, the newly declared class can only ever be a
sliver of any arm's supervision. Delivering 4 101 boxes rather than ~3 800 buys
**more `person`**, not more new class — `person` alone is ~789 boxes per 300
images against the new class's 10.

This is the same finding as v1 in a different guise, and section 6 quantifies
it: across the five seed-0 arms, mean `U_Recall50` and mean `new_class_AP50`
are **negatively** associated. Breadth of unknown discovery, and sheer quantity
of supervision, are not the currency of new-class AP.

---

## 3. Class imbalance inside the training set — structural, measured

Expected histogram for a 300-image draw (an arm's actual histogram is
tool-required):

| task | new class | boxes | **% of declared boxes** | vs most frequent (`person`) | vs median class |
|---|---|---:|---:|---:|---:|
| t2 | `traffic light` | ~70 | **5.99 %** | 1 : 11 | **5.29×** |
| t3 | `fire hydrant` | ~10 | **0.88 %** | 1 : 76 | 0.87× |
| t4 | `stop sign` | ~11 | **0.90 %** | 1 : 74 | 0.89× |

**The answer to "is the detector simply seeing too few positives?" is: at t3 and
t4 yes, at t2 no.** t2's class is over five times the median declared class and
one box in seventeen; t3 and t4 sit below the median at roughly one box in a
hundred. These are two different problems and they need different fixes.

---

## 4. Acquisition quality: current-new vs future vs already-known — *tool-required*

The breakdown you want — currently declared / becomes known next task / becomes
known later / remains unknown through t4 / already known / background — is
reconstructable per arm from `state.json` plus the candidate index, and the
runner already records three of its six buckets (`acquired_new_class`,
`acquired_becomes_known_t3/t4`, `acquired_stays_unknown`). The tool completes it.

The hypothesis it tests is precise and worth stating before the numbers arrive:
that the coverage arms spend budget on **breadth across future and never-declared
classes** while under-buying the one class the learner needs now. Two facts
already lean that way — the negative U-Recall/new-AP association of section 6,
and the earlier measurement that of the **42** unknown classes in the population
exactly **one** is declared at the next task.

---

## 5. `random`'s fire hydrant: supported, or variance?

**Not supportable as causal on current evidence.** What would be needed, and its
status:

| evidence | status |
|---|---|
| acquired fire hydrants, per arm | tool-required |
| supervised fire hydrants, unique images | tool-required |
| replay composition at t3 | tool-required (recorded in `replay_row`) |
| total iterations | tool-required |
| checkpoint lineage | already asserted correct, per-arm, by the dry run |
| per-class AP before/after t3 | needs `per_class_ap.csv` |
| **nondeterminism floor** | **never measured, by any experiment in this project** |

The last row is decisive. Even with every other column filled, a 6.82-vs-0.00
gap on a 101-object test set, from a ~10-instance training signal, cannot be
attributed to selection without knowing what two identical configurations do.
The earlier post-hoc audit established that PROB never calls
`torch.use_deterministic_algorithms`, that MSDeformAttn accumulates with
atomics, and that paired arms share no common random numbers. **The floor could
be several AP points and nobody has measured it.**

Three arms scoring *exactly* 0.00 is itself informative: it means no correct
detection cleared threshold at all, which is more consistent with a threshold
not being crossed than with a graded difference in supervision quality.

---

## 6. Correlations

**Computable now, n = 5 arms, arm-level means. Descriptive only.**

mean `U_Recall50` against mean `new_class_AP50`:

> **Spearman ρ = −0.500, Pearson r = −0.613** (n = 5)

Higher unknown-recall breadth goes with **lower** new-class AP. At n=5 this is a
direction, not an effect; but it is the same direction the two coverage
formulations produced independently, and it is what the mechanism predicts.

**The 15-point (5 arms × 3 tasks) analysis you asked for is tool-required** — it
needs per-task `new_class_AP50` and the acquisition columns together. When run,
it should report Pearson and Spearman for `new_class_AP50` against
current-new acquired / supervised / unique images / total boxes supervised /
U-Recall / class breadth, at n=15, and per-task at **n=5 per task**.

**Warn on the face of that table:** with a bimodal outcome (≈0 or ≈20) and
n=5 per task, a per-task correlation is dominated by which side of a threshold
each arm fell on. It should be read as a description of the sample and nothing
more.

---

## 7. Meeting decision

### A. The strongest result we have

**The annotation-cost design works, and it is the methodological contribution.**
Counting the budget in oracle answers under full-image labelling matched
`boxes_labelled` across arms to within ~6 % by construction, against the
predecessor experiment's 2.09× disparity — and the ledger separates what was
*bought* from what PROB was *handed*. That is a reusable result independent of
which selector wins.

Second: **the full sequential chain runs**, t2→t3→t4 with per-arm checkpoint
lineage, resume, and a frozen shared evaluation split — asserted, not assumed.

### B. The most important negative result

**Two independently designed coverage-based selectors both failed downstream,
from different mechanisms, and both produced the *highest* unknown-recall of any
arm while producing the *lowest* new-class AP.** Proposed-v1 maximised coverage
against a fixed labelled reference; Proposed-v2 used coverage only to
de-duplicate an uncertainty-filtered subset with REF-T1 removed. They share only
the farthest-first traversal. Mean new-class AP 0.00 and 0.06; mean U-Recall
18.05 and 18.76, the top two of five arms.

Stated at the right strength: on one development seed, **semantic breadth did
not convert into incremental new-class AP**, twice, under different objectives.

### C. The most likely bottleneck

**Two, and they are not the same bottleneck.**

* **t2 — the detector.** Supply is ample (6 % of boxes, 5.3× the median class,
  146 instances acquired) and AP is a well-measured zero on 534 test objects.
  A 17-pixel target, 5 epochs, frozen objectness head.
* **t3/t4 — supply *and* measurement.** ~10 acquired instances, 0.9 % of boxes,
  1:75 against `person`, evaluated against 101 and 75 test objects. The regime
  is too small to learn in *and* too small to measure in.

### D. Selection failure, learning failure, or both?

**Both, cleanly separated by task — and this is the report's main finding.**

* **t2 is a learning failure.** The selector delivered; the detector could not
  use it. No acquisition strategy fixes a 17-pixel object.
* **t3 and t4 are a supply-and-measurement failure.** No selector can acquire
  many instances of a class present in ~3 % of pool images at a 300-image
  budget, and no evaluation on 75 objects can resolve the difference if one did.

**What is *not* supported: that any arm's selection quality has been shown to
cause its new-class AP.** The t3/t4 differences that carry the entire arm
ordering sit in a regime where the noise floor is unmeasured.

### E. Three next experiments, by information gain per GPU hour

1. **Re-evaluate the existing checkpoints on a much larger evaluation split.**
   No training at all — evaluation passes only, on checkpoints already on Drive.
   Uncap declared-class test images and raise the remainder; `stop sign` goes
   from 75 objects toward its full 75+ and `fire hydrant` from 101, and
   `bear` from **2**. **Cost: under an hour. It is the cheapest thing on this
   list and it directly attacks the reason half the numbers are unreadable.**
   Caveat to state: this changes the split, so re-evaluated numbers form their
   own comparable set and are not mixed with the frozen-split ones.
2. **Measure the nondeterminism floor.** One fixed acquisition — say
   `admissibility` at t4 — retrained 3–4 times with everything else held. ~3 GPU
   hours. It decides whether *any* t3/t4 difference in this benchmark is
   interpretable, and every conclusion in section 7D depends on it. The Method V3
   audit recommended exactly this and it has never been run.
3. **Separate size from supply with a positive control.** One single-task run in
   which the declared class is both well-supplied *and* large — the opposite
   corner from `traffic light`. ~45 minutes per arm. It tests whether this
   pipeline can learn *any* new class at this budget, which is the assumption
   every negative result so far rests on.

Nothing on this list is a new selection method, and none of it should be run
before the seed-1 replication finishes.

---

## 8. A prediction of mine that has now failed

In `docs/banking_defect_forensics_2026-09-04.md` I recommended keeping the
protocol (option A) and named the condition that would reverse it:

> *"If per-task `new_class_AP50` shows t2 at ~0 for every arm and the whole
> spread lives at t3 and t4, then the defect is squarely in the causal path and
> the recommendation flips to (B)."*

**That condition is met.** t2 is ≈0 for all arms and the entire spread is at t3
and t4. My argument for (A) rested on t2 carrying the signal, and it does not.
**I withdraw that recommendation as stated.**

It does **not** follow that (B) — rerun everything under fixed banking — is now
right, and the reason is in section 1.1. Repairing banking would add roughly 7
recovered objects to a ~10-object supply, evaluated against 75 and 101 test
objects. That is a change inside the noise, bought for ~25 GPU-hours. **Neither
A nor B is the right next move.** The measurements above say the binding
constraints at t3/t4 are supply and evaluation resolution, so experiments 1 and
2 come first: they cost under four hours together and they determine whether a
banking fix could ever be detected. Deciding A/B before that would be choosing
without a readable instrument.
