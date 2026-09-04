# The deferred-label banking defect: forensics, protocol-v2 design, and a recommendation

2026-09-04. Read-only analysis. No detector was run, no result altered, no code
committed.

---

## 0. What I could and could not measure

**Could not:** the per-arm, per-task counts you asked for. They need each
trajectory's saved ledger, and those live in
`MyDrive/OWL/results/full_owod_active_benchmark_v1/`. The Google Drive mount
visible from this machine holds `OWL/checkpoints` and `OWL/work` (an older
experiment) and **has no `results/` directory at all**. So the forensic table is
produced by a tool, not by me: `audit_banking_defect.py`, written and verified
but **not committed**, per instruction.

**Could:** the model of the defect, checked against the implementation; the
**population-level** fate of a pre-declaration purchase, over the whole 28,800
-image candidate index; and — decisively for the recommendation — *which task's
new class is exposed to the defect at all*.

---

## 1. The model, stated so it can be checked rather than trusted

For an image `i` opened at task `k`, let `d(c)` be the task that declares class
`c`, and

    T(i) = k                                if i holds a class declared by k
         = min{ d(c) : c on i, d(c) > k }   otherwise
         = never                            if no class on i is ever declared

`T(i)` is the task at which the image is handed to PROB, whether directly or
through banking. Then for a class `c` on `i` with `d(c) > k`:

| | |
|---|---|
| **recovered** | `d(c) == T(i)` — the class is declared exactly when the image is trained |
| **lost** | `d(c) > T(i)` — the image is trained earlier, enters `trained_on`, and is never re-offered |
| **never declared** | this chain never declares `c` at all |

The root cause is one set difference in `owl/runner.py`:
`deferred = ledger − trained_on − opened`. Subtracting `trained_on` assumes an
image, once trained, has nothing left to give. A barren image is therefore
recovered **exactly once**, and any class on it declared later than its first
trainable task is lost too.

`audit_banking_defect.py --verify` (on by default) checks this model against the
trajectories: the set of images the runner actually deferred into task `k` must
equal the set the model predicts. It refuses to print counts if they disagree.

---

## 2. Which classes are exposed at all — and a correction

**`traffic light` is not exposed.** It is declared at **t2**, and t2 is the
first task at which anything is purchased. It can never be bought before its
own declaration.

> **Correction to my earlier report.** I listed "88.8 % of `traffic light`
> objects sit on mixed images" beside the `fire hydrant` and `stop sign`
> figures, which invited the reading that all three classes lose that share.
> They do not. That number described *mixedness*, not loss, and for
> `traffic light` the loss is **zero by construction**. The protocol and the
> supervisor note are corrected in the same commit as this document.

| class | declared | purchasable before declaration | exposed? |
|---|---|---|---|
| `traffic light` | t2 | — (t2 is the first purchase task) | **no** |
| `fire hydrant` | t3 | t2 | yes |
| `stop sign` | t4 | t2, t3 | yes |

---

## 3. Population-level fate of a pre-declaration purchase

Over all 28,800 candidate-index images. This is the structure the per-arm
numbers will sit inside, not a substitute for them.

| class | bought at | objects in index | recoverable | **lost** |
|---|---|---:|---:|---:|
| `fire hydrant` | t2 | 997 | 312 (31.3 %) | **685 (68.7 %)** |
| `stop sign` | t2 | 1 021 | 326 (31.9 %) | **695 (68.1 %)** |
| `stop sign` | t3 | 1 021 | 326 (31.9 %) | **695 (68.1 %)** |

Recovery requires the image to hold **no** class declared by the purchase task
*and* no class declared before the one in question — for `stop sign` bought at
t2 that means no task-1 class, no `traffic light` **and** no `fire hydrant`.

**Scale, for one arm selecting uniformly** (2 000-image pool, ~300 images opened
= 15 %):

| class | bought at | objects bought | lost | recovered |
|---|---|---:|---:|---:|
| `fire hydrant` | t2 | ~10.4 | **~7.1** | ~3.2 |
| `stop sign` | t2 | ~10.6 | **~7.2** | ~3.4 |
| `stop sign` | t3 | ~10.6 | **~7.2** | ~3.4 |

Against roughly 10 objects of the class supervised directly at its own
declaring task, the defect costs about 40 % of the total supply reaching
training for `fire hydrant` and `stop sign`. It costs nothing for
`traffic light`.

---

## 4. Does it affect the arms differently, and does it undermine the comparison?

The per-arm answer needs the tool. What can be settled now is whether it
*could* have produced the observed ordering, and the answer turns on one
structural fact:

**`mean new_class_AP50` averages t2, t3 and t4 — and t2's class is immune.**
`traffic light` also has by far the largest supply: 6,703 objects on 2,159
images, against ~1,000 objects each for `fire hydrant` and `stop sign`. The
frozen-pool simulation put per-arm `traffic light` acquisition at 21–112
objects and `fire hydrant`/`stop sign` at ~10.

Two consequences:

1. **`proposed-v1`'s 0.00 cannot be caused by the defect.** A mean of exactly
   zero requires ~zero at t2 as well, and t2 is untouched by banking. Whatever
   made the proposed method fail, it was not this.
2. **The 7.3-point spread between the arms is very likely dominated by a
   defect-immune task.** If t2 carries most of the signal, the defect changes
   every arm's t3/t4 ceiling and cannot reorder them.

So on present evidence the defect **mainly lowers everyone's future-supervision
ceiling** rather than undermining the four-arm comparison. It remains a
confound of *unknown sign* between arms with different barren shares — the
frozen-pool simulation put `admissibility` at 70 % barren against `entropy`'s
34 %, and a *barren* purchase is the one that gets recovered, so the arm that
opens more empty images is the one the defect treats more kindly. That
direction, if it matters at all, would flatter `admissibility` relative to
`entropy` — and those two are already within 0.2 of each other on mean
new-class AP.

**The falsifier, named in advance.** Read per-task `new_class_AP50` from
`per_task_metrics.csv`. If t2 is ~0 for every arm and the whole 7.3-point spread
lives at t3 and t4, then the defect is squarely in the causal path and the
recommendation below flips to (B).

---

## 5. Protocol-v2 banking — design only, not implemented

**Desired semantics.** Paying for an image buys every box on it, permanently. At
task `k`, boxes of classes declared by `k` are trainable; the rest are stored and
masked. When a stored class is later declared, its already-paid annotation
becomes trainable **without repurchasing the image**. No future-class identity
influences acquisition ranking. Replay stays separately defined and leaks
nothing.

**What is already correct and needs no change.** "Stored but masked" is what the
GPU path does today: PROB's `remove_unknown_instances` keeps
`category_id in range(0, prev + current)` and drops the rest, so a stored box is
masked by the loader rather than by us. Nothing needs to be written to disk, and
no annotation needs rewriting. Replay already filters to
`task.previous_classes`, so a future class cannot become an exemplar under
either protocol.

**The minimal change is one predicate.** Replace "has this image ever been
trained on?" with "has this image gained a trainable box since it was last
trained on?":

```
last_trained_at: dict[str, int]          # image -> task index it was last handed to PROB

def declared_boxes(image, upto_task):    # from candidate_index + the chain
    ...                                  # boxes whose class is declared by that task

def owes_supervision(image, k):
    seen = last_trained_at.get(image)
    if seen is None:
        return usable(image)             # unchanged first-time behaviour
    return declared_boxes(image, k) > declared_boxes(image, seen)

deferred = sorted(i for i in ledger - set(opened) if owes_supervision(i, k))
...
for image in trainable:
    last_trained_at[image] = k
```

`last_trained_at` joins `state.json` beside `ledger` and `trained_on`, so a
resumed chain reconstructs it. `trained_on` stays for the accounting it already
does. That is the whole change: one dict, one predicate, one persisted field.

**Leakage: unchanged.** `owes_supervision` reads per-image class *counts* — the
same quantity the cost function already reads — and only for images already
purchased, and only *after* the budget is committed, in the same place `deferred`
is computed today. No selector sees it. The existing source-order test that
pins "future labels are read only after the selector returns" continues to
cover it.

**Two consequences that must be recorded, not discovered.**

1. An image may now be handed to PROB up to three times in a four-task chain.
   `training_images` and `training_iterations` rise, unequally across arms, and
   the step-count asymmetry the protocol already reports gets larger. A new
   column — `images_retrained` — belongs in the row.
2. That repeated exposure is an *uncharged* rehearsal effect on top of the
   400-object replay memory. It should be expected to raise `known_mAP50`
   somewhat, and it is a new confound between arms that differ in how many
   images they re-offer. Protocol v2 must state it.

---

## 6. Recommendation: **(A) KEEP V1**

Keep the current banking behaviour as the frozen protocol, document the
limitation, and continue with Proposed-v2 seed 0 and the baseline replication
seeds.

**The scientific reason, and it is not the schedule.** Fixing banking would
raise `fire hydrant` and `stop sign` supply reaching training from roughly 3
recovered objects to roughly 10 per arm per task. **Ten is still far below what
this pipeline needs to learn a class.** The Method V3 audit measured
`new_class_AP50 ≈ 0` from **23–101** instances of a new class. A corrected
protocol would move t3 and t4 from *hopeless* to *still hopeless*, at the price
of invalidating four completed trajectories and every comparison built on them.

The binding constraint on new-class learning at t3 and t4 is **supply**, not
banking: `fire hydrant` and `stop sign` have ~1,000 objects each in a
28,800-image index, so a 2,000-image task pool holds ~63 and ~62 images with
one. Raising the candidate pool or the budget is what would make those tasks
measurable; recovering 7 more objects would not.

And the defect is **orthogonal to the contrast actually being reported**: t2's
class is immune and best-supplied, so the arm ordering rests on a task banking
cannot touch. `proposed-v1`'s 0.00 in particular cannot be explained by it.

**What (A) commits us to saying.** That `mAP50_tail` at t4 and
`new_class_AP50` at t3 and t4 carry a **known, quantified ceiling** — about 68 %
of pre-declaration purchases of the two tail classes never reach training — and
that this applies to every arm. That is a reportable limitation, not a hidden
one, and it is already in the protocol's section 14.

**What would change my mind — the falsifier from section 4.** If per-task
`new_class_AP50` shows t2 at ~0 for all four arms with the entire spread at t3
and t4, the defect is in the causal path of the headline result and **(B)
becomes correct**: the comparison would then rest on the two tasks the defect
distorts, and no amount of documentation would repair it. That check costs one
look at `per_task_metrics.csv`.

**Where banking belongs.** In a protocol v2 that changes **supply and banking
together** — a larger candidate pool or budget for the tail tasks, plus the
one-predicate fix of section 5 — run as its own experiment with its own seeds.
Fixing banking alone would spend the remaining time to move a metric that stays
unmeasurable.

---

## 7. How to produce the actual forensic table

`audit_banking_defect.py` is **not committed**, per instruction. It is read-only
on `--results`, refuses `--out == --results`, and verifies its model of the
runner against the trajectories before printing anything.

```
python audit_banking_defect.py \
    --results /content/drive/MyDrive/OWL/results/full_owod_active_benchmark_v1 \
    --out     /content/drive/MyDrive/OWL/results/banking_forensics
```

It prints, and writes as three CSVs:

1. the model check — observed vs predicted deferred images, per arm per task;
2. per arm, per purchase task, per class: `paid`, `recovered`, `lost`,
   `never_declared`, `recovery_rate`, `lost_fraction`;
3. the same restricted to classes this chain declares later;
4. per-arm rollups, and the spread in lost objects across arms.

Verified on a synthetic trajectory tree in the real layout with a hand-computed
answer: `fire hydrant` 8 paid → 5 recovered, 3 lost; `stop sign` 18 paid → 6
recovered, 12 lost; a never-declared class 9 paid → 9 never; model check
agreeing on every task.
