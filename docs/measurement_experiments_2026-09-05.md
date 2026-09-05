# Two measurement experiments: a larger evaluation, and a variance floor

2026-09-05. Experiment 1 is implemented and dry-run; **it has not been
executed**. Experiment 2 is designed and **not implemented**. No detector was
trained. The seed-1 replication was not touched.

---

## Experiment 1 — re-score existing checkpoints on a larger held-out split

### 1.1 Three findings that change the design. Read these first.

**(a) A larger split cannot fix t3/t4 new-class AP, which is what motivated it.**
The frozen 837-image split already contains **every** test image holding a
`fire hydrant` or a `stop sign`, because the 150-images-per-class cap never
binds for classes that rare. Their support *is* the entire benchmark test set.

| class | frozen split | | declared-complete | | **full test** | | gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| | objects | images | objects | images | objects | images | |
| `traffic light` | 534 | 160 | 637 | 191 | **637** | 191 | 1.2× |
| `fire hydrant` | 101 | 86 | 101 | 86 | **101** | 86 | **1.0×** |
| `stop sign` | 75 | 69 | 75 | 69 | **75** | 69 | **1.0×** |
| `bear` | **2** | 2 | 71 | 49 | **71** | 49 | **35.5×** |
| all 22 declared | 4 237 | — | — | — | **18 599** | — | 4.4× |
| unknown-class objects | 2 548 | — | — | — | **18 182** | — | 7.1× |
| **images** | **837** | | **3 864** | | **4 952** | | |

So the experiment is worth running, for different reasons than assumed:

* **`bear` 2 → 71 objects.** `bear` is one of three classes in the tail band, so
  `mAP50_tail` — a headline metric — is currently *partly a two-object
  measurement*, quantised, where one detection moves the reported tail by up to
  17 points. This repairs that.
* **known `mAP50` 4.4×** and **U-Recall / WI / A-OSE support 7.1×.**
* It does **not** repair t3/t4 new-class AP, and nothing can. There is no more
  test data for those two classes.

**(b) Only ten checkpoints exist, not fifteen.** `keep_checkpoints=2` retains
the newest two per arm, so after a three-task chain each arm keeps **t3 and t4**
and its **t2 checkpoint has been deleted**. t2 cannot be re-scored without
retraining, which is out of scope. The driver examines all three tasks by
default *precisely so that it reports this* rather than hiding it.

**(c) The name `large_eval` is a trap, and the guard catches it.** PROB routes a
split by substring, and `eval` contains `val`. A split named `large_eval` goes
to the `val` branch, where no annotation filtering runs at all: **U-Recall would
read zero everywhere and future-task objects would be scored as though their
class were already known.** A full table of plausible, wrong numbers. The split
is therefore `owl_large_test`, and `tests/test_large_eval.py` asserts that
`large_eval`, `owl_large_eval` and `owl_eval_test` are all refused.

### 1.2 Source, and the leakage check

**Source:** the benchmark's own committed test archive,
`data/staging/owdetr_test_annotations.tar.gz` — 4 952 images. Nothing new is
introduced; the frozen 837-image split is a *subset* of it.

**Leakage — measured, not argued:**

| | images | overlap with the test archive |
|---|---:|---:|
| candidate index (what acquisition draws from) | 28 800 | **0** |
| replay index (what rehearsal draws from) | 89 490 | **0** |
| candidate pool archive | 28 800 | **0** |

**Zero overlap with anything trainable.** Using the whole test split is
legitimate: it is held out by construction, identical for every arm, and it
changes no class declaration and no task semantics. The driver re-runs this
check at runtime and refuses to evaluate if it ever fails.

**Rejected alternative, and why.** Held-out *training* images would be a larger
pool still, but acquisition sampled from all 28 800 of them, so any of them
could have been opened by some arm — and the set "opened by nobody" is selected
conditional on what the selectors did, which biases it toward images no
selector wanted. Not clean; not used.

### 1.3 Cost

| scope | images | per checkpoint (1 pass) | per checkpoint (2 passes) | **10 checkpoints** |
|---|---:|---:|---:|---:|
| declared-complete | 3 864 | 25.3 min | 50.2 min | **4.2 h** / 8.4 h |
| full test | 4 952 | 32.3 min | 64.3 min | **5.4 h** / 10.7 h |

The second pass is needed **only** for U-Recall split by frequency group.
Aggregate `U_Recall50`, WI and A-OSE come from the metrics file without it, so
the one-pass column is the default. Add ~4 115 test JPEGs to fetch on first run.

**Recommended:** `--scope full`, one pass, **5.4 h**. If time is short,
`--scope declared` at 4.2 h keeps every object of all four focus classes and
loses only images with no declared class at all.

### 1.4 What it produces, and what it may not be used for

Per checkpoint: `known_mAP50`, `prev_mAP50`, `new_mAP50`, `new_class_AP50`,
`U_Recall50`, WI, A-OSE, `mAP50_head/medium/tail`, and per-class AP50 as its own
CSV. Written to a **separate** directory with its own manifest.

> **These numbers are a different measurement and may not be mixed into the
> frozen benchmark table.** A different evaluation set is a different endpoint.
> The manifest says so in its own `note` field. And nothing here may be used to
> redesign a selector — the driver cannot train, and
> `test_the_driver_never_trains` asserts the source contains no `.train(` call.

### 1.5 How to run it

```
python tools/run_large_eval.py --prob-root /content/PROB --data-root /content/data/OWOD --results /content/drive/MyDrive/OWL/results/full_owod_active_benchmark_v1 --out /content/drive/MyDrive/OWL/results/large_eval_v1 --scope full
```

`--plan-only` prints the split, the leakage check, the surviving checkpoints and
the cost, and stops. Run that first.

---

## Experiment 2 — training variance. Designed, NOT implemented, NOT run.

### 2.1 The two things that must not be conflated

**A — exact-repeat nondeterminism.** Same acquired images, same replay ids, same
input checkpoint, same nominal seed, same hyperparameters; train and evaluate
repeatedly. Measures the **irreducible floor**: PROB never calls
`torch.use_deterministic_algorithms`, and MSDeformAttn accumulates with atomics.

**B — training-seed sensitivity.** Same acquired images, same replay, same input
checkpoint; **only PROB's `--seed` changes**. Measures floor *plus* sensitivity
to initialisation and data order.

Note that the running seed-1 replication is **neither**: it varies the candidate
sample, the acquisition, the replay set *and* PROB's seed together. It bounds
total variance, and cannot separate these components.

### 2.2 Which is more valuable before the meeting: **A**

Three reasons.

1. **A is a lower bound, so a positive result settles the question outright.**
   The claim in dispute is whether `random`'s 6.82 against three arms at 0.00,
   or `admissibility`'s ~21 against `random`'s ~0 at t4, are interpretable. If
   *exact repeats alone* span that range, no further work is needed and every
   per-task arm ordering in the benchmark is unreadable.
2. **It is the cheapest thing that can produce that answer.**
3. There is prior reason to expect it is non-trivial: the Method V3 audit
   identified the atomics and the missing determinism flag, and nobody has
   measured the consequence.

**Stated honestly: if A comes back tight, the question is *not* settled.** A
tight floor would leave B unmeasured, and a legitimate re-run would vary the
seed. A is decisive only in the positive direction. B is the follow-up, same
harness, one flag different.

### 2.3 A hard constraint that dictates the target

**Only t4 can be repeated.** An exact repeat needs its input checkpoint, and
t3's input is the **pruned** t2 checkpoint. t4's input is the t3 checkpoint,
which survives. So `random` at t3 — the 6.82 itself — **cannot be exactly
repeated without retraining t2 first**, which would no longer be an exact
repeat.

**Most diagnostic feasible target: `admissibility` at t4.** Its reported
`new_class_AP50` is ~21, the largest per-task new-class value anywhere in the
benchmark, and the contrast it anchors (~21 against `random`'s ~0) is the one
carrying the arm ordering. Repeating a large value is far more informative than
repeating a zero: if a configuration that scored 21 can come out near 0, the
ordering collapses.

### 2.4 The design

| | |
|---|---|
| input checkpoint | `admissibility__seed0/t3_admissibility/checkpoint.pth` |
| training images | `admissibility__seed0/t4_admissibility/train/labelled_ids.txt`, **verbatim** |
| replay ids | `.../train/replay_ids.txt`, **verbatim** |
| everything else | `n_prev=21`, `n_current=1`, 5 epochs, lr 2e-4, batch 2, `ft`, freeze objectness |
| evaluation | the **frozen 837-image split**, so results are comparable to the benchmark |
| repeats | **4** |
| varies | nothing (A). For B, `--seed` only |

**Selection is not rerun.** `owl.bridge.Bridge.train` takes explicit
`labelled_ids` and `replay_ids` lists, so the frozen acquisition ledger is
replayed directly: no population is built, no DINOv2 pass runs, no selector is
constructed. That is the whole point — the acquisition is held *exactly* fixed
so that anything that moves is stochasticity.

**Cost:** train ~25.6 min + evaluate ~5.7 min (837 images, one pass) ≈ **31 min
per repeat**, so **~2.1 GPU-hours for 4**. The existing t4 result is arguably a
fifth sample of the same configuration, which would make it 3 new runs at
**~1.6 h** — with the caveat that it ran in a different session, so it is a
weaker sample.

**Minimum repeats: 3.** Three gives a range and a crude spread; four is the
first number at which the spread estimate is not dominated by a single draw.
Neither supports a significance test and none should be claimed.

**What it outputs:** four values of `new_class_AP50` at t4, their range, and the
same for `known_mAP50`. **The reading is fixed in advance:** if the range of
`new_class_AP50` across exact repeats is comparable to the between-arm
differences the benchmark reports, then those differences are not interpretable
and the report must say so.

---

## Claim hygiene — corrections to the 2026-09-05 forensic report

Two statements were stronger than the evidence and are corrected in
`docs/new_class_instability_forensics_2026-09-05.md`, which now carries an
explicit measured-versus-hypothesised list in its section 9.

**Withdrawn:** *"The selector did its job; the detector couldn't use it"* and
*"Not a supply failure. A learning failure, and the cause is object size."*

**What is measured:** the object-size distributions; the acquired counts; the
AP values; the evaluation support; the structural supply fractions; the −0.500 /
−0.613 correlation at n = 5; zero leakage.

**What is hypothesis:** that small object size, a frozen objectness head and a
five-epoch fine-tune jointly explain t2's zero. Nothing in the analysis varies
size, epochs or the frozen head, so the causal chain is inferred. What the
measurements *do* rule out, and only this, is the **supply** explanation at t2.

**Also downgraded:** "few-shot threshold behaviour" at t3/t4. The bimodality
(≈0 or ≈20, nothing between) is measured; that it reflects a threshold rather
than ordinary run-to-run variance is an interpretation, and experiment 2 is what
would test it.
