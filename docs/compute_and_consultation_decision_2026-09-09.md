# Should you buy 100 Colab compute units? — audit and decision

2026-09-09, ~48 h before the consultation. Read-only audit. No experiment was
launched; no frozen scientific code was modified.

## The answer

> ## BUY 100 COMPUTE UNITS: **NO**
> ## BUY ANY COMPUTE UNITS: **NO**

**One precondition, and it is the only thing that could reverse this.** Every
GPU experiment still worth running is either impossible in 48 h or unable to
change a decision — *provided seed 2 of the benchmark actually completed*. This
machine cannot see it (§1.0). If it did not complete, seed 2 alone is worth
buying compute for; nothing else is. §1.1 gives a 30-second check.

---

## 1.0 What I could and could not verify

`MyDrive/OWL` still exposes only `checkpoints/SOWODB/t1.pth` and `work/`. There
is **no `results/` directory**, no evidence bundle in Downloads or Desktop, and
nothing new in the repository since `e592c1c`. So **every benchmark and
diagnostic number remains user-reported.** Marked **VERIFIED** below only where I
read the file.

### 1.1 The one check that settles the purchase

```python
from google.colab import drive; drive.mount('/content/drive')
!ls /content/drive/MyDrive/OWL/results/full_owod_active_benchmark_v1/ | grep seed2
```

Three `*__seed2` directories with a `per_task_metrics.csv` in each ⇒ the decision
above stands, buy nothing. Nothing ⇒ seed 2 is P0 and §3B applies.

### 1.2 The three-seed means are internally consistent — a real check, not trust

Your reported three-seed means, against the seed-0/seed-1 values reported
earlier in this project, force what seed 2 must have been. That is an
arithmetic implication, **not a file read**, and it is a genuine consistency
test: a misremembered mean would almost certainly not land on a coherent seed 2.

| arm | seed 0 | seed 1 | seed 2 (implied) | mean |
|---|---:|---:|---:|---:|
| `random` | 2.40 | 1.20 | **0.12** | 1.24 |
| `admissibility` | 7.12 | 5.64 | **4.70** | 5.82 |
| `entropy` | 7.31 | 10.06 | **1.86** | 6.41 |

| paired contrast | s0 | s1 | s2 | **range** |
|---|---:|---:|---:|---:|
| `admissibility − random` | +4.72 | +4.44 | +4.58 | **0.28** |
| `entropy − random` | +4.91 | +8.86 | +1.74 | **7.12** |

**This is the most important number in the audit.** Each seed re-randomises the
acquisition *and* PROB's training seed, so the spread of the paired contrast
already contains training nondeterminism. `admissibility − random` varies by
**0.28 points across three seeds** while the arms' own values move by five. That
is a remarkably stable paired effect, and it is *independent evidence* for your
statement that entropy is the more variable arm — 25× the spread.

It also **downgrades the nondeterminism experiment from near-essential to
optional** (§2.2), which is the single biggest change from the 2026-09-07 plan.

---

## 1.3 Inventory of completed evidence

### A) Full sequential benchmark — USER-REPORTED

Three arms × three seeds, T1→T4, one new class per task, budgeted in oracle
answers. Chain, lineage, budget, evaluator and per-arm checkpoint descent are
**VERIFIED in code and tests** (`tests/test_full_benchmark_chain.py`); the
*numbers* are not readable here. Known limitations, all VERIFIED:

* it is a **controlled one-class-per-task chain, not published S-OWODB** — no
  number may be compared to a published S-OWODB result;
* `bear` has **2** test objects and is the only t1 tail class in this grouping,
  so `mAP50_tail` is partly a two-object measurement;
* the banking defect is present (§2.6);
* `proposed`, `proposed_v2`, `cost_aware` and both distribution-aware arms are
  **development-seed-informed, not pre-registered**;
* `coreset` OOM'd with no endpoint and **is not a result**.

### B) Labelling / forgetting — **VERIFIED**

`data/reference/measured/real_group_forgetting.csv`:

| condition | previous-19 mAP50 | forgetting |
|---|---:|---:|
| t1 boxes on the selected image **discarded** | 46.64 | **27.01** |
| t1 boxes **kept**, no replay | 70.96 | **2.69** |
| t1 boxes kept **+ uniform replay** | 70.45 | **3.20** |
| anchor, no training | 73.65 | 0 |

**What this proves.** In this configuration the dominant cause of catastrophic
forgetting was discarding annotation that was already on the image and cost the
annotator nothing. Keeping it is a **10× reduction**, free.

**What it does *not* prove, and you must say so.** It does **not** falsify
Contribution B. It shows that *one* allocation rule — **uniform** — added
nothing *after* labelling was fixed, on **one seed, one task step, in the
predecessor protocol**. A distribution-aware allocation was never tested in that
condition, and the ceiling for any allocation rule here is at most ~2.7 points
of forgetting, which is a statement about **headroom**, not about the idea.

### C) D/R/C — the honest classification

| item | class | evidence |
|---|---|---|
| PROB decoder-feature audit | falsified **proxy** | `data/results/decoder_layer_representation.csv` VERIFIED |
| DINOv2 representation (Stage 1) | falsified **proxy** for object/background; passed for semantics | `docs/method_v2_stage2_protocol_...md` VERIFIED |
| Stage-2 `D` gate | **completed negative** for one region-level ranking (AUC 0.6411 vs 0.65) | protocol VERIFIED |
| Stage-2 `R` gate | **completed negative** for three region-level scores | VERIFIED |
| Stage-2 `C` gate | **passed** (AUC 0.6101), then failed downstream **as a weight on a dense ranking** — an audited near-no-op | VERIFIED |
| DBSCAN coherence gate | **falsified proxy**: discards 92.0% of real unknowns vs 60.2% of background at eps 0.15, purity falls at every eps | `data/results/coherence_gate.csv` VERIFIED |
| Batch diversity (`mu_batch`) | **completed negative** at region level: distinct unknown objects 102 → 71, all three seeds | `data/results/batch_diversity_validation.csv` VERIFIED |
| `proposed` / `proposed_v2` | **falsified concrete methods** | USER-REPORTED |
| `distribution_aware_v1` one-shot | **falsified concrete method** under a pre-registered gate (0.51×, 0.85× of entropy) | USER-REPORTED |
| A1 early-future-class | **arithmetic available, never run** — `tools/close_part_a.py`, one command | VERIFIED (tool) |
| `distribution_aware_iterative_v1` | **incomplete**: seed 0 reported failing, seed 1 interrupted | USER-REPORTED |

**Not falsified, and this is the line to hold:** the *broader idea* that a
distribution-aware acquisition can help a long-tail OWOD learner. What has
failed is a specific family — coverage traversals and one cluster-rarity quota —
under specific representations. The supervisor's own additive score with **both**
`D` terms present has never run downstream at all.

### D) Replay / Contribution B

VERIFIED: the allocator tracks rarity (Spearman −0.995 at t2); object-level
materialisation fixed a 2.67× delivery disparity; the α comparison across three
seeds had between-seed spread exceeding the head-vs-tail difference. Plus §1.3B.
**Enough to discuss B responsibly without running anything** — as an open
question with a *measured headroom bound*, which is a stronger position than an
untested promise.

### E) Other completed work worth showing

`selection_arms.csv` (rounds are exactly flat for static scores; `consult`
+36%/+7%/+8%; `prior_consult_batch` worse on all three seeds);
`novelty_definitions.csv` (novelty vs the *growing labelled set* has
discrimination ratio 6.41 against 0.080/0.192 for anchor forms);
`clustering_contamination.csv` (the label-free contamination estimate is biased
low ~4×); the annotation-cost design itself. All VERIFIED.

---

## 2. What is missing, ruthlessly classified

| # | Experiment | GPU h | P(done in 48 h) | Priority | Verdict |
|---|---|---:|---:|---|---|
| 2.1 | iterative seed 1 | ~1 | 0.9 | **P2** | cannot change GO/NO-GO |
| 2.2 | nondeterminism floor | ~3 | 0.8 | **P2** | largely answered by §1.2 |
| 2.3 | Contribution B on the chain | ≥9 | 0.2 | **P3** | not mature, headroom small |
| 2.4 | `known_plus_selected` | ~5 + dev | 0.3 | **P3 now, P0 next** | needs per-box XML that does not exist |
| 2.5 | large evaluation | ~1 | 0.9 | **P3** | measured unable to help |
| 2.6 | banking Protocol V2 | ~18 | 0.05 | **P3** | invalidates everything |
| 2.7 | any new D/R/C variant | — | — | **P3** | result-chasing; forbidden |
| **2.0** | **extract the Drive evidence + build figures** | **0** | 1.0 | **P0** | the only P0, and it is CPU |

**2.1 — does seed 1 change the decision?** No. The gate is per-seed and
AND-required; seed 0 fails `tail_per_image_vs_entropy` at 0.51×. Seed 1 cannot
rescue it and nobody may suggest it might. Value is *completeness*: it turns a
single-seed negative into a replicated one. Worth ~1 free GPU-hour; **not worth
money.**

**2.2 — is the floor essential before presenting three seeds?** It was the plan
on 09-07. §1.2 changes that: the paired contrast's three-seed range is **0.28**,
and that range already contains training noise because each seed moves both. The
smallest useful version, if a free T4 appears, is **one arm, one task, three
repeats of a fixed acquisition** (~3 h) — it would bound the floor directly. But
you can defend the result without it by reporting the paired spread.

**2.3 — Contribution B.** No. Not enough time, and more importantly not enough
methodological maturity: the allocation rule, the memory size and the per-task
reallocation are three variables the consultation notes name as *an experiment
series*. Propose it; do not start it 48 h out.

**2.4 — `known_plus_selected`.** Highest scientific value of anything unrun, and
it is the supervisor's own flagged question. It is **not ready**: it needs
per-box filtered XML with unlabelled unknowns as *ignore*, which the repository
explicitly lists as "the next step". Rushing it would introduce exactly the kind
of confound this project has spent weeks removing. **Propose it as the next
experiment — that is a strength, not an omission.**

**2.5 — large evaluation.** Already measured unable to help: the frozen split
**already contains every test image holding `fire hydrant` or `stop sign`**
(1.0×). It would raise confidence on `bear` (35.5×) — a two-object class whose
metric you should be de-emphasising, not investing in.

**2.6 — banking V2.** The defect lowers *everyone's* future-supervision ceiling
and t2's class is immune, so it cannot plausibly have created the arm ordering.
**Documenting the limitation is sufficient**; a V2 rerun invalidates 9+
completed trajectories and cannot finish.

**2.7 — a new D/R/C variant.** No. Three pre-registered failures; a fourth built
now would be post-hoc by construction.

---

## 3. The three scenarios

**SCENARIO A — buy 0.** You present: the forgetting decomposition; the
cost-is-not-supervision accounting; a three-seed benchmark with a paired
contrast whose range is 0.28; the selection→learning transfer failure; four
carefully pre-registered negatives; and a precise next experiment. **This is a
strong package.** The one thing you cannot say is "we measured the training
noise floor directly" — and §1.2 lets you address that with the paired spread
instead.

**SCENARIO B — the minimum useful purchase.** ~1 GPU-hour, for iterative seed 1.
That is below the granularity at which buying compute makes sense, and Colab's
free tier can usually supply it. **Effective recommendation: buy nothing; run it
free if a T4 appears.** If §1.1 shows seed 2 missing, B becomes ~7 GPU-hours for
seed 2 and is then worth buying.

**SCENARIO C — buy 100.** There is nothing to spend them on. The list would be
iterative seed 1 (~1 h) and the floor (~3 h) — perhaps 4 GPU-hours of genuinely
optional work against a purchase sized for far more. In compute units the
conversion is not fixed and varies by accelerator and session, so I will not
invent one; reasoning in GPU-hours, **the large majority would go unused**, and
neither experiment changes a conclusion.

---

## 4. The story, with every claim labelled

1. **The annotation protocol is itself a major experimental variable** —
   discarding free on-image annotation costs 24 points of forgetting.
   **STRONG** (verified GPU measurement, though one seed / one step / predecessor
   protocol — say that).
2. **We built a true sequential OWOD active-learning benchmark** with per-arm
   checkpoint descent, a shared frozen evaluation split, and a budget in oracle
   answers so every arm pays the same annotator. **STRONG** for the design
   (asserted by tests); **MODERATE** for the numbers until §1.1 runs.
3. **Simple acquisition baselines beat random, and it replicated.**
   `admissibility − random` positive in 3/3 seeds. **MODERATE** — a 3/3 sign
   agreement, not significance.
4. **Admissibility is the more reliable of the two**, with a paired-contrast
   range of 0.28 against entropy's 7.12. **MODERATE**, and the claim about
   entropy costing more final known mAP is **NOT YET SUPPORTED** here — I could
   not read those columns.
5. **Unknown-discovery does not automatically become new-class AP** — 53 objects
   found vs random's 5, and new-class AP ≈ 0 either way. **STRONG** as a
   demonstrated dissociation; **DESCRIPTIVE** as to why.
6. **We tested concrete implementations of D/R/C rather than assuming them** —
   and reported each proxy's failure at proxy scope. **STRONG**.
7. **The current distribution-aware formulation fails its pre-registered gate.**
   **STRONG**, and its provenance (development-seed-informed) is stated.
8. **That narrows the next hypothesis rather than killing the direction.**
   **MODERATE** — defensible precisely because the untested forms are named.
9. **Contribution B is a separate open question with a measured headroom bound.**
   **MODERATE**.

---

## 5. Figures — all six generated without a GPU

`python tools/build_consultation_figures.py --out docs/figures`

Each prints its source file, columns and the caveat that belongs in the caption.
`fig7` (three-seed new AP) is **declared and skipped** unless you pass
`--benchmark <csv>`; it is never drawn from remembered numbers.

| fig | shows | source |
|---|---|---|
| 1 | forgetting 27.01 → 2.69 → 3.20 | `real_group_forgetting.csv` |
| 2 | cost is not supervision | `labelling_policy.csv` |
| 3 | acquisition vs learning | `real_group_forgetting.csv` |
| 4 | coherence gate rejects unknowns harder | `coherence_gate.csv` |
| 5 | batch diversity finds fewer objects | `batch_diversity_validation.csv` |
| 6 | rounds are a no-op / unreliable | `selection_arms.csv` |
| 7 | three-seed new AP **(needs Drive)** | benchmark per-task CSV |

---

## 6. Statistical language you can safely use

**Say:** "The direction replicated in three of three paired seeds." · "The
paired difference varied by 0.28 points across seeds, where the arms' own means
moved by five — so the effect is stable relative to everything the seed
changes." · "Each seed changes the acquisition and the detector's training seed
together, so the seed-to-seed spread already contains training noise." · "This
is a controlled one-class-per-task chain, not the published S-OWODB split."

**A sign test:** you may mention it — 3/3 one-sided is **p = 0.125** — but say in
the same breath that with n = 3 it cannot reach conventional significance, so it
is a consistency statement, not evidence of significance.

**Never say:** "statistically significant" · "we proved" · "entropy is best" ·
"tail performance improved" without noting `bear` has 2 test objects ·
"distribution-aware active learning failed" · "coreset showed the gate matters".

**Disclose unprompted:** the budget equalises oracle answers, not gradient steps
or supervised boxes; the banking defect; the development-seed-informed
provenance of four arms; and that `mAP50_tail` rests on a two-object class.

---

## 7. The 48 hours

**Today, 30 min — the only P0.** Run §1.1's check, then the extractor from
`docs/three_day_consultation_plan_2026-09-07.md` §11 and
`tools/close_part_a.py`. Until this runs, half the package is unverified.

**Today, 2 h.** Drop the extracted CSV in and run
`tools/build_consultation_figures.py --benchmark <csv>` for all seven figures.
Read every caveat line; they are the captions.

**Today evening, 1 h.** *Only if a free T4 appears*: iterative seed 1. Do not
buy compute for it, and do not wait up for it.

**Tomorrow morning, 3 h.** Verify the three-seed table against §1.2's implied
seed 2. If it disagrees, trust the file and tell me.

**Tomorrow midday, 2 h.** Write the nine-point narrative with the labels from §4.

**Tomorrow afternoon, 2 h.** Rehearse — *out loud* — the four negatives and the
three "what this does not prove" statements. Those are where a supervisor pushes.

**Tomorrow evening.** Stop. Do not watch Colab.
