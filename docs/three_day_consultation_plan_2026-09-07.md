# Three days to the consultation — audit and resource decision

2026-09-07. Read-only audit. Nothing was trained, tuned or launched to write this.
Every number below is marked **VERIFIED** (read from a committed artefact or from
source on this machine) or **USER-REPORTED** (told to me, not yet checkable here).

**What this machine cannot see.** `MyDrive/OWL` exposes only `checkpoints/SOWODB/t1.pth`
and `work/`; there is **no `results/` directory at all**. So the full benchmark's
seed-0 and seed-1 numbers, and both distribution-aware diagnostics, are
**unverifiable from here**. Three historical six-task trajectories
(`work/{random,objectness,prior_consult_batch}/t2..t6`) exist but are cloud-only
placeholders that time out on read. §11 gives a one-command extractor.

---

## 1. Executive decision

* **Run seed 2 of `random` / `admissibility` / `entropy` tonight.** ~6.7 h, zero
  implementation risk, notebook already committed and pinned. It is the only
  experiment that turns our one *positive* pattern into a three-seed replication.
* **Resume the iterative diagnostic's seed 1 tomorrow — ~1 h.** It cannot change
  the NO-GO (the gate is per-seed and seed 0 fails at 0.51×), but it converts a
  single-seed negative into a replicated one for ~1 h of already-cached compute.
* **Do NOT start Contribution B (distribution-aware replay).** Its headroom has
  already been measured, and it is small: once the selected image's *existing*
  task-1 boxes are kept, forgetting is **2.69**, and adding uniform replay makes
  it **3.20** — replay bought nothing. Spending 72 h on allocation rules on top
  of that is optimising a term worth well under one AP point here.
* **Do NOT extend to T1→T10.** A four-task chain with three arms × three seeds is
  more defensible than a rushed ten-task chain with one seed, and the supervisor's
  per-task question list is already answerable at t2–t4.
* **Do NOT implement banking Protocol V2.** It invalidates every completed
  trajectory (9+), and re-running the minimum fair set is ~18 T4-hours we do not
  have. Declare the limitation instead.
* **Do NOT run the large evaluation split.** It was already measured to be unable
  to help the classes that need it: the frozen split **already contains every
  test image holding `fire hydrant` or `stop sign`** (1.0×).
* **Do NOT build another D/R/C selector, re-choose the clusterer, or sweep α.**
  Three formulations have failed under pre-registered gates. A fourth built now
  would be post-hoc by construction.
* **The strongest thing we can show is not the proposed method.** It is the
  labelling-policy result: **keeping the free task-1 annotations already present
  on a selected image cuts forgetting 27.01 → 2.69, a 10× reduction, at zero
  annotation cost.** That is a direct answer to the question the supervisor
  himself flagged as the one that distorts everything else.
* **Two claims I have made in committed memos are wrong and are corrected in §5.**
  The "+38 % from rounds" figure is one seed of three (+36 %, +7 %, +8 %), and for
  the arm combining a prior with batch diversity rounds are **consistently worse**.
* **Reserve the whole final day for figures and talking points.** The biggest risk
  to this meeting is not missing results; it is that the results we have are
  scattered across an unreadable Drive.

---

## 2. What the supervisor asked, and where we stand

Source of truth: `~/Desktop/konzultacio_2026-08-25_uj_iranyok.md` and
`~/Desktop/Aug 25 kutatás konzultáció ötletek.docx` (the raw notes). The score
under discussion was `s(x) = U(x) + λ·D(x) + γ·w(ĉ(x))·coh(x)`.

| # | Requested | Implemented | Experiment | Result | Evidence | Status |
|---|---|---|---|---|---|---|
| 1a | `D` = novelty vs the **growing labelled** set, not a fixed anchor | `scoring.novelty`, `diversity_source='labelled'` | region-level definition audit, 3 seeds | `nearest_labelled` discrimination ratio **6.41** vs `nearest_known_prototype` **0.080** and `nearest_known_cluster` **0.192**; only `nearest_labelled` drifts as the pool grows (0.036) | `data/results/novelty_definitions.csv` **VERIFIED** | **PARTIALLY CLOSED** — the definition change is measured and clearly right at region level; never run downstream |
| 1b | `D` = **batch diversity** among the selected, k-means++-like, updated during selection | `selection._greedy`, `mu_batch` | `mu_batch` 0 vs 0.3, 3 seeds | pairwise similarity 0.900 → 0.876 (batch is genuinely less redundant) but **distinct unknown objects 102 → 71 (−30 %)**, proposals per distinct object 1.17 → 1.79, tail objects 41 → 30. Same direction on all three seeds | `data/results/batch_diversity_validation.csv` **VERIFIED** | **PARTIALLY CLOSED** — measured negative for this proxy at region level; never run downstream |
| 2 | `coh ∈ {0,1}`, DBSCAN noise → 0 | `clustering.noise_gate`, `scoring.coherence('binary')` | eps sweep on PROB features | the gate discards **92.0 %** of real unknown objects against **60.2 %** of background at eps 0.15; unknown purity **falls** at every eps (0.0345 → 0.0084 / 0.0134 / 0.0281), and at eps 0.45 it is inert (2 clusters) | `data/results/coherence_gate.csv` **VERIFIED** | **CLOSED for PROB features.** Open for other representations |
| 3 | One clustering over known+candidates giving both `D` and `w`, minimising **known contamination** | `clustering.fit`, `contamination()` | K sweep, 3 seeds | at K=1600 verified contamination **0.116**, unknown recall **0.816**; the label-free estimate (0.028) **underestimates the verified value ~4×** | `data/results/clustering_contamination.csv` **VERIFIED** | **PARTIALLY CLOSED** — the diagnostic exists and works; it was never used to *choose* the clustering the benchmark ran |
| 4 | Replay: distribution-aware, per-task reallocated, memory size × rule × static/dynamic | `replay.allocate`, `exemplars.select`, protocol V3 | old GPU chain; α allocation audit | **the decisive number: forgetting 27.01 (drop the image's t1 boxes) → 2.69 (keep them, no replay) → 3.20 (keep them + uniform replay).** Allocator verified to follow rarity (Spearman −0.995 at t2) | `data/reference/measured/real_group_forgetting.csv` **VERIFIED**; `docs/eredmenyek_vazlat.md` | **PARTIALLY CLOSED** — and the result argues replay is *not* the bottleneck once labelling is fixed |
| 5 | **Image vs region labelling — flagged by the supervisor as the question that distorts every other measurement** | `labelling.py`, three rules | policy audit, 306 images / 600 regions | `box_only`: 423 labelled, **20.7 % of its background sits on a real annotated object**, 0.705 supervision per oracle unit. `full_image`: 3 498 labelled, **1.80× the cost**, 3.23/unit. **`known_plus_selected`: 2 729 labelled, 769 ignored, cost 1.00×, 4.55/unit** | `data/results/labelling_policy.csv` **VERIFIED** | **PARTIALLY CLOSED** — the accounting is settled; the supervisor's own middle option has **never reached the detector** |
| 6 | Real sequential chain, T1 → … ≈ T10, with per-task U-Recall / forgetting / new AP / head-medium-tail / selection composition, plus baselines | `owl/runner.run_chain`, Benchmark V1 | T1→**T4**, 3 baselines + 3 proposed arms | numbers **USER-REPORTED only** (see §3); chain, lineage, budget and evaluator are **VERIFIED** in code and tests | `docs/full_owod_active_benchmark_v1_protocol_2026-09-03.md`; `tests/test_full_benchmark_chain.py` | **PARTIALLY CLOSED** — 4 of ~10 tasks, and it is a controlled one-class-per-task chain, **not** published S-OWODB |
| 7 | One-shot vs rounds (600×1 · 6×100 · 12×50) | `selection.select(rounds=)`; `budget.spend_ranking_in_rounds` | region-level, 3 seeds; then a downstream-shaped diagnostic at 6×500 | `entropy`/`objectness`/`plan` **exactly flat** — rounds are provably a no-op for a static score. `consult` **+36 % / +7 % / +8 %**. `prior_consult_batch` **−10 % / −10 % / −20 %**, worse on every seed | `data/results/selection_arms.csv` **VERIFIED** | **PARTIALLY CLOSED** — the region-level answer exists and is weak-to-negative; the sequential version is the interrupted diagnostic |

**Two requests never tested downstream at all**: the supervisor's literal additive
score with **both** `D` terms present and separately measured (1a + 1b), and the
`known_plus_selected` labelling rule (5). Everything downstream so far has used
`full_image` and no `D` term.

---

## 3. Research state

**VERIFIED complete (A — completed scientific result).** The labelling-policy
accounting; the forgetting decomposition above; the coherence-gate eps sweep; the
novelty-definition audit; the batch-diversity audit; the clustering-contamination
audit; the rounds audit. All CPU or old-GPU, all committed, all reproducible here.

**USER-REPORTED, not verifiable from this machine (must be re-extracted before the
meeting).** Benchmark seed 0 and seed 1 for `random`/`admissibility`/`entropy`;
`proposed` and `proposed_v2` seed 0; both distribution-aware diagnostics. The
recollections in the brief — random 2.40/1.20, admissibility 7.12/5.64, entropy
7.31/10.06 mean new-class AP50 — are **consistent with** the frozen gate values
also reported, but I have not read the files.

**Incomplete (E).** Iterative diagnostic seed 1 (interrupted during the t4 DINOv2
embed; caches intact, resume verified byte-identical in tests). Benchmark seed 2
(prepared, pinned, never launched).

**Failed for engineering reasons, never a result (F).** `coreset` seed 0 — CUDA
OOM, no endpoint. It must never appear in a table.

**Diagnostic only (B), and it does not become preregistered evidence
retroactively (C).** Every candidate-side number in §2, the ceiling audit, the
banking forensic, the new-class instability forensic.

**Superseded / not comparable (G).** The pre-benchmark `owod-longtail` single-step
runs and the `work/*` six-task chains use a different protocol, budget unit and
evaluator. They are *evidence for their own questions* (forgetting, labelling) and
must not be placed in a table beside Benchmark V1.

---

## 4. Strongest existing findings, with provenance

Ranked by (importance × supervisor relevance × reliability × presentability).

**1. Discarding the free annotations already on a selected image is the dominant
cause of catastrophic forgetting here — not the absence of replay.**
Previous-19 mAP50 / forgetting: drop them **46.64 / 27.01** → keep them, no replay
**70.96 / 2.69** → keep them + uniform replay **70.45 / 3.20**. Anchor 73.65.
*VERIFIED, `data/reference/measured/real_group_forgetting.csv`. One seed, one task
step, predecessor protocol.* This answers supervisor point 5 on the detector and
reframes point 4.

**2. Equal annotation cost does not mean equal supervision, and the middle policy
the supervisor proposed is the efficient one.** At 600 oracle units:
`box_only` 0.705 supervision/unit with **20.7 %** of its background on real
objects; `known_plus_selected` **4.55**/unit at the **same** cost with zero
half-labelling; `full_image` 3.23/unit at **1.80×** cost.
*VERIFIED, `data/results/labelling_policy.csv`.*

**3. Selection quality does not transfer to new-class AP.** Distribution-aware
selection found **53** unknown objects to random's **5** at 600 regions (10.6×) and
still produced new-class AP50 of 0.0010 vs 0.0000.
*VERIFIED, same CSV.* Reproduced independently in the current benchmark, where the
two coverage arms had the **highest** U-Recall and the **lowest** new-class AP
(USER-REPORTED).

**4. The DBSCAN coherence gate, on PROB features, rejects real unknowns harder
than background at every threshold tested** — 92.0 % vs 60.2 % at eps 0.15 — so
unknown purity falls whichever eps is chosen.
*VERIFIED, `data/results/coherence_gate.csv`.*

**5. Batch diversity reduces distinct-object discovery.** `mu_batch` 0 → 0.3 moves
distinct unknown objects 102 → 71 (−30 %) and proposals-per-object 1.17 → 1.79, on
all three seeds, while the batch does become less redundant (0.900 → 0.876).
*VERIFIED, `data/results/batch_diversity_validation.csv`.*

**6. Iterative rounds are a no-op for static scores and weak-to-negative for the
rest.** Exactly flat for `entropy`, `objectness`, `plan`; +36 %/+7 %/+8 % for
`consult`; **−10 %/−10 %/−20 %** for `prior_consult_batch`.
*VERIFIED, `data/results/selection_arms.csv`.*

**7. Novelty relative to the growing labelled set is the right definition, and it
is measurably different from the anchor forms** — discrimination ratio 6.41 vs
0.080 / 0.192, and it is the only one that moves as the pool grows.
*VERIFIED, `data/results/novelty_definitions.csv`.*

**8. Cluster-size rarity cannot be read off a label-free contamination estimate.**
The estimate is 0.028 where the verified contamination is 0.116 — biased low ~4×.
*VERIFIED, `data/results/clustering_contamination.csv`.*

**9. The annotation-cost design works.** Budgeting in oracle answers under
full-image labelling matched `boxes_labelled` across arms to ~6 %, against the
predecessor's 2.09× disparity. *Design VERIFIED in code and tests; the 6 % figure
is USER-REPORTED.*

**10. Tail supply is structurally tiny, and the evaluation cannot be enlarged.**
`fire hydrant` and `stop sign` are ~0.9 % of declared boxes; the frozen eval split
already holds **every** test image containing them.
*VERIFIED in `docs/new_class_instability_forensics_2026-09-05.md` and the
committed index.*

---

## 5. What we must not say — and two corrections to my own earlier claims

**CORRECTION 1.** `docs/iterative_decision_memo_2026-09-06.md` §3 justified the
iterative experiment partly with "rounds help exactly the arms with something to
update (`consult` 26 → 36, **+38 %**)". The committed table says 25 → 34, 30 → 32,
37 → 40 across seeds 0/1/2 — **+36 %, +7 %, +8 %** — and `prior_consult_batch`
gets *worse* with rounds on all three seeds. The prior that rounds help is much
weaker than I stated, and it strengthens the adverse prediction in that memo's
§3.1 rather than the case for running it.

**CORRECTION 2.** That memo's §2.2 said per-class early purchase was not
recoverable from `diagnostic_rows.csv`. It is exactly recoverable; already
corrected in the memo's §8 and implemented in `tools/close_part_a.py`.

| Claim | Verdict | Why |
|---|---|---|
| "semantic diversity does not work" | **UNSUPPORTED** | two coverage traversals and one cluster-quota allocator failed. The additive score with `D` as a *term*, and batch diversity downstream, were never run |
| "`D` is falsified" | **TOO STRONG** | `D_NO_GO` was one region-level ranking, one reference, one AUC threshold (0.6411 vs 0.65) |
| "`C` is falsified" | **TOO STRONG** | verified false for *DBSCAN density on PROB features*. View-consistency `C` passed its own gate (AUC 0.6101) and then failed downstream **as a weight on a dense ranking**, which the audit showed is a near-no-op |
| "distribution-aware active learning failed" | **UNSUPPORTED** | three implementations of one family failed. Finding 1 above is itself a distribution-relevant positive |
| "entropy is definitively best" | **TOO STRONG** | two seeds, and the whole arm ordering rests on t3/t4 where the noise floor is **unmeasured** |
| "tail-favouring replay is worse" | **DESCRIPTIVE ONLY** | three seeds where the between-seed spread exceeded the head-vs-tail difference |
| "small object size causes the t2 failure" | **TOO STRONG** | `traffic light` median 282 px² and 78.5 % COCO-small are measured; the causal link is a hypothesis, and nothing varied size |
| "this is S-OWODB" | **FORBIDDEN** | one class per task, controlled chain. No number may be compared to a published S-OWODB result |
| "we proved X statistically" | **FORBIDDEN** | at three seeds the strongest honest statement is a sign test, 3/3, p = 0.125 one-sided |
| "`coreset` shows the gate matters" | **FORBIDDEN** | CUDA OOM, no endpoint. Not a result |
| "U-Recall does not matter" | **TOO STRONG** | the *negative association* with new-class AP is descriptive at n=5 arms |

Standing caveats to state out loud: one or two seeds; PROB never calls
`torch.use_deterministic_algorithms` and MSDeformAttn accumulates with atomics, so
the noise floor is unmeasured; `bear` has **2** test objects and is the only t1
tail class in this grouping, so `mAP50_tail` is partly a two-object measurement;
`proposed_v2`, `cost_aware`, `distribution_aware_v1` and the iterative arm are all
**development-seed-informed, not pre-registered**.

---

## 6. Candidate experiment ranking

Scored on relevance / finishes in 72 h / interpretable / new information /
seed-protection / engineering risk / confound risk / p-hacking risk / useful if
negative / GPU cost.

| # | Experiment | GPU | Verdict |
|---|---|---:|---|
| **1** | **Benchmark seed 2, `random`/`admissibility`/`entropy`** | **6.7 h** | **RUN.** Already pinned and tested; third paired seed on our only positive pattern; useful whichever way it goes — if the sign breaks, that is the most important result of the week |
| **2** | **Resume iterative diagnostic seed 1** | **~1 h** | **RUN.** Cannot change the NO-GO but makes it a *replicated* negative; caches intact, resume proven byte-identical |
| **3** | Nondeterminism floor (one fixed acquisition, retrained 3–4×) | ~3 h | **RUN IF NIGHT 2 IS FREE.** It is the only thing that makes the t3/t4 ordering interpretable — but it produces a caveat, not a result, and seed 2 partially subsumes it |
| 4 | `known_plus_selected` vs `full_image` on the chain | ~5 h + real dev | **DO NOT START NOW.** Highest supervisor relevance of any *new* experiment, but it needs per-box filtered XML that does not exist yet. **Propose it as the next step** |
| 5 | Distribution-aware replay (Contribution B) | ~9 h+ | **DO NOT RUN.** Headroom measured at well under 1 AP point once labelling is fixed (2.69 vs 3.20) |
| 6 | Large evaluation split | ~1 h | **DO NOT RUN.** Already measured as unable to help the classes that need it (1.0× for both tail classes) |
| 7 | Banking Protocol V2 | ~18 h | **DO NOT RUN.** Invalidates every completed trajectory |
| 8 | T1→T10 extension | ~15 h+ | **DO NOT RUN.** Cannot finish; would replace a replicated 4-task result with a single-seed 10-task one |
| 9 | α sweep / memory-size ablation | ~6 h | **DO NOT RUN.** Between-seed spread already exceeded the effect |
| 10 | Another D/R/C selector, or re-choosing the clusterer | any | **FORBIDDEN.** Post-hoc by construction after three pre-registered failures |

---

## 7. The chosen 72-hour plan

**STRATEGY C — result-maximisation.** Chosen over *A (replication-first + replay)*
because replay's headroom is already measured small, and over *B
(supervisor-coverage-first)* because the labelling arm cannot be implemented,
validated and run safely in the time left — and a rushed version of the
supervisor's most important question is worse than a clean proposal to run it.

| Block | Wall clock | Action | Who |
|---|---|---|---|
| Tonight | 18:00–18:30 | Extract the Drive evidence (§11 command). **This is the single highest-value 30 minutes in the plan** — until it runs, nothing in §3 is verified | you |
| Tonight | 18:30–19:00 | Launch **seed 2** from the committed notebook. Fresh T4, Run all, walk away | you |
| Overnight 1 | ~6.7 h | seed 2 runs | GPU |
| Day 2 morning | 2 h | Verify seed 2 landed; compute the three-seed paired table; **sign test only**, no averaging across a disagreement | me |
| Day 2 midday | ~1 h | **Resume the iterative diagnostic** (seed 1). Same notebook, Run all — it reuses every cache | you |
| Day 2 afternoon | 3 h | Build the 6 figures in §10 from verified artefacts | me |
| Overnight 2 | ~3 h | **Nondeterminism floor**, only if seed 2 completed cleanly and the figures are drafted. If either slipped, skip it | GPU |
| Day 3 morning | 3 h | Final tables; write the §9 talking track; rehearse the negative results out loud | you + me |
| Day 3 afternoon | — | **Reserve. Do not start anything.** | — |

Total GPU: **~10.7 h** across two nights, of which 7.7 h is committed and 3 h is
conditional.

---

## 8. Stop / go rules, fixed before running

**Seed 2** — *confirmatory*. Endpoint: mean `new_class_AP50` per arm, and the sign
of `admissibility − random` and `entropy − random`. Stop when the chain completes
or the session budget expires; a partial seed 2 is reported as partial, never as a
third seed. **Negative outcome is informative**: if either sign flips, we say the
two-seed pattern did not replicate and the benchmark's arm ordering is not
established — and that is a genuine finding, not a failure.

**Iterative seed 1** — *confirmatory of a NO-GO already declared*. Endpoint: the
six frozen gates, unchanged. Seed 0 fails `tail_per_image_vs_entropy` at 0.51×;
the gate is per-seed and AND-required, so **seed 1 cannot rescue it and no one may
suggest otherwise.** We run it for replication, and we stop there: if it also
fails, this D/R/C formulation is closed and no v4 follows.

**Nondeterminism floor** — *exploratory*. Endpoint: the spread of
`new_class_AP50` at t4 across 3–4 identical retrainings. It answers "is a 20-point
t4 gap interpretable?" and **does not** answer which arm is better. Stop after 3
repeats regardless of what they show.

---

## 9. The consultation story

**What we discussed two weeks ago.** Four changes to the score — a real `D`, a
binary DBSCAN `coh`, one clustering giving both `D` and rarity, and entropy kept —
plus three protocol questions: replay, image-versus-region labelling, and one-shot
versus rounds. You flagged labelling as the one that distorts everything else.

**What I built.** A full sequential benchmark, T1→T4, one new class per task,
budgeted in **oracle answers** under full-image labelling so every arm pays the
same annotator; per-arm checkpoint lineage; a shared frozen evaluation split; and
three pre-registered baselines plus three proposed selectors, each frozen behind a
decision rule written before it ran.

**What worked — and this is the result I would lead with.** The labelling question
you flagged turned out to be the one with the largest measured effect anywhere in
the project. When a selected image's *already existing* task-1 annotations are
discarded, forgetting is **27.0** points. Keep them — which costs the annotator
nothing, because the detector already knows those classes — and forgetting is
**2.7**. Adding uniform replay on top changed it to 3.2, i.e. nothing. So in this
configuration the dominant cause of catastrophic forgetting was *throwing away
free supervision*, not the absence of a memory. And the middle policy you
suggested — label the knowns, ignore the unselected unknowns — is the efficient
one: same oracle cost as box-only, zero half-labelling, **6.5× the supervision**.

**What failed, under rules fixed in advance.** Three distribution-aware selectors.
Two coverage traversals produced the *highest* unknown-recall and the *lowest*
new-class AP of any arm. A cluster-rarity quota allocator — your "one clustering,
both terms" idea, with HDBSCAN noise as the binary gate — bought genuinely
different images (Jaccard 0.09 against entropy) and beat a cost-matched control by
**5×** on tail supply per image, yet reached only **0.51×** of entropy on the same
endpoint and failed its gate. Making it iterative changed that number by under
half a percent.

**What I learned.** Two things, and they are the transferable part. First, good
acquisition does not imply good learning: one arm found 10× more unknown objects
than random and still learned nothing new. Second, the binary DBSCAN gate on the
detector's own features rejects real unknown objects *harder than background* —
92 % versus 60 % — because in a pool that is three-quarters background, "you have
many neighbours" means "you look like background".

**What remains open.** Your labelling policy has never reached the detector inside
the sequential chain — only its accounting has. The additive score with *both*
diversity terms present has never been run downstream at all. And the training
noise floor at t3/t4 is unmeasured, so I will not claim the arm ordering is
established.

**What I propose next.** Run the labelling policy on the chain as a first-class
arm; that is the highest-value experiment we have and it is your question. Then
revisit replay, with the expectation — already measured — that its headroom is
small once labelling is fixed.

---

## 10. Figures

| # | Title | x | y | Arms / seeds | Message | Source |
|---|---|---|---|---|---|---|
| 1 | Where forgetting actually comes from | policy (drop t1 boxes / keep / keep+replay) | previous-19 mAP50 and forgetting | 1 arm, 1 seed | 27.0 → 2.7 → 3.2. The headline | `real_group_forgetting.csv` |
| 2 | Cost is not supervision | labelling rule | supervision per oracle unit; half-labelled share | 3 rules, 1 seed | `known_plus_selected` is free and clean | `labelling_policy.csv` |
| 3 | New-class AP per task, three seeds | task t2/t3/t4 | `new_class_AP50` | random/admissibility/entropy × seeds 0,1,2 | replication, per seed, no averaging | Drive `per_task_metrics.csv` |
| 4 | Acquisition does not transfer | distinct unknown objects acquired | new-class AP50 | every arm we have | the central negative | `real_group_forgetting.csv` + benchmark |
| 5 | The coherence gate rejects the wrong things | DBSCAN eps | noise share, background vs real unknown | — | 92 % vs 60 % | `coherence_gate.csv` |
| 6 | The frozen gates and what the allocator did | gate | measured value vs threshold | one-shot and iterative | an honest pre-registered NO-GO | Drive `verdict.json` ×2 |

Optional 7th if the floor runs: t4 `new_class_AP50` across identical retrainings,
with the arm spread drawn as a band behind it.

---

## 11. The one thing to do immediately

Open a fresh T4 Colab and run **this single cell** before anything else. It costs
no GPU and turns §3's unverifiable half into evidence:

```python
from google.colab import drive; drive.mount('/content/drive')
!cd /content && rm -rf owod-active && git clone -q https://github.com/gubiczam/owod-active
!cd /content/owod-active && python tools/close_part_a.py \
    /content/drive/MyDrive/OWL/results/distribution_aware_diagnostic/diagnostic_rows.csv
!cd /content/drive/MyDrive/OWL/results && \
  tar -czf /content/drive/MyDrive/OWL/evidence_2026-09-07.tar.gz \
      --exclude='*.pth' --exclude='*.npz' --exclude='*.jpg' . && \
  echo OK && ls -la /content/drive/MyDrive/OWL/evidence_2026-09-07.tar.gz
```

Then download `evidence_2026-09-07.tar.gz` and drop it in the repo root, and every
number in §3 becomes checkable. **After that, launch seed 2.**
