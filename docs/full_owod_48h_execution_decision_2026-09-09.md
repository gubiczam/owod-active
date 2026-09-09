# Full T1→T10 before the consultation — execution decision

2026-09-09. Read-only audit plus additive engineering. **No experiment was
launched.** Every number below was re-derived against the implementation on
2026-09-09 and is reproducible with the commands in §9.

> ## FULL T1→T10 BEFORE CONSULTATION: **YES**
> **arms `random` + `admissibility`, seed 0, 9 incremental tasks. ~15–18 T4-hours.**

This reverses the 2026-09-09 morning recommendation to buy no compute. The
reversal is caused by new information — the supervisor's standing requirement for
a complete trajectory, and a scientific argument (§4) that only a longer chain
can settle — not by convenience. The earlier decision was correct *given the goal
it assumed*.

---

## 1. Does the runner support a correct T1→T10 run?

**Yes, verified by dry-running the real launcher**, not by reading the code:
`random` and `admissibility` each reached **9 of 9 tasks**, writing checkpoints
`t2 … t10` with per-arm lineage.

| requirement | status |
|---|---|
| true checkpoint lineage | **yes** — each task fine-tunes the previous task's own checkpoint; `owl/runner.py` fails closed if the expected parent is missing |
| task-specific known/new classes | **yes** — `protocol.build_chain` has always defaulted to 10 and supports 62 |
| replay | **yes** — `uniform`, M = 400 objects, at every task |
| evaluation after every task | **yes** — 9 evaluations on one shared split |
| forgetting · U-Recall · WI · A-OSE | **yes** — PROB's own evaluator; all present in `metrics.json` |
| head/medium/tail | **yes** — the tail band grows from 3 classes at t4 to 4 at t10 |
| annotation/supervision accounting | **yes** — the answer ledger, `boxes_trained_on`, `training_iterations` per task |
| resume | **yes** — finished trajectories skip, an interrupted one restarts at its task |

## 2. Was anything hard-coded to T4?

**No pervasive hardcoding. Exactly one constant**: `benchmark.N_TASKS = 4`.
Everything else derives from `bm.chain()`. But that constant is *load-bearing in
two ways*: `check_protocol()` pins it against the protocol document, and
`declared_classes()` derives from it — which determines the shared evaluation
split. So it could not simply be edited.

## 3. What changed (additive only)

| file | change |
|---|---|
| `owl/active_selection/benchmark.py` | `chain(n_tasks=None)`, `declared_classes(n_tasks=None)`, `cycle_config(..., n_tasks=None)`, `manifest(..., n_tasks=None)`; degenerate lengths refused |
| `owl/evaluation_subset.py` | `shared_test_set_name(n_tasks)` — four tasks keeps the bare name, others are suffixed; the result is passed through `check_split_name` |
| `tools/prepare_full_owod_benchmark.py` | `--n-tasks`, threaded into the split builder |
| `tools/run_full_owod_benchmark.py` | `--n-tasks`, threaded into the split, the chain, `cycle_config` and the manifest |

**`N_TASKS` stays 4 and `check_protocol()` still agrees (21 frozen fields).**
Every V1 default is unchanged and asserted by tests.

**Two provenance defects found and fixed while verifying.** The manifest recorded
`chain()` — the *four*-task chain — and the experiment name
`full_owod_active_benchmark_v1`, so a ten-task session would have written a
manifest describing the wrong experiment. It now records `chain_length`,
`full_owod_chain_t10`, `comparable_with_benchmark_v1: false` and a
`comparability_note` in words. `frozen.n_tasks` deliberately still reports **4**,
because that is the protocol's value and `check_protocol` pins it.

## 4. Why it is worth the GPU — the scientific reason

Not presentation. The new-class supply per declaring task, from the committed
candidate index:

| task | new class | band | pool objects |
|---|---|---|---:|
| t2 | traffic light | head | 6,703 |
| t3 | fire hydrant | **tail** | **997** |
| t4 | stop sign | **tail** | **1,021** |
| t5 | parking meter | **tail** | **672** |
| t6 | bench | head | 5,057 |
| t7 | chair | head | **19,584** |
| t8 | diningtable | head | 8,112 |
| t9 | pottedplant | medium | 4,489 |
| t10 | backpack | head | 4,631 |

Benchmark V1 stops at t4, so **every** incremental task it measures except t2
declares a class with ~1,000 pool objects. A weak new-class AP there cannot be
separated from "too few positives" — which is exactly the confound
`docs/new_class_instability_forensics_2026-09-05.md` identified and could not
resolve. t6–t10 declare classes with **4,489–19,584** objects. This is the first
configuration in the project where supply is *not* the binding constraint.

## 5. **T1→T10 is a SECOND EXPERIMENT, not a comparable extension**

| | Benchmark V1 | this run |
|---|---|---|
| chain | t1 → t4 | t1 → t10 |
| declared classes | 3 | 9 |
| shared split name | `owl_shared_test` | `owl_shared_test_t10` |
| **shared split size** | **837 images** | **2,817 images (3.37×)** |

Every metric — known mAP50, new-class AP50, U-Recall, forgetting, tail — is
computed on that split. **A different split means a different measurement.** So:

* **no number from this run may appear in the same table as a V1 number**, not
  even for t2/t3/t4, which exist in both chains;
* the two are reported as two experiments: one replicated over three seeds on a
  four-task chain, one complete nine-task trajectory on one seed;
* the manifest states this in machine-readable form and the notebook states it
  in the first cell a reader sees.

## 6. The banking defect on a longer chain — re-verified 2026-09-09

Independently recomputed from the committed index and the rule in
`docs/banking_defect_forensics_2026-09-04.md` §1:

| chain | exposed object-slots | recoverable | **lost** |
|---|---:|---:|---:|
| 4 tasks | 3,039 | 964 (31.7 %) | **2,075 (68.3 %)** |
| 10 tasks | **240,346** | 45,090 (18.8 %) | **195,256 (81.2 %)** |

Both figures reproduce the earlier audit exactly. The defect is **more exposed**
on a long chain, because head classes such as `chair` sit on many images that
also carry an earlier-declared class.

**What it does and does not cost.** It costs the *banked bonus* — objects bought
before their class was declared. It does **not** touch the supply delivered at a
class's own declaring task, which is the base term and is what §4's argument
rests on. It falls equally on both arms. It is **not fixed here**: fixing it is a
new protocol and would invalidate every completed trajectory.

## 7. Runtime — re-derived from the measured cost basis

`data/reference/gpu_cost_basis.json` (T4): predict 3.97 min/1000 images,
evaluate 6.46 min/1000 images + 0.3, train 0.9 s/iteration + 2.05.

**Calibration.** The model predicts **327 min** for three arms × T1→T4 against
the **331 min** actually observed — **1.2 % error**. That is what licenses the
projection.

| | optimistic (train 550 img) | pessimistic (train 800 img) |
|---|---:|---:|
| per arm-task | 49.1 min | 58.5 min |
| **1 arm, 9 tasks** | **7.4 h** | **8.8 h** |
| **2 arms** | **14.7 h** | **17.6 h** |
| 3 arms | 22.1 h | 26.3 h |

Add ~0.5–1 h once per session for candidate JPEG fetching (up to 18,000 images
per arm; `/content` does not survive a disconnect). **Two arms: ~15.5–18.5 h.**
Three arms would not leave margin in 36 hours and one disconnect cascade could
sink it — which is why `entropy` is out.

## 8. Compute

Reasoning in **T4 GPU-hours**, because Colab publishes no guaranteed
compute-unit-per-hour rate and I will not invent one: **~16–19 T4-hours**,
including one or two reconnects.

**Buy the 100-unit block.** On the commonly reported T4 consumption it is
comfortably more than 19 hours of headroom; if consumption is at the pessimistic
end it is roughly the right size. Some units will remain — that is insurance
against a mid-run disconnect, not waste. **Do not run this on the free tier:** it
will not sustain a 15-hour session.

## 9. Reproducing every number here

```bash
python -c "from owl.active_selection import benchmark as bm; print(bm.check_protocol())"
python tools/validate_notebook_freshness.py notebooks/full_owod_chain_t10.ipynb
python -m pytest tests/test_long_chain.py -q
```

## 10. Caveats to state at the consultation

1. **One seed.** No error bar. The V1 three-seed replication is the only place a
   direction has been replicated.
2. **Not comparable with V1** (§5), and not the published S-OWODB split.
3. **Banking** (§6) depresses the banked bonus, more so than at four tasks.
4. **Replay is M = 400 for all nine tasks** — ~14 exemplars per class by t10. The
   consultation asked for per-task re-allocation; this does not do it. That is
   Contribution B.
5. **Equal oracle answers is not equal gradient steps.** `training_iterations` is
   in every row and must be quoted whenever an AP difference is discussed.
6. `bear` still has 2 test objects and remains in the tail band.
