# Remaining experiment matrix

2026-09-04. Runtime from `tools/plan_full_owod_benchmark.py`, priced on the
project's own measured basis (`data/reference/gpu_cost_basis.json`) with a
deliberately conservative DINOv2 crop rate. All figures T4.

---

## 1. What is already done, and what it costs to touch it

| trajectory | status | cost to re-enter |
|---|---|---|
| `random__seed0` | COMPLETE | seconds — restored task-by-task from `state.json` |
| `admissibility__seed0` | COMPLETE | seconds |
| `proposed__seed0` | COMPLETE | seconds |
| `entropy__seed0` | COMPLETE | seconds |
| `coreset__seed0` | **attempted, incomplete — CUDA OOM, no endpoint** | see §5 |
| `proposed_v2__seed0` | not run | 2.33 h |

A completed task is one with both `state.json` and `metrics.json`; the launcher
prints `already done; restored from state.json` and makes **no** detector call —
no predict, no train, no evaluate, no image fetch. Their configuration
fingerprints differ from each other and from `proposed_v2` in the `arm` field
alone, so nothing about them is at risk from a new arm.

---

## 2. Per-trajectory cost

| arm | DINOv2 crops/task | predict | DINO | train | eval | per task | per trajectory |
|---|---:|---:|---:|---:|---:|---:|---:|
| `random` | — | 7.9 | — | 25.6 | 11.1 | 44.7 min | **2.23 h** |
| `admissibility` | — | 7.9 | — | 25.6 | 11.1 | 44.7 min | **2.23 h** |
| `entropy` | — | 7.9 | — | 25.6 | 11.1 | 44.7 min | **2.23 h** |
| `proposed_v2` | 12 000 | 7.9 | 2.0 | 25.6 | 11.1 | 46.7 min | **2.33 h** |
| *`proposed` (done)* | *24 000* | *7.9* | *4.0* | *25.6* | *11.1* | *48.7 min* | *2.43 h* |
| *`coreset` (OOM)* | *80 000* | *7.9* | *13.3* | *25.6* | *11.1* | *58.0 min* | *2.90 h* |

Add ~5 minutes per session for re-fetching the 837 shared evaluation images into
a fresh `/content`.

---

## 3. The matrix

### Mandatory — 6 trajectories, 13.40 h

| arm | seeds | trajectories | GPU |
|---|---|---:|---:|
| `random` | 1, 2 | 2 | 4.47 h |
| `admissibility` | 1, 2 | 2 | 4.47 h |
| `entropy` | 1, 2 | 2 | 4.47 h |
| | | **6** | **13.40 h** |

### Decision gate — 1 trajectory, 2.33 h

| arm | seed | trajectories | GPU |
|---|---|---:|---:|
| `proposed_v2` | 0 | 1 | 2.33 h |

### Conditional — 2 trajectories, 4.66 h

Run **only** if `proposed_v2` seed 0 satisfies the frozen kill rule: mean
`new_class_AP50` ≥ **3.56** *and* final `known_mAP50` ≥ **44.89**. The
summariser prints the verdict mechanically. On `STOP`, preserve the negative
result, do not tune, do not run these.

| arm | seeds | trajectories | GPU |
|---|---|---:|---:|
| `proposed_v2` | 1, 2 | 2 | 4.66 h |

### Totals

* if the kill rule says **STOP**: **7 trajectories, 15.73 h**
* if it says **PROCEED**: **9 trajectories, 20.39 h**

---

## 4. Recommended session grouping

One `(seeds, arms)` combination per session. Mixing seeds in one launcher call
runs the cross product, which would start `proposed_v2` seeds 1–2 before the
verdict exists.

| session | `SEEDS` | `SESSION_ARMS` | GPU | why grouped this way |
|---|---|---|---:|---|
| **1** | `(0,)` | `(..., "proposed_v2")` | **2.4 h** | the decision gate, alone and cheap |
| **2** | `(1,)` | `("random", "admissibility", "entropy")` | **6.9 h** | one seed of the mandatory baselines |
| **3** | `(2,)` | `("random", "admissibility", "entropy")` | **6.9 h** | the other seed |
| **4** *(conditional)* | `(1, 2)` | `("proposed_v2",)` | **4.8 h** | only after a `PROCEED` |

Sessions 1 and 2 fit in one sitting (9.3 h) if you run all, then edit `SEEDS`
to `(1,)` and re-run cells 8–10. Sessions 2 and 3 cannot be merged: 13.8 h
exceeds a Colab session.

**Priority if time runs short:** sessions 2 and 3 before session 4. Two
replication seeds on three baselines is a defensible result; a third seed on a
development-seed-informed method is not.

---

## 5. Exact CLI

The notebook is the supported path; these are the equivalents. `$C` is the
common prefix:

```bash
C="--prob-root /content/PROB --data-root /content/data/OWOD --checkpoint /content/drive/MyDrive/OWL/checkpoints/SOWODB/t1.pth --ref-t1 /content/drive/MyDrive/OWL/features/ref_t1_dinov2_vitb14_cap1000_v1.npz --out /content/drive/MyDrive/OWL/results/full_owod_active_benchmark_v1"
```

Session 1 — the decision gate:

```bash
python tools/run_full_owod_benchmark.py $C --seeds 0 --arms proposed_v2 --time-budget-minutes 200
```

Session 2 and session 3 — the mandatory baselines:

```bash
python tools/run_full_owod_benchmark.py $C --seeds 1 --arms random admissibility entropy --time-budget-minutes 480
```

```bash
python tools/run_full_owod_benchmark.py $C --seeds 2 --arms random admissibility entropy --time-budget-minutes 480
```

Session 4 — conditional, only after a `PROCEED`:

```bash
python tools/run_full_owod_benchmark.py $C --seeds 1 2 --arms proposed_v2 --time-budget-minutes 360
```

After every session:

```bash
python tools/summarize_full_owod_benchmark.py --results /content/drive/MyDrive/OWL/results/full_owod_active_benchmark_v1
```

```bash
python tools/plot_full_owod_benchmark.py --results /content/drive/MyDrive/OWL/results/full_owod_active_benchmark_v1
```

---

## 6. Resume and skip behaviour

**Skips entirely, no detector call:** any `(arm, seed)` whose every task has
`state.json` and `metrics.json`. Naming a completed arm in `--arms` is free and
is the normal way to keep the manifest complete — the manifest **merges**, so a
later session never erases an earlier one's entries.

**Resumes at the task it died on:** a trajectory whose later tasks are absent.
The checkpoint lineage is fail-closed — a task refuses to train unless the
checkpoint it extends is the one the previous task wrote, so a resumed chain
cannot silently restart from the anchor.

**Refused rather than blended:** a workspace stamped with a different
configuration fingerprint. The launcher stops and names the differing fields.

**`coreset__seed0` needs a decision before it is ever re-entered.** Its
workspace may hold a partially written task from the OOM. It is deliberately
absent from `SESSION_ARMS`; do not add it back casually. If it is retried, first
inspect `coreset__seed0/` for a task directory with a `state.json` but no
`metrics.json` and remove that directory, and expect the memory fix in
`owl.active_selection.semantic.release` plus
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to be what makes 80 000
crops per task survivable.

**Never skipped:** the shared 837-image evaluation fetch at session start, which
is a validating no-op when the files are already present, and hard-fails if any
are missing — the split is frozen, so a missing test image changes what every
arm is scored on.
