# Controlled S-OWODB long-tail protocol v1

## Decision gate

The deterministic dataset protocol is **GO for review and regeneration**.  The
Day-2 GPU study is **NO-GO at this commit**: it needs four provenance-complete,
condition-specific T1 anchors and a reviewed OWL commit containing this module.
The repository contains neither a recipe for retraining PROB's T1 anchor nor the
three controlled anchors.  Reusing the published natural T1 checkpoint for an
LT condition would mean that the detector had never been trained on that LT
condition, so it would not test the stated question.

`tools/prepare_longtail_no_replay.py` makes that distinction executable.  It is
read-only and refuses an execution-ready plan until every anchor exists and its
SHA-256, the manifest scientific hash, the source hash, the exact OWL commit,
and pinned PROB commit are in the run fingerprint.

## Pipeline audit

1. Canonical T1 image IDs originate in the external S-OWODB split
   `ImageSets/OWDETR/owdetr_t1_train.txt`.  The committed provenance record is
   `data/staging/owdetr_replay_manifest.json`; it records 89,490 eligible and
   retained image IDs (`max_images=0`).
2. Object annotations originate in the canonical VOC XML directory.  The exact
   XML set used here is committed as
   `data/staging/owdetr_replay_annotations.tar.gz`; its per-image T1 counts are
   committed as `data/reference/t1_replay_class_counts.json`.
3. `owl.protocol.TASK1` fixes 19 pretrained classes.  `build_chain(6)` then
   declares one evaluator-order class at each of T2--T6: traffic light, fire
   hydrant, stop sign, parking meter, and bench.
4. `CLASS_ORDER = TASK1 + TASK2 + TASK3 + TASK4`; PROB's positional evaluator
   requires this exact order.  A class cannot become known before earlier
   positions.
5. T1 counts are the number of canonical T1 objects in each source XML.  Later
   task counts come from the candidate index and the images actually purchased
   by the active annotation loop; they are endogenous outcomes, not fixed task
   splits.
6. Historical groups are read from `data/reference/class_groups.csv`.
7. Those historical groups are global train-frequency rank thirds over all 80
   classes (27/26/27 after deterministic count ordering), not T1-specific
   thirds.  Consequently only `bear` is a historical tail class within T1.
8. `Bridge.train` receives image IDs.  PROB resolves one XML per ID and trains
   on every currently known box represented by that XML.
9. The bridge's physical sampling unit is an image, but its supervision is a
   set of individual boxes.  Replay Protocol V3 already materializes
   object-filtered XML aliases to enforce an exact object memory.
10. Images are multi-object and may contain several classes and several objects
    of one class.
11. Whole-image per-class dropping therefore changes other classes at the same
    time and cannot independently attain a preregistered class schedule.
12. At T2--T6, `known_plus_selected` trains on all currently known objects in an
    opened image; future-class-only images are banked and become usable when
    their class is declared.  The no-replay arm adds no exemplar images.
13. Current-task material uses canonical annotations.  Replay uses isolated
    object aliases and never mutates the canonical XML.  The controlled T1
    protocol likewise defines a separate training-only filtered view.
14. The completed `random__none`, `random__uniform`, and
    `random__tail_favouring` workspaces, Replay Protocol V3 outputs, historical
    grouping, published T1 checkpoint, candidate/evaluation splits, and
    `owl/runner.py`, `owl/exemplars.py`, and `owl/replay.py` remain frozen.

The 4,952 committed evaluation XML IDs are disjoint from all 89,490 T1 source
IDs.  The generator verifies that fact and hashes the evaluation archive before
and after generation.  Source JPEGs are not committed: their provenance record
says they come from COCO train2017/val2017 and the Colab bootstrap fetches them.
Therefore local XML existence is fully proven; bulk JPEG availability must be
proven by the future anchor-training preflight and is not claimed here.

## Controlled population and sampling unit

The controlled population is the 19 fixed T1 classes.  These are the only
classes known at the forgetting anchor and the same population can be measured
after every later task.  Controlling T2--T6 independently would alter the
active selector's acquired data and mix causal input with an experimental
outcome.  Controlling the full candidate pool would likewise change discovery
opportunities.  Both stay frozen.

The sampling unit is an **annotated object**, selected without replacement into
a training-only filtered XML view.  The source image and source XML remain
unchanged.  Object identity is `(image_id, canonical_class_name,
within-class ordinal in source XML)`.  Objects for each class are ordered by
SHA-256 of protocol name/version, seed, class, image ID, and ordinal, so output
does not depend on Python, NumPy, filesystem, archive, or dictionary order.

This choice gives exact independent class counts and follows the established
Replay V3 object-alias semantics.  It does *not* claim to change natural scene
prevalence: an unselected object may remain visually present while absent from
the training alias.  Thus the intervention is controlled **annotated
supervision frequency**.  Whole-image sampling would preserve complete image
annotation but cannot isolate class counts in multi-class images; a hybrid
would retain the same omission issue after its filtering step while adding an
optimizer-dependent approximation.  A future scene-prevalence study should be
registered as a separate image-level protocol, not silently substituted here.

## Schedule, total supervision, and groups

Classes are ranked by decreasing original T1 object count, with evaluator order
as the tie-break.  For rank `r = 0,...,C-1`, `C=19`, and requested ratio `rho`:

```text
w_r = rho^(-r/(C-1))
n_r* = N w_r / sum_j w_j
```

Integer counts use deterministic largest-remainder rounding and must not exceed
source capacity.  `N=79,233` is the maximum common total feasible for LT-10,
LT-50, and LT-100 without oversampling.  It is therefore fixed across the three
causal severity conditions.  ORIGINAL is the complete natural T1 source and is
not downsampled.

The controlled study's primary groups are T1 rank thirds: ranks 0--6 are head
(7 classes), 7--12 medium (6), and 13--18 tail (6).  Historical global groups
are preserved and may be reported as secondary metadata only.

| Rank | Class | Group | Original | LT-10 | LT-50 | LT-100 |
|---:|---|---|---:|---:|---:|---:|
| 0 | person | head | 262465 | 10432 | 15730 | 18025 |
| 1 | car | head | 43867 | 9179 | 12658 | 13956 |
| 2 | bird | head | 10806 | 8077 | 10185 | 10806 |
| 3 | boat | head | 10759 | 7107 | 8195 | 8367 |
| 4 | truck | head | 9973 | 6254 | 6595 | 6478 |
| 5 | sheep | head | 9509 | 5503 | 5306 | 5016 |
| 6 | motorbike | head | 8725 | 4842 | 4270 | 3883 |
| 7 | cow | medium | 8147 | 4261 | 3436 | 3007 |
| 8 | bicycle | medium | 7113 | 3749 | 2765 | 2328 |
| 9 | horse | medium | 6587 | 3299 | 2225 | 1802 |
| 10 | bus | medium | 6069 | 2903 | 1790 | 1396 |
| 11 | elephant | medium | 5513 | 2554 | 1440 | 1081 |
| 12 | dog | medium | 5508 | 2248 | 1159 | 837 |
| 13 | zebra | tail | 5303 | 1978 | 933 | 648 |
| 14 | aeroplane | tail | 5135 | 1740 | 750 | 501 |
| 15 | giraffe | tail | 5131 | 1531 | 604 | 388 |
| 16 | cat | tail | 4768 | 1347 | 486 | 301 |
| 17 | train | tail | 4571 | 1186 | 391 | 233 |
| 18 | bear | tail | 1294 | 1043 | 315 | 180 |

| Condition | Requested rho | Achieved rho | Images | Objects | max | min | H/M/T |
|---|---:|---:|---:|---:|---:|---:|---:|
| ORIGINAL | natural | 202.832303 | 89,490 | 421,243 | 262,465 | 1,294 | 7/6/6 |
| LT-10 | 10 | 10.001918 | 37,429 | 79,233 | 10,432 | 1,043 | 7/6/6 |
| LT-50 | 50 | 49.936508 | 35,808 | 79,233 | 15,730 | 315 | 7/6/6 |
| LT-100 | 100 | 100.138889 | 35,460 | 79,233 | 18,025 | 180 | 7/6/6 |

ORIGINAL is an ecological reference, not the balanced endpoint of the schedule:
its natural ratio is already about 203 and it contains 5.32 times as many
objects as a controlled condition.  Therefore ORIGINAL-versus-LT differences
are descriptive and confounded by total supervision and distribution shape.
The preregistered primary severity contrast is matched-total LT-10 -> LT-50 ->
LT-100.  A later matched-total natural-shape control can separate “exponential
schedule” from “natural shape”, but must receive its own manifest and name.

Object matching does not exactly match images: LT-10 uses 37,429 images and
LT-100 35,460, a difference of 1,969 (5.26% of the LT-10 count), because the
selected objects have different image co-occurrence/density.  Fixed object
total is the fairest primary control because object frequency is the manipulated
quantity, but image-context/negative exposure remains a preregistered residual
confound.  Matching both totals is generally impossible without duplicating
images, discarding additional objects, or changing the target schedule.

## Reproducibility artefacts

`tools/build_longtail_splits.py` verifies the source index against all source
XMLs, train/evaluation ID disjointness, exact positive monotonic counts, unique
object identities, source hashes before/after, and deterministic ordering.  It
writes:

- four human-readable manifests with a scientific-content SHA-256;
- four deterministic gzip object ledgers (blank filename, gzip mtime 0);
- `summary.csv`, `per_class.csv`, and `frequency_curves.png`.

Re-running with the same sources, protocol, and seed produces identical
manifest and ledger bytes.  Plot bytes are diagnostic, not part of a scientific
fingerprint.

## Day-2 no-replay matrix and gate

Every run freezes the completed pilot settings: six tasks, 600-region budget,
six rounds, 2,000 candidate images/task, 50 proposals/image, random selection,
`known_plus_selected`, no replay, no replay reallocation, Replay Protocol V3
marker, five epochs, learning rate 2e-4, batch size 2, 1,600 clusters, grouped
recall enabled, and seed 0.

| Condition | Workspace | Required start anchor |
|---|---|---|
| ORIGINAL | `random__none__original` | `t1_original.pth` |
| LT-10 | `random__none__lt10` | `t1_lt10.pth` |
| LT-50 | `random__none__lt50` | `t1_lt50.pth` |
| LT-100 | `random__none__lt100` | `t1_lt100.pth` |

Seeds 1 and 2 are prepared by appending `__seed1` / `__seed2`; they are not to
be launched until seed 0 validates the protocol.  An execution fingerprint is
the full historical `CycleConfig` result-affecting fingerprint plus LT protocol
version, condition, manifest scientific hash, source-annotation SHA-256,
condition-specific anchor SHA-256, exact OWL commit, and exact PROB commit
`4c66be1a52cad9360e09c729e9134aba8fe0b531`.

Exact execution fingerprints intentionally do not exist yet: the anchors and
reviewed OWL SHA are missing.  Dataset scientific hashes do exist and are:

| Condition | Manifest scientific SHA-256 |
|---|---|
| ORIGINAL | `f25ae1b235f87cefe2044e81ca6753cc46bd5b81b3085b019459de4a8113b032` |
| LT-10 | `b3c751fa1034a499d592391d87afc145b0cc8b11bf4a255512fd4f52ca094f0f` |
| LT-50 | `9525f6f40958c1282b739a79c6c196d4c7942e35dc605a777a0f45b245547f0f` |
| LT-100 | `5d5e9b2287c97748c135f4412696201449b7086683f58facc876c3c62d8a4e2d` |

Protocol-only audit (expected to report the gate, without writing anything):

```bash
python tools/prepare_longtail_no_replay.py \
  --anchor-root /content/drive/MyDrive/OWL/checkpoints/SOWODB/longtail \
  --work-root /content/drive/MyDrive/OWL/work \
  --protocol-only
```

After anchor training is independently specified, reviewed, and completed, omit
`--protocol-only` and pass the reviewed commit with `--owl-commit`.  The command
then produces four exact fingerprints and refuses mismatched existing
workspaces.  It still launches nothing.

The existing notebook budget is 420 minutes per chain.  Four seed-0 chains are
therefore bounded at 28 T4 GPU-hours; the three matched-total LT chains are
bounded at 21 hours.  ORIGINAL may reuse historical `random__none` only after
its checkpoint SHA, full config, evaluation split, code commits, and manifest
semantics are proven identical.  The names and current provenance differ, so
the default is no reuse.

## Metrics and interpretation

For every condition report task-level known mAP50, previous-class mAP50,
new-class AP50, forgetting, and U-Recall; controlled head/medium/tail AP50 and
forgetting; and per-class anchor/final AP50, absolute/relative forgetting,
original/controlled frequency, rank, and group.  Required figures are the
committed frequency curves, severity versus final group AP50, severity versus
group forgetting, per-class forgetting versus log controlled frequency, and
anchor AP versus forgetting.

- **Support:** across matched-total LT-10, LT-50, LT-100, tail forgetting rises
  monotonically (with corresponding retention loss) while anchor quality and
  head behavior do not explain the trend.
- **Partial support:** only some rare classes worsen, or the aggregate tail
  trend is non-monotonic but frequency remains associated with forgetting after
  accounting for anchor AP.  This supports class-specific vulnerability, not a
  universal rarity rule.
- **Rejection:** no reproducible severity/frequency effect across seeds after
  controlling anchor AP, or head and tail degrade similarly.  Strong head loss
  would instead indicate broad distribution/anchor-quality effects.

No result should be forced into the supporting category.  Anchor AP must be
reported because a class that begins poorly cannot supply the same retention
evidence as one that begins well.
