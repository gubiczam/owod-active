# Controlled-long-tail T1 anchor training protocol

## Decision and scope

The historical S-OWODB T1 artifact is usable as a descriptive baseline, but its exact
training launch and initialization cannot be recovered. It must not be described as an
exactly reproduced anchor. The three primary anchors are therefore **controlled re-training
under a fixed reproduced recipe**.

The launcher is operationally ready for a live T4 benchmark. Full training remains
**NO-GO** until all of these gates are satisfied:

1. the notebook's immutable reviewed OWL SHA is fetched and verified;
2. one seed-0 initialization is created on the target Colab stack and its SHA-256 recorded;
3. all three conditions reference that byte-identical initialization;
4. a live T4 passes the real CUDA/compiled-MSDA training smoke for the selected condition;
5. the live benchmark fits the declared budget or the explicit overrun control is enabled.

This protocol does not authorize T2–T6 training.

## Historical artifact audit

The locally hydrated artifact is
`~/Downloads/results/SOWODB/t1.pth` (478,682,895 bytes, local mtime
2022-11-12 12:05:14) with newly computed SHA-256
`dba5390bffdfdf63058a995f241696df8d06b7fb859aecc8292d9ea02d459a22`.
No prior SHA record and no file literally named `prob_sowod_t1_final.pth` was found. Drive
aliases found during the audit were unhydrated macOS cloud placeholders and were not used as
hash evidence. No external SSD was mounted: `/Volumes` contained only `Macintosh HD`.

The checkpoint is a complete epoch checkpoint: `model` (579 tensors), `optimizer` (298
states), `lr_scheduler`, `epoch=40`, and an 85-field `args` namespace. It is not model-only.
The optimizer state contains 183,434 updates, exactly
`floor((89,490 / 4) / 5) * 41`, corroborating four ranks, batch 5 per rank, 41 epochs, and
the named official T1 train split. It contains no RNG state.

Pinned PROB's README publishes S-OWODB weights and a DINO ResNet-50 source, and
`configs/S_OWOD_BENCHMARK.sh` contains a T1 command. The local artifact is therefore most
likely a copy of the published pretrained checkpoint, but no download receipt or original
training log survives. This classification remains **INFERRED**, not proven.

### Provenance table

`PROVEN` means embedded in the artifact or directly hashed/observed. `INFERRED` means
documented by the repository and compatible with the artifact but not embedded. `UNKNOWN`
means the audit found no defensible evidence.

| Setting | Value | Status | Evidence |
|---|---|---:|---|
| Historical filename/path | Hydrated local copy at `Downloads/results/SOWODB/t1.pth`; historical PROB convention `exps/SOWODB/PROB/t1.pth` | PROVEN | Local file and pinned evaluation config; literal requested filename was not found |
| Historical SHA-256 | `dba5390b...9a22` | PROVEN | Newly computed over the hydrated 478,682,895-byte file; no older recorded hash found |
| Artifact origin | Probably official published S-OWODB weight copied locally | INFERRED | PROB README publishes S-OWODB weights; no acquisition log |
| Checkpoint contents | Full model, optimizer, StepLR, epoch, args | PROVEN | Top-level checkpoint keys |
| PROB code identity used to train it | Exact commit unknown; state is fully compatible with `4c66be1...b531` | UNKNOWN | Checkpoint predates the first public commit by about 14 days; strict state/shape audit passed on the pinned model |
| Exact launch command/config dump | Not recovered | UNKNOWN | No original log, shell history, or config dump found |
| Public S-OWODB T1 script | `configs/S_OWOD_BENCHMARK.sh` exists | PROVEN | Pinned PROB tree |
| Public-script/artifact agreement | Partial, with material contradictions | PROVEN | Script says `model_type=prob`, `obj_temp=1.3`, `PROB_V1`; artifact args say `ebdetr`, `1`, `PROB_V2` |
| Architecture | PROB Deformable-DETR, 40,841,158 parameters | PROVEN | 18 `prob_obj_head` keys and strict 579-key load into pinned `build_model(..., mode="prob")` |
| Backbone | DINO ResNet-50 | PROVEN | Checkpoint args `dino_resnet50`; compatible state keys |
| Detector initialization before T1 | No whole-detector pretrain/resume | PROVEN | Artifact args contain empty `pretrain` and `resume` |
| Historical DINO bytes | DINO self-supervised ResNet-50 expected; exact historical file hash not recoverable | UNKNOWN | README/code load the file, but checkpoint does not embed its source hash |
| Deformable-DETR detector weights | No evidence they were loaded | PROVEN | Empty whole-model pretrain/resume; backbone-only DINO path |
| Dataset/task | `OWDETR`, T1, previous 0/current 19, `num_classes=81` | PROVEN | Checkpoint args |
| T1 train split | Named `owdetr_t1_train`, 89,490 images | PROVEN | Checkpoint args plus exact optimizer-state step arithmetic; historical split-file bytes not embedded |
| T1 eval split | Named `owdetr_test` | PROVEN | Checkpoint args; historical split-file bytes are UNKNOWN |
| T1 class mapping/order | The fixed 19-class S-OWODB order listed below | PROVEN | Pinned `OWDetection` mapping selected by embedded dataset/task args |
| Input preprocessing | ImageNet normalization; max size 1333 | INFERRED | Compatible pinned dataset transforms; not stored in checkpoint args |
| Augmentation | Horizontal flip and PROB's resize/crop/resize branch | INFERRED | Compatible pinned `make_coco_transforms("train")`; exact training commit unknown |
| Optimizer | AdamW, betas `(0.9,0.999)`, epsilon `1e-8`, three parameter groups | PROVEN | Serialized optimizer state |
| Learning rates | `2e-4`; backbone `2e-5`; linear projection `2e-5` | PROVEN | Args and optimizer initial LRs |
| Weight decay | `1e-4` | PROVEN | Args and all optimizer groups |
| Batch/world size | 5 per rank, 4 ranks, effective global batch 20 | PROVEN | Args and optimizer-step arithmetic |
| Epochs | 41, completed epoch 40 | PROVEN | Args/checkpoint epoch/scheduler state |
| Scheduler | StepLR at epoch 31, gamma 0.1 | PROVEN | Args and serialized scheduler |
| Warmup | None in compatible public path | INFERRED | Pinned optimizer/scheduler construction; exact training commit unknown |
| Gradient clipping | max norm 0.1 | PROVEN | Checkpoint args |
| Matcher costs | class/bbox/GIoU = 2/5/2 | PROVEN | Checkpoint args |
| Loss weights | classification/bbox/GIoU = 2/5/2; objectness `1e-3`; focal alpha 0.25 | PROVEN | Checkpoint args |
| Objectness temperature | 1 in artifact, 1.3 in public T1 script | PROVEN | Direct contradiction between artifact args and script |
| Seed | Base seed 42 | PROVEN | Checkpoint args |
| Seed behavior | Compatible code uses `seed + rank` for Python, NumPy and torch | INFERRED | Pinned `main_open_world.py`; exact training commit unknown |
| Workers | 3 per process | PROVEN | Checkpoint args |
| Evaluation/checkpoint interval | `eval_every=5`; checkpoint every epoch, numbered around evaluations/end | INFERRED | Arg plus compatible code; exact training commit unknown |
| Exemplar flag | `exemplar_replay_selection=True`; T1 has no replay, flag controls post-training exemplar selection | PROVEN | Checkpoint args and pinned control flow |
| Resume quality | Model/optimizer/scheduler/epoch only; no RNG state | PROVEN | Serialized keys and pinned resume path |
| MSDA | Historical environment had compiled `MultiScaleDeformableAttention`; current anchor requires compiled execution | INFERRED | Public environment and old operator dispatch; no historical runtime receipt |
| Historical Python/torch/CUDA | Python 3.10.4, torch 1.12 + CUDA 11.3, torchvision 0.13 documented for repo | INFERRED | Initial public `env.yml`; not embedded in the checkpoint |
| OS/compiler | README reports Ubuntu 16.04, CUDA 11.1/11.3, GCC 5.4 for released models | INFERRED | README; not embedded in checkpoint |
| Historical logs/experiment folder | Not found locally, on Drive, or on a mounted SSD | UNKNOWN | Filesystem audit; no external SSD mounted |

The material unknowns—exact training commit/command, exact historical DINO bytes, and exact
runtime—rule out the phrase “exact historical reproduction.”

## Fixed controlled re-training recipe

The code-owned canonical representation is `owl.t1_anchor.AnchorRecipe`. Its JSON is sorted
and serialized without incidental whitespace; SHA-256 of those canonical bytes is the recipe
fingerprint. `condition` and `manifest_sha256` intentionally make the three fingerprints
different. Runtime identity and the common initialization state hash are also recorded and
must match across anchors; every fixed scientific field remains identical.

| Component | Fixed value |
|---|---|
| Code | OWL `c46ffe193c7f1ab0edc282214d720f08461736f9`; PROB `4c66be1a52cad9360e09c729e9134aba8fe0b531` |
| Conditions | LT-10, LT-50, LT-100 only |
| Architecture | PROB Deformable-DETR, `model_type=prob`, DINO ResNet-50, 100 queries, four feature levels |
| Transformer | 6 encoder + 6 decoder layers, hidden 256, FFN 1024, 8 heads, dropout 0.1; single-stage/no box refinement; auxiliary losses on |
| Classes | `PREV=0`, `CUR=19`, 81 output classes, exact order below |
| Initialization | one saved seed-0 whole-model state, epoch -1, reused byte-for-byte |
| Dataset | `OWDETR`; condition-filtered T1 XML aliases; unchanged COCO JPEG bytes |
| Duration | 41 epochs; same-epoch policy |
| Batch | 2 on one T4; `RandomSampler`, drop-last; two workers |
| Optimizer | AdamW, LR `2e-4`; backbone `2e-5`; linear projection `2e-5`; weight decay `1e-4` |
| Scheduler | StepLR, drop at 31, gamma 0.1 |
| Gradient clipping | max norm 0.1 |
| Matcher/loss | costs 2/5/2; class/bbox/GIoU weights 2/5/2; objectness `1e-3`, temperature 1; focal alpha 0.25 |
| Augmentation | horizontal flip; random resize 480–800 or resize 400/500/600 + crop 384–600 + resize; max 1333; ImageNet normalization |
| Seed | 0 for Python, NumPy and torch; `PYTHONHASHSEED=0` in the launcher environment |
| Forbidden | replay, active/exemplar selection, oversampling, class-balanced sampling, loss reweighting, tail-specific augmentation |
| Evaluation | fixed `owl_shared_test`, seed 0, max 150/class, remainder multiplier 1 |

This deliberately differs from the artifact's distributed global batch 20, seed 42 and
post-training selection flag. Batch 2 is the already exercised single-T4 setting. Those
differences are why the result is a fixed controlled re-training rather than historical
equivalence.

PyTorch's `torch.manual_seed` seeds CPU and CUDA, `RandomSampler` consumes the seeded torch
generator, and modern DataLoader workers deterministically derive Python/NumPy/torch worker
seeds from the loader seed. The initialization creator also seeds Python, NumPy and torch
before model construction. This is controlled stochasticity, not a claim of bitwise GPU
determinism: CUDA reduction/atomic kernels and library versions may remain nondeterministic.
The initialization sidecar captures Python, torch, torchvision and CUDA versions, and every
run refuses a mismatch.

### Shared initialization

The pinned current backbone source is the official DINO ResNet-50 file at
`PROB/models/dino_resnet50_pretrain.pth`, SHA-256
`156f8c4166a23dc2951ae811e39d76a06269c565932edf647c0187e65cd7aa7c`.
The initialization creator loads that backbone, constructs the exact 81-output PROB T1
detector at seed 0, and saves the entire 579-tensor model state. Random detector/head tensors
are consequently frozen into one artifact instead of being regenerated per condition.

Proposed persistent path:

`/content/drive/MyDrive/OWL/anchors/controlled_lt_v1/seed0/prob_t1_seed0_init.pth`

Its file SHA-256 is **PENDING** because creating scientific output before the reviewed OWL
commit and live Colab runtime would violate the commit gate. The adjacent
`prob_t1_seed0_init.initialization.json` records both the file SHA and a canonical tensor-state
hash. Full training cannot start without both. All three conditions must use the exact file
SHA printed by the first creation command.

## Population, budget, and class order

| Condition | Manifest scientific SHA-256 | Images | Objects | Optimizer steps at batch 2 × 41 epochs |
|---|---|---:|---:|---:|
| LT-10 | `b3c751fa...94f0f` | 37,429 | 79,233 | 767,274 |
| LT-50 | `9525f6f4...547f0f` | 35,808 | 79,233 | 734,064 |
| LT-100 | `5d5e9b22...4e2d` | 35,460 | 79,233 | 726,930 |

The LT-10/LT-100 step difference is 40,344, or 5.55% relative to LT-100. The primary policy
is **same epochs**: it preserves the recovered 41-epoch exposure schedule and gives every
selected annotation the same number of epoch-level opportunities. Same steps would change
the number of exposures per selected object by condition and would silently add a second
intervention. The step difference must be reported and may motivate a preregistered
same-step sensitivity analysis later; this launcher does not implement one.

This same-epoch design is essential to the controlled intervention. Deterministically
cycling a smaller condition to force equal optimizer steps would expose some selected
objects more often than others and change the effective sampling distribution. The launcher
therefore never shortens, repeats, or dynamically tunes the schedule to fit a Colab session.

Exact evaluator/model order:

1. aeroplane
2. bicycle
3. bird
4. boat
5. bus
6. car
7. cat
8. cow
9. dog
10. horse
11. motorbike
12. sheep
13. train
14. elephant
15. bear
16. zebra
17. giraffe
18. truck
19. person

## Isolated artifacts and invariants

The dedicated notebook uses local ephemeral storage for downloaded JPEGs and filtered XML,
and persistent Drive storage only for initialization, receipts, checkpoints, metrics and
combined reports:

```text
/content/drive/MyDrive/OWL/anchors/controlled_lt_v1/seed0/
├── prob_t1_seed0_init.pth
├── prob_t1_seed0_init.initialization.json
└── t1_anchor__lt10__seed0/                 # lt50/lt100 analogous
    ├── config.json                          # immutable scientific config
    ├── recipe.json                          # immutable recipe fingerprint
    ├── provenance.json                      # source/runtime/hash ledger
    ├── training_view.json                   # local data-root receipt
    ├── smoke/ and cuda_training_smoke.json  # exact-path 20-iteration benchmark
    ├── train/checkpoint.pth                 # full resumable epoch state
    ├── train/checkpointNNNN.pth             # periodic snapshots
    ├── t1_lt10.pth and metadata JSON        # validated final alias
    ├── anchor_bridge_metrics.json           # raw evaluator output
    ├── anchor_metrics.json and per_class.csv
    └── DONE.json                            # written last, with output hashes
```

The materializer selects individual object ordinals, removes every unselected/future object
from each copied training XML, validates all 19 achieved counts, and hashes the filtered tree
and split. It never edits the committed source archives. Evaluation XMLs and the shared split
are identical across severities. `JPEGImages` is a symlink to one verified canonical JPEG
directory; JPEGs are never rewritten. Existing workspace/checkpoint paths and known
historical workspace names are refused.

## Dedicated one-click Colab launcher

Use `notebooks/train_controlled_lt_anchors.ipynb`; the replay notebook is not modified or
repurposed. The only user-editable cell selects conditions and declares a GPU-hour budget.
Run All mounts Drive, verifies immutable OWL/PROB pins and origins, installs the minimal
Python-3.13 stack, functionally exercises COCOeval, requires compiled MSDA on a T4, downloads
and validates the exact JPEG union, creates or verifies one shared initialization, and
materializes each isolated XML view.

For every unfinished condition it then performs the exact 20-iteration CUDA benchmark and
evaluation-reload smoke before considering full training. The measured estimate is compared
with `GPU_BUDGET_HOURS`. By default, a run that cannot fit prints a budget gate and starts no
full training; `ALLOW_BUDGET_OVERRUN=True` is an explicit instruction to begin the unchanged,
resumable 41-epoch recipe. Conditions run sequentially and are evaluated immediately.

On rerun, the notebook classifies each workspace as `READY`, `INCOMPLETE RESUMABLE`,
`INCOMPLETE NON-RESUMABLE`, or `DONE`. It only resumes from a valid full epoch checkpoint,
refuses ambiguous partial state, and never relabels a benchmark or checkpoint as a final
anchor. A final comparison is written only after every requested condition has a validated
`DONE.json`.

## CUDA smoke and resume guarantee

The smoke uses 20 real drop-last training batches from the selected condition and a real
shared-evaluation subset. Pinned PROB performs strict model-state initialization load,
Deformable-DETR forward, matcher, criterion, weighted finite-loss and finite-gradient checks,
backward, gradient clipping, optimizer steps, bbox postprocessing, COCO/open-world
evaluation, checkpoint save, then actual bridge reload and evaluation of that checkpoint.
The compiled MSDA wrapper and downstream dispatch must both be true; a fallback is not
accepted for anchor training. GPU memory must be at least 14 GiB.

PROB writes `checkpoint.pth` with model, AdamW, StepLR, epoch and args after every epoch, in
the persistent Drive workspace. `--resume` restores those fields and continues at the next
epoch. It does **not** serialize Python, NumPy, torch, CUDA, sampler or DataLoader RNG state.
This is an epoch-state resume, not `RESUME-EXACT`; augmentation/order after a disconnect may
diverge from an uninterrupted run. It is nevertheless stronger than restart-from-last-model.

## Runtime estimate

The only local project measurement is the 2026-08-26 T4 pilot in
`data/reference/gpu_cost_basis.json`: batch-2 PROB training about 0.9 seconds/iteration and
evaluation about 6.462 minutes/1,000 images + 0.3 minute fixed overhead. Extrapolation to a
41-epoch T1 run is uncertain and must be replaced by the real anchor smoke throughput before
launch. The fixed shared evaluation contains 4,308 images, approximately 28.1 minutes per
evaluation. Compatible PROB evaluates at epochs 0, 1, 5, …, 40 (10 times), and the protocol
performs one final evaluation.

| Condition | Training point estimate | In-run + final evaluation | Total point estimate |
|---|---:|---:|---:|
| LT-10 | 191.8 h | 5.2 h | 197.0 h |
| LT-50 | 183.5 h | 5.2 h | 188.7 h |
| LT-100 | 181.7 h | 5.2 h | 186.9 h |

These are roughly 7.8–8.2 uninterrupted days per condition, so a normal single Colab session
cannot finish one anchor. The run is operationally NO-GO until either a measured smoke
invalidates this extrapolation, a sufficiently persistent T4 runtime is available, or a
separately reviewed distributed/step-budget protocol is approved. The previous seven-hour
incremental-chain estimate is not applicable.

## Anchor evaluation and learnability gate

Every final checkpoint is evaluated immediately on the same `owl_shared_test` split:
4,308 deterministic images, seed 0, maximum 150 per declared class, remainder multiplier 1,
split SHA-256 `f37a3bb0916dd8462fceb35f60364fed75d3a00cebd3e0ce72775dbf79d76c27`.
The evaluator writes overall T1 known mAP50, controlled head/medium/tail mAP50, and AP50 for
all 19 classes with rank, group and achieved training count. It also preserves the raw bridge
metrics. Evaluation distribution never depends on LT severity.

`anchor_metrics.json` sets `incremental_training_authorized=false`. `per_class.csv` records
the same 19-class schema, and the cross-condition tool writes `anchor_summary.csv`,
`anchor_per_class.csv`, and `anchor_comparison.json` with minima, zero classes and the pure
Spearman correlation between AP50 and log training frequency. Before T2,
compare all 57 class rows and the three overall/group summaries; report minimum, median,
maximum and exact-zero AP classes; inspect whether AP declines with frequency and whether
LT-100 tails collapsed. No post-hoc AP threshold is introduced. Degenerate LT-100 results
must stop the incremental study for an explicit decision: retain learnability collapse as
the question, preregister another severity/capacity recipe, or abandon LT-100. The code never
softens the manifest automatically.

Downstream forgetting is condition-specific only:

```text
F_abs(c,s) = AP_T1(c,s) - AP_T6(c,s)
F_rel(c,s) = F_abs(c,s) / (AP_T1(c,s) + epsilon)
```

The helper refuses cross-condition anchors and retains final AP. In particular, LT-100 T6
can never be normalized by the historical ORIGINAL T1 artifact.

## ORIGINAL recommendation

Use the historical checkpoint only as a clearly labelled descriptive S-OWODB reference. Do
not include it in numerical claims of matched training protocol, and do not launch an
ORIGINAL re-training now. ORIGINAL has 421,243 objects and natural rho about 203, so it is not
total-supervision matched and would add another approximately week-long run. The primary
causal comparison remains LT-10/LT-50/LT-100. A new ORIGINAL anchor should be considered only
as a separately budgeted descriptive study under this fixed recipe.
