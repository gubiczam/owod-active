# Controlled T1 Anchor FAST Recipe V1

`controlled_t1_anchor_fast_v1` is a fixed-compute controlled experiment. It is not intended
to reproduce the historically fully trained PROB T1 anchor, and it does not establish
convergence. Its purpose is to compare LT-10, LT-50, and LT-100 under one identical,
predeclared optimization and image-presentation budget.

The controlled imbalance is in annotated training supervision. Filtered XML views can contain
visually present but unselected instances, so this protocol does not claim to alter natural
scene prevalence.

## Budget frozen before science

The default is 12,000 optimizer updates per condition with batch size 2: exactly 24,000 image
presentations each. A real Tesla T4 benchmark runs five warm-up updates and twenty measured
updates for every condition. Each measured interval includes data loading, PROB augmentation,
model forward, matcher, criterion, weighted loss, backward, finite-gradient validation, and
AdamW step. The receipt also records peak CUDA memory, GPU, Python, Torch, Torchvision, CUDA,
condition, and recipe. All three receipts must come from one identical live runtime stack.

The immutable plan uses only these timing receipts. It budgets the slowest observed condition
for all three runs, three 30-minute final evaluations, 15 minutes of setup, and a 30-minute
safety reserve against `TOTAL_GPU_BUDGET_HOURS` (14.0 by default, 15.0 maximum). If 12,000 does
not fit, the planner rounds an equal reduced count down to a thousand-update boundary. It never
increases above 12,000 automatically, never reads AP, and never selects different counts by
condition. If fewer than 1,000 updates fit, the decision is NO-GO and training cannot start.

## Unique-image sampling and LR

For a frozen count `S`, each condition deterministically samples exactly `2*S` distinct indices
uniformly without replacement from its full condition split. The SHA-256-derived seed includes
recipe, seed 0, and condition; a full deterministic permutation is truncated to `2*S`, and
sequential pairs form batches. No image is presented twice
within a condition's FAST run. Therefore `S` is hard-capped at 17,730, half the smallest split.

The LR drop is `floor(S * 31 / 41)` completed updates. At the 12,000 default this is 9,073:
one-based updates 1–9,073 use the base LR and update 9,074 first uses the 0.1 multiplier.

## Resume, budget, and finality

`train/resume_latest.pth` is written atomically every 1,000 updates, at the global-budget stop,
and at condition completion. It contains model, optimizer, explicit scheduler, global step,
batch offset, selected-order identity, all Python/NumPy/Torch RNG states, recipe and plan
fingerprints, benchmark hash, manifest, initialization, PROB and split identities. Resume starts
at the exact next unseen batch without reshuffling or replay.

Execution order is LT-100, LT-50, LT-10. Before starting a condition, the notebook estimates
whether every unfinished condition and final evaluation still fits in the one global deadline.
If not, it starts nothing further. Evaluation is final-only on the immutable 4,308-image
`owl_shared_test`; there is no early stopping or validation-driven tuning.

States are `READY`, `TRAINING` while the process is active, `INCOMPLETE_RESUMABLE`,
`TRAINED_PENDING_EVAL`, `DONE`, or `FAILED`. DONE requires the exact frozen global step, final
checkpoint and hash, full evaluation, 19-class AP50, grouped metrics, per-class CSV, provenance,
and hash-linked DONE receipt. The descriptive comparison is withheld until all three conditions
are DONE.

FAST lives under `anchors/controlled_lt_fast_v1/seed0` with workspace names such as
`t1_anchor_fast__lt100__seed0`. Full Recipe V2 remains under its separate version, schema,
receipts, and workspaces; its reviewed implementation remains commit
`8a9c5a97d23f4532240d7be4852e3bc98dc2060b`. Both use PROB commit
`4c66be1a52cad9360e09c729e9134aba8fe0b531` and the exact shared initialization bytes.
