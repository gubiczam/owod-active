# Controlled T1 anchor training protocol — Recipe V2

Recipe V2 (`controlled_t1_anchor_v2`) is the only scientific anchor-training path. The
legacy dataset-length Recipe V1 cannot validate or produce final V2 metadata.

## Fixed scientific budget

Every condition (`lt10`, `lt50`, `lt100`) uses the same seed-0 initialization, PROB commit
`4c66be1a52cad9360e09c729e9134aba8fe0b531`, AdamW configuration, loss, PROB augmentation,
shared 4,308-image `owl_shared_test`, and these exact budgets:

| Quantity | Value |
|---|---:|
| Optimizer updates | 183,434 |
| Reference epochs | 41 |
| Updates/reference epoch | 4,474 |
| Batch size | 2 |
| Images/reference epoch | 8,948 |
| Total image presentations | 366,868 |
| LR drop after completed updates | 138,694 |
| Seed | 0 |

The condition manifest and its filtered supervision distribution are the only
condition-dependent scientific inputs.

## Sampling and global step

`global_step` is the number of completed optimizer updates and ranges from 0 through
183,434. Before the next update:

```text
reference_epoch = global_step // 4474
step_in_reference_epoch = global_step % 4474
```

At each reference epoch, a SHA-256-derived seed over recipe version, seed, condition, and
reference epoch drives Python's deterministic `random.sample`. It selects and orders exactly
8,948 unique indices without replacement from the condition's real split. Consecutive pairs
are batches. A restart reconstructs the same selection and starts at the saved batch offset;
it never replays the beginning of a reference epoch.

The data loader uses zero workers. This deliberately keeps PROB's unchanged random transform
stream in the main process so the saved Python, NumPy, Torch CPU, and Torch CUDA RNG states
make a mid-epoch resume exact.

## LR boundary

The scheduler is explicit in update space. One-based updates 1–138,694 use the base LR.
After 138,694 updates have completed, the multiplier becomes 0.1, so the lower LR first
applies to one-based update **138,695** (zero-based update index 138,694). A targeted unit test
guards this boundary.

## Checkpoints, sessions, and finality

`train/resume_latest.pth` is written every 1,000 updates, at every reference-epoch boundary,
on the budget soft stop, and on completion. The complete temporary file is loaded and its
identity checked before atomic replacement, so an interrupted write does not replace the last
known-good checkpoint. It stores model, optimizer, explicit scheduler state, global position,
sampling identity, all RNG states, hashes, condition, seed, recipe fingerprint, and PROB args.

The Colab notebook defaults to an 8.5-hour session and stops ten minutes early for checkpoint
and Drive flush. An unfinished session is `INCOMPLETE RESUMABLE`. Re-running all cells resumes
from Drive. No intermediate full evaluation or early stopping is used.

Run one condition at a time in the preregistered operational order LT-100, LT-10, LT-50. The
published notebook therefore starts with `CONDITIONS = ["lt100"]`.

A condition becomes `DONE` only after all 183,434 updates, final checkpoint publication and
hashing, the full shared-test evaluation, `anchor_metrics.json`, `per_class.csv`, provenance,
and the hash-linked `DONE.json`. The final report contains overall, head, medium, tail, and all
19 per-class AP50 values plus descriptive Spearman correlation with log training frequency.

## Live benchmark and ETA

Each condition's CUDA preflight uses five warm-up updates followed by 20 measured updates on
the real filtered LT batch and executes the real model forward, matcher, criterion, weighted
loss, backward pass, gradient validation, and AdamW step. Receipts are stored in each workspace
as `cuda_training_smoke_v2.json`; a combined `live_benchmarks_v2.json` persists the exact
seconds/update values. Training-only ETA is `183434 * seconds_per_optimizer_update / 3600`;
checkpointing, data materialization, and the one mandatory final evaluation are reported as
additional overhead.
