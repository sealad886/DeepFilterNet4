# `train_dynamic.py` Counter Semantics and Resume Invariants

This note documents the canonical counter model used by the MLX dynamic trainer.

## Counter definitions

- **micro-batch**: one dataloader iteration (`for batch in ...`).
- **optimizer step** (`global_step`): one parameter update.
  - With gradient accumulation, many micro-batches may map to one optimizer step.

## Epoch boundaries

Per epoch, training consumes a bounded number of micro-batches:

- `micro_batches_per_epoch = len(dataset) // batch_size`
- `epoch_target_micro_batches = min(micro_batches_per_epoch, max_train_batches)` if `max_train_batches` is set
- `train_total = epoch_target_micro_batches - resume_micro_batches`

The training loop iterates over a bounded iterator so progress cannot overshoot epoch totals.

## Checkpoint semantics

Checkpoint metadata stores:

- `batch_idx`: micro-batches completed in the current epoch (count)
- `micro_batches_completed`: same value as `batch_idx` (explicit field)
- `global_step`: optimizer steps completed
- `optimizer_steps_completed`: same value as `global_step` (explicit field)
- `counter_semantics_version = 2`

Legacy checkpoints without `counter_semantics_version` are interpreted as:

- `batch_idx` = last processed micro-batch index (0-based)
- Converted during resume to completed-count by adding 1

## Model/data checkpoint reconciliation

When resuming from an in-progress model checkpoint, model and data checkpoint positions must agree exactly:

- expected: `(epoch=start_epoch, micro_batch=resume_batch_idx)`
- actual from data checkpoint: `(epoch, batch)`

Mismatch is a hard error to prevent silent data duplication or skipping.

For epoch-boundary resumes (`epoch_end`, `best`, `final`), mid-epoch data checkpoints are ignored deterministically.

## Sync mode and counter semantics

The `sync_mode` setting (`fast`/`normal`/`debug`/`profile`) affects how often
evaluation barriers fire, but does **not** change counter semantics:

- `eval_frequency` varies by mode: `debug`→1, `profile`→5, `normal`→10, `fast`→25–50.
- All counters (`batch_idx`, `global_step`, `micro_batches_completed`) remain
  micro-batch-based regardless of sync mode.
- Checkpoint metadata and resume behavior are identical across all sync modes.
