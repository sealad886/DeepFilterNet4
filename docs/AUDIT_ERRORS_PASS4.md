# Correctness/Errors Audit — Pass 4

**Scope:** `DeepFilterNet/df_mlx/` (66+ Python source files)  
**Date:** 2026-02-19  
**Auditor:** Checker agent (Pass 4)  
**Status:** FINDINGS REPORTED

---

## Methodology

Systematic line-by-line reading of all high-criticality files:
- `train_dynamic.py` (4415 lines — complete)
- `training_losses.py`, `training_checkpoints.py`, `training_ops.py`, `training_session.py`, `training_signals.py`, `training_waveform.py` (complete)
- `dynamic_dataset.py` (1728 lines — complete)
- `loss.py` (1031 lines — complete)
- `ops.py`, `grad_utils.py`, `feature_ops.py`, `lr.py`, `discriminator.py`, `enhance.py`, `evaluation.py`, `modules.py` (complete or substantially read)

Focus: runtime exceptions, incorrect control flow, resource leaks, race conditions, incorrect math, signature mismatches, concurrency bugs, and logic errors NOT previously reported in Passes 1–3.

---

## Previously Fixed (NOT re-reported)

| Pass | Finding | Status |
|------|---------|--------|
| P2 | Atomic checkpoint writes | FIXED |
| P2 | float16 → float32 for loss computation | FIXED |
| P3 | ZeroDivisionError in metric averaging | FIXED |
| P3 | Overwrite guard in `save_audio` | FIXED |
| P3 | Temp file cleanup in checkpoints | FIXED |
| Audit-2025-01 | Mask saturation penalty inverted | FIXED |
| Audit-2025-01 | Sigma variance floor missing | FIXED |
| Audit-2025-01 | Single-frame edge cases | FIXED |
| Audit-2025-01 | Empty array mean | FIXED |

---

## New Findings

### AUDIT-P4-P1-001: Interfering speaker same-file guard compares wrong indices

**Severity:** P1 (correctness — silent data corruption)  
**File:** [dynamic_dataset.py](../DeepFilterNet/df_mlx/dynamic_dataset.py#L1064-L1068)  
**Lines:** 1064–1068

**Problem:**  
The interfering speaker augmentation generates a random index and compares it against the wrong value for same-file exclusion:

```python
interfer_idx = rng.randint(0, len(self._indices) - 1)       # raw position
if interfer_idx != self._indices[idx]:                        # resolved file index
    interfer_speech = self._load_speech(interfer_idx, rng)    # raw position used
```

`interfer_idx` is an index *into* `self._indices` (i.e., a position in the shuffled index array), but `self._indices[idx]` is the *resolved file index* at position `idx`. These are from different domains. The comparison should be `interfer_idx != self._indices[idx]` → `self._indices[interfer_idx] != self._indices[idx]` (comparing resolved file indices), or the raw index should be used consistently.

**Impact:**  
- The same-file exclusion guard is unreliable. The target speech uses `self._indices[idx]` as a file index, but the interferer uses `interfer_idx` directly as a file index. So the guard compares a random position (0–N) against a shuffled file index — they may coincidentally match even when the files differ, and may differ even when the same file would be loaded.
- Net effect: the target speaker can interfere with *itself*, violating the training objective of having a distinct interferer.

**Fix:**
```python
interfer_idx = rng.randint(0, len(self._indices) - 1)
interfer_file_idx = self._indices[interfer_idx]          # resolve to file index
if interfer_file_idx != self._indices[idx]:               # compare file indices
    interfer_speech = self._load_speech(interfer_file_idx, rng)
```

Alternatively, sample directly from the file index space and skip the indirection:
```python
interfer_file_idx = rng.randint(0, len(self.splits[self._current_split]) - 1)
if interfer_file_idx != self._indices[idx]:
    interfer_speech = self._load_speech(interfer_file_idx, rng)
```

**Test:** Unit test creating a DynamicDataset with 2 speech files, fixed seed, verify that when `idx` selects file 0 and random picks position that resolves to file 0, the interferer is skipped.

---

### AUDIT-P4-P1-002: Signal handler performs non-async-signal-safe I/O

**Severity:** P1 (correctness/safety — potential deadlock or corruption)  
**File:** [training_signals.py](../DeepFilterNet/df_mlx/training_signals.py)  
**Also:** [train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py#L1079-L1082)

**Problem:**  
`_handle_sigint` (registered via `signal.signal(SIGINT, ...)`) calls `save_checkpoint()` and `train_stream.save_checkpoint()`. These functions:
- Open files for writing (`open()`, `safetensors.save_file()`)
- Create temporary files and do `os.replace()`
- Serialize model state to JSON
- Access `train_stream._checkpoint` and `train_stream._batch_count` which may be concurrently modified by the data loader thread

Signal handlers in Python should only set flags or call `os.write()` to a pipe. While CPython's GIL provides some protection, the handler can interrupt the main thread mid-checkpoint-write, causing:
1. **Corruption:** If the main thread is already writing a checkpoint, the signal handler writes a second one simultaneously with shared mutable state.
2. **Deadlock:** If the main thread holds a lock (e.g., inside `mx.eval`), and the handler tries to acquire the same lock.
3. **Race on data stream state:** `_batch_count` is read without synchronization while the data thread may be incrementing it.

**Impact:** Graceful shutdown under CTRL+C can corrupt checkpoints or hang. Risk is low in practice (Python GIL serializes most code), but the window exists and increases with concurrent data loading.

**Fix:**  
Replace direct checkpoint-save in the signal handler with a flag-based approach:

```python
_interrupt_requested = False

def _handle_sigint(signum, frame):
    global _interrupt_requested
    _interrupt_requested = True
    print("\n⚠️ Interrupt received. Will save checkpoint at next sync point...")

# In training loop, after each sync point:
if _interrupt_requested:
    save_checkpoint(...)
    if train_stream is not None:
        train_stream.save_checkpoint(data_checkpoint_path)
    sys.exit(0)
```

This is the standard pattern for safe signal handling.

---

### AUDIT-P4-P2-001: PrefetchDataLoader silently drops errors until iteration completes

**Severity:** P2 (correctness — delayed error propagation)  
**File:** [dynamic_dataset.py](../DeepFilterNet/df_mlx/dynamic_dataset.py#L1239-L1256)  
**Lines:** 1239–1256

**Problem:**  
When `strict_failures=True` (the default), if a worker thread hits an exception:

```python
worker_errors.append(RuntimeError(...))
return  # Worker exits
```

The worker appends to `worker_errors` and returns. But `worker_errors` is only checked *after* the main iterator's `while True: batch = prefetch_queue.get()` loop completes (line 1266). The worker does call `_queue_put(None)` in its `finally` block, which eventually breaks the main loop. However:

1. Errors are only raised after all already-queued batches are consumed and the None sentinel arrives.
2. If the failure happens early and `prefetch_factor` is large, multiple stale batches may still be processed with a silently-invalidated data pipeline before the error surfaces.
3. If `_queue_put(None)` in the `finally` block fails (stop_event already set), the error is lost entirely.

**Impact:** Training may proceed with a partial dataset after an early data failure, producing subtly wrong metrics before the error finally propagates. The training results from those intermediate batches are wasted.

**Fix:**  
Check `worker_errors` after *every* batch retrieval:

```python
while True:
    batch = prefetch_queue.get()
    if batch is None:
        break
    if worker_errors:
        raise worker_errors[0]
    yield batch
```

---

### AUDIT-P4-P2-002: `compiled_step` with gradient accumulation skips `mx.eval` on non-update batches

**Severity:** P2 (performance/correctness — unbounded lazy graph accumulation)  
**File:** [train_dynamic.py](../DeepFilterNet/df_mlx/train_dynamic.py#L3205-L3212)  
**Lines:** 3205–3212

**Problem:**  
In the compiled path with `grad_accumulation_steps > 1`:

```python
if should_sync:
    if did_optimizer_update:
        mx.eval(loss, model.parameters(), optimizer.state)
    else:
        mx.eval(loss)
```

When `should_sync` is True but `did_optimizer_update` is False (i.e., we're mid-accumulation window), only `mx.eval(loss)` is called. But the accumulated gradients (`accumulated_grads`) are never materialized until the accumulation window completes. With high `grad_accumulation_steps` (e.g., 8–16) and `eval_frequency > 1`, the lazy computation graph grows linearly with the number of accumulated micro-batches, potentially consuming excessive memory.

More critically, if `should_sync` is False for several consecutive micro-batches (when `eval_frequency > 1`), *no* sync happens at all during accumulation, and the graph grows unbounded for `eval_frequency * grad_accumulation_steps` iterations.

**Impact:** OOM during gradient accumulation with high `eval_frequency` settings. The memory usage scales as O(eval_frequency × grad_accumulation_steps × model_size) because each micro-batch's full forward+backward graph is retained.

**Fix:**  
Add a periodic sync within the accumulation window, or force `eval_frequency=1` when `grad_accumulation_steps > 1` in compiled mode:

```python
# After each compiled_loss_and_grad_step, sync the loss at minimum:
mx.eval(loss)
```

Or document that `eval_frequency` must be 1 when using gradient accumulation in compiled mode.

---

### AUDIT-P4-P2-003: `CosineScheduler.load_state_dict` replays all steps sequentially (O(N) resume)

**Severity:** P2 (performance — slow checkpoint resume)  
**File:** [lr.py](../DeepFilterNet/df_mlx/lr.py#L241-L249)  
**Lines:** 241–249

**Problem:**

```python
def load_state_dict(self, state: dict) -> None:
    ...
    target_global_step = state["current_epoch"] * self.steps_per_epoch + state["current_step"]
    for _ in range(target_global_step):
        try:
            next(self._generator)
        except StopIteration:
            break
```

Resuming from a checkpoint at step 500,000 calls `next()` 500,000 times. This is O(N) in the number of completed steps and takes several seconds for long training runs.

**Impact:** Slow checkpoint resume for long training runs. Not a correctness bug, but a performance issue that grows linearly with training duration.

**Fix:**  
Use the closed-form cosine schedule formula to compute the LR for any step directly instead of replaying the generator:

```python
def load_state_dict(self, state: dict) -> None:
    # Restore parameters
    ...
    # Recreate generator and advance to saved position
    self._generator = cosine_scheduler(...)
    target = state["current_epoch"] * self.steps_per_epoch + state["current_step"]
    # Skip the generator ahead by dropping elements
    # Or better: compute LR analytically and set state
    self.current_lr = self._compute_lr_at_step(target)
```

---

### AUDIT-P4-P2-004: `weight_norm_conv1d` ignores `dilation` and `groups` parameters

**Severity:** P2 (correctness — API contract violation)  
**File:** [discriminator.py](../DeepFilterNet/df_mlx/discriminator.py#L63-L80)  
**Lines:** 63–80

**Problem:**

```python
def weight_norm_conv1d(
    in_channels, out_channels, kernel_size,
    stride=1, padding=0, dilation=1, groups=1,
) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        # dilation and groups are silently dropped!
    )
```

The function accepts `dilation` and `groups` parameters but never passes them to `nn.Conv1d`. Any caller relying on dilation or grouped convolution would get incorrect results silently.

Similarly, `weight_norm_conv2d` accepts `stride` and `padding` tuples but the actual parameters may not be forwarded correctly to MLX's `nn.Conv2d` depending on whether MLX supports tuple padding.

**Impact:** Currently no callers pass non-default `dilation` or `groups`, so the bug is latent. But the API is misleading and any future use of these parameters would fail silently.

**Fix:**

```python
return nn.Conv1d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    # groups not supported in MLX Conv1d — raise if non-default
)
if groups != 1:
    raise NotImplementedError("MLX Conv1d does not support groups != 1")
```

---

### AUDIT-P4-P3-001: `ShardedAudioCache._get_shard` has TOCTOU race with double load

**Severity:** P3 (concurrency — redundant work, wasted memory)  
**File:** [dynamic_dataset.py](../DeepFilterNet/df_mlx/dynamic_dataset.py)  

**Problem:**  
The shard loading pattern is:

```python
with self._lock:
    if shard_key in self._loaded_shards:
        return self._loaded_shards[shard_key]

# Load shard outside lock (expensive I/O)
data = np.load(shard_path)

with self._lock:
    if shard_key in self._loaded_shards:
        return self._loaded_shards[shard_key]  # Another thread loaded first
    self._loaded_shards[shard_key] = data
```

Two threads requesting the same cold shard simultaneously will both load it from disk because the first lock block releases before loading. The second lock block's re-check prevents *cache corruption*, but the redundant load wastes I/O and memory during the load period.

**Impact:** Minor — duplicated disk reads on cold cache starts with concurrent workers. No data corruption. The double-check-locking pattern used here is actually reasonable for the use case (I/O outside lock, re-verify inside lock). The cost is bounded and transient.

**Fix (optional):**  
Use a per-shard lock or `threading.Lock` per shard key to ensure only one thread loads a given shard:

```python
# Use a dictionary of per-shard events
if shard_key not in self._shard_events:
    self._shard_events[shard_key] = threading.Event()
```

Low priority given the bounded cost.

---

### AUDIT-P4-P3-002: `ASRLoss.__init__` raises but `_lazy_init` and `__call__` have dead code

**Severity:** P3 (dead code — maintenance hazard)  
**File:** [loss.py](../DeepFilterNet/df_mlx/loss.py#L882-L927)  
**Lines:** 882–927

**Problem:**  
`ASRLoss.__init__` unconditionally raises `NotImplementedError`, making the `_lazy_init` and `__call__` methods unreachable dead code. The methods reference `self._initialized`, `self.model`, and `self.whisper` which are never set by `__init__`.

If someone later removes the `raise` to enable the class, it would crash immediately because `self._initialized`, `self.model`, etc. are not initialized.

**Impact:** No runtime effect (class can't be instantiated), but the dead code creates a false sense of implementation readiness and will crash if un-guarded.

**Fix:**  
Either remove the dead methods entirely (mark as TODO stub), or add proper `__init__` attribute initialization before the `raise`:

```python
def __init__(self, ...):
    self.factor = factor
    self.factor_lm = factor_lm
    self.model = model
    self._initialized = False
    self.whisper = None
    raise NotImplementedError("MLX ASRLoss is not yet implemented.")
```

---

## Summary

| ID | Severity | File | Issue | Type | Status |
|----|----------|------|-------|------|--------|
| P4-P1-001 | P1 | dynamic_dataset.py:1064 | Interfering speaker same-file guard compares wrong index domains | Logic bug | OPEN |
| P4-P1-002 | P1 | training_signals.py | Signal handler does I/O (not async-signal-safe), risk of deadlock/corruption | Safety | OPEN |
| P4-P2-001 | P2 | dynamic_dataset.py:1239 | Worker errors propagated only after all queued batches consumed | Error handling | OPEN |
| P4-P2-002 | P2 | train_dynamic.py:3205 | Compiled gradient accumulation may cause unbounded lazy graph | Memory/perf | OPEN |
| P4-P2-003 | P2 | lr.py | O(N) step replay on checkpoint resume | Performance | **FIXED** (859759c) |
| P4-P2-004 | P2 | discriminator.py | `dilation`/`groups` params silently dropped | API contract | **FIXED** (859759c) |
| P4-P3-001 | P3 | dynamic_dataset.py | Shard double-load race in thread-safe cache | Concurrency | OPEN |
| P4-P3-002 | P3 | loss.py:882 | Dead code in `ASRLoss` after unconditional raise | Dead code | OPEN |

**P1:** 2 findings (1 correctness, 1 safety) — both OPEN  
**P2:** 4 findings — 2 FIXED, 2 OPEN  
**P3:** 2 findings — both OPEN  
**Total:** 8 findings, 2 fixed, 6 open
